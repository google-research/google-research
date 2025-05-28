# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline for CelebA."""
import abc
import dataclasses
from typing import Callable, Sequence, Optional

from absl import logging

from clu import preprocess_spec
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

Features = preprocess_spec.Features


class ImagePreprocessOp(abc.ABC, preprocess_spec.PreprocessOp):
  """Base class for all image preprocess ops."""

  image_key: str = "image"  # tf.float32 in [0, 1]

  def __call__(self, features):
    features[self.image_key] = self.apply(features[self.image_key])
    return features

  @abc.abstractmethod
  def apply(self, image):
    """Returns transformed image."""


@dataclasses.dataclass(frozen=True)
class LabelMapping(preprocess_spec.PreprocessOp):
  """Use a specific attribute for label."""

  label_key: str = "label"

  def __call__(self, features):
    features[self.label_key] = tf.cast(features["attributes"]["Blond_Hair"],
                                       tf.int64)
    return features


@dataclasses.dataclass(frozen=True)
class DecodeAndRandomResizedCrop(ImagePreprocessOp):
  """Decodes the images and extracts a random crop."""
  resize_size: int

  def apply(self, image):
    shape = tf.io.extract_jpeg_shape(image)
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(0.05, 1.0),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    top, left, _ = tf.unstack(begin)
    h, w, _ = tf.unstack(size)
    image = tf.image.decode_and_crop_jpeg(image, [top, left, h, w], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.image.resize(image, (self.resize_size, self.resize_size))


@dataclasses.dataclass(frozen=True)
class RandomFlipLeftRight(ImagePreprocessOp):

  def apply(self, image):
    return tf.image.random_flip_left_right(image)


@dataclasses.dataclass(frozen=True)
class RescaleValues(ImagePreprocessOp):
  """Rescales values from `min/max_input` to `min/max_output`.

  Attr:
    min_output: The minimum value of the output.
    max_output: The maximum value of the output.
    min_input: The minimum value of the input.
    max_input: The maximum value of the input.
    clip: Whether to clip the output value, in case of input out-of-bound.
  """

  min_output: float = 0.
  max_output: float = 1.
  min_input: float = 0.
  max_input: float = 255.0

  def __post_init__(self):
    assert self.min_output < self.max_output
    assert self.min_input < self.max_input

  def apply(self, image):
    image = tf.cast(image, tf.float32)
    image = (image - self.min_input) / (self.max_input - self.min_input)
    image = self.min_output + image * (self.max_output - self.min_output)
    return image


@dataclasses.dataclass(frozen=True)
class ResizeSmall(ImagePreprocessOp):
  """Resizes the smaller side to `size` keeping aspect ratio.

  Attr:
    size: Smaller side of an input image (might be adjusted if max_size given).
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.
  """
  size: int
  antialias: bool = False

  def apply(self, image):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    # Figure out the necessary h/w.
    ratio = (
        tf.cast(self.size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32))
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
    return tf.image.resize(image, [h, w], antialias=self.antialias)


@dataclasses.dataclass(frozen=True)
class CentralCrop(ImagePreprocessOp):
  """Makes a central crop of a given size."""
  size: int

  def apply(self, image):
    h, w = self.size, self.size
    top = (tf.shape(image)[0] - h) // 2
    left = (tf.shape(image)[1] - w) // 2
    return tf.image.crop_to_bounding_box(image, top, left, h, w)


def predicate(features, all_subclasses):
  isallowed = tf.equal(
      tf.constant(all_subclasses), tf.cast(features["label"], tf.int32))
  reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
  return tf.greater(reduced, tf.constant(0.))


def _preprocess_with_per_example_rng(ds,
                                     preprocess_fn, *,
                                     rng):
  """Maps `ds` using the preprocess_fn and a deterministic RNG per example.

  Args:
    ds: Dataset containing Python dictionary with the features. The 'rng'
      feature should not exist.
    preprocess_fn: Preprocessing function that takes a Python dictionary of
      tensors and returns a Python dictionary of tensors. The function should be
      convertible into a TF graph.
    rng: Base RNG to use. Per example RNGs will be derived from this by folding
      in the example index.

  Returns:
    The dataset mapped by the `preprocess_fn`.
  """

  def _fn(example_index, features):
    example_index = tf.cast(example_index, tf.int32)
    features["rng"] = tf.random.experimental.stateless_fold_in(
        tf.cast(rng, tf.int64), example_index)
    processed = preprocess_fn(features)
    if isinstance(processed, dict) and "rng" in processed:
      del processed["rng"]
    return processed

  return ds.enumerate().map(
      _fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def pad_dataset(dataset,
                *,
                batch_dims,
                pad_up_to_batches = None,
                cardinality = None):
  """Adds padding to a dataset.

  Args:
    dataset: The dataset to be padded.
    batch_dims: List of size of batch dimensions. Multiple batch dimension can
      be used to provide inputs for multiple devices. E.g.
      [jax.local_device_count(), batch_size // jax.device_count()].
    pad_up_to_batches: Set this option to process the entire dataset. When set,
      then the dataset is first padded to the specified number of batches. A new
      feature called "mask" is added to every batch. This feature is set to
      `True` for every example that comes from `dataset_builder`, and to `False`
      for every example that is padded to get to the specified number of
      batches. Note that the specified `dataset_builder` and `split` must result
      in at least `pad_up_to_batches` (possibly partial) batches. If `None`,
      derives from `batch_dims` and `cardinality` such that `pad_up_to_batches *
      batch_dims == cardinality`. Note that `cardinality` is what you pass in,
      not necessarily the original full dataset size if you decide to shard it
      per host.
    cardinality: Number of examples in the dataset. Only needed when the
      cardinality cannot be retrieved via `ds.cardinalty()` (e.g. because of
      using `ds.filter()`).

  Returns:
    The padded dataset, with the added feature "mask" that is set to `True` for
    examples from the original `dataset` and to `False` for padded examples.
  """
  if not isinstance(dataset.element_spec, dict):
    raise ValueError("The dataset must have dictionary elements.")
  if cardinality is None:
    cardinality = dataset.cardinality()
    if cardinality == tf.data.UNKNOWN_CARDINALITY:
      raise ValueError(
          "Cannot determine dataset cardinality. This can happen when you use "
          "a `.filter()` on the dataset. Please provide the cardinality as an "
          "argument to `create_dataset()`.")
  if "mask" in dataset.element_spec:
    raise ValueError("Dataset already contains a feature named \"mask\".")
  if pad_up_to_batches is None:
    pad_up_to_batches = int(np.ceil(cardinality / np.prod(batch_dims)))

  filler_element = tf.nest.map_structure(
      lambda spec: tf.zeros(spec.shape, spec.dtype)[None], dataset.element_spec)
  filler_element["mask"] = [False]
  filler_dataset = tf.data.Dataset.from_tensor_slices(filler_element)

  dataset = dataset.map(
      lambda features: dict(mask=True, **features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  padding = pad_up_to_batches * np.prod(batch_dims) - int(cardinality)
  assert padding >= 0, (
      f"Invalid padding={padding} (batch_dims={batch_dims}, cardinality="
      f"{cardinality}, pad_up_to_batches={pad_up_to_batches})")
  return dataset.concatenate(filler_dataset.repeat(padding))


def create_dataset_helper(dataset_builder,
                          data_rng,
                          filter_fn,
                          preprocess_fn,
                          cache,
                          shuffle_buffer_size,
                          batch_dims,
                          shuffle,
                          is_train,
                          n_repeat,
                          pad_up_to_batches=None):
  """Helper function for creating train and val datasets.

  Args:
    dataset_builder: TensorFlow dataset builder.
    data_rng: PRNGKey for seeding operations.
    filter_fn: Dataset filtering function (if there is).
    preprocess_fn: Preprocessing function.
    cache: Boolean flag on whether to cache the dataset.
    shuffle_buffer_size: Size of shuffling buffer.
    batch_dims: Batch size.
    shuffle: Boolean flag on whether to shuffle the dataset.
    is_train: Boolean flag on whether the dataset is for training.
    n_repeat: Number of repetitions of the dataset.
    pad_up_to_batches: Boolean flag on whether to pad (val) dataset to get all
      the examples.

  Returns:
    A tuple with the dataset and the corresponding number of examples.
  """
  rng_available = data_rng is not None
  # if not rng_available and shuffle:
  #   raise ValueError("Please set 'rng' when shuffling.")
  if rng_available:
    if isinstance(data_rng, tf.Tensor):
      rngs = [
          x.numpy()
          for x in tf.random.experimental.stateless_split(data_rng, 3)
      ]
    else:
      rngs = list(jax.random.split(data_rng, 3))
  else:
    rngs = 3 * [[None, None]]

  dataset_options = tf.data.Options()
  dataset_options.experimental_optimization.map_parallelization = True
  dataset_options.experimental_threading.private_threadpool_size = 48
  dataset_options.experimental_threading.max_intra_op_parallelism = 1

  read_config = tfds.ReadConfig(
      shuffle_seed=rngs.pop()[0], options=dataset_options)
  if is_train:
    split = tfds.Split.TRAIN
    decoders = {"image": tfds.decode.SkipDecoding()}
  else:
    split = tfds.Split.VALIDATION
    decoders = None
  ds = dataset_builder.as_dataset(
      split=split,
      shuffle_files=False,
      read_config=read_config,
      decoders=decoders)

  if filter_fn is not None:
    ds = ds.filter(filter_fn)
  logging.info("num_devices=%d, num_process=%d", jax.local_device_count(),
               jax.process_count())
  num_examples = int(ds.reduce(0, lambda x, _: x + 1).numpy())
  if is_train:
    logging.info("num_train_examples after filtering=%d", num_examples)
  else:
    logging.info("num_eval_examples after filtering=%d", num_examples)

  if preprocess_fn is not None:
    if cache:
      ds = ds.cache()
    ds = ds.shard(jax.process_count(), jax.process_index())
    if shuffle:
      ds = ds.shuffle(shuffle_buffer_size, seed=rngs.pop()[0])

    ds = ds.repeat(n_repeat)
    if rng_available:
      ds = _preprocess_with_per_example_rng(ds, preprocess_fn, rng=rngs.pop())
    else:
      ds = ds.map(
          preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if pad_up_to_batches is not None:
    assert isinstance(pad_up_to_batches, int) or pad_up_to_batches == "auto"
    ds = pad_dataset(
        ds,
        batch_dims=batch_dims,
        pad_up_to_batches=(None if pad_up_to_batches == "auto" else
                           pad_up_to_batches),
        cardinality=None)

  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds, num_examples


def create_datasets(config):
  """Create datasets for training and evaluation.

  Args:
    config: Configuration to use.

  Returns:
    A tuple with the total number of training batches info, the training dataset
    and the evaluation dataset.
  """
  if config.dataset_name == "celeb_a":
    dataset_builder = tfds.builder("celeb_a", try_gcs=True)
    num_classes = 2
    train_preprocess = preprocess_spec.PreprocessFn([
        DecodeAndRandomResizedCrop(resize_size=224),
        RandomFlipLeftRight(),
        LabelMapping(),
    ],
                                                    only_jax_types=True)
    eval_preprocess = preprocess_spec.PreprocessFn([
        RescaleValues(),
        ResizeSmall(256),
        CentralCrop(224),
        LabelMapping(),
    ],
                                                   only_jax_types=True)
  elif config.dataset_name == "imagenet2012":
    dataset_builder = tfds.builder("imagenet2012", try_gcs=True)
    num_classes = 1000
    train_preprocess = preprocess_spec.PreprocessFn([
        DecodeAndRandomResizedCrop(resize_size=224),
        RandomFlipLeftRight(),
    ],
                                                    only_jax_types=True)
    eval_preprocess = preprocess_spec.PreprocessFn([
        RescaleValues(),
        ResizeSmall(256),
        CentralCrop(224),
    ],
                                                   only_jax_types=True)

  train_ds, num_train_examples = create_dataset_helper(
      dataset_builder,
      filter_fn=None,
      data_rng=None,
      preprocess_fn=train_preprocess,
      cache=False,
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=[jax.local_device_count(), config.per_device_batch_size],
      shuffle=True,
      is_train=True,
      n_repeat=None)

  num_train_steps = config.num_epochs * (
      num_train_examples // (jax.process_count() * jax.local_device_count() *
                             config.per_device_batch_size))

  num_validation_examples = (
      dataset_builder.info.splits["validation"].num_examples)
  eval_num_batches = None
  if config.eval_pad_last_batch:
    # This is doing some extra work to get exactly all 50k examples in the
    # ImageNet validation split. Without this the dataset would first be split
    # between the different hosts and then into batches (both times dropping the
    # remainder). If you don't mind dropping a few extra examples you can omit
    # the `pad_up_to_batches` argument.
    eval_batch_size = jax.local_device_count() * config.per_device_batch_size
    eval_num_batches = int(
        np.ceil(num_validation_examples / eval_batch_size /
                jax.process_count()))
  eval_ds, num_validation_examples = create_dataset_helper(
      dataset_builder,
      filter_fn=None,
      data_rng=None,
      preprocess_fn=eval_preprocess,
      cache=jax.process_count() > 1,
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=[jax.local_device_count(), config.per_device_batch_size],
      shuffle=False,
      is_train=False,
      n_repeat=1,
      pad_up_to_batches=eval_num_batches)

  num_val_steps = eval_num_batches

  return num_train_steps, num_val_steps, num_classes, train_ds, eval_ds
