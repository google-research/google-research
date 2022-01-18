# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Input pipeline.

Read and process each example
"""
import functools
from typing import Dict

from class_balanced_distillation import deterministic_data
import ml_collections
import numpy as np
import simclr.tf2.data_util as simclr_data
import tensorflow as tf
import tensorflow_datasets as tfds


def random_color_jitter(image, p=1.0):

  def _transform(image):
    color_jitter_t = functools.partial(
        simclr_data.color_jitter, strength=1.0, impl="simclrv2")
    image = simclr_data.random_apply(color_jitter_t, p=0.8, x=image)
    return simclr_data.random_apply(simclr_data.to_grayscale, p=0.2, x=image)

  return simclr_data.random_apply(_transform, p=p, x=image)


def oversample_classes(example, base_probs, target_probs):
  """Returns the number of copies of given example."""
  cur_label = example["label"]
  cur_prob = base_probs[cur_label]
  cur_target_prob = target_probs[cur_label]

  # Add tiny to initial_probs to avoid divide by zero.
  denom = (cur_prob + np.finfo(cur_prob.dtype.as_numpy_dtype).tiny)
  ratio_l = tf.cast(cur_target_prob / denom, tf.float32)

  # Return 1 for head classes, we only want to sample them once
  max_ratio = tf.maximum(ratio_l, 1)

  # This is the number of duplicates we want to add for oversampling
  num_oversamples = tf.floor(ratio_l)

  # Maybe add one more based on the residual probability
  residual = max_ratio - num_oversamples
  residual_sel = tf.less_equal(
      tf.random.uniform([], dtype=tf.float32), residual
  )

  return tf.cast(num_oversamples, tf.int64) + tf.cast(residual_sel, tf.int64)


def undersampling_filter(example, base_probs, target_probs):
  """Computes if given example is rejected or not."""
  cur_label = example["label"]
  cur_prob = base_probs[cur_label]
  cur_target_prob = target_probs[cur_label]

  # Add tiny to initial_probs to avoid divide by zero.
  denom = (cur_prob + np.finfo(cur_prob.dtype.as_numpy_dtype).tiny)
  ratio_l = tf.cast(cur_target_prob / denom, tf.float32)
  acceptance_prob = tf.minimum(ratio_l, 1.0)

  acceptance = tf.less_equal(tf.random.uniform([], dtype=tf.float32),
                             acceptance_prob)

  return acceptance


def resize_small(image,
                 size,
                 *,
                 antialias = False):
  """Resizes the smaller side to `size` keeping aspect ratio.

  Args:
    image: Single image as a float32 tensor.
    size: an integer, that represents a new size of the smaller side of an input
      image.
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.
  """
  h, w = tf.shape(image)[0], tf.shape(image)[1]

  # Figure out the necessary h/w.
  ratio = (tf.cast(size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32))
  h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
  w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
  image = tf.image.resize(image, [h, w], antialias=antialias)
  return image


def central_crop(image, size):
  """Makes central crop of a given size."""
  h, w = size, size
  top = (tf.shape(image)[0] - h) // 2
  left = (tf.shape(image)[1] - w) // 2
  image = tf.image.crop_to_bounding_box(image, top, left, h, w)
  return image


def decode_and_random_resized_crop(image, rng,
                                   resize_size,
                                   do_normalize = True):
  """Decodes the images and extracts a random crop."""
  shape = tf.io.extract_jpeg_shape(image)
  begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
      shape,
      tf.zeros([0, 0, 4], tf.float32),
      seed=rng,
      area_range=(0.05, 1.0),
      min_object_covered=0,  # Don't enforce a minimum area.
      use_image_if_no_bounding_boxes=True)
  top, left, _ = tf.unstack(begin)
  h, w, _ = tf.unstack(size)
  image = tf.image.decode_and_crop_jpeg(image, [top, left, h, w], channels=3)
  if do_normalize:
    image = tf.cast(image, tf.float32) / 255.0
  image = tf.image.resize(image, (resize_size, resize_size))
  return image


def train_preprocess(features,
                     add_jitter = False):
  """Processes a single example for training."""
  image = features["image"]
  label = features["label"]
  # This PRNGKey is unique to this example. We can use it with the stateless
  # random ops in TF.
  rng = features.pop("rng")
  rng, rng_crop, rng_flip = tf.unstack(
      tf.random.experimental.stateless_split(rng, 3))

  image = decode_and_random_resized_crop(image, rng_crop, resize_size=224)
  image = tf.image.stateless_random_flip_left_right(image, rng_flip)

  if add_jitter:
    image = random_color_jitter(image)
    image = tf.clip_by_value(image, 0., 1.)

  features = {"image": image, "label": label}
  return features


def eval_preprocess(features):
  """Process a single example for evaluation."""
  image = features["image"]
  assert image.dtype == tf.uint8
  image = tf.cast(image, tf.float32) / 255.0
  image = resize_small(image, size=256)
  image = central_crop(image, size=224)
  return {"image": image, "label": features["label"]}


def create_datasets(
    config, data_rng, *,
    strategy):
  """Create datasets for training and evaluation.

  For the same data_rng and config this will return the same datasets. The
  datasets only contain stateless operations.

  Args:
    config: Configuration to use.
    data_rng: PRNGKey for seeding operations in the training dataset.
    strategy: Distribution strategy to use. Each replica will run a separate
      input pipeline.

  Returns:
    A tuple with the dataset info, the training dataset and the evaluation
    dataset.
  """

  if config.dataset == "imagenet-lt":
    dataset_builder = (
        tfds.builder("imagenet_lt",
                     data_dir="~/data/imagenet-lt/tfds/"))

  elif config.dataset == "inaturalist18":
    dataset_builder = (
        tfds.builder("i_naturalist2018",
                     data_dir="~/data/inaturalist18/tfds/"))

  else:
    raise ValueError(f"Dataset {config.dataset} not supported.")

  def train_split(host_id, host_count):
    return deterministic_data.get_read_instruction_for_host(
        "train",
        dataset_builder.info.splits["train"].num_examples,
        host_id=host_id,
        host_count=host_count)

  def val_split(host_id, host_count):
    return deterministic_data.get_read_instruction_for_host(
        "validation",
        dataset_builder.info.splits["validation"].num_examples,
        host_id=host_id,
        host_count=host_count,
        drop_remainder=False
        )

  def test_split(host_id, host_count):
    return deterministic_data.get_read_instruction_for_host(
        "test",
        dataset_builder.info.splits["test"].num_examples,
        host_id=host_id,
        host_count=host_count,
        drop_remainder=False
        )

  if config.sampling == "class_balanced" or config.sampling == "sqrt":
    if config.dataset == "imagenet-lt":
      img_per_class = np.load(
          tf.io.gfile.GFile(
              "class_balanced_distillation/data/ImageNet_LT/train_img_per_class.npy",
              "rb"))
    elif config.dataset == "inaturalist18":
      img_per_class = np.load(
          tf.io.gfile.GFile(
              "class_balanced_distillation/data/iNaturalist18/train_img_per_class.npy",
              "rb"))
    else:
      raise ValueError(f"Dataset {config.dataset} not supported.")

    base_probs = img_per_class / img_per_class.sum()
    target_probs = np.ones_like(img_per_class, dtype=np.float32)
    target_probs = target_probs / target_probs.sum()

    base_probs = tf.convert_to_tensor(
        base_probs, dtype=tf.float32)
    target_probs = tf.convert_to_tensor(
        target_probs.astype(np.float32), dtype=tf.float32)

    oversampling_fn = functools.partial(
        oversample_classes,
        base_probs=base_probs,
        target_probs=target_probs,
        )

    undersampling_filter_fn = functools.partial(
        undersampling_filter,
        base_probs=base_probs,
        target_probs=target_probs,
        )
  elif config.sampling == "uniform":
    undersampling_filter_fn = None
    oversampling_fn = None
  else:
    raise ValueError(f"Sampling {config.sampling} not supported.")

  train_preprocess_fn = functools.partial(
      train_preprocess, add_jitter=config.add_color_jitter)

  train_ds = deterministic_data.create_distributed_dataset(
      dataset_builder,
      strategy=strategy,
      global_batch_size=config.global_batch_size,
      split=train_split,
      num_epochs=config.num_epochs,
      shuffle=True,
      cache=False,
      filter_fn=undersampling_filter_fn,
      preprocess_fn=train_preprocess_fn,
      decoders={"image": tfds.decode.SkipDecoding()},
      shuffle_buffer_size=config.shuffle_buffer_size,
      prefetch_size=8,
      rng=data_rng,
      oversampling_fn=oversampling_fn)

  eval_preprocess_fn = functools.partial(
      eval_preprocess)

  if config.dataset != "inaturalist18":
    test_num_batches = int(
        np.ceil(dataset_builder.info.splits["test"].num_examples /
                (config.global_batch_size))) * strategy.num_replicas_in_sync

    test_ds = deterministic_data.create_distributed_dataset(
        dataset_builder,
        strategy=strategy,
        global_batch_size=config.global_batch_size,
        split=test_split,
        num_epochs=1,
        shuffle=False,
        cache=True,
        preprocess_fn=eval_preprocess_fn,
        pad_up_to_batches=test_num_batches)

  val_num_batches = int(
      np.ceil(dataset_builder.info.splits["validation"].num_examples /
              (config.global_batch_size))) * strategy.num_replicas_in_sync

  val_ds = deterministic_data.create_distributed_dataset(
      dataset_builder,
      strategy=strategy,
      global_batch_size=config.global_batch_size,
      split=val_split,
      num_epochs=1,
      shuffle=False,
      cache=True,
      preprocess_fn=eval_preprocess_fn,
      pad_up_to_batches=val_num_batches)

  if config.dataset == "inaturalist18":
    test_ds = val_ds

  return dataset_builder.info, train_ds, val_ds, test_ds
