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

"""Defines dataloader functionalities."""

import re
from typing import Any, Callable, Optional, Tuple

from clu import deterministic_data
import jax
from lib.datasets import billiard
from lib.datasets import trafficsigns
from lib.preprocess import image_ops
from lib.preprocess import preprocess_spec
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset(
    dataset: str,
    global_batch_size: int,
    rng: np.ndarray,
    train_preprocessing_fn: Optional[Callable[[Any], Any]] = None,
    eval_preprocessing_fn: Optional[Callable[[Any], Any]] = None,
    num_epochs: Optional[int] = None,
    filter_fn: Optional[Callable[[Any], Any]] = None,
    **kwargs,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
  """Creates training and eval datasets.

  The train dataset will be shuffled, while the eval dataset won't be.

  Args:
    dataset: Name of the dataset.
    global_batch_size: Global batch size to use.
    rng: PRNG seed used for shuffling.
    train_preprocessing_fn: Function that will be applied on each single sample.
    eval_preprocessing_fn: Optional preprocessing function specifically for eval
      samples (if None, use train_preprocessing_fn for eval samples as well).
    num_epochs: Number of epochs to repeat dataset (default=None, optional).
    filter_fn: Funtion that filters samples according to some criteria.
    **kwargs: Optional keyword arguments for specific datasets.

  Returns:
    A tuple consisting of a train dataset, an eval dataset as well as the
    number of classes.
  """
  del kwargs
  rng, preprocess_rng = jax.random.split(rng)
  if dataset.startswith("test_image_classification"):
    # The shape of the dataset can be provided as suffix of the name of the
    # dataset test_image_classification_batch_height_width.
    match = re.search(r"test_image_classification_(\d+)_(\d+)_(\d+)_(\d+)",
                      dataset)
    if match:
      shape = tuple(int(match.group(i)) for i in [1, 2, 3, 4])
    else:
      shape = (13, 32, 32, 3)

    images_tensor = tf.random.uniform(
        shape, minval=0, maxval=256, dtype=tf.int32, seed=22432)
    images_tensor = tf.cast(images_tensor, tf.uint8)
    labels_tensor = tf.random.uniform((shape[0],),
                                      minval=0,
                                      maxval=10,
                                      dtype=tf.int32,
                                      seed=4202)

    ds_image = tf.data.Dataset.from_tensor_slices(images_tensor)
    ds_label = tf.data.Dataset.from_tensor_slices(labels_tensor)

    ds = tf.data.Dataset.zip({"image": ds_image, "label": ds_label})
    train_ds = ds
    eval_ds = ds
    num_classes = 10
  elif dataset == "trafficsigns":
    train_ds = trafficsigns.load("train")
    eval_ds = trafficsigns.load("test")
    num_classes = 4
  elif dataset.startswith("billiard"):
    # The format of the dataset string is "billiard-label_fn_str-{valid,test}"
    # where label_fn_str options are specified in data/billiard.py
    # Example: billiard-left-color-min-max-valid
    parts = dataset.split("-")
    label_fn_str = "-".join(parts[1:-1])
    evaluation_split = parts[-1]
    train_ds, num_classes = billiard.load_billiard("train", label_fn_str)
    eval_ds, _ = billiard.load_billiard(evaluation_split, label_fn_str)
  elif dataset.startswith("caltech_birds2011"):
    mode = dataset[len("caltech_birds2011") + 1:]
    train_ds, eval_ds, num_classes = _get_birds200_dataset(mode, rng)
  elif dataset.startswith("test_image_classification"):
    # The shape of the dataset can be provided as suffix of the name of the
    # dataset test_image_classification_batch_height_width.
    match = re.search(r"test_image_classification_(\d+)_(\d+)_(\d+)_(\d+)",
                      dataset)
    if match:
      shape = tuple(int(match.group(i)) for i in [1, 2, 3, 4])
    else:
      shape = (13, 32, 32, 3)

    with tf.device("/CPU:0"):
      images_tensor = tf.random.uniform(
          shape, minval=0, maxval=256, dtype=tf.int32, seed=22432)
      images_tensor = tf.cast(images_tensor, tf.uint8)
      labels_tensor = tf.random.uniform((shape[0],),
                                        minval=0,
                                        maxval=10,
                                        dtype=tf.int32,
                                        seed=4202)

      ds_image = tf.data.Dataset.from_tensor_slices(images_tensor)
      ds_label = tf.data.Dataset.from_tensor_slices(labels_tensor)

      ds = tf.data.Dataset.zip({"image": ds_image, "label": ds_label})
    train_ds = ds
    eval_ds = ds
    num_classes = 10
  else:  # Should be a TFDS dataset.
    train_ds, eval_ds, num_classes = _get_tfds_dataset(dataset, rng)

  # Set up a preprocessing function.
  if train_preprocessing_fn is None:

    @tf.autograph.experimental.do_not_convert  # Usually fails anyway.
    def _image_preprocess_fn(features):
      if "image" in features:
        features["image"] = tf.cast(features["image"], tf.float32) / 255.0
      if "id" in features:  # Included in some TFDS datasets, breaks JAX.
        del features["id"]
      return features

    train_preprocessing_fn = _image_preprocess_fn

  if eval_preprocessing_fn is None:
    eval_preprocessing_fn = train_preprocessing_fn

  rng_train, rng_eval = jax.random.split(preprocess_rng)
  train_ds = _prepare_dataset(
      train_ds,
      global_batch_size,
      True,
      rng_train,
      train_preprocessing_fn,
      num_epochs=num_epochs,
      filter_fn=filter_fn)
  eval_ds = _prepare_dataset(
      eval_ds,
      global_batch_size,
      False,
      rng_eval,
      eval_preprocessing_fn,
      num_epochs=1,
      filter_fn=filter_fn)
  return train_ds, eval_ds, num_classes


def _get_birds200_dataset(
    mode: str,
    rng: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
  """Load the caltech_birds2011 dataset."""
  assert jax.host_count() == 1, (
      "caltech_birds2011 dataset does not support multihost training. "
      "Found {} hosts.".format(jax.host_count()))

  dataset_builder = tfds.builder("caltech_birds2011")
  num_classes = 200

  # Make sure each host uses a different RNG for the training data.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.host_id())
  data_rng, shuffle_rng = jax.random.split(data_rng)

  if mode == "train-val":
    read_config = tfds.ReadConfig(shuffle_seed=shuffle_rng[0])
    ds = dataset_builder.as_dataset(
        split="train", shuffle_files=False, read_config=read_config)

    train_ds = ds.take(5000).shuffle(5000, seed=shuffle_rng[0])
    eval_ds = ds.skip(5000)

  elif mode == "train-test":
    train_split = "train"
    eval_split = "test"

    train_read_config = tfds.ReadConfig(shuffle_seed=shuffle_rng[0])
    train_ds = dataset_builder.as_dataset(
        split=train_split, shuffle_files=True, read_config=train_read_config)

    eval_read_config = tfds.ReadConfig(shuffle_seed=shuffle_rng[1])
    eval_ds = dataset_builder.as_dataset(
        split=eval_split, shuffle_files=False, read_config=eval_read_config)
  else:
    raise ValueError(f"Unknown mode: {mode}.")

  return train_ds, eval_ds, num_classes


def _get_tfds_dataset(
    dataset: str,
    rng: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
  """Loads a TFDS dataset."""

  dataset_builder = tfds.builder(dataset)
  num_classes = 0
  if "label" in dataset_builder.info.features:
    num_classes = dataset_builder.info.features["label"].num_classes

  # Make sure each host uses a different RNG for the training data.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.host_id())
  data_rng, shuffle_rng = jax.random.split(data_rng)
  train_split = deterministic_data.get_read_instruction_for_host(
      "train", dataset_builder.info.splits["train"].num_examples)
  train_read_config = tfds.ReadConfig(shuffle_seed=shuffle_rng[0])
  train_ds = dataset_builder.as_dataset(
      split=train_split, shuffle_files=True, read_config=train_read_config)

  eval_split_name = {
      "cifar10": "test",
      "imagenet2012": "validation"
  }.get(dataset, "test")

  eval_split_size = dataset_builder.info.splits[eval_split_name].num_examples
  eval_split = deterministic_data.get_read_instruction_for_host(
      eval_split_name, eval_split_size)
  eval_read_config = tfds.ReadConfig(shuffle_seed=shuffle_rng[1])
  eval_ds = dataset_builder.as_dataset(
      split=eval_split, shuffle_files=False, read_config=eval_read_config)
  return train_ds, eval_ds, num_classes


def _prepare_dataset(
    dataset: tf.data.Dataset,
    global_batch_size: int,
    shuffle: bool,
    rng: np.ndarray,
    preprocess_fn: Optional[Callable[[Any], Any]] = None,
    num_epochs: Optional[int] = None,
    filter_fn: Optional[Callable[[Any], Any]] = None) -> tf.data.Dataset:
  """Batches, shuffles, prefetches and preprocesses a dataset.

  Args:
    dataset: The dataset to prepare.
    global_batch_size: The global batch size to use.
    shuffle: Whether the shuffle the data on example level.
    rng: PRNG for seeding the shuffle operations.
    preprocess_fn: Preprocessing function that will be applied to every example.
    num_epochs: Number of epochs to repeat the dataset.
    filter_fn: Funtion that filters samples according to some criteria.

  Returns:
    The dataset.
  """
  if shuffle and rng is None:
    raise ValueError("Shuffling without RNG is not supported.")

  if global_batch_size % jax.host_count() != 0:
    raise ValueError(f"Batch size {global_batch_size} not divisible by number "
                     f"of hosts ({jax.host_count()}).")
  local_batch_size = global_batch_size // jax.host_count()
  batch_dims = [jax.local_device_count(), local_batch_size]

  # tf.data uses single integers as seed.
  if rng is not None:
    rng = rng[0]

  ds = dataset.repeat(num_epochs)
  if shuffle:
    ds = ds.shuffle(1024, seed=rng)

  if preprocess_fn is not None:
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if filter_fn is not None:
    ds = ds.filter(filter_fn)

  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size, drop_remainder=True)
  return ds.prefetch(tf.data.experimental.AUTOTUNE)


def parse_preprocessing_strings(training_string, eval_string):
  """Parses conjurer preprocessing strings."""
  print(training_string)
  print(image_ops.all_ops(), flush=True)
  train_preprocessing_fn = preprocess_spec.parse(training_string,
                                                 image_ops.all_ops())
  eval_preprocessing_fn = preprocess_spec.parse(eval_string,
                                                image_ops.all_ops())
  return train_preprocessing_fn, eval_preprocessing_fn
