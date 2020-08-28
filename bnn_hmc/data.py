# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Data loaders."""

from typing import Generator, Tuple

import jax
import numpy as onp
import tensorflow as tf
import tensorflow_datasets as tfds

SupervisedDataset = Tuple[onp.ndarray, onp.ndarray]
SupervisedDatasetGen = Generator[SupervisedDataset, None, None]

# Format: (img_mean, img_std)
_ALL_DS_STATS = {
    "cifar10": ((0.49, 0.48, 0.44), (0.2, 0.2, 0.2))
}


def load_dataset(
    split,
    batch_size,
    name = "cifar10"
):
  """Loads the dataset as a generator of batches."""
  # Do no data augmentation.
  name = name.lower()
  ds, dataset_info = tfds.load(name, split=split, as_supervised=True,
                               with_info=True)
  num_classes = dataset_info.features["label"].num_classes
  num_examples = dataset_info.splits[split].num_examples

  def img_to_float32(image, label):
    return tf.image.convert_image_dtype(image, tf.float32), label

  ds = ds.map(img_to_float32).cache()
  ds_stats = _ALL_DS_STATS[name]

  def img_normalize(image, label):
    """Normalize the image to zero mean and unit variance."""
    mean, std = ds_stats
    image -= tf.constant(mean, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(std, shape=[1, 1, 3], dtype=image.dtype)
    return image, label

  ds = ds.map(img_normalize)
  if batch_size == -1:
    batch_size = num_examples
  ds = ds.batch(batch_size)
  return tfds.as_numpy(ds), num_classes, num_examples


def batch_split_axis(batch,
                     n_split):
  """Reshapes batch to have first axes size equal n_split."""
  x, y = batch
  n = x.shape[0]
  n_new = n / n_split
  assert n_new == int(n_new), (
      "First axis cannot be split: batch dimension was {} when "
      "n_split was {}.".format(x.shape[0], n_split))
  n_new = int(n_new)
  return tuple(arr.reshape([n_split, n_new, *arr.shape[1:]]) for arr in (x, y))


def make_ds_pmap_fullbatch(
    name = "cifar10",
    n_devices = None):
  """Make train and test sets sharded over batch dim."""

  train_set, n_classes, _ = load_dataset("train", -1, name)
  train_set = next(iter(train_set))

  test_set, _, _ = load_dataset("test", -1, name)
  test_set = next(iter(test_set))

  n_devices = n_devices or len(jax.local_devices())

  def pmap_ds(ds):
    return jax.pmap(lambda x: x)(batch_split_axis(ds, n_devices))
  train_set, test_set = tuple(pmap_ds(ds) for ds in (train_set, test_set))
  return train_set, test_set, n_classes
