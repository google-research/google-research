# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Datasets.

All data generated should be in [0, 255].
Data loaders should iterate through the data in the same order for all hosts,
and sharding across hosts is done here.
"""

from typing import Sequence, Tuple
from absl import logging
import jax.numpy as jnp
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from d3pm.images import utils


def batch_dataset(dataset, batch_shape):
  for b in reversed(batch_shape):
    dataset = dataset.batch(b, drop_remainder=True)
  return dataset


def shard_dataset(dataset, *, shard_id, num_shards):
  """Shard a dataset, ensuring that all shards have equal cardinality."""
  assert 0 <= shard_id < num_shards
  logging.info('Sharding dataset: shard_id=%d num_shards=%d', shard_id,
               num_shards)

  if num_shards == 1:
    return dataset

  def get_current_shard(z):
    assert z.shape[0] == num_shards
    return z[shard_id]

  dataset = dataset.batch(num_shards, drop_remainder=True)
  dataset = dataset.map(lambda x: tf.nest.map_structure(get_current_shard, x))
  return dataset


class Dataset:
  """Generic dataset.

  All generated image data should be in [0, 255], and these subclasses are
  responsible for sharding across hosts.
  """

  @property
  def data_shape(self):
    """Data shape, e.g. (32, 32, 3) for an image."""
    raise NotImplementedError

  @property
  def num_train(self):
    """Size of training set."""
    raise NotImplementedError

  @property
  def num_eval(self):
    """Size of eval set."""
    raise NotImplementedError

  @property
  def num_classes(self):
    """Number of classes."""
    raise NotImplementedError

  def get_tf_dataset(self, *, batch_shape, split,
                     global_rng, repeat, shuffle,
                     augment, shard_id, num_shards):
    """Training dataset function.

    Args:
      batch_shape: tuple: leading batch dims
      split: str: 'train' or 'eval'
      global_rng: Jax PRNG for shuffling (equal across all hosts)
      repeat: bool: enables repeating the dataset
      shuffle: bool: enables shuffling
      augment: bool: data augmentation
      shard_id: int: the current shard (jax.host_id())
      num_shards: int: total number of shards (jax.host_count())

    Returns:
      tf.data.Dataset
    """
    raise NotImplementedError


class CIFAR10(Dataset):
  """CIFAR10 dataset."""

  def __init__(self, *, class_conditional, randflip, rot90=False):
    self._class_conditional = class_conditional
    self._randflip = randflip
    self._rot90 = rot90

  @property
  def data_shape(self):
    return (32, 32, 3)

  @property
  def num_train(self):
    return 50000

  @property
  def num_eval(self):
    return 10000

  @property
  def num_classes(self):
    return 10 if self._class_conditional else 1

  def _preprocess_and_batch(self, ds, *, batch_shape, augment):
    """Data preprocessing (and augmentation) and batching."""

    def preprocess(x):
      img = tf.cast(x['image'], tf.float32)
      aug = None
      if augment:  # NOTE: this makes training nondeterministic
        if self._randflip:
          augment_img = tf.image.flip_left_right(img)
          aug = tf.random.uniform(shape=[]) > 0.5
          img = tf.where(aug, augment_img, img)
        if self._rot90:
          u = tf.random.uniform(shape=[])
          k = tf.cast(tf.floor(4. * u), tf.int32)
          img = tf.image.rot90(img, k=k)
          aug = aug | (k > 0)
      if aug is None:
        aug = tf.convert_to_tensor(False, dtype=tf.bool)
      out = {'image': img, 'augmented': aug}
      if self._class_conditional:
        out['label'] = tf.cast(x['label'], tf.int32)
      return out

    ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = batch_dataset(ds, batch_shape=batch_shape)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def get_tf_dataset(self, *, batch_shape, split, global_rng, repeat,
                     shuffle, augment, shard_id,
                     num_shards):
    """Training dataset."""
    if split == 'train':
      split_str = 'train'
    elif split == 'eval':
      split_str = 'test'
    else:
      raise NotImplementedError
    if shuffle:
      global_rng = utils.RngGen(global_rng)
    ds = tfds.load(
        'cifar10',
        split=split_str,
        shuffle_files=shuffle,
        read_config=None if not shuffle else tfds.ReadConfig(
            shuffle_seed=utils.jax_randint(next(global_rng))))
    if repeat:
      ds = ds.repeat()
    if shuffle:
      ds = ds.shuffle(50000, seed=utils.jax_randint(next(global_rng)))
    ds = shard_dataset(ds, shard_id=shard_id, num_shards=num_shards)
    return self._preprocess_and_batch(
        ds, batch_shape=batch_shape, augment=augment)


class MockCIFAR10(CIFAR10):
  """Mocked version of CIFAR10 dataset."""

  @property
  def data_shape(self):
    return (8, 8, 3)

  @property
  def num_train(self):
    return 10

  @property
  def num_eval(self):
    return 10

  def get_tf_dataset(self, *, batch_shape, split, global_rng, repeat,
                     shuffle, augment, shard_id,
                     num_shards):
    del split, global_rng, repeat, shuffle
    ds = tf.data.Dataset.from_tensors({
        'image':
            tf.fill(
                dims=self.data_shape,
                value=tf.constant(127, dtype=tf.uint8),
            ),
        'label':
            tf.zeros(shape=(), dtype=tf.int64),
    }).repeat()
    ds = shard_dataset(ds, shard_id=shard_id, num_shards=num_shards)
    return self._preprocess_and_batch(
        ds, batch_shape=batch_shape, augment=augment)
