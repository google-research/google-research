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

"""Datasets.

The general design philosophy of these dataset loaders is to keep them as simple
as possible. Data processing or manipulation of conditioning information should
be kept in an experiment's main.py, not here.

When data augmentation is enabled, nondeterministic behavior is expected.
"""

# pylint: disable=logging-format-interpolation
# pylint: disable=g-long-lambda

import functools
from typing import Any, Mapping, Optional, Tuple

from . import utils
from absl import logging
from clu import deterministic_data
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def batch_dataset(dataset, batch_shape):
  for b in reversed(batch_shape):
    dataset = dataset.batch(b, drop_remainder=True)
  return dataset


class Dataset:
  """Generic dataset."""

  @property
  def info(self):
    raise NotImplementedError

  @property
  def data_shape(self):
    return self.info['data_shape']

  @property
  def num_train(self):
    return self.info['num_train']

  @property
  def num_eval(self):
    return self.info['num_eval']

  @property
  def num_classes(self):
    return self.info['num_classes']

  def _load_tfds(self, *, split, shuffle_seed):
    raise NotImplementedError

  def _preprocess(self, x, *, split, augment):
    """Preprocess one example."""
    raise NotImplementedError

  def _shuffle_buffer_size(self, split):
    del split
    return 50000

  def get_shuffled_repeated_dataset(self, *, batch_shape,
                                    split, local_rng, augment):
    """Shuffled and repeated dataset suitable for training.

    Shuffling is determined by local_rng, which should be different for
    each shard.

    Args:
      batch_shape: leading shape of batches
      split: which dataset split to load
      local_rng: rng for shuffling (should be different for each host/shard)
      augment: whether to enable data augmentation

    Returns:
      dataset
    """
    local_rng = utils.RngGen(local_rng)
    ds = self._load_tfds(  # file-level shuffling here
        split=split, shuffle_seed=utils.jax_randint(next(local_rng)))
    ds = ds.shuffle(
        self._shuffle_buffer_size(split),
        seed=utils.jax_randint(next(local_rng)))
    ds = ds.repeat()
    ds = ds.map(
        functools.partial(self._preprocess, split=split, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = batch_dataset(ds, batch_shape=batch_shape)
    return ds.prefetch(tf.data.AUTOTUNE)

  def get_padded_one_shot_dataset(self, *, batch_shape,
                                  split, shard_id, num_shards):
    """Non-repeated non-shuffled sharded dataset with padding.

    Should not drop any examples. Augmentation is disabled.

    Args:
      batch_shape: leading shape of batches
      split: which dataset split to load
      shard_id: current shard id (e.g. process_index)
      num_shards: number of shards (e.g. process_count)

    Returns:
      dataset
    """
    ds = self._load_tfds(split=split, shuffle_seed=None)
    ds = ds.map(
        functools.partial(self._preprocess, split=split, augment=False),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = deterministic_data.pad_dataset(
        ds, batch_dims=(num_shards, *batch_shape),
        cardinality={'train': self.num_train, 'eval': self.num_eval}[split])
    ds = ds.shard(index=shard_id, num_shards=num_shards)
    ds = batch_dataset(ds, batch_shape=batch_shape)
    return ds.prefetch(tf.data.AUTOTUNE)


class CIFAR10(Dataset):
  """CIFAR10 dataset."""

  def __init__(self, *, class_conditional, randflip):
    self._class_conditional = class_conditional
    self._randflip = randflip
    self._info = {
        'data_shape': (32, 32, 3),
        'num_train': 50000,
        'num_eval': 10000,
        'num_classes': 10 if self._class_conditional else 1
    }

  @property
  def info(self):
    return self._info

  def _load_tfds(self, *, split, shuffle_seed):
    return tfds.load(
        'cifar10',
        split={'train': 'train', 'eval': 'test'}[split],
        shuffle_files=shuffle_seed is not None,
        read_config=None if shuffle_seed is None else tfds.ReadConfig(
            shuffle_seed=shuffle_seed))

  def _preprocess(self, x, *, split, augment):
    del split
    img = tf.cast(x['image'], tf.float32)
    if augment:  # NOTE: this makes training nondeterministic
      if self._randflip:
        aug_img = tf.image.flip_left_right(img)
        aug = tf.random.uniform(shape=[]) > 0.5
        img = tf.where(aug, aug_img, img)
    out = {'image': img}
    if self._class_conditional:
      out['label'] = tf.cast(x['label'], tf.int32)
    return out


def central_square_crop(img):
  """Crop to square along the long edge."""
  h, w, _ = tf.unstack(tf.shape(img))
  box = tf.where(h > w, [h // 2 - w // 2, 0, w, w], [0, w // 2 - h // 2, h, h])
  offset_height, offset_width, target_height, target_width = tf.unstack(box)
  return tf.image.crop_to_bounding_box(
      img, offset_height, offset_width, target_height, target_width)


def decode_and_central_square_crop(img):
  """Crop to square along the long edge."""
  h, w, _ = tf.unstack(tf.io.extract_jpeg_shape(img))
  box = tf.where(h > w, [h // 2 - w // 2, 0, w, w], [0, w // 2 - h // 2, h, h])
  return tf.image.decode_and_crop_jpeg(img, box, channels=3)


class ImageNet(Dataset):
  """ImageNet dataset."""

  def __init__(self,
               *,
               class_conditional,
               image_size,
               randflip,
               extra_image_sizes=()):
    """ImageNet dataset.

    Args:
      class_conditional: bool: class conditional generation problem; if True,
        generated examples will contain a label.
      image_size: int: size of image to model
      randflip: bool: random flip augmentation
      extra_image_sizes: Tuple[int]: also provide image at these resolutions
    """
    self._class_conditional = class_conditional
    self._image_size = image_size
    self._randflip = randflip
    self._extra_image_sizes = extra_image_sizes
    self._info = {
        'data_shape': (self._image_size, self._image_size, 3),
        'num_train': 1281167,
        'num_eval': 50000,
        'num_classes': 1000 if self._class_conditional else 1
    }

  @property
  def info(self):
    return self._info

  def _load_tfds(self, *, split, shuffle_seed):
    return tfds.load(
        'imagenet2012',
        split={'train': 'train', 'eval': 'validation'}[split],
        shuffle_files=shuffle_seed is not None,
        read_config=None if shuffle_seed is None else tfds.ReadConfig(
            shuffle_seed=shuffle_seed),
        decoders={'image': tfds.decode.SkipDecoding()})

  def _preprocess(self, x, *, split, augment):
    del split  # unused
    out = {}

    # Decode the image and resize
    img = tf.cast(decode_and_central_square_crop(x['image']), tf.float32)

    if augment:
      # NOTE: this makes training nondeterministic
      if self._randflip:
        logging.info('ImageNet: randflip=True')
        img = tf.image.random_flip_left_right(img)

    # Standard area resizing
    out['image'] = tf.clip_by_value(
        tf.image.resize(img, [self._image_size, self._image_size], 'area'),
        0, 255)

    # Optionally provide the image at other resolutions too
    for s in self._extra_image_sizes:
      assert isinstance(s, int)
      out[f'extra_image_{s}'] = tf.clip_by_value(
          tf.image.resize(img, [s, s], 'area'), 0, 255)

    # Class label
    if self._class_conditional:
      out['label'] = tf.cast(x['label'], tf.int32)

    return out


class LSUN(Dataset):
  """LSUN dataset."""

  def __init__(self, *, subset, image_size, randflip,
               extra_image_sizes=()):
    """LSUN datasets.

    Args:
      subset: str: 'church' or 'bedroom'
      image_size: int: size of image to model, 64 or 128
      randflip: bool: random flip augmentation
      extra_image_sizes: optional extra image sizes
    """
    self._subset = subset
    self._image_size = image_size
    self._randflip = randflip
    self._extra_image_sizes = extra_image_sizes

    self._info = {
        'data_shape': (self._image_size, self._image_size, 3),
        'num_train': {'bedroom': 3033042, 'church': 126227}[self._subset],
        'num_eval': 300,
        'num_classes': 1,
    }

  @property
  def info(self):
    return self._info

  def _load_tfds(self, *, split, shuffle_seed):
    tfds_name = {'church': 'lsun/church_outdoor',
                 'bedroom': 'lsun/bedroom'}[self._subset]
    return tfds.load(
        tfds_name,
        split={'train': 'train', 'eval': 'validation'}[split],
        shuffle_files=shuffle_seed is not None,
        read_config=None if shuffle_seed is None else tfds.ReadConfig(
            shuffle_seed=shuffle_seed),
        decoders={'image': tfds.decode.SkipDecoding()})

  def _preprocess(self, x, *, split, augment):
    del split  # unused

    # Decode the image and resize
    img = tf.cast(decode_and_central_square_crop(x['image']), tf.float32)
    if augment:  # NOTE: nondeterministic
      if self._randflip:
        aug_img = tf.image.flip_left_right(img)
        aug = tf.random.uniform(shape=[]) > 0.5
        img = tf.where(aug, aug_img, img)

    out = {}
    out['image'] = tf.clip_by_value(tf.image.resize(
        img, [self._image_size, self._image_size], antialias=True), 0, 255)

    # Optionally provide the image at other resolutions too
    for s in self._extra_image_sizes:
      assert isinstance(s, int)
      out[f'extra_image_{s}'] = tf.clip_by_value(
          tf.image.resize(img, [s, s], antialias=True), 0, 255)

    return out
