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

"""ImageNet dataset with typical pre-processing."""
# pylint: skip-file

import enum
# from tkinter.tix import IMAGE
from typing import Generator, Mapping, Optional, Sequence, Text, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

Batch = Mapping[Text, np.ndarray]


class ImageNetSplit(enum.Enum):
  """Imagenet dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name):
    return {
        'TRAIN': ImageNetSplit.TRAIN,
        'TRAIN_AND_VALID': ImageNetSplit.TRAIN_AND_VALID,
        'VALID': ImageNetSplit.VALID,
        'VALIDATION': ImageNetSplit.VALID,
        'TEST': ImageNetSplit.TEST
    }[name.upper()]

  @property
  def num_examples(self):
    return {
        ImageNetSplit.TRAIN_AND_VALID: 1281167,
        ImageNetSplit.TRAIN: 1271167,
        ImageNetSplit.VALID: 10000,
        ImageNetSplit.TEST: 50000
    }[self]


class ImageNetteSplit(enum.Enum):
  """ImageNetteSplit dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name):
    return {
        'TRAIN': ImageNetteSplit.TRAIN,
        'TRAIN_AND_VALID': ImageNetteSplit.TRAIN_AND_VALID,
        'VALID': ImageNetteSplit.VALID,
        'VALIDATION': ImageNetteSplit.VALID,
        'TEST': ImageNetteSplit.TEST
    }[name.upper()]

  @property
  def num_examples(self):
    return {
        ImageNetteSplit.TRAIN_AND_VALID: 9469,
        ImageNetteSplit.TRAIN: 9469,
        ImageNetteSplit.VALID: 0,
        ImageNetteSplit.TEST: 3925
    }[self]


class CifarSplit(enum.Enum):
  """ImageNetteSplit dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name):
    return {
        'TRAIN': CifarSplit.TRAIN,
        'TRAIN_AND_VALID': CifarSplit.TRAIN_AND_VALID,
        'VALID': CifarSplit.VALID,
        'VALIDATION': CifarSplit.VALID,
        'TEST': CifarSplit.TEST
    }[name.upper()]

  @property
  def num_examples(self):
    return {
        CifarSplit.TRAIN_AND_VALID: 50000,
        CifarSplit.TRAIN: 50000,
        CifarSplit.VALID: 0,
        CifarSplit.TEST: 10000
    }[self]


class PreprocessMode(enum.Enum):
  """Preprocessing modes for the dataset."""
  PRETRAIN = 1  # Generates two augmented views (random crop + augmentations).
  LINEAR_TRAIN = 2  # Generates a single random crop.
  EVAL = 3  # Generates a single center crop.


def normalize_images(images):
  """Normalize the image using ImageNet statistics."""
  mean_rgb = (0.485, 0.456, 0.406)
  stddev_rgb = (0.229, 0.224, 0.225)
  normed_images = images - jnp.array(mean_rgb).reshape((1, 1, 1, 3))
  normed_images = normed_images / jnp.array(stddev_rgb).reshape((1, 1, 1, 3))
  return normed_images


def load(dataset,
         subset,
         *,
         preprocess_mode,
         batch_dims,
         transpose = False,
         allow_caching = False):
  """Loads the given split of the dataset."""
  if dataset == "imagenette":
    img_size = 128
    split = ImageNetteSplit.from_string(subset)
  elif dataset in ["cifar-10", "cifar-100"]:
    img_size = 32
    split = ImageNetteSplit.from_string(subset)
  elif dataset == "imagenet":
    img_size = 224
    split = ImageNetSplit.from_string(subset)
  start, end = _shard(split, jax.host_id(), jax.host_count())

  total_batch_size = np.prod(batch_dims)

  tfds_split = tfds.core.ReadInstruction(_to_tfds_split(split),
                                         from_=start,
                                         to=end,
                                         unit='abs')

  if dataset == "imagenette":
    ds = tfds.load('imagenette/160px-v2',
                   split=tfds_split,
                   data_dir="../tensorflow_datasets",
                   decoders={'image': tfds.decode.SkipDecoding()})
  elif dataset == "imagenet":
    ds = tfds.load('imagenet2012:5.*.*',
                   split=tfds_split,
                   data_dir="../tensorflow_datasets",
                   decoders={'image': tfds.decode.SkipDecoding()})
  else:
    raise ValueError("Unknown dataset")

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1

  if preprocess_mode is not PreprocessMode.EVAL:
    options.experimental_deterministic = False
    if jax.host_count() > 1 and allow_caching:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

  else:
    if split.num_examples % total_batch_size != 0:
      raise ValueError(f'Test/valid must be divisible by {total_batch_size}')

  ds = ds.with_options(options)

  def preprocess_pretrain(example):
    view1 = _preprocess_image(example['image'], img_size, mode=preprocess_mode)
    view2 = _preprocess_image(example['image'], img_size, mode=preprocess_mode)
    label = tf.cast(example['label'], tf.int32)
    return {'view1': view1, 'view2': view2, 'labels': label}

  def preprocess_linear_train(example):
    image = _preprocess_image(example['image'], img_size, mode=preprocess_mode)
    label = tf.cast(example['label'], tf.int32)
    return {'images': image, 'labels': label}

  def preprocess_eval(example):
    image = _preprocess_image(example['image'], img_size, mode=preprocess_mode)
    label = tf.cast(example['label'], tf.int32)
    return {'images': image, 'labels': label}

  if preprocess_mode is PreprocessMode.PRETRAIN:
    ds = ds.map(preprocess_pretrain,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif preprocess_mode is PreprocessMode.LINEAR_TRAIN:
    ds = ds.map(preprocess_linear_train,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    ds = ds.map(preprocess_eval,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def transpose_fn(batch):
    # We use the double-transpose-trick to improve performance for TPUs. Note
    # that this (typically) requires a matching HWCN->NHWC transpose in your
    # model code. The compiler cannot make this optimization for us since our
    # data pipeline and model are compiled separately.
    batch = dict(**batch)
    if preprocess_mode is PreprocessMode.PRETRAIN:
      batch['view1'] = tf.transpose(batch['view1'], (1, 2, 3, 0))
      batch['view2'] = tf.transpose(batch['view2'], (1, 2, 3, 0))
    else:
      batch['images'] = tf.transpose(batch['images'], (1, 2, 3, 0))
    return batch

  for i, batch_size in enumerate(reversed(batch_dims)):
    ds = ds.batch(batch_size)
    if i == 0 and transpose:
      ds = ds.map(transpose_fn)  # NHWC -> HWCN

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  yield from tfds.as_numpy(ds)


def _to_tfds_split(split):
  """Returns the TFDS split appropriately sharded."""
  # NOTE: Imagenet did not release labels for the test split used in the
  # competition, we consider the VALID split the TEST split and reserve
  # 10k images from TRAIN for VALID.
  if split in (split.TRAIN, split.TRAIN_AND_VALID, split.VALID):
    return tfds.Split.TRAIN
  else:
    assert split == split.TEST
    return tfds.Split.VALIDATION


def _shard(split, shard_index, num_shards):
  """Returns [start, end) for the given shard index."""
  assert shard_index < num_shards
  arange = np.arange(split.num_examples)
  shard_range = np.array_split(arange, num_shards)[shard_index]
  start, end = shard_range[0], (shard_range[-1] + 1)
  if split == split.TRAIN:
    # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000].
    offset = split.VALID.num_examples
    start += offset
    end += offset
  return start, end


def _preprocess_image(
    image_bytes,
    img_size,
    mode,
):
  """Returns processed and resized images."""
  if mode is PreprocessMode.PRETRAIN:
    image = _decode_and_random_crop(image_bytes)
    # Random horizontal flipping is optionally done in augmentations.preprocess.
  elif mode is PreprocessMode.LINEAR_TRAIN:
    image = _decode_and_random_crop(image_bytes)
    image = tf.image.random_flip_left_right(image)
  else:
    image = _decode_and_center_crop(image_bytes, img_size)
  # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
  # clamping overshoots. This means values returned will be outside the range
  # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
  assert image.dtype == tf.uint8
  image = tf.image.resize(image, [img_size, img_size],
                          tf.image.ResizeMethod.BICUBIC)
  image = tf.clip_by_value(image / 255., 0., 1.)
  return image


def _decode_and_random_crop(image_bytes, area_lb=0.08):
  """Make a random crop of 224."""
  img_size = tf.image.extract_jpeg_shape(image_bytes)
  area = tf.cast(img_size[1] * img_size[0], tf.float32)
  target_area = tf.random.uniform([], area_lb, 1.0, dtype=tf.float32) * area

  log_ratio = (tf.math.log(3 / 4), tf.math.log(4 / 3))
  aspect_ratio = tf.math.exp(
      tf.random.uniform([], *log_ratio, dtype=tf.float32))

  w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
  h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

  w = tf.minimum(w, img_size[1])
  h = tf.minimum(h, img_size[0])

  offset_w = tf.random.uniform((),
                               minval=0,
                               maxval=img_size[1] - w + 1,
                               dtype=tf.int32)
  offset_h = tf.random.uniform((),
                               minval=0,
                               maxval=img_size[0] - h + 1,
                               dtype=tf.int32)

  crop_window = tf.stack([offset_h, offset_w, h, w])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image


def transpose_images(batch):
  """Transpose images for TPU training.."""
  new_batch = dict(batch)  # Avoid mutating in place.
  if 'images' in batch:
    new_batch['images'] = jnp.transpose(batch['images'], (3, 0, 1, 2))
  else:
    new_batch['view1'] = jnp.transpose(batch['view1'], (3, 0, 1, 2))
    new_batch['view2'] = jnp.transpose(batch['view2'], (3, 0, 1, 2))
  return new_batch


def _decode_and_center_crop(
    image_bytes,
    img_size,
    jpeg_shape = None,
    crop_padding = 32,
):
  """Crops to center of image with padding then scales."""
  if jpeg_shape is None:
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]

  padded_center_crop_size = tf.cast(
      ((img_size / (img_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
  ])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image
