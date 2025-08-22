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

"""Return training and evaluation/test datasets from config files.

This code is adapted from
https://github.com/yang-song/score_sde/blob/main/datasets.py.
"""
from typing import Any, Optional, Tuple

import jax
import ml_collections
from score_sde import datasets as default_datasets
import tensorflow as tf
import tensorflow_datasets as tfds

SUPPORTED_DATASETS = ['CIFAR10', 'CELEBA', 'SVHN']

get_data_scaler = default_datasets.get_data_scaler
get_data_inverse_scaler = default_datasets.get_data_inverse_scaler


def get_dataset_builder_and_resize_op(config):
  """Create dataset builder and image resizing function for dataset."""
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      if config.data.num_channels == 1:
        img = tf.image.rgb_to_grayscale(img)
      return tf.image.resize(
          img, [config.data.image_size, config.data.image_size])

  elif config.data.dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a')

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = default_datasets.central_crop(img, 140)
      if config.data.num_channels == 1:
        img = tf.image.rgb_to_grayscale(img)
      return tf.image.resize(
          img, [config.data.image_size, config.data.image_size])

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      if config.data.num_channels == 1:
        img = tf.image.rgb_to_grayscale(img)
      return tf.image.resize(
          img, [config.data.image_size, config.data.image_size])

  else:
    raise ValueError(
        f'Dataset {config.data.dataset} not supported.')
  return dataset_builder, resize_op


def get_preprocess_fn(config, resize_op, uniform_dequantization=False,
                      evaluation=False):
  """Create preprocessing function for dataset."""
  @tf.autograph.experimental.do_not_convert
  def preprocess_fn(d):
    """Basic preprocessing function scales data to [0, 1) and randomly flips."""
    img = resize_op(d['image'])
    if config.data.random_flip and not evaluation:
      img = tf.image.random_flip_left_right(img)
    if uniform_dequantization:
      img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
    return dict(image=img, label=d.get('label', None))

  return preprocess_fn


def get_dataset(
    config,
    additional_dim = None,
    uniform_dequantization = False,
    evaluation = False,
    shuffle_seed = None,
    device_batch = True
):
  """Create data loaders for training, validation, and testing.

  Most of the logic from `score_sde/datasets.py` is kept.

  Args:
    config: The config.
    additional_dim: If not `None`, add an additional dimension
      to the output data for jitted steps.
    uniform_dequantization: If `True`, uniformly dequantize the images.
      This is usually only used when evaluating log-likelihood [bits/dim]
      of the data.
    evaluation: If `True`, fix number of epochs to 1.
    shuffle_seed: Optional seed for shuffling dataset.
    device_batch: If `True`, divide batch size into device batch and
      local batch.

  Returns:
    train_ds, val_ds, test_ds.
  """
  if config.data.dataset not in SUPPORTED_DATASETS:
    raise NotImplementedError(
        f'Dataset {config.data.dataset} not yet supported.')

  # Compute batch size for this worker.
  batch_size = (
      config.training.batch_size if not evaluation else config.eval.batch_size)
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by '
                     f'the number of devices ({jax.device_count()})')

  per_device_batch_size = batch_size // jax.device_count()
  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1
  # Create additional data dimension when jitting multiple steps together
  if not device_batch:
    batch_dims = [batch_size]
  elif additional_dim is None:
    batch_dims = [jax.local_device_count(), per_device_batch_size]
  else:
    batch_dims = [
        jax.local_device_count(), additional_dim, per_device_batch_size
    ]

  # Get dataset builder.
  dataset_builder, resize_op = get_dataset_builder_and_resize_op(config)

  # Get preprocessing function.
  preprocess_fn = get_preprocess_fn(
      config, resize_op, uniform_dequantization, evaluation)

  def create_dataset(dataset_builder, split,
                     take_val_from_train = False,
                     train_split = 0.9):
    # Some datasets only include train and test sets, in which case we take
    # validation data from the training set.
    if split == 'test':
      take_val_from_train = False
    source_split = 'train' if take_val_from_train else split

    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(
        options=dataset_options, shuffle_seed=shuffle_seed)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
          split=source_split, shuffle_files=True, read_config=read_config)
    elif config.data.dataset in [
        'DeadLeaves', 'Eigenfaces', 'fastMRI'
    ]:
      ds = dataset_builder[source_split].with_options(dataset_options)
    else:
      ds = dataset_builder.with_options(dataset_options)

    if take_val_from_train:
      train_size = int(train_split * len(ds))
      # Take the first `train_split` pct. for training and the rest for val.
      ds = ds.take(train_size) if split == 'train' else ds.skip(train_size)

    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for batch_size in reversed(batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  # Set the correct split names.
  if config.data.dataset == 'CIFAR10':
    train_ds = create_dataset(
        dataset_builder, 'train', take_val_from_train=True)  # 50,000 * 0.9
    val_ds = create_dataset(
        dataset_builder, 'validation', take_val_from_train=True)  # 50,000 * 0.1
    test_ds = create_dataset(dataset_builder, 'test')  # 10,000
  elif config.data.dataset == 'CELEBA':
    train_ds = create_dataset(dataset_builder, 'train')  # 162,770
    test_ds = create_dataset(dataset_builder, 'test')  # 19,962
    val_ds = create_dataset(dataset_builder, 'validation')  # 19,867
  elif config.data.dataset == 'SVHN':
    train_ds = create_dataset(dataset_builder, 'train')  # 73,257
    test_ds = create_dataset(dataset_builder, 'test')  # 26,032
    val_ds = create_dataset(dataset_builder, 'extra')  # 531,131

  return train_ds, val_ds, test_ds
