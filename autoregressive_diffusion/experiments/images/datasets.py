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

"""Contains the methods to get different datasets.

This file contains a collection of datasets, returning train and test
partitions.
"""

from clu import deterministic_data
import jax
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds


def create_datasets(config, data_rng):
  """Create datasets for training and evaluation."""
  # Compute batch size per device from global batch size.
  if config.batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch size ({config.batch_size}) must be divisible by '
                     f'the number of devices ({jax.device_count()}).')
  per_device_batch_size = config.batch_size // jax.device_count()

  dataset_builder = tfds.builder(config.dataset)

  def cast_int32(batch):
    img = tf.cast(batch['image'], tf.int32)
    out = batch.copy()
    out['image'] = img
    return out

  def drop_info(batch):
    """Removes unwanted keys from batch."""
    if 'id' in batch:
      batch.pop('id')
    if 'rng' in batch:
      batch.pop('rng')
    return batch

  if config.data_augmentation:
    should_augment = True
    should_randflip = True
    should_rotate = True
  else:
    should_augment = False
    should_randflip = False
    should_rotate = False

  def augment(batch):
    img = tf.cast(batch['image'], tf.float32)
    aug = None
    if should_augment:
      if should_randflip:
        img_flipped = tf.image.flip_left_right(img)
        aug = tf.random.uniform(shape=[]) > 0.5
        img = tf.where(aug, img_flipped, img)
      if should_rotate:
        u = tf.random.uniform(shape=[])
        k = tf.cast(tf.floor(4. * u), tf.int32)
        img = tf.image.rot90(img, k=k)
        aug = aug | (k > 0)
    if aug is None:
      aug = tf.convert_to_tensor(False, dtype=tf.bool)

    out = batch.copy()
    out['image'] = img
    return out

  def preprocess_train(batch):
    return cast_int32(augment(drop_info(batch)))

  def preprocess_eval(batch):
    return cast_int32(drop_info(batch))

  # Read instructions to shard the dataset!
  print('train', dataset_builder.info.splits['train'].num_examples)
  # TODO(emielh) use dataset_info instead of num_examples.
  train_split = deterministic_data.get_read_instruction_for_host(
      'train', num_examples=dataset_builder.info.splits['train'].num_examples)
  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      num_epochs=1,
      shuffle=True,
      batch_dims=[jax.local_device_count(), per_device_batch_size],
      preprocess_fn=preprocess_train,
      rng=data_rng,
      prefetch_size=tf.data.AUTOTUNE,
      drop_remainder=True
      )

  # TODO(emielh) check if this is necessary?

  # Test batches are _not_ sharded. In the worst case, this simply leads to some
  # duplicated information. In our case, since the elbo is stochastic we get
  # multiple passes over the test data.
  if config.test_batch_size % jax.local_device_count() != 0:
    raise ValueError(f'Batch size ({config.batch_size}) must be divisible by '
                     f'the number of devices ({jax.local_device_count()}).')
  test_device_batch_size = config.test_batch_size // jax.local_device_count()

  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split='test',
      # Repeated epochs for lower variance ELBO estimate.
      num_epochs=config.num_eval_passes,
      shuffle=False,
      batch_dims=[jax.local_device_count(), test_device_batch_size],
      preprocess_fn=preprocess_eval,
      # TODO(emielh) Fix this with batch padding instead of dropping.
      prefetch_size=tf.data.AUTOTUNE,
      drop_remainder=False)

  return dataset_builder.info, train_ds, eval_ds


def get_dataset(config, data_rng):
  """Function that combines data loading for different datasets."""
  _, train_ds, test_ds = create_datasets(config, data_rng)

  if config.dataset == 'mnist':
    shape = (28, 28, 1)
    n_classes = 256
  elif config.dataset == 'binarized_mnist':
    shape = (28, 28, 1)
    n_classes = 2
  elif config.dataset == 'cifar10':
    shape = (32, 32, 3)
    n_classes = 256
  else:
    raise ValueError

  return train_ds, test_ds, shape, n_classes
