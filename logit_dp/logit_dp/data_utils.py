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

"""Utilities for creating various testing and training datsets."""

import collections
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_cifar10_data_split(
    train_batch_size,
    # Equal to the size of the CIFAR10 test dataset
    buffer_size=10000,
    seed=0,
    # Optax's version of ResNet assumes each sample has shape (1, 3, 32, 32).
    expand_channel_dim=False,
):
  """Returns CIFAR10 training and testing datasets."""
  num_classes = 10
  key = jax.random.PRNGKey(seed)

  def process_data(split):
    """Normalizes and reshapes raw CIFAR10 data."""
    dataset = tfds.load('cifar10', split=split)
    images, labels = [], []
    for example in dataset.prefetch(tf.data.experimental.AUTOTUNE):
      images.append(example['image'])
      labels.append(example['label'])
    # Stack examples into (N, 32, 32, 3) and (N,) tensors respectively.
    images = jnp.array(np.stack(images))
    labels = jnp.array(np.stack(labels))
    # Cast features to float and scale to (0.0, 1.0).
    images = images.astype(jnp.float32) / 255
    # Encoder receives CHW so transpose HWC -> CHW now.
    images = jnp.transpose(images, (0, 3, 1, 2))
    if expand_channel_dim:
      images = jnp.expand_dims(images, axis=1)
    return images, labels

  def make_lookup_arr(labels):
    """Create lookup table from class to samples."""
    class_idx_to_examples_idxs = collections.defaultdict(list)
    for y_idx, y in enumerate(labels):
      class_idx_to_examples_idxs[int(y)].append(y_idx)
    return class_idx_to_examples_idxs

  def make_paired_train_data_iterator(images, lookup_arr, key):
    """Creates a `tf.data.Dataset iterator` that generates training pairs."""
    all_pairs = []
    total_samples = 0
    for i in range(num_classes):
      num_samples = len(lookup_arr[i])
      total_samples += num_samples
      num_pairs = num_samples // 2
      key, new_key = jax.random.split(key)

      shuffle_idx = jax.random.permutation(
          new_key, np.arange(num_samples), independent=True
      )
      class_examples = images[jnp.array(lookup_arr[i])]
      class_examples = class_examples[shuffle_idx]
      all_pairs.append(
          jnp.stack(
              [class_examples[:num_pairs], class_examples[num_pairs:]], axis=1
          )
      )
    all_pairs = jnp.concatenate(all_pairs, axis=0)
    _, new_key = jax.random.split(key)
    shuffle_idx = jax.random.permutation(
        new_key, np.arange(total_samples), independent=True
    )
    tf_cifar_dataset = (
        tf.data.Dataset.from_tensor_slices(all_pairs[shuffle_idx])
        .shuffle(buffer_size, reshuffle_each_iteration=True)
        .batch(train_batch_size)
        .prefetch(1)
    )
    return tf_cifar_dataset

  # Get relevant train objects.
  train_images, train_labels = process_data('train')
  train_class_to_idx = make_lookup_arr(train_labels)
  _, new_key = jax.random.split(key)
  train_data_iterator = make_paired_train_data_iterator(
      train_images, train_class_to_idx, new_key
  )
  # Get relevant test objects.
  test_images, test_labels = process_data('test')
  test_class_to_idx = make_lookup_arr(test_labels)

  return train_data_iterator, (test_images, test_labels), test_class_to_idx


def get_cifar100_data_split(train_batch_size, buffer_size=10000):
  """Returns CIFAR100 training and testing datasets."""
  data_builder = tfds.builder('cifar100')
  data_builder.download_and_prepare()

  def _pp(data):
    im = tf.cast(data['image'], tf.float32)
    im = im / 255.0
    # Encoder receives CHW so transpose HWC -> CHW now.
    im = tf.transpose(im, [2, 0, 1])
    data['image'] = im
    # Encoder receives CHW so transpose HWC -> CHW now.
    return {'image': data['image'], 'label': data['coarse_label']}

  train_data = data_builder.as_dataset(split='train')
  train_data = train_data.map(_pp)
  train_data = (
      train_data.shuffle(
          buffer_size,
          reshuffle_each_iteration=True,
      )
      .batch(train_batch_size)
      .prefetch(1)
  )
  test_data = data_builder.as_dataset(split='test')
  test_data = test_data.map(_pp)
  test_data = test_data.batch(
      buffer_size
  )  # `buffer_size` is the number of test records in CIFAR100.
  test_inputs = test_data.get_single_element()
  return train_data, (
      jnp.array(test_inputs['image']),
      jnp.array(test_inputs['label']),
  )
