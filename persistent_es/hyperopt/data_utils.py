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

"""Data loading utilities.
"""
from functools import partial

import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds


NUM_TRAIN = { 'mnist': 50000,
              'fashion_mnist': 50000,
              'cifar10': 40000,
              'cifar100': 45000,
            }

means = { 'mnist': 0.13066,
          'fashion_mnist': 0.28604,
          'cifar10': jnp.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
          'cifar100': jnp.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), }

stds = { 'mnist': 0.30811,
         'fashion_mnist': 0.35302,
         'cifar10': jnp.array([x / 255.0 for x in [63.0, 62.1, 66.7]]),
         'cifar100': jnp.array([x / 255.0 for x in [63.0, 62.1, 66.7]]), }


def load_data(dataset_name):
  train_dataset = tfds.load(dataset_name, batch_size=-1, split='train')

  all_train_images = tfds.as_numpy(train_dataset['image'])[:NUM_TRAIN[dataset_name]] / 255.0
  all_train_labels = tfds.as_numpy(train_dataset['label'])[:NUM_TRAIN[dataset_name]]

  all_val_images = tfds.as_numpy(train_dataset['image'])[NUM_TRAIN[dataset_name]:] / 255.0
  all_val_labels = tfds.as_numpy(train_dataset['label'])[NUM_TRAIN[dataset_name]:]

  test_dataset = tfds.load(dataset_name, batch_size=-1, split='test')
  all_test_images = tfds.as_numpy(test_dataset['image']) / 255.0
  all_test_labels = tfds.as_numpy(test_dataset['label'])

  train_data = jnp.array((all_train_images - means[dataset_name]) / stds[dataset_name]).transpose((0,3,1,2))
  train_targets = jnp.array(all_train_labels)

  val_data = jnp.array((all_val_images - means[dataset_name]) / stds[dataset_name]).transpose((0,3,1,2))
  val_targets = jnp.array(all_val_labels)

  test_data = jnp.array((all_test_images - means[dataset_name]) / stds[dataset_name]).transpose((0,3,1,2))
  test_targets = jnp.array(all_test_labels)

  return (train_data, train_targets), (val_data, val_targets), (test_data, test_targets)

@partial(jax.jit, static_argnums=3)
def create_minibatches(key, data, targets, batch_size):
  N, channels, width, height = data.shape
  idx_perm = jax.random.permutation(key, N)
  data = data[idx_perm].reshape((N//batch_size, batch_size, channels, width, height))
  targets = targets[idx_perm].reshape((N//batch_size, batch_size))
  return data, targets
