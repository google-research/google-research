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

"""Methods and data structures used to test this library."""

from typing import Any, Callable, Optional, Sequence

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp

from ev3.utils import data_util


def theta2label(angle, num_labels):
  # Divides the interval [-π,π] into NUM_LABELS equal intervals and returns
  # the index of the segment that θ falls into.
  pi = jnp.pi
  return jnp.floor(num_labels * (angle - (-pi)) / (pi - (-pi)))


def get_data_iterator(
    num_features = 0,
    num_labels = 0,
    data_size = 0,
    data_rand_key = None,
    param_rand_key = None,
    batch_size = 0,
    dataset_name = 'test',
    process_fn = None,
):
  """Returns a test data iterator that generates batches of data."""
  assert batch_size > 0
  if dataset_name == 'test':
    assert min(num_labels, num_features) == 2, (
        'In this test, if you want to have more than 2 labels, the number of '
        'features needs to be equal to 2. (num_labels, num_features)='
        f'{(num_labels, num_features)}'
    )
    assert data_size > 0

    # Sample features
    data_iter_key, feature_key = jax.random.split(data_rand_key, 2)
    features = jax.random.normal(feature_key, [data_size, num_features])

    # Compute labels
    if num_features == 2 and num_labels > 2:
      angles = jnp.arctan2(features[:, 1], features[:, 0])
      labels = theta2label(angles, num_labels)
    else:
      true_weights = jax.random.normal(param_rand_key, [num_features])
      labels = (features @ true_weights >= 0) + 0

    all_data = {'feature': features, 'label': labels}

    ds = data_util.TestDataIterator(
        all_data=all_data,
        batch_size=batch_size,
        n_all_data=data_size,
        rand_key=data_iter_key,
    )
  else:
    if process_fn is None and dataset_name in ['mnist', 'cifar10']:
      process_fn = data_util.normalize_img
    ds = data_util.TFDataIterator(
        ds_name=dataset_name, batch_size=batch_size, process_fn=process_fn
    )

  return ds


def get_batch_size_for_xent(args, batch_axes):
  """Extract the number of samples in a batch of data."""
  arg_batch_sizes = [
      arg.shape[ax] for ax, arg in zip(batch_axes, args) if ax is not None
  ]
  assert max(arg_batch_sizes) == min(arg_batch_sizes)
  return arg_batch_sizes[0]


class TwoLayerPerceptron(nn.Module):
  num_hidden_nodes: int
  num_labels: int

  @nn.compact
  def __call__(self, inputs, **kwargs):
    x = inputs
    h = nn.Dense(self.num_hidden_nodes, name='hidden_layer')(x)
    h = nn.relu(h)
    y = nn.Dense(self.num_labels, name='output')(h)
    return y


class MLP(nn.Module):
  layer_widths: Sequence[int]
  num_labels: int

  @nn.compact
  def __call__(self, inputs, **kwargs):
    x = inputs.reshape(inputs.shape[0], -1)
    for i, n_nodes in enumerate(self.layer_widths):
      x = nn.Dense(n_nodes, name=f'layer_{i}')(x)
      x = nn.relu(x)
    y = nn.Dense(self.num_labels, name='output')(x)
    return y


class CNN(nn.Module):
  """A simple CNN model."""

  num_classes: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, **kwargs):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_classes)(x)
    return x
