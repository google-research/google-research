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

"""Different model architectures used in the CIFAR-10 MoE experiment."""

import functools
from flax import linen as nn


class CNN(nn.Module):
  """A simple CNN model."""
  proj_dim: int

  @nn.compact
  def __call__(self, x, train = True):
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.99,
        epsilon=1e-5)
    c = 128
    x = nn.Conv(features=c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.Conv(features=2 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=4 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=8 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(8, 8), strides=(8, 8))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=self.proj_dim)(x)
    return x


class CNNNorm(nn.Module):
  """A simple CNN model with Batch norm at the final layer."""
  proj_dim: int

  @nn.compact
  def __call__(self, x, train = True):
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.99,
        epsilon=1e-5)
    c = 128
    x = nn.Conv(features=c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.Conv(features=2 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=4 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=8 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(8, 8), strides=(8, 8))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=self.proj_dim)(x)
    x = norm()(x)
    return x


class CNNDense200(nn.Module):
  """A simple CNN model."""
  proj_dim: int

  @nn.compact
  def __call__(self, x, train = True):
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.99,
        epsilon=1e-5)
    c = 128
    x = nn.Conv(features=c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.Conv(features=2 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=4 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=8 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(8, 8), strides=(8, 8))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=200)(x)
    x = norm()(x)
    z = nn.relu(x)
    x = nn.Dense(features=self.proj_dim)(z)
    return x, z


class CNNSoftmax200(nn.Module):
  """An CNN model with MoE."""
  proj_dim: int

  @nn.compact
  def __call__(self, x, train = True):
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.99,
        epsilon=1e-5)

    c = 128
    x = nn.Conv(features=c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.Conv(features=2 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=4 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=8 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(8, 8), strides=(8, 8))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=200)(x)
    x = 4. * nn.LayerNorm(use_scale=False, use_bias=False)(x)
    z = nn.softmax(x)
    x = nn.LayerNorm()(z)
    x = nn.Dense(features=self.proj_dim)(x)
    return x, z


class CNNMax200(nn.Module):
  """A CNN model with another variant of MoE."""
  proj_dim: int

  @nn.compact
  def __call__(self, x, train = True):
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.99,
        epsilon=1e-5)
    c = 128
    x = nn.Conv(features=c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.Conv(features=2 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=4 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=8 * c, kernel_size=(3, 3), padding=1)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(8, 8), strides=(8, 8))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=200)(x)
    x = norm()(x)
    x = nn.softmax(100 * x)
    x = nn.Dense(features=self.proj_dim)(x)
    x = norm()(x)
    return x
