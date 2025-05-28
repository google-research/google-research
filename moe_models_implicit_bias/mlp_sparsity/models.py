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

"""Different model architectures used in this experiment."""

import functools
from flax import linen as nn
import jax
import ml_collections


class MLP2BNLN(nn.Module):
  """A 2 layer MLP model with Batch Norm and Layer Norm."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)
    norm2 = functools.partial(
        nn.LayerNorm, epsilon=1e-5, use_bias=(self.config.sparsity == 1))
    c = self.config.width
    x = nn.Dense(features=c)(x)
    x = norm1()(x)
    x = norm2()(x)
    z1 = nn.relu(x)
    x = nn.Dense(features=1)(z1)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, [z1]


class MLP2Usual(nn.Module):
  """A 2 layer MLP model with Layer Norm."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)
    c = self.config.width
    x = nn.Dense(features=c)(x)
    x = norm1()(x)
    z1 = nn.relu(x)
    x = nn.Dense(features=1)(z1)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, [z1]


class MLP2NoNorm(nn.Module):
  """A 2 layer MLP model with no Batch Norm or Layer Norm.."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    c = self.config.width
    x = nn.Dense(features=c)(x)
    z1 = nn.relu(x)
    x = nn.Dense(features=1)(z1)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, [z1]


class MLP3(nn.Module):
  """A 3 layer MLP model."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)
    norm2 = functools.partial(
        nn.LayerNorm, epsilon=1e-5, use_bias=(self.config.sparsity == 1))
    c = self.config.width
    x = nn.Dense(features=c)(x)
    x = norm1()(x)
    x = norm2()(x)
    z1 = nn.relu(x)
    x = nn.Dense(features=c)(z1)
    x = norm1()(x)
    x = norm2()(x)
    z2 = nn.relu(x)
    x = nn.Dense(features=1)(z2)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, [z1, z2]


class MLPTransformer5(nn.Module):
  """A Transformer-like MLP model."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)

    c = self.config.width
    x = nn.Dense(features=c // 4)(x)
    x = norm1()(x)
    z = []

    for _ in range(5):
      x2 = nn.Dense(features=c)(x)
      x2 = nn.relu(x2)
      z += [jax.lax.stop_gradient(x2)]
      x2 = nn.Dense(features=c // 4)(x2)
      x2 = nn.relu(x2)
      z += [jax.lax.stop_gradient(x2)]
      x += x2
      x = norm1()(x)

    x = nn.Dense(features=1)(x)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, z


class MLPTransformer51(nn.Module):
  """A Transformer-like MLP model."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)

    c = self.config.width
    x = nn.Dense(features=c)(x)
    x = norm1()(x)
    z = []

    for _ in range(5):
      x2 = nn.Dense(features=c)(x)
      x2 = nn.relu(x2)
      z += [jax.lax.stop_gradient(x2)]
      x2 = nn.Dense(features=c)(x2)
      x2 = nn.relu(x2)
      z += [jax.lax.stop_gradient(x2)]
      x += x2
      x = norm1()(x)

    x = nn.Dense(features=1)(x)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, z


class MLPTransformer52(nn.Module):
  """A Transformer-like MLP model."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)

    c = self.config.width
    x = nn.Dense(features=c)(x)
    x = norm1()(x)
    z = []

    for _ in range(10):
      x2 = nn.Dense(features=c)(x)
      x2 = nn.relu(x2)
      z += [jax.lax.stop_gradient(x2)]
      x += x2
      x = norm1()(x)

    x = nn.Dense(features=1)(x)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, z


class MLPTransformer53(nn.Module):
  """A Transformer-like MLP model."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)

    c = self.config.width
    x = nn.Dense(features=c // 4)(x)
    x = norm1()(x)
    z = []

    for _ in range(5):
      x = nn.Dense(features=c)(x)
      x = nn.relu(x)
      z += [jax.lax.stop_gradient(x)]
      x = nn.Dense(features=c // 4)(x)
      x = nn.relu(x)
      z += [jax.lax.stop_gradient(x)]
      x = norm1()(x)

    x = nn.Dense(features=1)(x)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, z


class MLPTransformer54(nn.Module):
  """A Transformer-like MLP model."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)

    c = self.config.width
    x = nn.Dense(features=c)(x)
    x = norm1()(x)
    z = []

    for _ in range(5):
      x = nn.Dense(features=c)(x)
      x = nn.relu(x)
      z += [jax.lax.stop_gradient(x)]
      x = nn.Dense(features=c)(x)
      x = nn.relu(x)
      z += [jax.lax.stop_gradient(x)]
      x = norm1()(x)

    x = nn.Dense(features=1)(x)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, z


class MLPTransformer55(nn.Module):
  """A Transformer-like MLP model."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)

    c = self.config.width
    x = nn.Dense(features=c)(x)
    x = norm1()(x)
    z = []

    for _ in range(10):
      x = nn.Dense(features=c)(x)
      x = nn.relu(x)
      z += [jax.lax.stop_gradient(x)]
      x = norm1()(x)

    x = nn.Dense(features=1)(x)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, z


class MLP12(nn.Module):
  """A 12-layer MLP model."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)

    c = self.config.width
    z = []

    for _ in range(12):
      x = nn.Dense(features=c)(x)
      x = norm1()(x)
      x = nn.relu(x)
      z += [jax.lax.stop_gradient(x)]

    x = nn.Dense(features=1)(x)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, z


class MLPTransformer10(nn.Module):
  """A Transformer-like MLP model."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):
    norm1 = functools.partial(nn.LayerNorm, epsilon=1e-5)

    c = self.config.width
    x = nn.Dense(features=c // 4)(x)
    x = norm1()(x)
    z = []

    for _ in range(5):
      x2 = nn.Dense(features=c)(x)
      x2 = nn.relu(x2)
      z += [jax.lax.stop_gradient(x2)[:100]]
      x2 = nn.Dense(features=c // 4)(x2)
      x2 = nn.relu(x2)
      z += [jax.lax.stop_gradient(x2)[:100]]
      x += x2
      x = norm1()(x)

    x = nn.Dense(features=1)(x)

    norm_fake = functools.partial(
        nn.BatchNorm, use_running_average=False, momentum=0.99, epsilon=1e-5)

    _ = norm_fake()(x)

    return x, z
