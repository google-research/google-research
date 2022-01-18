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

"""Normalization layers."""
import flax.deprecated.nn as nn
import jax.nn.initializers as init
import jax.numpy as jnp


def get_normalization(config, conditional=False):
  """Obtain normalization modules from the config file."""
  norm = config.model.normalization
  if conditional:
    if norm == 'InstanceNorm++':
      return ConditionalInstanceNorm2dPlus.partial(
          num_classes=config.model.num_classes)
    else:
      raise NotImplementedError(f'{norm} not implemented yet.')
  else:
    if norm == 'InstanceNorm':
      return InstanceNorm2d
    elif norm == 'InstanceNorm++':
      return InstanceNorm2dPlus
    elif norm == 'VarianceNorm':
      return VarianceNorm2d
    elif norm == 'GroupNorm':
      return nn.GroupNorm
    else:
      raise ValueError('Unknown normalization: %s' % norm)


class VarianceNorm2d(nn.Module):
  """Variance normalization for images."""

  def apply(self, x, bias=False):

    variance = jnp.var(x, axis=(1, 2), keepdims=True)
    h = x / jnp.sqrt(variance + 1e-5)

    normal_init = init.normal(0.02)
    def scale_init(key, shape, dtype=jnp.float32):
      return normal_init(key, shape, dtype=dtype) + 1.
    bias_init = init.zeros

    h = h * self.param('scale', (1, 1, 1, x.shape[-1]), scale_init)
    if bias:
      h = h + self.param('bias', (1, 1, 1, x.shape[-1]), bias_init)

    return h


class InstanceNorm2d(nn.Module):
  """Instance normalization for images."""

  def apply(self, x, bias=True):
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    variance = jnp.var(x, axis=(1, 2), keepdims=True)
    h = (x - mean) / jnp.sqrt(variance + 1e-5)
    h = h * self.param('scale', (1, 1, 1, x.shape[-1]), init.ones)
    if bias:
      h = h + self.param('bias', (1, 1, 1, x.shape[-1]), init.zeros)

    return h


class InstanceNorm2dPlus(nn.Module):
  """InstanceNorm++ as proposed in the original NCSN paper."""

  def apply(self, x, bias=True):
    means = jnp.mean(x, axis=(1, 2))
    m = jnp.mean(means, axis=-1, keepdims=True)
    v = jnp.var(means, axis=-1, keepdims=True)
    means_plus = (means - m) / jnp.sqrt(v + 1e-5)

    h = (x - means[:, None, None, :]
        ) / jnp.sqrt(jnp.var(x, axis=(1, 2), keepdims=True) + 1e-5)

    normal_init = init.normal(0.02)
    def scale_init(key, shape, dtype=jnp.float32):
      return normal_init(key, shape, dtype=dtype) + 1.

    bias_init = init.zeros

    h = h + means_plus[:, None, None, :] * self.param(
        'alpha', (1, 1, 1, x.shape[-1]), scale_init)
    h = h * self.param('gamma', (1, 1, 1, x.shape[-1]), scale_init)
    if bias:
      h = h + self.param('beta', (1, 1, 1, x.shape[-1]), bias_init)

    return h


class ConditionalInstanceNorm2dPlus(nn.Module):
  """Conditional InstanceNorm++ as in the original NCSN paper."""

  def apply(self, x, y, num_classes=10, bias=True):
    means = jnp.mean(x, axis=(1, 2))
    m = jnp.mean(means, axis=-1, keepdims=True)
    v = jnp.var(means, axis=-1, keepdims=True)
    means_plus = (means - m) / jnp.sqrt(v + 1e-5)
    h = (x - means[:, None, None, :]
        ) / jnp.sqrt(jnp.var(x, axis=(1, 2), keepdims=True) + 1e-5)
    normal_init = init.normal(0.02)
    zero_init = init.zeros
    if bias:
      def init_embed(key, shape, dtype=jnp.float32):
        feature_size = shape[1] // 3
        normal = normal_init(
            key, (shape[0], 2 * feature_size), dtype=dtype) + 1.
        zero = zero_init(key, (shape[0], feature_size), dtype=dtype)
        return jnp.concatenate([normal, zero], axis=-1)

      embed = nn.Embed.partial(
          num_embeddings=num_classes,
          features=x.shape[-1] * 3,
          embedding_init=init_embed)
    else:
      def init_embed(key, shape, dtype=jnp.float32):
        return normal_init(key, shape, dtype=dtype) + 1.
      embed = nn.Embed.partial(num_embeddings=num_classes,
                               features=x.shape[-1] * 2,
                               embedding_init=init_embed)

    if bias:
      gamma, alpha, beta = jnp.split(embed(y), 3, axis=-1)
      h = h + means_plus[:, None, None, :] * alpha[:, None, None, :]
      out = gamma[:, None, None, :] * h + beta[:, None, None, :]
    else:
      gamma, alpha = jnp.split(embed(y), 2, axis=-1)
      h = h + means_plus[:, None, None, :] * alpha[:, None, None, :]
      out = gamma[:, None, None, :] * h

    return out
