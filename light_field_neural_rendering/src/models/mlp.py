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

"""File contaning MLP models."""

import functools

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from light_field_neural_rendering.src.utils import config_utils


class SimpleMLP(nn.Module):
  """A simple MLP.

  Adapted from jaxnerf with following modifications.
  (https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/model_utils.py)
  1. No conditioning option.
  2. Only return rgb.
  3. Use DenseGeneral instead of Dense.
  """
  config: config_utils.MLPParams

  @nn.compact
  def __call__(self, x):
    """Evaluate the MLP.

    Args:
      x: jnp.ndarray(float32), [batch, num_samples, feature], points.

    Returns:
      raw_rgb: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_rgb_channels].
    """
    dense_layer = functools.partial(
        nn.DenseGeneral, kernel_init=jax.nn.initializers.glorot_uniform())
    inputs = x
    for i in range(self.config.net_depth):
      x = dense_layer(self.config.net_width)(x)
      x = self.config.net_activation(x)
      if i % self.config.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)
    raw_rgb = dense_layer(self.config.num_rgb_channels)(x)
    return raw_rgb
