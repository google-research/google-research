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

"""Implements a sketch layer with learnable sketches."""

from flax import linen as nn
import jax.numpy as jnp


class FFSketch(nn.Module):
  """Implements a simple 1 hidden layer network.

  Attributes:
    sketch_size: Number of features in the output vector
    expansion_factor: Multiplicative factor in the hidden layer
  """

  sketch_size: int
  expansion_factor: int

  @nn.compact
  def __call__(self, x):
    x = nn.LayerNorm(epsilon=1e-5)(x)
    x = nn.Dense(self.expansion_factor * self.sketch_size, use_bias=False)(x)
    x = nn.gelu(x)
    x = nn.Dense(self.sketch_size, use_bias=False)(x)
    return x


class SketchLayer(nn.Module):
  """Implements a sketch layer with learnable sketches.

  Attributes:
    sketch_dim: Number of features in the output vector
    sketch_levels: Number of sketch applications. We use two.
    expansion_factor: Expansion factor in hidden layer of each sketch level.
  """

  sketch_dim: int
  sketch_levels: int
  expansion_factor: int

  def setup(self):
    self.ff1 = [
        FFSketch(self.sketch_dim, self.expansion_factor)
        for _ in range(self.sketch_levels)
    ]

    self.ff2 = [
        FFSketch(self.sketch_dim, self.expansion_factor)
        for _ in range(self.sketch_levels)
    ]

  @nn.compact
  def __call__(self, x):
    x1 = x
    x2 = x
    for i in range(self.sketch_levels):
      x1 = self.ff1[i](x1)
      x2 = self.ff2[i](x2)
    return jnp.tanh(x1 * x2 / jnp.sqrt(self.sketch_dim)) * jnp.sqrt(
        self.sketch_dim
    )
