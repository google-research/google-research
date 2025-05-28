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

"""Utility functions for defining the transformer model."""

import jax
import jax.numpy as jnp


class RoPE:
  """Applies Rotary Position Embeddings.

  This class can be used to apply Rotary Position Embeddings (RoPE) to an
  array of vectors. The implementation follows https://arxiv.org/abs/2104.09864.

  Attributes:
    context_length: Context length of the language model.
    d: dimension of the vectors.
  """

  def __init__(self, context_length, d):
    assert d % 2 == 0
    thetas = 10000 ** (-2 * jnp.arange(d // 2) / d)
    positions = jnp.arange(context_length)
    positions_times_thetas = jnp.outer(positions, thetas)
    self._sin = jnp.sin(positions_times_thetas)
    self._cos = jnp.cos(positions_times_thetas)
    self._sin = jnp.expand_dims(self._sin, axis=-2)
    self._cos = jnp.expand_dims(self._cos, axis=-2)

  def apply(self, x):
    """Rotates the vectors using the precomputed sin and cos matrices.

    Args:
      x: Array of shape [..., context_length, num_heads, d]

    Returns:
      An array of the same shape obtained after rotating each row in input
      appropriately.
    """
    context_length, d = x.shape[-2:]
    batch_shape = x.shape[:-2]
    x = x.reshape(
        batch_shape + (context_length, d // 2, 2)
    )  # [..., context_length, d // 2, 2]
    x0, x1 = x[Ellipsis, 0], x[Ellipsis, 1]  # [..., context_length, d//2] each
    x = jnp.concatenate(
        [x0 * self._cos - x1 * self._sin, x1 * self._cos + x0 * self._sin],
        axis=-1,
    )
    return x


def sinusoidal_position_embedding(context_length, d):
  """Returns a sinusoidal position embedding.

  This function takes two arguments : context_length and an even integer d
  and returns an array of shape [context_length, d] denoting the position
  embeddings we add at each position of the context. The implementation follows
  the definition in Section 3.5 of https://arxiv.org/abs/1706.03762.

  Args:
    context_length: Context length of the model. Usually a power of 2.
    d: Embedding dimension of the model.

  Returns:
    A jax array of shape [context_length, d].
  """
  thetas = 10000 ** (-jnp.arange(0, d // 2) / (d // 2))  # [d // 2]
  positions = jnp.arange(0, context_length)  # [context_length]
  positions_times_thetas = jnp.outer(
      positions, thetas
  )  # [context_length, d // 2]
  sin = jnp.sin(positions_times_thetas)
  cos = jnp.cos(positions_times_thetas)
  pe = jnp.concatenate(
      [sin[Ellipsis, None], cos[Ellipsis, None]], axis=-1
  )  # [context_length, d // 2, 2]
  pe = pe.reshape((context_length, d))  # [context_length, d]
  return pe
