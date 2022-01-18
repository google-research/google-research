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

"""Time embedding taken from diffusion/v3.

This contains a function to obatin a time embedding to utilize in diffusion or
ARM models.
"""
from typing import Callable

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp


Array = jnp.ndarray


def get_timestep_embedding(timesteps, embedding_dim,
                           max_time=1000., dtype=jnp.float32):
  """Build sinusoidal embeddings (from Fairseq).

  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".

  Args:
    timesteps: jnp.ndarray: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    max_time: float: largest time input
    dtype: data type of the generated embeddings

  Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  timesteps *= (1000. / max_time)

  half_dim = embedding_dim // 2
  emb = jnp.log(10000) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
  emb = timesteps.astype(dtype)[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


class TimeEmbedding(nn.Module):
  """Time embedding module."""
  num_channels: int
  max_time: float
  activation: Callable[[Array], Array] = jax.nn.swish

  @nn.compact
  def __call__(self, t):
    """Gets time embedding via the sinusoidal embedding and two dense layers.

    Args:
      t: Input timesteps that need to be embedded.

    Returns:
      Embeddings with shape `(len(timesteps), 4 * embedding_dim)`.
    """
    logging.info('model max_time: %f', self.max_time)
    temb = get_timestep_embedding(
        t, self.num_channels, max_time=self.max_time)
    temb = nn.Dense(
        features=self.num_channels * 4, name='dense0')(temb)
    temb = nn.Dense(
        features=self.num_channels * 4, name='dense1')(self.activation(temb))
    assert temb.shape == (t.shape[0], self.num_channels * 4)
    return temb

