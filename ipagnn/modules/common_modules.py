# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Common modules used by Learned Interpreter models."""

from absl import logging  # pylint: disable=unused-import
from flax import nn
import jax
import jax.numpy as jnp


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """

  def apply(self,
            inputs,
            num_embeddings,
            features,
            mode='input',
            emb_init=nn.initializers.normal(stddev=1.0)):
    """Applies Embed module.

    Args:
      inputs: input data
      num_embeddings: number of embedding
      features: size of the embedding dimension
      mode: either 'input' or 'output' -> to share input/output embedding
      emb_init: embedding initializer

    Returns:
      output which is embedded input data
    """
    embedding = self.param('embedding', (num_embeddings, features), emb_init)
    if mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      return jnp.take(embedding, inputs, axis=0)
    if mode == 'output':
      return jnp.einsum('bld,vd->blv', inputs, embedding)


class Tag(nn.Module):
  """Save a value to global state when running in stateful mode."""

  def apply(self, x):
    if self.is_stateful():
      tagged = self.state('tag')
      tagged.value = x
    return x


class StackedRNNCell(nn.Module):
  """Stacked RNN Cell."""

  def apply(self, carry, inputs, cells):
    new_carry = []
    for c, cell in zip(carry, cells):
      new_c, inputs = cell(c, inputs)
      new_carry.append(new_c)
    return new_carry, inputs

  @staticmethod
  def initialize_carry(rng, cells, batch_dims, size,
                       init_fn=nn.initializers.zeros):
    keys = jax.random.split(rng, len(cells))
    return [
        cell.initialize_carry(key, batch_dims, size, init_fn=init_fn)
        for key, cell in zip(keys, cells)
    ]
