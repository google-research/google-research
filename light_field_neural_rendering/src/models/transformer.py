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

"""Transformer model."""

import dataclasses
import functools
from typing import Any, Callable, Optional, Tuple

from flax import linen as nn
from jax import lax
import jax.numpy as jnp

from light_field_neural_rendering.src.utils import config_utils

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


def _resolve(a, b):
  """Returns a if a is not None, else returns b."""
  if a is not None:
    return a
  else:
    return b


# The function is tweaked from https://github.com/google/flax/blob/main/examples/wmt/models.py
class LearnedPositionEmbs(nn.Module):
  """Learned positional embeddings."""
  max_length: int  # Max length of the 2nd dimension

  @nn.compact
  def __call__(
      self,
      inputs,
      input_positions=None,
  ):
    """Add a leaned positional embeding to the input Args:

      inputs: input data. (bs, near_view, num_proj, in_dim)

    Returns:
      output: `(bs, near_view, num_proj, in_dim)`
    """
    input_shape = inputs.shape
    pos_emb_shape = (1, self.max_length, inputs.shape[-1])

    pos_embedding = self.param('pos_embedding',
                               nn.initializers.normal(stddev=1e-6),
                               pos_emb_shape)
    if input_positions is not None:
      pos_embedding = jnp.take(pos_embedding, input_positions, axis=1)

    return pos_embedding


class Mlp(nn.Module):
  """Transformer MLP block with single hidden layer."""

  hidden_params: Optional[int] = None
  out_params: Optional[int] = None
  dropout_rate: float = 0.
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      nn.initializers.xavier_uniform())
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      nn.initializers.normal(stddev=1e-6))

  @nn.compact
  def __call__(self, inputs, deterministic):
    h = nn.Dense(
        features=_resolve(self.hidden_params, inputs.shape[-1]),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    h = nn.gelu(h)
    h = nn.Dense(
        features=_resolve(self.out_params, inputs.shape[-1]),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            h)
    return h


class SelfAttentionTransformerLayer(nn.Module):
  """Transformer layer."""
  attention_heads: int
  qkv_params: Optional[int] = None
  mlp_params: Optional[int] = None
  dropout_rate: float = 0.

  @nn.compact
  def __call__(self, query, deterministic):
    out_params = query.shape[-1]
    aux = {}

    # Attention from query to value
    attention_output = nn.SelfAttention(
        num_heads=self.attention_heads,
        qkv_features=self.qkv_params,
        out_features=out_params,
        dropout_rate=self.dropout_rate)(
            query, deterministic=deterministic)
    normalized_attention_output = nn.LayerNorm()(query + attention_output)

    mlp_output = Mlp(
        hidden_params=self.mlp_params,
        out_params=out_params,
        dropout_rate=self.dropout_rate)(
            normalized_attention_output, deterministic=deterministic)
    return nn.LayerNorm()(normalized_attention_output + mlp_output)


class SelfAttentionTransformer(nn.Module):
  """Self Attention Transformer."""
  params: config_utils.TransformerParams  # Network parameters.

  @nn.compact
  def __call__(self, points, deterministic):
    """Call the transformer on a set of inputs."""
    for i in range(self.params.num_layers):
      points = SelfAttentionTransformerLayer(
          attention_heads=self.params.attention_heads,
          qkv_params=self.params.qkv_params,
          mlp_params=self.params.mlp_params,
          dropout_rate=self.params.dropout_rate)(
              query=points, deterministic=deterministic)
    return points
