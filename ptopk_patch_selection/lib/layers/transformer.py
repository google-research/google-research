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

"""Self-Attention layer.

This layer contains all the machinery around the attention mechanism such as
the stacked feed-forward network, the layer norms, the residual connections
and the dropout.
"""

import enum
from typing import Callable, Optional, Tuple

import flax.deprecated.nn as nn
import jax
import jax.numpy as jnp
from lib.layers import stacked_layers
from lib.typing import AuxOutput


class Pooling(enum.Enum):
  NONE = None
  MAX = "max"
  MEAN = "mean"
  CLS = "CLS"
  SUM = "sum"


class Transformer(nn.Module):
  """Multi-layer transformer."""

  def apply(self,
            x,
            *,
            num_layers,
            num_heads,
            dim_hidden = None,
            pooling = Pooling.NONE,
            is_training):
    """Transformer.

    Args:
      x: Input tensor of shape (batch, sequence_length, dim).
      num_layers: Number of layers.
      num_heads: Number of attention heads.
      dim_hidden: Dimension of the representations, default to last dimension of
        `x`.
      pooling: Optional pooling of the output tokens representations to obtain a
        single representation of the sequence.
      is_training: Whether the model is being trained.

    Returns:
      The sequences representations (batch, sequence_length, dim_hidden).
    """
    pooling = Pooling(pooling)
    dim_hidden = dim_hidden or x.shape[-1]

    if dim_hidden != x.shape[-1]:
      x = nn.Dense(x, features=dim_hidden)

    if pooling == Pooling.CLS:
      cls_token = self.param(
          "cls_token",
          shape=(1, 1, dim_hidden),
          initializer=jax.nn.initializers.normal(1. / dim_hidden))
      batch_size = x.shape[0]
      cls_token = cls_token.repeat(batch_size, axis=0)
      x = jnp.concatenate([cls_token, x], axis=1)

    self_attention = nn.MultiHeadDotProductAttention.partial(
        num_heads=num_heads,
        deterministic=not is_training,
        inputs_kv=None)

    layer = TransformerLayer.partial(
        self_attention_module=self_attention,
        dim_intermediate=2 * dim_hidden,
        with_aux_outputs=False)

    transformer = stacked_layers.StackedLayers.partial(layer=layer,
                                                       num_layers=num_layers,
                                                       with_aux_outputs=False)

    representations = transformer(x)

    if pooling == Pooling.NONE:
      output = representations
    if pooling == Pooling.MEAN:
      output = representations.mean(axis=1)
    if pooling == Pooling.SUM:
      output = representations.sum(axis=1)
    if pooling == Pooling.MAX:
      output = representations.max(axis=1)
    if pooling == Pooling.CLS:
      output = representations[:, 0, :]

    return output


class TransformerLayer(nn.Module):
  """Transformer Layer with attention, ffn, dropout, layernorm, residuals."""

  def apply(self,
            x,
            *,
            self_attention_module,
            dim_intermediate,
            is_training,
            dropout_rate = 0.1,
            use_pre_layernorm = False,
            layernorm_epsilon = 1e-6,
            with_aux_outputs = True):
    """Compute self-attention with a feed-forward network on top.

    Args:
      x: Input representations.
      self_attention_module: Self-Attention layer.
      dim_intermediate: Size of the intermediate layer of the feed forward.
      is_training: Wether to enable dropout.
      dropout_rate: Dropout probability.
      use_pre_layernorm: Use pre layer norm from
        https://arxiv.org/abs/2002.04745.
      layernorm_epsilon: Epsilon parameter for all the layer norms.
      with_aux_outputs: Whether the self_attention_module has an aux output.

    Returns:
      New representations in a jnp.array of same shape as `x`.
    """
    dim_hidden = x.shape[-1]
    use_pre_ln = use_pre_layernorm
    use_post_ln = not use_pre_ln

    def apply_ln_if(pred, x, name):
      if pred:
        return nn.LayerNorm(x, epsilon=layernorm_epsilon, name=name)
      else:
        return x

    # attention
    x = apply_ln_if(use_pre_ln, x, "ln_pre_att")
    x_att = self_attention_module(x)
    if with_aux_outputs:
      x_att, output_aux = x_att

    # dropout norm and add
    x_att = nn.dropout(x_att, dropout_rate, deterministic=not is_training)
    x = x + x_att
    x = apply_ln_if(use_post_ln, x, "ln_post_att")

    # feed forward
    x_ffn = x
    x_ffn = apply_ln_if(use_pre_ln, x, "ln_pre_ffn")
    x_ffn = nn.Dense(x_ffn, dim_intermediate, name="ff_1")
    x_ffn = jax.nn.relu(x_ffn)
    x_ffn = nn.Dense(x_ffn, dim_hidden, name="ff_2")

    # dropout norm and add
    x_ffn = nn.dropout(x_ffn, dropout_rate, deterministic=not is_training)
    x = x + x_ffn
    x = apply_ln_if(use_post_ln, x, "ln_post_ffn")

    if with_aux_outputs:
      output = x, output_aux
    else:
      output = x
    return output
