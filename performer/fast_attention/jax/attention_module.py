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

"""Fast Attention class for Flax.

This class is designed to replace MultiHeadDotProductAttention in Flax.
It provides a simple 1D weight mask for attention calculations.
"""

import functools
from typing import (Any, Callable, Tuple, Optional)
from flax.linen.attention import dot_product_attention
from flax.linen.initializers import zeros
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.module import compact
from flax.linen.module import Module
import jax.numpy as jnp

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


class MultiHeadDotProductFastAttention(Module):
  """Multi-head dot-product fast attention.

    Similar to regular multi-head dot-product fast attention, but only accepts
    vector masks to prevent the L^2 calculation

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
  """
  num_heads: int
  dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: bool = False
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention
  decode: bool = False

  @compact
  def __call__(self,
               inputs_q,
               inputs_kv,
               mask_k = None):
    """Multi-head dot product attention with 1D attention mask.

    Applies multi-head dot product attention on the input data.
    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.
    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask_k: attention mask of shape
        `[batch_sizes..., num_heads, key/value_length]`.
    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(DenseGeneral,
                              axis=-1,
                              features=(self.num_heads, head_dim),
                              kernel_init=self.kernel_init,
                              bias_init=self.bias_init,
                              use_bias=self.use_bias,
                              precision=self.precision)
    # Project inputs_q to multi-headed q/k/v
    # Dimensions are then [batch, ..., length, n_heads, n_features_per_head]
    query, key, value = (dense(dtype=self.dtype, name='query')(inputs_q),
                         dense(dtype=self.dtype, name='key')(inputs_kv),
                         dense(dtype=self.dtype, name='value')(inputs_kv))

    # TODO(tomprom): Enforce use of unidirectional mask for decoding
    if self.decode:
      raise NotImplementedError

    # Convert the boolean attention mask to an attention bias.
    if mask_k is not None:
      # Element-wise multiply the key mask by the value matrix
      # Dim: batch, sequence length, heads, qkv dimension (per head)
      value = jnp.einsum('...l,...lhd->...lhd', mask_k, value)
    else:
      # If no mask is provided, leave as is.
      pass

    dropout_rng = None
    if not self.deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply Attention
    x = self.attention_fn(
        query,
        key,
        value,
        bias=None,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=self.deterministic,
        dtype=self.dtype,
        precision=self.precision)

    # Back to the original inputs' dimensions.
    out = DenseGeneral(features=features,
                       axis=(-2, -1),
                       kernel_init=self.kernel_init,
                       bias_init=self.bias_init,
                       use_bias=self.use_bias,
                       dtype=self.dtype,
                       precision=self.precision,
                       name='out')(x)

    return out
