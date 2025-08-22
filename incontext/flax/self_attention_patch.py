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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patch for self attention."""

import functools
from typing import Callable, Optional

from flax.linen import attention
from jax import lax
import jax.numpy as jnp

Array = attention.Array
Module = attention.Module
Dtype = attention.Dtype
PRNGKey = attention.PRNGKey
Shape = attention.Shape
DenseGeneral = attention.DenseGeneral
PrecisionLike = attention.PrecisionLike
combine_masks = attention.combine_masks
promote_dtype = attention.promote_dtype
dot_product_attention_weights = attention.dot_product_attention_weights
merge_param = attention.merge_param
compact = attention.compact
zeros = attention.zeros
default_kernel_init = attention.default_kernel_init


def dot_product_attention(
    query,
    key,
    value,
    bias = None,
    mask = None,
    broadcast_dropout = True,
    dropout_rng = None,
    dropout_rate = 0.0,
    deterministic = False,
    dtype = None,
    precision = None,
    return_attention = False,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    return_attention: returns attention weights.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
    Attention weigts (optional).
  """
  query, key, value = promote_dtype(query, key, value, dtype=dtype)
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], ("q, k, v "
                                                                  "batch dims "
                                                                  "must match.")
  assert query.shape[-2] == key.shape[-2] == value.shape[
      -2], "q, k, v num_heads must match."
  assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

  # compute attention weights
  attn_weights = dot_product_attention_weights(
      query,
      key,
      bias,
      mask,
      broadcast_dropout,
      dropout_rng,
      dropout_rate,
      deterministic,
      dtype,
      precision,
  )

  # return weighted sum over values for each query position
  out = jnp.einsum(
      "...hqk,...khd->...qhd", attn_weights, value, precision=precision)

  if return_attention:
    return out, attn_weights
  else:
    return out, None


class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: infer from inputs and
        params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
      return_attention: returns attention weights.
  """

  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention
  decode: bool = False

  @compact
  def __call__(
      self,
      inputs_q,
      inputs_kv,
      mask = None,
      deterministic = None,
      return_attention = False,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output
    vector.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape `[batch_sizes..., length, features]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      return_attention: returns attention weights.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert (qkv_features % self.num_heads == 0
           ), "Memory dimension must be divisible by number of heads."
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        DenseGeneral,
        axis=-1,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (
        dense(name="query")(inputs_q),
        dense(name="key")(inputs_kv),
        dense(name="value")(inputs_kv),
    )

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable("cache", "cached_key")
      cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape,
                                 key.dtype)
      cached_value = self.variable("cache", "cached_value", jnp.zeros,
                                   value.shape, value.dtype)
      cache_index = self.variable("cache", "cache_index",
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError("Autoregressive cache shape error, "
                           "expected query shape %s instead got %s." %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
            mask,
            jnp.broadcast_to(
                jnp.arange(max_length) <= cur_index,
                tuple(batch_dims) + (1, 1, max_length)),
        )

    dropout_rng = None
    if self.dropout_rate > 0.0:  # Require `deterministic` only if using drpt.
      m_deterministic = merge_param("deterministic", self.deterministic,
                                    deterministic)
      if not m_deterministic:
        dropout_rng = self.make_rng("dropout")
    else:
      m_deterministic = True

    # apply attention
    x, attn_weights = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
        return_attention=return_attention,
    )  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        name="out",
    )(
        x)

    if return_attention:
      return out, attn_weights

    return out, None


class SelfAttention(MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @compact
  def __call__(
      self,
      inputs_q,
      inputs_qv = None,
      mask = None,
      deterministic = None,
      return_attention = False,
  ):
    """Applies multi-head dot product self-attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output
    vector.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_qv: not used for compatability with the superclass.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      return_attention: returns attention weights.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if inputs_qv is None:
      inputs_qv = inputs_q
    return super().__call__(
        inputs_q,
        inputs_qv,
        mask,
        deterministic=deterministic,
        return_attention=return_attention)
