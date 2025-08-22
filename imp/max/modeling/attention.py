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

"""Attention modules using Jax/Flax."""

import functools

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp

from imp.max.modeling import linear
from imp.max.modeling import normalization
from imp.max.modeling import scalable_attention
from imp.max.utils import sharding
from imp.max.utils import typing


class MultiHeadAttention(nn.Module):
  """A general-purpose Multi-Head Attention module."""

  d_head: int
  num_heads: int
  d_model: int
  use_bias: bool = True
  dropout_rate: float = 0.1
  kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
  bias_init: nn.initializers.Initializer = nn.initializers.zeros
  dtype: jax.typing.DTypeLike = jnp.float32
  precision: typing.Precision = None
  qk_layernorm: bool = False
  qkv_kernel_shardings: typing.ShardingAxes = ()
  out_kernel_shardings: typing.ShardingAxes = ()
  activation_shardings: typing.ShardingAxes = ()
  layernorm_shardings: typing.ShardingAxes = ()
  qkv_dot_general: typing.DotGeneral = lax.dot_general
  out_dot_general: typing.DotGeneral = lax.dot_general
  einsum_dot_general: typing.DotGeneral = lax.dot_general
  lora_rank: int = 4
  lora_scale: float = 0.
  efficient_attention: bool = False

  def setup(self):
    # query, key, and value projections
    mh_dense = functools.partial(
        linear.DenseGeneral,
        features=(self.num_heads, self.d_head),
        axis=-1,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        kernel_shardings=self.qkv_kernel_shardings,
        dot_general=self.qkv_dot_general,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
    )
    self.q = mh_dense(name='q')
    self.k = mh_dense(name='k')
    self.v = mh_dense(name='v')
    self.o = mh_dense(
        features=self.d_model,
        axis=(-2, -1),
        kernel_shardings=self.out_kernel_shardings,
        dot_general=self.out_dot_general,
        name='o')

    if self.qk_layernorm:
      self.layer_norm_q = normalization.LayerNorm(
          use_bias=self.use_bias,
          dtype=self.dtype,
          shardings=self.layernorm_shardings,
          name='layer_norm_q'
      )
      self.layer_norm_k = normalization.LayerNorm(
          use_bias=self.use_bias,
          dtype=self.dtype,
          shardings=self.layernorm_shardings,
          name='layer_norm_k'
      )

    self.dropout = nn.Dropout(self.dropout_rate, broadcast_dims=(-2, -1))

  @nn.compact
  def _cache_kv(
      self,
      query,
      key,
      value,
      attention_mask = None,
      attention_bias = None
  ):
    """Caches current key/value tensors for future attention calls.

    This piece of code is borrowed from `t5x.examples.scalable_t5.layers`.
    This method is called twice, first to initialize the cache and then for
    an actual decoding process. The two calls are differentiated by the
    presence of 'cached_key' in the variable dict. In the cache initialization
    stage, the cache variables are initialized as zeros and will be filled
    in the subsequent decoding process.

    In the cache initialization call, `query` has a shape:
    [batch, instance, q_length, d_q] and `key`/`value` have a shape:
    [batch, instance, kv_length, d_kv]. During the incremental decoding stage,
    query, key and value all have the shape [batch, instance, 1, qkv_features]
    corresponding to a single step.

    Args:
      query: input queries of shape `[batch, instance, q_length, d_q]`
      key: input keys of shape `[batch, instance, kv_length, d_kv]`
      value: input values of shape `[batch, instance, kv_length, d_kv]`
      attention_mask: attention mask of shape
        `[batch, instance, 1, q_length, kv_length]`.
      attention_bias: attention bias of shape
        `[1, num_heads, q_length, kv_length]`.

    Returns:
      key: output keys of shape `[batch, instance, k_length, d_kv]`
      value: output values of shape `[batch, instance, kv_length, d_kv]`
      attention_mask: attention mask of shape
        `[batch, instance, 1, 1, kv_length]`.
      attention_bias: attention bias of shape
        `[1, num_heads, 1, kv_length]`.
    """

    # Detect if we're initializing by absence of existing cache data.
    is_initialized = self.has_variable('cache', 'cached_key')
    # The key and value are [batch, instance, length, num_heads, head_dim]-size
    # but we cache them as [batch, instance, num_heads, head_dim, length] as a
    # TPU fusion optimization. This also enables the 'scatter via one-hot
    # broadcast' trick, which means we do a one-hot broadcast instead of a
    # scatter/gather operation, resulting in a 3-4x speedup in practice.
    swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
    cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                               swap_dims(key.shape), key.dtype)
    cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                 swap_dims(value.shape), value.dtype)
    cache_index = self.variable('cache', 'cache_index',
                                lambda: jnp.array(0, dtype=jnp.int32))
    if is_initialized:
      batch, instance, num_heads, head_dim, length = (cached_key.value.shape)
      # During fast autoregressive decoding, we feed one position at a time,
      # and cache the keys and values step by step.
      # Sanity shape check of cached key against input query.
      expected_shape = (batch, instance, 1, num_heads, head_dim)
      if expected_shape != query.shape:
        raise ValueError(
            'Autoregressive cache shape error, expected query shape '
            f'{expected_shape} instead got {query.shape}.'
        )

      # Create a OHE of the current index. NOTE: the index is increased below.
      cur_index = cache_index.value
      one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
      # In order to update the key, value caches with the current key and
      # value, we move the length axis to the back, similar to what we did for
      # the cached ones above.
      # Note these are currently the key and value of a single position, since
      # we feed one position at a time.
      one_token_key = jnp.moveaxis(key, -3, -1)
      one_token_value = jnp.moveaxis(value, -3, -1)
      # Update key, value caches with our new 1d spatial slices.
      # We implement an efficient scatter into the cache via one-hot
      # broadcast and addition.
      key = cached_key.value + one_token_key * one_hot_indices
      value = cached_value.value + one_token_value * one_hot_indices
      cached_key.value = key
      cached_value.value = value
      cache_index.value = cache_index.value + 1
      # Move the keys and values back to their original shapes.
      key = jnp.moveaxis(key, -1, -3)
      value = jnp.moveaxis(value, -1, -3)

      # Causal mask for cached decoder self-attention: our single query
      # position should only attend to those key positions that have already
      # been generated and cached, not the remaining zero elements.
      attention_mask_for_cached_keys = jnp.broadcast_to(
          jnp.arange(length) <= cur_index,
          (batch, instance, 1, 1, length)
          # (1, 1, length) represent (heads, q_length, kv_length)
          # q_length is 1 because during decoding we deal with one index.
          # The same mask is applied to all batch, instance, and heads elements.
      )
      if attention_mask is None:
        attention_mask = attention_mask_for_cached_keys
      else:
        attention_mask = jnp.logical_and(
            attention_mask, attention_mask_for_cached_keys
            ).astype(self.dtype)

      # Grab the correct relative attention bias during decoding. This is
      # only required during single step decoding.
      if attention_bias is not None:
        # The bias is a full attention matrix, but during decoding we only
        # have to take a slice of it.
        # This is equivalent to attention_bias[..., cur_index:cur_index+1, :].
        attention_bias = jax.vmap(
            lax.dynamic_slice_in_dim,
            in_axes=(None, 0, None, None),
        )(
            operand=jnp.squeeze(attention_bias, axis=0),
            start_index=jnp.reshape(cur_index, (-1)),
            slice_size=1,
            axis=-2,
        )

    return key, value, attention_mask, attention_bias

  def __call__(self,
               query,
               key,
               value,
               decode = False,
               deterministic = True,
               attention_mask = None,
               attention_bias = None):
    """Standard Multi-Head Attention based on pair-wise similarity scores.

    Args:
      query: Q matrix `[batch, instance, q_length, dim]`.
      key: K matrix `[batch, instance, kv_length, dim]`.
      value: V matrix `[batch, instance, kv_length, dim]`.
      decode: bool, whether the attention is performed for decoding
      deterministic: bool, deterministic or not (to apply dropout)
      attention_mask: mask for the attention scores.
        `[batch, instance, None, q_length, kv_length]`.
      attention_bias: bias for the attention scores.
        `[None, None, num_heads, q_length, kv_length]`.

    Returns:
      Output of shape `[batch, instance, q_length, num_heads*d_head]`.
    """

    # multi-head projection
    query = self.q(query)
    key = self.k(key)
    value = self.v(value)

    if self.qk_layernorm:
      query = self.layer_norm_q(query)
      key = self.layer_norm_k(key)

    query = sharding.shard_array(query, self.activation_shardings)
    key = sharding.shard_array(key, self.activation_shardings)
    value = sharding.shard_array(value, self.activation_shardings)

    if self.efficient_attention:
      coords = jnp.arange(query.shape[-3])
      coords = coords[Ellipsis, jnp.newaxis]

      value = scalable_attention.general_favor_attention(
          query=query,
          key=key,
          value=value,
          coords=coords,
      )
      # final output concatentation + projection
      output = self.o(value)
      return output

    # scale query by depth
    dk = query.shape[-1]
    # TODO(b/230247492): add option to push this to kernel init (in T5 config)
    query = query / jnp.sqrt(dk).astype(self.dtype)

    # perform caching, if decoding mode is enabled
    if decode:
      key, value, attention_mask, attention_bias = self._cache_kv(
          query=query, key=key, value=value,
          attention_mask=attention_mask, attention_bias=attention_bias,
      )

    # calculate similarity scores with the following shape:
    #        (batch, instance, num_heads, q_length, kv_length)
    scores = jnp.einsum('bnqhd,bnkhd->bnhqk', query, key,
                        precision=self.precision,
                        _dot_general=self.einsum_dot_general)

    # apply attention bias
    if attention_bias is not None:
      scores += attention_bias

    # apply attention mask
    if attention_mask is not None:
      scores += jnp.where(attention_mask > 0, 0.0, -1e10).astype(self.dtype)

    # apply softmax on scores
    scores = jax.nn.softmax(scores).astype(self.dtype)

    # apply attention dropout
    scores = self.dropout(scores, deterministic)

    # weighted sum over values for each query position
    value = jnp.einsum('bnhqk,bnkhd->bnqhd', scores, value,
                       precision=self.precision,
                       _dot_general=self.einsum_dot_general)

    # final output concatentation + projection
    output = self.o(value)

    return output
