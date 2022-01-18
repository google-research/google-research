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

"""Implements relative self-attention."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

import functools
from typing import Any, Callable, Iterable, Optional

from flax.linen import attention
from flax.linen import initializers
from flax.linen import linear
from flax.linen import module
from jax import lax
import jax.numpy as jnp

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


def make_relative_position_bucket(relative_position, bidirectional=False,
                                  num_buckets=32, max_distance=128):
  """Translate relative position to a bucket number for relative attention."""
  # Adapted from Mesh Tensorflow:
  # https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py
  min_relative_position = -max_distance
  max_relative_position = max_distance if bidirectional else 0
  num_possible_positions = max_relative_position - min_relative_position + 1
  if num_buckets >= num_possible_positions:
    # Use one bucket for each position.
    relative_position = jnp.clip(relative_position,
                                 a_min=min_relative_position,
                                 a_max=max_relative_position)
    if bidirectional:
      bucket = ((relative_position > 0) * max_distance
                + jnp.abs(relative_position))
    else:
      bucket = -relative_position
    return bucket.astype(jnp.int32)

  relative_buckets = 0
  if bidirectional:
    num_buckets //= 2
    relative_buckets += (relative_position > 0) * num_buckets
    relative_position = jnp.abs(relative_position)
  else:
    relative_position = -jnp.clip(relative_position, a_max=0)

  max_exact = num_buckets // 2
  is_small = relative_position < max_exact

  relative_position_if_large = max_exact + (
      jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact)
      * (num_buckets - max_exact)
  )
  relative_position_if_large = jnp.clip(relative_position_if_large,
                                        a_max=num_buckets - 1)

  relative_buckets += jnp.where(is_small, relative_position,
                                relative_position_if_large)

  return relative_buckets.astype(jnp.int32)


class RelativeMultiHeadDotProductAttention(module.Module):
  """Dot-product attention with relative positional encodings.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: float32)
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: if false, the attention weight is masked randomly
      using dropout, whereas if true, the attention weights
      are deterministic.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    decode: whether to prepare and use an autoregressive cache.
    bidirectional: whether to attend forward and backward (True) or only to past
      tokens (False).
    num_relative_position_buckets: number of buckets for relative positions
      for attention.
    max_distance: maximum value of relative position.
  """
  num_heads: int
  dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      linear.default_kernel_init)
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  use_bias: bool = True
  decode: bool = False
  bidirectional: bool = False
  num_relative_position_buckets: int = 32
  max_distance: int = 128

  @module.compact
  def __call__(self,
               inputs_q,
               inputs_kv,
               mask = None,
               custom_relative_position = None,
               deterministic = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      custom_relative_position: relative positions tensor
        `[batch_sizes..., query_length, key/value_length]'
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      deterministic = module.merge_param('deterministic', self.deterministic,
                                         deterministic)
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(linear.DenseGeneral,
                              axis=-1,
                              features=(self.num_heads, head_dim),
                              kernel_init=self.kernel_init,
                              bias_init=self.bias_init,
                              use_bias=self.use_bias,
                              precision=self.precision)
    relative_attention_embed = linear.Embed(
        num_embeddings=self.num_relative_position_buckets,
        features=self.num_heads,
        embedding_init=initializers.normal(stddev=1.0),
        dtype=self.dtype)

    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(dtype=self.dtype, name='query')(inputs_q),
                         dense(dtype=self.dtype, name='key')(inputs_kv),
                         dense(dtype=self.dtype, name='value')(inputs_kv))

    if custom_relative_position is None:
      query_length = inputs_q.shape[-2]
      key_length = inputs_kv.shape[-2]
      context_position = jnp.arange(query_length, dtype=jnp.int32)[:, None]
      memory_position = jnp.arange(key_length, dtype=jnp.int32)[None, :]

      relative_position = memory_position - context_position
      relative_position_bucket = make_relative_position_bucket(
          relative_position,
          bidirectional=self.bidirectional,
          num_buckets=self.num_relative_position_buckets,
          max_distance=self.max_distance)

      bias = relative_attention_embed(relative_position_bucket)
      bias = bias.transpose((2, 0, 1))
      # Expand batch dimensions.
      bias = jnp.broadcast_to(
          bias, (1,) * len(inputs_q.shape[:-2]) + bias.shape)

    else:
      relative_position = custom_relative_position
      relative_position_bucket = make_relative_position_bucket(
          relative_position,
          bidirectional=self.bidirectional,
          num_buckets=self.num_relative_position_buckets,
          max_distance=self.max_distance)

      bias = relative_attention_embed(relative_position_bucket)
      permute = tuple(map(lambda i: len(inputs_q.shape) + 1 + i, (-1, -3, -2)))
      bias = bias.transpose(
          tuple(range(len(inputs_q.shape[:-2]))) + permute)

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key',
                                 jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value',
                                   jnp.zeros, value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
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
        mask = attention.combine_masks(
            mask,
            jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                             tuple(batch_dims) + (1, 1, max_length)))

        bias = lax.dynamic_slice(
            bias,
            (0, 0, cur_index, 0),
            (1, self.num_heads, 1, max_length))

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      bias += lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # apply attention
    x = attention.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = linear.DenseGeneral(features=features,
                              axis=(-2, -1),
                              kernel_init=self.kernel_init,
                              bias_init=self.bias_init,
                              use_bias=self.use_bias,
                              dtype=self.dtype,
                              precision=self.precision,
                              name='out')(x)
    return out


class RelativeSelfAttention(RelativeMultiHeadDotProductAttention):
  """Self-attention special case."""

  @module.compact
  def __call__(self,
               inputs_q,
               mask = None,
               custom_relative_position = None,
               deterministic = None):
    return super().__call__(inputs_q, inputs_q,
                            mask=mask,
                            custom_relative_position=custom_relative_position,
                            deterministic=deterministic)
