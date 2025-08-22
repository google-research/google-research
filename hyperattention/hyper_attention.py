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

"""JAX Implementation of HyperAttention and Its Useful Subroutines.

HyperAttention is an efficient self-attention mechanism developed in the paper:
https://arxiv.org/abs/2310.05869.
"""

from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = Any


def get_softmax_attention_and_normalizers(
    query,
    key,
    value,
    causal = False,
    mask = None,
    precision = None,
):
  """Computes standard softmax attention and the row normalizers.

  This function has a similar functionality as flax.linen.dot_product_attention.
  There are two differences: (1) This function also outputs the normalization
  term used to normalize each row of the attention matrix. (2) The ordering of
  the axes of the inputs are different.

  Args:
    query: queries for calculating attention with shape of `[batch...,
      num_heads, q_length, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., num_heads,
      kv_length, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., num_heads,
      kv_length, v_depth_per_head]`.
    causal: if true, we do not compute the attention between the query and the
      key where the token of the key appears later than the token of the query.
    mask: mask has the shape of `[batch..., num_heads, q_length, kv_length]`.
      Attention weights are masked out if their corresponding mask value is
      `False`.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Two Array. The first has shape [batch..., num_heads, q_length,
    v_depth_per_head] indicating the self-attention output. The second has shape
    [batch..., num_heads, q_length, 1] indicating the normalizing value used to
    normalize the last dimension.
  """
  q_length = query.shape[-2]
  qk_depth_per_head = query.shape[-1]
  dtype = query.dtype

  scale = 1.0 / jnp.sqrt(qk_depth_per_head).astype(dtype)
  attn_weights = (
      jnp.einsum('...qd,...kd->...qk', query, key, precision=precision) * scale
  )

  if causal or mask is not None:
    final_mask = jnp.ones(attn_weights.shape)
    if causal:
      final_mask = nn.make_causal_mask(jnp.ones(query.shape[:-3] + (q_length,)))
    if mask is not None:
      final_mask *= mask
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(final_mask, attn_weights, big_neg)

  attn_weights_max = jnp.max(attn_weights, -1, keepdims=True)
  unnormalized = jnp.exp(attn_weights - attn_weights_max)
  scaling = jnp.sum(unnormalized, -1, keepdims=True)
  normalized = unnormalized / scaling

  output = jnp.einsum(
      '...qk,...kd->...qd', normalized, value, precision=precision
  )
  lse = (
      jnp.log(scaling) + attn_weights_max
  )  # lse stands for log sum exponentials

  return output, lse


def add_self_attentions(
    attn1, lse1, attn2, lse2
):
  """Merge self attentions with respect to two different sets of (keys, values).

  Given self attention attn1 and normalizer lse1 obtained by computing attention
  between queries and (keys1, values1), and self attention attn2 and normalizer
  lse2 obtained by computing attention between queries and (keys2, values2),
  this function computes the attention between queries and ([keys1, keys2],
  [values1, values2]), and it outputs the corresponding normalizer as well.

  The detailed computation is as follows:
  - attn
  = (attn1 * exp(lse1) + attn2 * exp(lse2)) / (exp(lse1) + exp(lse2))
  = (attn1 + attn2 * exp(lse2 - lse1)) / (1 + exp(lse2-lse1))
  = attn1 * c + attn2 * (1-c), where c=1/(1 + exp(lse2-lse1)),
  - lse
  = log(exp(lse1) + exp(lse2))
  = log(exp(lse1) * (1 + exp(lse2 - lse1)))
  = lse1 + log(1 + exp(lse2 - lse1)) = lse1 - log(c)

  Args:
    attn1: the first self attention matrix with shape [batch..., num_heads,
      q_length, v_depth_per_head]
    lse1: the corresponding normalizer for the first self attention matrix. It
      has shape [batch..., num_heads, q_length, 1]
    attn2: the second self attention matrix with shape [batch..., num_heads,
      q_length, v_depth_per_head]
    lse2: the corresponding normalizer for the second self attention matrix. It
      has shape [batch..., num_heads, q_length, 1]

  Returns:
    Two Array. The first has shape [batch..., num_heads, q_length,
    v_depth_per_head] indicating the merged self-attention output. The second
    has shape [batch..., num_heads, q_length, 1] indicating the merged
    normalizing value.
  """
  c = (1 / (1 + jnp.exp(lse2 - lse1))).astype(attn1.dtype)
  attn = c * attn1 + (1 - c) * attn2
  lse = lse1 - jnp.log(c + jnp.finfo(lse1.dtype).eps)
  return attn, lse


def select_features_by_indices(x, indices):
  """subsample the second to the last axis of the input by indices.

  For example, a conceptually equivlanet implementation in numpy when x.ndim=4:
  out = np.zeros((num_batchs, num_heads, length, dimension))
  for i in range(num_batchs):
    for j in range(num_heads):
      out[i,j] = x[i,j][indices[i,j],:]
  return out

  Args:
    x: the input data has shape [batch..., length, dimension].
    indices: it has shape [batch..., num_samples] where each entry should be in
      [0, length - 1].

  Returns:
    out: the selected data with shape [batch..., num_samples, dimension].
  """
  dimension = x.shape[-1]
  indices = jnp.broadcast_to(
      jnp.expand_dims(indices, -1), indices.shape + (dimension,)
  )
  return jnp.take_along_axis(x, indices, axis=-2)


class SimHash:
  """This class is used to apply rotary position embeddings."""

  def __init__(
      self,
      dimension,
      num_projection = 32,
      random_key = None,
      precision = None,
  ):
    """Initializes parameters for applying SimHash.

    Args:
      dimension: feature dimension, i.e., the last axis of the shape of the
        input.
      num_projection: the number of sampled random projection vectors used by
        SimHash.
      random_key: the random key used to sample random projection vectors.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
    """
    if random_key is None:
      random_key = jax.random.PRNGKey(0)
    self.projections = jax.random.normal(
        random_key, (dimension, num_projection)
    )
    self.precision = precision
    self.powers_of_two = jnp.power(2.0, jnp.arange(num_projection))

  def apply(self, x):
    """Apply SimHash.

    Args:
      x: the input data has shape [batch..., dimension].

    Returns:
      output: the output data has shape [batch...], i.e., each feature vector is
      mapped to a hash value.
    """
    hashed_vectors = (
        jnp.einsum(
            '...nd,...dr->...nr', x, self.projections, precision=self.precision
        )
        > 0
    )
    return jnp.matmul(
        hashed_vectors, self.powers_of_two, precision=self.precision
    )


class HyperAttention:
  """This class implements HyperAttention: https://arxiv.org/abs/2310.05869."""

  def __init__(
      self,
      dimension = 64,
      num_projection = 32,
      min_bucket_size = 128,
      max_bucket_size = 512,
      bucket_size_ratio = 1 / 32.0,
      min_sample_size = 128,
      max_sample_size = 256,
      sample_size_ratio = 1 / 64.0,
      min_seq_len = 1024,
      use_sorting = True,
      use_sampling = True,
      random_key = None,
      precision = jax.lax.Precision.DEFAULT,
  ):
    """Initializes parameters for applying HyperAttention.

    Args:
      dimension: feature dimension of query and key embeddings, i.e.,
        qk_depth_per_head.
      num_projection: the number of sampled random projection vectors used by
        SimHash.
      min_bucket_size: the minimum bucket size used by SortingLSH.
      max_bucket_size: the maximum bucket size used by SortingLSH.
      bucket_size_ratio: the desired (1 / number of buckets produced by
        SortingLSH).
      min_sample_size: the minimum number of samples sampled by the sampling
        process.
      max_sample_size: the maximum number of samples sampled by the sampling
        process.
      sample_size_ratio: the desired (1 / number of samples sampled by the
        sampling process).
      min_seq_len: run vanilla softmax if the context length is at most
        min_seq_len.
      use_sorting: enable SortingLSH.
      use_sampling: enable uniform sampling.
      random_key: the random key used by SimHash and the uniform sampling
        process.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
    """
    self.min_bucket_size = min_bucket_size
    self.max_bucket_size = max_bucket_size
    self.bucket_size_ratio = bucket_size_ratio
    self.min_sample_size = min_sample_size
    self.max_sample_size = max_sample_size
    self.sample_size_ratio = sample_size_ratio
    self.min_seq_len = min_seq_len
    self.use_sorting = use_sorting
    self.use_sampling = use_sampling
    if random_key is None:
      random_key = jax.random.PRNGKey(0)
    hash_key, self.sampling_key = jax.random.split(random_key, 2)
    self.lsh = SimHash(
        dimension=dimension,
        num_projection=num_projection,
        random_key=hash_key,
        precision=precision,
    )
    self.precision = precision

  def attention_without_causal_mask(
      self,
      query,
      key,
      value,
  ):
    """Computes HyperAttention without causal mask.

    Args:
      query: queries for calculating attention with shape of `[batch...,
        num_heads, q_length, qk_depth_per_head]`.
      key: keys for calculating attention with shape of `[batch..., num_heads,
        kv_length, qk_depth_per_head]`.
      value: values to be used in attention with shape of `[batch..., num_heads,
        kv_length, v_depth_per_head]`.

    Returns:
      Two Array. The first has shape [batch..., num_heads, q_length,
      v_depth_per_head] indicating the self-attention output. The second has
      shape [batch..., num_heads, q_length, 1] indicating the normalizing value
      used to normalize the last dimension.
    """
    dimension = query.shape[-1]
    q_length = query.shape[-2]
    num_heads = query.shape[-3]
    batch_shape = query.shape[:-3]
    kv_length = key.shape[-2]
    v_depth_per_head = value.shape[-1]

    query_sort_idx = jnp.argsort(self.lsh.apply(query), axis=-1, stable=True)
    key_sort_idx = jnp.argsort(self.lsh.apply(key), axis=-1, stable=True)
    query_sort_idx_inv = jnp.argsort(query_sort_idx, axis=-1, stable=True)

    # SortingLSH process.
    if self.use_sorting:
      query_sorted = select_features_by_indices(query, query_sort_idx)
      key_sorted = select_features_by_indices(key, key_sort_idx)
      value_sorted = select_features_by_indices(value, key_sort_idx)
    else:
      query_sorted = query
      key_sorted = key
      value_sorted = value
    key_bucket_size = min(
        max(int(kv_length * self.bucket_size_ratio), self.min_bucket_size),
        self.max_bucket_size,
        q_length,
    )
    num_buckets = kv_length // key_bucket_size
    query_bucket_size = q_length // num_buckets
    query_split_per_block = query_sorted.reshape(
        -1, 1, query_bucket_size, dimension
    )
    key_split_per_block = key_sorted.reshape(-1, 1, key_bucket_size, dimension)
    value_split_per_block = value_sorted.reshape(
        -1, 1, key_bucket_size, v_depth_per_head
    )
    attn_block, lse_block = get_softmax_attention_and_normalizers(
        query_split_per_block,
        key_split_per_block,
        value_split_per_block,
        precision=self.precision,
    )
    attn_block = attn_block.reshape(batch_shape + (num_heads, q_length, -1))
    lse_block = lse_block.reshape(batch_shape + (num_heads, q_length, -1))

    # Uniform sampling process.
    sample_size = min(
        max(int(kv_length * self.sample_size_ratio), self.min_sample_size),
        self.max_sample_size,
    )
    if (
        sample_size > 0
        and (q_length > query_bucket_size)
        and (kv_length > key_bucket_size)
        and self.use_sampling
    ):
      sampled_set = jax.random.randint(
          key=jax.random.fold_in(self.sampling_key, kv_length),
          shape=batch_shape + (num_heads, sample_size),
          minval=0,
          maxval=kv_length,
      )
      offset_q_length = jnp.expand_dims(jnp.arange(q_length), (-1, -3))
      block_mask = (offset_q_length // query_bucket_size) != (
          sampled_set // key_bucket_size
      ).reshape(-1, 1, sample_size)
      block_mask = block_mask.reshape(
          batch_shape + (num_heads, -1, sample_size)
      )
      weights = kv_length / sample_size
      value_subset = select_features_by_indices(value_sorted, sampled_set)
      key_subset = select_features_by_indices(key_sorted, sampled_set)
      attn_residual, lse_residual = get_softmax_attention_and_normalizers(
          query_sorted,
          key_subset,
          value_subset,
          mask=block_mask,
          precision=self.precision,
      )
      lse_residual = lse_residual + jnp.log(weights)
      attn, lse = add_self_attentions(
          attn_block, lse_block, attn_residual, lse_residual
      )
    else:
      attn, lse = attn_block, lse_block

    # Re-order rows with the inverse order for query_sorted -> query.
    if self.use_sorting:
      attn = select_features_by_indices(attn, query_sort_idx_inv)
      lse = select_features_by_indices(lse, query_sort_idx_inv)
    return attn, lse

  def get_attention_and_normalizers(
      self,
      query,
      key,
      value,
      causal = False,
  ):
    """Computes HyperAttention without causal mask.

    Args:
      query: queries for calculating attention with shape of `[batch...,
        num_heads, q_length, qk_depth_per_head]`.
      key: keys for calculating attention with shape of `[batch..., num_heads,
        kv_length, qk_depth_per_head]`.
      value: values to be used in attention with shape of `[batch..., num_heads,
        kv_length, v_depth_per_head]`.
      causal: if true, we do not compute the attention between the query and the
        key where the token of the key appears later than the token of the
        query.

    Returns:
      Two Array. The first has shape [batch..., num_heads, q_length,
      v_depth_per_head] indicating the self-attention output. The second has
      shape [batch..., num_heads, q_length, 1] indicating the normalizing value
      used to normalize the last dimension.
    """
    dimension = query.shape[-1]
    q_length = query.shape[-2]
    num_heads = query.shape[-3]
    batch_shape = query.shape[:-3]
    kv_length = key.shape[-2]
    v_depth_per_head = value.shape[-1]

    if not causal:
      attn, lse = self.attention_without_causal_mask(query, key, value)
    else:
      if kv_length <= self.min_seq_len:
        attn, lse = get_softmax_attention_and_normalizers(
            query,
            key,
            value,
            causal=True,
            precision=self.precision,
        )
      else:
        q_bd = query.reshape(
            batch_shape + (2 * num_heads, q_length // 2, dimension)
        )
        k_bd = key.reshape(
            batch_shape + (2 * num_heads, kv_length // 2, dimension)
        )
        v_bd = value.reshape(
            batch_shape + (2 * num_heads, kv_length // 2, v_depth_per_head)
        )
        attn_bd, lse_bd = self.get_attention_and_normalizers(
            q_bd,
            k_bd,
            v_bd,
            causal=True,
        )
        attn_bd = attn_bd.reshape(batch_shape + (num_heads, q_length, -1))
        lse_bd = lse_bd.reshape(batch_shape + (num_heads, q_length, -1))
        attn_unmasked, lse_unmasked = self.attention_without_causal_mask(
            jnp.take(
                query, jnp.arange(kv_length // 2, query.shape[-2]), axis=-2
            ),
            jnp.take(key, jnp.arange(kv_length // 2), axis=-2),
            jnp.take(value, jnp.arange(kv_length // 2), axis=-2),
        )
        attn_up, lse_up = (
            jnp.take(attn_bd, jnp.arange(q_length // 2), axis=-2),
            jnp.take(lse_bd, jnp.arange(q_length // 2), axis=-2),
        )
        attn_down, lse_down = add_self_attentions(
            jnp.take(
                attn_bd, jnp.arange(q_length // 2, attn_bd.shape[-2]), axis=-2
            ),
            jnp.take(
                lse_bd, jnp.arange(q_length // 2, lse_bd.shape[-2]), axis=-2
            ),
            attn_unmasked,
            lse_unmasked,
        )
        attn = jnp.concatenate((attn_up, attn_down), axis=-2)
        lse = jnp.concatenate((lse_up, lse_down), axis=-2)
    return attn, lse
