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

"""Multiquery attention with incremental processing of KV cache.

We provide a function `attend` which calculates attention of a query `q` over
some k/v context. The k/v context is provided as a list of chunks, each chunk
potentially having a different batch size. All chunks except the last are
provided as `KVCache` objects; the last chunk is provided directly as k/v
tensors. For example:

```
# Batch-size-1 KVCache
few_shot_examples_cache = KVCache(...)
# Batch-size-8 KVCache
tasks = KVCache(...)
# q/k/v tensors all have batch size 64:
q, k, v = ...
attn_out = attend(q, k, v, [few_shot_examples_cache, tasks])
```
"""
import functools
from typing import Sequence, Tuple, Optional

from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy
from jax.sharding import PartitionSpec as P
import numpy as np

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import special2

HParams = checkpoint.HParams
P = jax.sharding.PartitionSpec


@struct.dataclass
class KVCache:
  """Key-value cache for a batch.

  Different sequences in a batch can have different lengths. Tokens outside the
  range `0:self.lengths` are invalid and should be masked out during attention.

  If the cache is circular, then we write to it in the same fashion, but attend
  differently - understanding that the length represents the number of tokens
  behind the curent offset position, wrapping around the buffer if necessary.
  This is so that we can write with a single slice instead of a scatter.
  """
  lengths: jnp.ndarray  # int32[batch]
  k: jnp.ndarray  # bf16[seqlen, layers, batch, qkv]
  v: jnp.ndarray  # bf16[seqlen, layers, batch, qkv]
  offset: Optional[jnp.ndarray]
  circular: Optional[bool] = struct.field(
      pytree_node=False, kw_only=True, default=False
  )

  @property
  def max_length(self):
    return self.k.shape[2]  # length

  @classmethod
  def zeros(
      cls, h, batch, length, circular = False
  ):
    """Constructs an empty KV cache of the specified batch and length."""
    return KVCache(
        lengths=jnp.zeros((batch,), jnp.int32),
        k=jnp.zeros((h.layers, batch, length, h.qkv), jnp.bfloat16),
        v=jnp.zeros((h.layers, batch, length, h.qkv), jnp.bfloat16),
        offset=jnp.zeros((1,), jnp.int32),
        circular=circular,
    )

  def write_token(self, token_i, token, idx=2):
    """Note offset in the circular buffer is where we would be writing next."""
    assert token.k.shape[idx] == 1
    return KVCache(
        lengths=self.lengths + 1,
        k=lax.dynamic_update_index_in_dim(self.k, token.k, token_i, idx),
        v=lax.dynamic_update_index_in_dim(self.v, token.v, token_i, idx),
        offset=jnp.array([token_i,]) + 1 % self.max_length
        if self.circular
        else jnp.zeros((1,), jnp.int32),
        circular=self.circular,
    )

  @classmethod
  def logical_axes(cls, circular=False):
    """Returns the logical axis spec for the KV cache."""
    return KVCache(
        lengths=P('attn_batch'),
        k=P('prefix_layers', 'attn_batch', 'prefix_time', 'prefix_qkv'),
        v=P('prefix_layers', 'attn_batch', 'prefix_time', 'prefix_qkv'),
        offset=P(),
        circular=circular,
    )  # pytype: disable=wrong-arg-types

  @classmethod
  def physical_axes(cls, circular=False):
    """Returns the partition specs for the weights in their physical axes."""
    return jax.tree.map(
        partitioning.logical_to_physical, KVCache.logical_axes(circular)
    )


def prefix_lengths(prefix):
  """Gets the token count, per batch element. int32[batch]."""
  lengths = jnp.array([0], jnp.int32)
  for kv in prefix:
    batch, = kv.lengths.shape
    lengths = flat_broadcast(lengths, batch) + kv.lengths
  return lengths


def flat_broadcast(x, size):
  """Broadcasts an array from shape [old_size, ...] to shape [size, ...].

  The value `size` must be a multiple of `old_size`.

  Args:
    x: The array to broadcast.
    size: The target size of the first dimension.

  Returns:
    x, broadcasted for the target shape.
  """
  xsize = x.shape[0]
  assert size % xsize == 0, f'Incompatible batch sizes: {size} vs {xsize}'
  mul = size // xsize
  x = lax.expand_dims(x, (1,))
  sh = list(x.shape)
  sh[1] = mul
  x = jnp.broadcast_to(x, tuple(sh))
  return lax.collapse(x, 0, 2)


def _attend_chunk_multihead(
    q,
    k,
    v,
    mask,
):
  """Multihead attention over a single chunk.

  Result still needs cross-chunk normalization, because we have only normalized
  the softmax within this single chunk.

  Args:
    q: Query. f32[qbatch, qlen, num_heads, d_qkv].
    k: Key. f32[kbatch, klen, num_heads, d_qkv].
    v: Value. f32[kbatch, klen, num_heads, d_qkv].
    mask: Attention mask. Broadcastable to
      bool[kbatch, qbatch // kbatch, num_heads, qlen, klen].

  Returns:
    (local_out, local_max, local_sum), where:
    local_out: unnormalized output for this chunk.
      f32[qbatch, qlen, num_heads, d_qkv]
    local_max: max of exponentials for this chunk.
      f32[qbatch, qlen, num_heads, 1]
    local_sum: sum of exponentials for this chunk, divided by exp(local_max).
      f32[qbatch, qlen, num_heads, 1]
  """
  q_batch, qlen, num_heads, qkv = q.shape
  kv_batch, klen, num_heads, qkv = k.shape
  assert v.shape == (kv_batch, klen, num_heads, qkv)
  num_samples = q_batch // kv_batch

  # When qbatch>kbatch, this means that the kbatch needs to be broadcasted up to
  # the qbatch. We never actually materialize this broadcast: instead, we make
  # this broadcast happen in a fused way, directly within the einsum. This
  # achieves the critical optimization that the load of k and v from the old kv
  # caches is reused `qbatch//kbatch` many ways, aka lazy-prefix-broadcast.
  # print(f"q {q.shape}, k: {k.shape}, v: {v.shape}")
  q = jnp.reshape(q, (kv_batch, num_samples, qlen, num_heads, qkv))
  attn_logits = jnp.einsum('bsqhd,bkhd->bshqk', jnp.float32(q), jnp.float32(k))
  if mask is not None:
    # attn_mask: bool[cache_batch, cache_beam=1, heads=1, qlength=1, klength]
    attn_logits += jnp.where(mask, 0, -1e10)
  # local_max: f32[cache_batch, cache_beam, heads, qlength, 1=klength]
  local_max = jnp.max(attn_logits, axis=-1, keepdims=True)
  local_exps = special2.exp2(attn_logits - local_max)
  # local_sum: f32[cache_batch, cache_beam, heads, qlength]
  local_sum = jnp.sum(local_exps, axis=-1, keepdims=True)
  # local_sum: f32[batch, qlength, heads, 1]
  local_sum = jnp.swapaxes(jax.lax.collapse(local_sum, 0, 2), 1, 2)
  # local_max: f32[batch, qlength, heads, 1]
  local_max = jnp.swapaxes(jax.lax.collapse(local_max, 0, 2), 1, 2)
  local_out = jnp.einsum('bshqk,bkhd->bsqhd', jnp.float32(local_exps),
                         jnp.float32(v))
  # local_out: f32[batch, qlength, heads, qkv]
  local_out = jax.lax.collapse(local_out, 0, 2)
  return local_out, local_max, local_sum


def _attend_chunk(
    q,
    k,
    v,
    mask,
):
  """Multiquery attention over a single chunk.

  Result still needs cross-chunk normalization, because we have only normalized
  the softmax within this single chunk.

  Args:
    q: Query. f32[qbatch, qlen, num_heads, d_qkv].
    k: Key. f32[kbatch, klen, (num_heads,)? d_qkv].
    v: Value. f32[kbatch, klen, (num_heads,)? d_qkv].
    mask: Attention mask. Broadcastable to bool[kbatch, qbatch // kbatch,
      num_heads, qlen, klen].

  Returns:
    (local_out, local_max, local_sum), where:
    local_out: unnormalized output for this chunk.
      f32[qbatch, qlen, num_heads, d_qkv]
    local_max: max of exponentials for this chunk.
      f32[qbatch, qlen, num_heads, 1]
    local_sum: sum of exponentials for this chunk, divided by exp(local_max).
      f32[qbatch, qlen, num_heads, 1]
  """
  if k.ndim == 4:
    return _attend_chunk_multihead(
        q,
        k,
        v,
        mask,
    )

  q_batch, qlen, num_heads, qkv = q.shape
  kv_batch, klen, qkv = k.shape
  assert v.shape == (kv_batch, klen, qkv)
  num_samples = q_batch // kv_batch

  # When qbatch>kbatch, this means that the kbatch needs to be broadcasted up to
  # the qbatch. We never actually materialize this broadcast: instead, we make
  # this broadcast happen in a fused way, directly within the einsum. This
  # achieves the critical optimization that the load of k and v from the old kv
  # caches is reused `qbatch//kbatch` many ways, aka lazy-prefix-broadcast.

  q = jnp.reshape(q, (kv_batch, num_samples, qlen, num_heads, qkv))
  attn_logits = jnp.einsum('bsqhd,bkd->bshqk', jnp.float32(q), jnp.float32(k))
  if mask is not None:
    # attn_mask: bool[cache_batch, cache_beam=1, heads=1, qlength=1, klength]
    attn_logits += jnp.where(mask, 0, -1e10)
  # local_max: f32[cache_batch, cache_beam, heads, qlength, 1=klength]
  local_max = jnp.max(attn_logits, axis=-1, keepdims=True)
  local_exps = special2.exp2(attn_logits - local_max)
  # local_sum: f32[cache_batch, cache_beam, heads, qlength]
  local_sum = jnp.sum(local_exps, axis=-1, keepdims=True)
  # local_sum: f32[batch, qlength, heads, 1]
  local_sum = jnp.swapaxes(jax.lax.collapse(local_sum, 0, 2), 1, 2)
  # local_max: f32[batch, qlength, heads, 1]
  local_max = jnp.swapaxes(jax.lax.collapse(local_max, 0, 2), 1, 2)
  local_out = jnp.einsum('bshqk,bkd->bsqhd', jnp.float32(local_exps),
                         jnp.float32(v))
  # local_out: f32[batch, qlength, heads, qkv]
  local_out = jax.lax.collapse(local_out, 0, 2)
  return local_out, local_max, local_sum


def attend(
    q,
    k,
    v,
    prefix,
    layer,
    causal_masking = True,
    custom_mask = None,
    custom_prefix_masks = None,
):
  """Computes attention of q over `prefix` followed by `k` and `v`.

  Args:
    q: Query. bf16[batch, len, num_heads, qkv].
    k: Keys, of the same length and batch size as q. bf16[batch, len,
      (num_heads,)?, qkv].
    v: Values, of the same length and batch size as q. bf16[batch, len,
      (num_heads,)?, qkv].
    prefix: Additional attention context, which may have different length than q
      and smaller batch size than q. Lengths can be arbitrary; batch sizes must
      divide q's batch size. Masking will be applied according to
      `KVCache.lengths`.
    layer: Which layer of the KVCache in `prefix` to use.
    causal_masking: Triangular (causal) masking will be applied.
    custom_mask: Optional custom attention mask for the current q and k.
    custom_prefix_masks: Optional custom attention mask for the caches.

  Returns:
    Attention of q over the k/v/prefix context.
  """
  # Compare
  # flaxformer/components/attention/dense_attention.py
  #
  # Relative to that implementation, our differences are:
  # * for us, the k/v attention context is split up into many tensors: all of
  #   prefix, plus `k` and `v`. In general, the k/v context can often have a
  #   smaller batch size earlier in the sequence: for example, during few-shot
  #   prompting, the few-shot examples typically have batch size 1, whereas the
  #   task has a larger batch size. In our implementation here, we use a list of
  #   k/v tensors, each of potentially different batch size, in order to support
  #   this without having to pre-broadcast all tensors to the max batch size.
  # * we use base-2 exponentials rather than base-e exponentials, to save a
  #   multiplication by LOG2_E.
  # * we skip the sqrt(d_qkv) correction in Flaxformer's rescale_logits,
  #   by instead arranging for it already to have been folded into `q`.
  # * we do not support dropout.
  # * we directly calculate masking here, rather than having masking provided by
  #   the caller as an attention bias. This is a stylistic choice in code
  #   organization.
  #
  # We need to run attention over several different KV caches, including our
  # current KV cache. We need to normalize the attention probabilities relative
  # to sum of all caches, which requires two passes through all caches.
  #
  # We organize it as follows:
  # * first pass through all caches: compute attention (consuming both k and v),
  #   normalizing locally, within just this KV cache. As part of this pass, sum
  #   up normalization terms.
  # * second pass through: renormalize per-cache attention results to global
  #   attention results
  # For each KV cache, we need to keep one of attn_logits, attn_probs, or
  # attn_out live from the first pass to the second pass. It is possible to do
  # renormalization on any of these tensors. When cache.max_length>qkv,
  # it is optimal to keep attn_out live (it is smaller); otherwise, it is
  # optimal to keep attn_logits or attn_probs live. For now we keep it simple
  # and universally choose to keep attn_out live. This simplifies the code, and
  # also is never _too_ bad, because qkv is always fairly small.

  # First pass: consume k, v, sum up normalizers, store attn_out.
  # local_outs: Sequence[f32[batch, qlength, heads, qkv]]
  local_outs = []
  # local_maxes: Sequence[f32[batch, qlength, heads, 1]]
  local_maxes = []
  # local_sums: Sequence[f32[batch, qlength, heads, 1]]
  local_sums = []

  # Attention to previous steps of tokens. Here we don't need any triangular
  # attention mask (since they're all strictly in the past), but we do need
  # length-dependent masking.
  def my_layer(t, layer_idx=0):
    return lax.dynamic_index_in_dim(t, layer, axis=layer_idx, keepdims=False)

  if custom_prefix_masks is None:
    for kv_cache in prefix:
      # cache_k, cache_v: [klen, kbatch, qkv]
      # cache_k, cache_v: [kbatch, klen, qkv]
      cache_k = my_layer(kv_cache.k)
      cache_v = my_layer(kv_cache.v)

      # Attention mask so we don't attend past the length of each prefix.
      k_iota = lax.iota(jnp.int32, kv_cache.max_length)
      # mask: [kbatch, klen]
      if kv_cache.circular:
        # Gives us wraparound then checks it against lengths
        # as when it is circular, the lengths trail the active offset
        # Array([ 2,  1,  0, 11, 10,  9,  8,  7,  6,  5,  4,  3], dtype=int32)
        mask = (
            -(k_iota - kv_cache.offset[0] + 1) % kv_cache.max_length
        ) < kv_cache.lengths[:, jnp.newaxis]
      else:
        mask = k_iota < kv_cache.lengths[:, np.newaxis]
      # mask: [kbatch, num_samples, num_heads, qlen, klen]
      mask = mask[:, np.newaxis, np.newaxis, np.newaxis, :]

      local_out, local_max, local_sum = _attend_chunk(
          q,
          cache_k,
          cache_v,
          mask,
      )
      local_outs.append(local_out)
      local_maxes.append(local_max)
      local_sums.append(local_sum)
  elif len(prefix) == len(custom_prefix_masks):
    for kv_cache, custom_prefix_mask in zip(prefix, custom_prefix_masks):
      # cache_k, cache_v: [klen, kbatch, qkv]
      # cache_k, cache_v: [kbatch, klen, qkv]
      cache_k = my_layer(kv_cache.k)
      cache_v = my_layer(kv_cache.v)
      local_out, local_max, local_sum = _attend_chunk(
          q,
          cache_k,
          cache_v,
          custom_prefix_mask,
      )
      local_outs.append(local_out)
      local_maxes.append(local_max)
      local_sums.append(local_sum)

  if k is not None and v is not None:
    if causal_masking:
      # Attention within this step of tokens. We need the triangular attention
      # mask here, and no length-dependent masking.
      qk = (q.shape[1], k.shape[1])
      q_iota = lax.broadcasted_iota(jnp.int32, qk, 0)
      k_iota = lax.broadcasted_iota(jnp.int32, qk, 1)
      mask = q_iota >= k_iota
    else:
      mask = custom_mask
    local_out, local_max, local_sum = _attend_chunk(
        q,
        k,
        v,
        mask,
    )
    local_outs.append(local_out)
    local_maxes.append(local_max)
    local_sums.append(local_sum)

  # End of first pass: sum locals to global
  global_max = functools.reduce(jnp.maximum, local_maxes)
  global_sum = sum([
      special2.exp2(local_max - global_max) * local_sum
      for (local_sum, local_max) in zip(local_sums, local_maxes)
  ])

  # Second pass: renormalize attn_outs
  attn_out = 0
  for local_max, local_out in zip(local_maxes, local_outs):
    # local_outs has been divided by exp(local_max). Overall it needs to be
    # divided by exp(global_max)*global_sum. So our normalizer is
    # exp(local_max-global_max)/global_sum.
    local_normalizer = special2.exp2(local_max - global_max) / global_sum
    # ^ local_normalizer: f32[batch, qlength, heads, 1]
    attn_out += local_normalizer * local_out

  return attn_out  # pytype: disable=bad-return-type  # jax-ndarray
