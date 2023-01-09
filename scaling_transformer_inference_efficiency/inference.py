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

"""Minimalist codebase for PaLM model inference.

Relative to the t5x implementation of PaLM, this codebase does not aim for
configurability, and instead aims for peak performance inference, including in
ways that would require significant changes to how t5x's APIs are structured.

Test this with :inference_test
"""

from typing import Callable, Optional, Sequence

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.chunk import FullChunkResult
from scaling_transformer_inference_efficiency.layers import layers_pjit
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap
from scaling_transformer_inference_efficiency.partitioning import _with_sharding_constraint

CheckpointSpec = checkpoint.CheckpointSpec

HParams = checkpoint.HParams
Weights = weights.Weights
QuantizedWeights = weights.QuantizedWeights
Layer = weights.Layer
QuantizedLayer = weights.QuantizedLayer

################################################################################
################################################################################
################################################################################


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
def infer(
    h,
    _transformer_layer_fn,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    chunk,
    intermediate_dtype = jnp.bfloat16):
  """Forward pass through model, returning per-token logits."""

  # flaxformer/architectures/t5/t5_architecture.py;l=1516;

  # Start indices are the sums of the lengths of the KV caches.
  start_indices = attention.prefix_lengths(kv_caches)
  prefix_batch, = start_indices.shape
  batch, max_length = chunk.tokens.shape
  assert batch % prefix_batch == 0, 'Incompatible batch sizes'

  # Do RoPE lookups in the sin/cos tables. Only needed once per prefix_batch.
  def slice_at(index, table):
    # table: [precomputed_length, qkv // 2]
    return lax.dynamic_slice_in_dim(table, index, max_length)

  def slices_at(indices, table):
    # print(f'table: {table.shape}')
    return jax.vmap(slice_at, in_axes=(0, None))(indices, table)

  sin = slices_at(start_indices, params.sin)
  cos = slices_at(start_indices, params.cos)
  # sin, cos: bf16[prefix_batch, max_length, qkv // 2]

  token_ids = _with_sharding_constraint(chunk.tokens, (None, None))
  x = params.embedding[token_ids, :]

  x = _with_sharding_constraint(x, (None, None, 'table_embed'))
  x = _with_sharding_constraint(
      x, ('residual_batch', 'residual_time', 'residual_embed'))

  def loop_body(layer, carry):
    x, k, v = carry

    x, layer_k, layer_v = _transformer_layer_fn(h, layer, params.layer, sin,
                                                cos, kv_caches, x)
    x, layer_k, layer_v = intermediate_dtype(x), intermediate_dtype(
        layer_k), intermediate_dtype(layer_v)
    k = lax.dynamic_update_index_in_dim(k, jnp.swapaxes(layer_k, 0, 1), layer,
                                        0)
    v = lax.dynamic_update_index_in_dim(v, jnp.swapaxes(layer_v, 0, 1), layer,
                                        0)

    return x, k, v

  # Initialize output KV cache.
  # TODO(sholto): attn_batch should be contingent
  k = jnp.zeros((h.layers, max_length, batch, h.qkv), intermediate_dtype)
  k = _with_sharding_constraint(k, ('layers', 'time', 'attn_batch', None))
  v = jnp.zeros((h.layers, max_length, batch, h.qkv), intermediate_dtype)
  v = _with_sharding_constraint(v, ('layers', 'time', 'attn_batch', None))
  x, k, v = jax.lax.fori_loop(0, h.layers, loop_body, (x, k, v))

  k = jnp.swapaxes(k, 0, 1)
  v = jnp.swapaxes(v, 0, 1)

  # TODO(sholto): Should this ever be scaled when quantised?
  x = layers_pjit._layernorm(x)  # pylint: disable = protected-access

  x = _with_sharding_constraint(x, (None, None, None))
  x = _with_sharding_constraint(x, (None, None, 'params_embed'))
  logits = jnp.einsum('bte,ve->btv', jnp.float32(x),
                      jnp.float32(params.embedding))

  # TODO(sholto): Vocab may need to go xyz on 62B VL.
  # END GOOGLE-INTERAL
  logits = _with_sharding_constraint(logits, (None, None, 'vocab'))
  k, v = jnp.bfloat16(k), jnp.bfloat16(v)
  return FullChunkResult(
      logits=logits, kv_cache=attention.KVCache(chunk.lengths, k, v))


def div_up(x, y):
  return (x + y - 1) // y


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
def infer_xmap(
    h,
    _transformer_layer_fn,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    chunk,
    attn_all_to_all,
    latency_collectives,
    shard_seqlen_vs_batch,
    batch_unsharded = False,
    intermediate_dtype = jnp.bfloat16,
    pre_embedded_inputs = None):
  """Forward pass through xmap path, returning per-token logits."""

  # flaxformer/architectures/t5/t5_architecture.py;l=1516;
  x_axis = lax.psum(1, 'x')
  y_axis = lax.psum(1, 'y')
  z_axis = lax.psum(1, 'z')

  if attn_all_to_all == partitioning.AttnAllToAll.NONE:
    attn_batch_sharding = 1
  elif attn_all_to_all == partitioning.AttnAllToAll.AXIS_Z:
    attn_batch_sharding = z_axis
  elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZ:
    attn_batch_sharding = y_axis * z_axis
  elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZX:
    attn_batch_sharding = y_axis * z_axis * x_axis
  else:
    raise NotImplementedError('Ensure you pass in a matching object')

  batch, max_length = chunk.tokens.shape

  # Start indices are the sums of the lengths of the KV caches.
  x, sin, cos = two_d_parallel_xmap.xmap_embed(params, kv_caches, chunk,
                                               shard_seqlen_vs_batch,
                                               batch_unsharded)
  # Used for prompt tuning (where we want to take gradients w.r.t the inputs)
  if pre_embedded_inputs is not None:
    x = pre_embedded_inputs

  def loop_body(layer, carry):
    x, k, v = carry
    x, layer_k, layer_v = _transformer_layer_fn(
        h,
        layer,
        params.layer,
        sin,
        cos,
        kv_caches,
        x,
        x_axis,
        y_axis,
        z_axis,
        attn_all_to_all,
        latency_collectives,
        shard_seqlen_vs_batch,
        batch_unsharded,
        intermediate_dtype)
    k = lax.dynamic_update_index_in_dim(k, jnp.swapaxes(layer_k, 0, 1), layer,
                                        0)
    v = lax.dynamic_update_index_in_dim(v, jnp.swapaxes(layer_v, 0, 1), layer,
                                        0)
    return x, k, v

  # Initialize output KV cache.
  k = jnp.zeros(
      (h.layers, max_length, div_up(batch, attn_batch_sharding), h.qkv),
      intermediate_dtype)
  v = jnp.zeros(
      (h.layers, max_length, div_up(batch, attn_batch_sharding), h.qkv),
      intermediate_dtype)
  x, k, v = jax.lax.fori_loop(0, h.layers, loop_body, (x, k, v))

  k = jnp.swapaxes(k, 0, 1)
  v = jnp.swapaxes(v, 0, 1)

  # [batch, maxlen, embed.X]
  xnorm, _ = two_d_parallel_xmap.allgather_layernorm(x, shard_seqlen_vs_batch,
                                                     batch_unsharded)

  # x: bfloat16[batch, maxlen, dmodel.X] # [vocab.YZ, embed.X]
  with jax.named_scope('unembed'):
    logits_unreduced = jnp.einsum('bte,ve->btv', jnp.float32(xnorm),
                                  jnp.float32(params.embedding))
    # x: [batch, maxlen, vocab.YZ] {X unreduced}
    if batch_unsharded:
      # logits: float32[batch.X, maxlen, vocab.YZ]
      logits = lax.psum_scatter(
          logits_unreduced, 'x', scatter_dimension=2, tiled=True)
    else:
      # logits: float32[batch, maxlen, vocab.YZX]
      logits = lax.psum_scatter(
          logits_unreduced, 'x', scatter_dimension=0, tiled=True)

  k, v = jnp.bfloat16(k), jnp.bfloat16(v)

  # We need to get only the part of lengths which corresponds to that
  # shard of the kv cache, which can be sharded across batch
  # NOTE: This will not currently support MHA being sharded over heads
  #  -only multiquery attention but neither will any of the code above
  # where k and v are sliced into.
  # A MHA kv cache would require a heads dimension!
  # That being said, we don't have any parallel-layers MHA models.

  # chunk.lengths: [batch] -> [batch.attn_batch_sharding]
  # TODO(sholto): Make this simpler
  if attn_all_to_all == partitioning.AttnAllToAll.NONE:
    cache_lengths = chunk.lengths
  elif attn_all_to_all == partitioning.AttnAllToAll.AXIS_Z:
    slice_size = batch // attn_batch_sharding
    z_index = lax.axis_index('z') * slice_size
    cache_lengths = lax.dynamic_slice_in_dim(chunk.lengths, z_index, slice_size)
  elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZ:
    slice_size = batch // attn_batch_sharding
    yz_index = (lax.axis_index('y') * z_axis + lax.axis_index('z')) * slice_size
    cache_lengths = lax.dynamic_slice_in_dim(chunk.lengths, yz_index,
                                             slice_size)
  elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZX:
    slice_size = batch // attn_batch_sharding
    yzx_index = (lax.axis_index('y') *
                 (z_axis * x_axis) + lax.axis_index('z') * x_axis +
                 lax.axis_index('x')) * slice_size
    cache_lengths = lax.dynamic_slice_in_dim(chunk.lengths, yzx_index,
                                             slice_size)
  # should equal batch dim as sharded for kv cache
  assert cache_lengths.shape[0] == batch // attn_batch_sharding
  assert cache_lengths.shape[0] == k.shape[2]

  return FullChunkResult(
      logits=logits, kv_cache=attention.KVCache(cache_lengths, k, v))

