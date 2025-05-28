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

"""Minimalist codebase for PaLM model inference.

Relative to the t5x implementation of PaLM, this codebase does not aim for
configurability, and instead aims for peak performance inference, including in
ways that would require significant changes to how t5x's APIs are structured.

Test this with :inference_test
"""

from functools import partial  # pylint: disable = g-importing-member
from typing import Callable, Optional, Sequence

import jax
from jax import lax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import jax.scipy
from jax.sharding import PartitionSpec as P

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
    x, layer_k, layer_v = intermediate_dtype(x), intermediate_dtype(  # pytype: disable=not-callable  # jnp-type
        layer_k), intermediate_dtype(layer_v)  # pytype: disable=not-callable  # jnp-type
    k = lax.dynamic_update_index_in_dim(k, jnp.swapaxes(layer_k, 0, 1), layer,
                                        0)
    v = lax.dynamic_update_index_in_dim(v, jnp.swapaxes(layer_v, 0, 1), layer,
                                        0)

    return x, k, v

  # Initialize output KV cache.
  k = jnp.zeros((h.layers, max_length, batch, h.qkv), intermediate_dtype)
  k = _with_sharding_constraint(k, ('layers', 'attn_batch', 'time', None))
  v = jnp.zeros((h.layers, max_length, batch, h.qkv), intermediate_dtype)
  v = _with_sharding_constraint(v, ('layers', 'attn_batch', 'time', None))
  x, k, v = jax.lax.fori_loop(0, h.layers, loop_body, (x, k, v))

  k = jnp.swapaxes(k, 1, 2)
  v = jnp.swapaxes(v, 1, 2)

  # TODO(sholto): Should this ever be scaled when quantised?
  x = layers_pjit._layernorm(x)  # pylint: disable = protected-access

  x = _with_sharding_constraint(x, (None, None, None))
  x = _with_sharding_constraint(x, (None, None, 'embedding_embed'))
  logits = jnp.einsum('bte,ve->btv', jnp.float32(x),
                      jnp.float32(params.embedding))

  logits = _with_sharding_constraint(logits, (None, None, 'vocab'))
  k, v = jnp.bfloat16(k), jnp.bfloat16(v)
  return FullChunkResult(
      logits=logits, kv_cache=attention.KVCache(chunk.lengths, k, v, offset=0))  # pytype: disable=wrong-arg-types  # jax-ndarray


def manual_fwd_pass(
    h,
    sharding_config,
    embed_fn,
    _transformer_layer_fn,
    unembed_fn,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    chunk,
    k,
    v,
    pre_embedded_inputs = None,
    intermediate_dtype = jnp.bfloat16,
):
  """Fwd pass to use within shard_map."""

  x_axis = lax.psum(1, 'x')
  y_axis = lax.psum(1, 'y')
  z_axis = lax.psum(1, 'z')

  # Start indices are the sums of the lengths of the KV caches.
  x, sin, cos = embed_fn(params, kv_caches, chunk)

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
        intermediate_dtype=intermediate_dtype,
    )
    k = lax.dynamic_update_index_in_dim(
        k, jnp.swapaxes(layer_k, 0, 1), layer, 0
    )
    v = lax.dynamic_update_index_in_dim(
        v, jnp.swapaxes(layer_v, 0, 1), layer, 0
    )
    return x, k, v
  x, k, v = jax.lax.fori_loop(0, h.layers, loop_body, (x, k, v))
  k = jnp.swapaxes(k, 1, 2)
  v = jnp.swapaxes(v, 1, 2)

  # [batch, maxlen, embed.X]
  xnorm, _ = two_d_parallel_xmap.allgather_layernorm(
      x, sharding_config.shard_seqlen_vs_batch, sharding_config.batch_unsharded
  )

  logits = unembed_fn(xnorm, params)

  return logits, k, v


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
def infer_template(
    h,
    sharding_config,
    fwd_pass,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    chunk,
    pre_embedded_inputs = None,
    intermediate_dtype = jnp.bfloat16,
):
  """Forward pass through xmap path, returning per-token logits."""

  # flaxformer/architectures/t5/t5_architecture.py;l=1516;

  batch, max_length = chunk.tokens.shape

  # Initialize output KV cache.
  k = jnp.zeros((h.layers, max_length, batch, h.qkv), intermediate_dtype)
  v = jnp.zeros((h.layers, max_length, batch, h.qkv), intermediate_dtype)

  fwd_pass = partial(
      fwd_pass,
      intermediate_dtype=intermediate_dtype,
  )

  # 2D: logits: [batch.x, time, vocab.YZ]
  #     kv_cache: [.., ..., prefixbatch, ...]
  cache_sharding = [
      kv.physical_axes(kv.circular) for kv in kv_caches
  ]
  # Match static values with actual cache

  # if a dimension is insufficient to be sharded, replicate
  cache_sharding = jax.tree.map(
      partial(partitioning.safe_sharding, mesh=sharding_config.mesh),
      kv_caches,
      cache_sharding,
  )

  logit_logical = P('logit_batch', 'time', 'vocab')
  logit_sharding = jax.tree.map(
      partitioning.logical_to_physical, logit_logical
  )

  # input/output cache, where we write the per layer kv cache results
  in_cache_sharding = jax.tree.map(
      partitioning.logical_to_physical,
      P('prefix_layers', 'prefix_time', 'attn_batch', 'prefix_qkv'),
  )

  embedding_sharding = jax.tree.map(
      partitioning.logical_to_physical,
      P('residual_batch', 'residual_time', 'residual_embed'),
  )

  pre_embedded_sharding = (
      None if pre_embedded_inputs is None else embedding_sharding
  )

  logits, k, v = shard_map(
      fwd_pass,
      sharding_config.mesh,
      in_specs=(
          params.physical_axes(),
          cache_sharding,
          chunk.physical_axes(),
          in_cache_sharding,
          in_cache_sharding,
          pre_embedded_sharding,
      ),
      out_specs=(
          logit_sharding,
          attention.KVCache.physical_axes().k,
          attention.KVCache.physical_axes().v,
      ),
      check_rep=False,
  )(params, kv_caches, chunk, k, v, pre_embedded_inputs)
  k, v = jnp.bfloat16(k), jnp.bfloat16(v)

  logits = _with_sharding_constraint(logits, logit_logical)

  return FullChunkResult(
      logits=logits, kv_cache=attention.KVCache(chunk.lengths, k, v, offset=0)  # pytype: disable=wrong-arg-types  # jax-ndarray
  )
