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

import functools
from typing import Callable, Sequence

from absl.testing import absltest
import jax
from jax import lax
from jax.experimental.maps import xmap
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import global_to_per_device
from scaling_transformer_inference_efficiency import layers_parallel
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.global_to_per_device import shard_map
from scaling_transformer_inference_efficiency.partitioning import _with_sharding_constraint

jax.config.update('jax_array', True)  # required for jax < 0.4.0

# jax.config.update('experimental_xmap_spmd_lowering', True)
# jax.config.update('experimental_xmap_spmd_lowering_manual', True)
CheckpointSpec = checkpoint.CheckpointSpec

HParams = checkpoint.HParams
Weights = weights.Weights
QuantizedWeights = weights.QuantizedWeights
Layer = weights.Layer
QuantizedLayer = weights.QuantizedLayer


def setup(batch_size, seq_len):
  """Sets up necessary inputs."""
  X, Y, Z = 2, 2, 2  # slice sizes pylint: disable = invalid-name
  assert len(jax.devices()) == X * Y * Z

  mesh = partitioning.make_mesh()
  fold_out_for_mesh = functools.partial(global_to_per_device.fold_out, mesh)

  key = jax.random.PRNGKey(0)
  dtype = jnp.float32
  h = checkpoint.HParams(
      layers=8, embed=8, ff=32, heads=16, qkv=4, max_len=256, vocab=1024)
  key, k2, k3, k4, k5 = jax.random.split(key, 5)
  q_wi = jax.random.normal(k2, (h.layers, h.heads, h.embed, h.q_wi_per_head),
                           dtype)
  kv = jax.random.normal(k3, (h.layers, h.embed, 1, 2 * h.qkv), dtype)
  o_wo = jax.random.normal(k4, (h.layers, h.heads, h.o_wo_per_head, h.embed),
                           dtype)
  embedding = jax.random.normal(k5, (h.vocab, h.embed), dtype)
  sin = jnp.ones((h.max_len, h.qkv // 2), dtype)
  cos = jnp.ones((h.max_len, h.qkv // 2), dtype)

  # create the params
  params_pjit = weights.Weights(
      weights.Layer(q_wi, kv, o_wo), sin, cos, embedding)
  params_logical = weights.Weights.logical_axes()
  params_sharding = jax.tree_util.tree_map(partitioning.logical_to_physical,
                                           params_logical)

  # create the token inputs
  token_chunk = chunk.Chunk(
      tokens=jnp.reshape(
          jnp.arange(batch_size * seq_len), (batch_size, seq_len)),
      lengths=jnp.array([seq_len] * batch_size))
  chunk_logical = chunk.Chunk(tokens=P(None, None), lengths=P(None))
  chunk_sharding = jax.tree_util.tree_map(partitioning.logical_to_physical,
                                          chunk_logical)

  # TODO(sholto): Make dynamic based on batch sharding of kv
  result_logical = chunk.FullChunkResult.logical_axes()
  result_sharding = jax.tree_util.tree_map(partitioning.logical_to_physical,
                                           result_logical)

  result_layout = jax.tree_map(global_to_per_device.logical_to_layout,
                               result_logical)

  def to_named_sharding(mesh, spec):
    return jax.sharding.NamedSharding(mesh, spec)

  to_named_sharding = functools.partial(to_named_sharding, mesh)

  # pjit sharding
  chunk_spec = jax.tree_util.tree_map(to_named_sharding, chunk_sharding)
  param_spec = jax.tree_util.tree_map(to_named_sharding, params_sharding)

  token_chunk = jax.device_put(token_chunk, chunk_spec)
  params_pjit = jax.device_put(params_pjit, param_spec)

  # xmap sharding
  folded_out = jax.tree_map(fold_out_for_mesh, params_pjit, params_logical)
  params_xmap, params_layouts = global_to_per_device.unzip_tree(
      params_pjit, folded_out)
  folded_out = jax.tree_map(fold_out_for_mesh, token_chunk, chunk_logical)
  chunk_xmap, chunk_layout = global_to_per_device.unzip_tree(
      token_chunk, folded_out)

  kv_caches = []

  return (dtype, h, mesh, params_pjit, kv_caches, token_chunk, params_xmap,
          params_layouts, chunk_xmap, chunk_layout, result_layout,
          chunk_sharding, params_sharding, result_sharding)


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
def pjit_embed_and_layer(
    h,
    _transformer_layer_fn,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    token_chunk,
    intermediate_dtype = jnp.bfloat16):
  """Forward pass through model, returning per-token logits."""

  # Start indices are the sums of the lengths of the KV caches.
  start_indices = attention.prefix_lengths(kv_caches)
  prefix_batch, = start_indices.shape
  batch, max_length = token_chunk.tokens.shape
  assert batch % prefix_batch == 0, 'Incompatible batch sizes'

  # Do RoPE lookups in the sin/cos tables. Only needed once per prefix_batch.
  def slice_at(index, table):
    # table: [precomputed_length, qkv // 2]
    return lax.dynamic_slice_in_dim(table, index, max_length)

  def slices_at(indices, table):
    return jax.vmap(slice_at, in_axes=(0, None))(indices, table)

  sin = slices_at(start_indices, params.sin)
  cos = slices_at(start_indices, params.cos)
  # sin, cos: bf16[prefix_batch, max_length, qkv // 2]

  token_ids = _with_sharding_constraint(token_chunk.tokens, (None, None))
  x = params.embedding[token_ids, :]

  x = _with_sharding_constraint(x, (None, None, 'table_embed'))
  x = _with_sharding_constraint(x, ('residual_batch', 'time', 'residual_embed'))

  z, layer_k, layer_v = _transformer_layer_fn(h, 0, params.layer, sin, cos,
                                              kv_caches, x)
  # TODO(sholto): Repro an interesting jax casting issue where not casting here
  # causes it to be very wrong!
  return intermediate_dtype(x), intermediate_dtype(z), intermediate_dtype(
      layer_k), intermediate_dtype(layer_v)


def xmap_embed(
    h,
    _transformer_layer_fn,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    token_chunk):
  del h

  z_axis = lax.psum(1, 'z')
  # Start indices are the sums of the lengths of the KV caches.
  start_indices = attention.prefix_lengths(kv_caches)
  prefix_batch, = start_indices.shape
  batch, max_length = token_chunk.tokens.shape
  assert batch % prefix_batch == 0, 'Incompatible batch sizes'

  # Do RoPE lookups in the sin/cos tables. Only needed once per prefix_batch.
  def slice_at(index, table):
    # table: [precomputed_length, qkv // 2]
    return lax.dynamic_slice_in_dim(table, index, max_length)

  def slices_at(indices, table):
    return jax.vmap(slice_at, in_axes=(0, None))(indices, table)

  sin = slices_at(start_indices, params.sin)
  cos = slices_at(start_indices, params.cos)
  # sin, cos: bf16[prefix_batch, max_length, qkv // 2]

  # x: int32[batch, maxlen]
  # embed: bfloat16[vocab.YZ, dmodel.X]
  x = token_chunk.tokens
  vocab_yz, _ = params.embedding.shape

  yz_index = lax.axis_index('y') * z_axis + lax.axis_index('z')
  vocab_start = yz_index * vocab_yz

  # Initial embedding lookup:
  with jax.named_scope('embed'):
    one_x = x - vocab_start
    embeds = params.embedding[one_x, :]
    one_x = one_x[:, :, jnp.newaxis]
    embeds = jnp.where((one_x >= 0) & (one_x < vocab_yz), embeds, 0)
    # [batch, time, embed.XY]
    embeds = lax.psum_scatter(embeds, 'y', scatter_dimension=2, tiled=True)
    # [batch.Z, time, embed.XY]
    embeds = lax.psum_scatter(embeds, 'z', scatter_dimension=0, tiled=True)

  return embeds, sin, cos


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
def xmap_embed_and_layer(
    h,
    _transformer_layer_fn,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    token_chunk,
    attn_all_to_all,
    shard_seqlen_vs_batch,
    intermediate_dtype = jnp.bfloat16):
  """Forward pass through xmap path, returning per-token logits."""
  x_axis = lax.psum(1, 'x')
  y_axis = lax.psum(1, 'y')
  z_axis = lax.psum(1, 'z')

  x, sin, cos = xmap_embed(h, _transformer_layer_fn, params, kv_caches,
                           token_chunk)

  z, layer_k, layer_v = _transformer_layer_fn(
      h,
      0,
      params.layer,
      sin,
      cos,
      kv_caches,
      x,
      x_axis,
      y_axis,
      z_axis,
      attn_all_to_all,
      latency_collectives=False,
      shard_seqlen_vs_batch=shard_seqlen_vs_batch,
      intermediate_dtype=intermediate_dtype)

  return intermediate_dtype(x), intermediate_dtype(z), intermediate_dtype(
      layer_k), intermediate_dtype(layer_v)


class LayersTest(absltest.TestCase):
  """Tests for inference fwd pass."""

  def xmap_pjit_equivalency(self,
                            batch_size=4,
                            seq_len=32,
                            attn_sharding=partitioning.AttnAllToAll.NONE):
    """Tests equivalency of a pjit and xmap forward pass."""

    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding))
    with rules:
      (dtype, h, mesh, params_pjit, kv_caches, token_chunk, params_xmap,
       params_layouts, chunk_xmap, chunk_layout, _, _, _,
       _) = setup(batch_size, seq_len)

      @functools.partial(pjit)
      def fwd_pjit(token_chunk, params):
        x, z, k, v = pjit_embed_and_layer(
            h,
            layers_parallel.pjit_transformer_layer,
            params,
            kv_caches,
            token_chunk,
            intermediate_dtype=dtype)
        return x, z, k, v

      with mesh:
        x_baseline, z_baseline, k_baseline, _ = fwd_pjit(
            token_chunk, params_pjit)

      z_sharding = P('residual_batch', 'time', 'residual_embed')
      kv_sharding = P('attn_batch', 'length', 'qkv')
      z_layout = global_to_per_device.logical_to_layout(z_sharding)
      kv_layout = global_to_per_device.logical_to_layout(kv_sharding)

      @functools.partial(
          xmap,
          in_axes=(chunk_layout, params_layouts),
          out_axes=(z_layout, z_layout, kv_layout, kv_layout),
          axis_resources={
              'x': 'x',
              'y': 'y',
              'z': 'z'
          })
      def fwd_xmap(token_chunk, params):
        x, z, k, v = xmap_embed_and_layer(
            h,
            layers_parallel.transformer_layer_weight_stationary,
            params,
            kv_caches,
            token_chunk,
            attn_all_to_all=attn_sharding,
            shard_seqlen_vs_batch=False,
            intermediate_dtype=dtype)

        return x, z, k, v

      with mesh:
        x_xm, z_xm, k_xm, _ = fwd_xmap(chunk_xmap, params_xmap)

      x_folded = global_to_per_device.fold_in(x_xm, z_sharding)
      k_folded = global_to_per_device.fold_in(k_xm, kv_sharding)
      z_folded = global_to_per_device.fold_in(z_xm, z_sharding)

      np.testing.assert_allclose(x_baseline, x_folded, rtol=1e-03, atol=1e-03)
      np.testing.assert_allclose(k_baseline, k_folded, rtol=1e-03, atol=1e-03)
      np.testing.assert_allclose(z_baseline, z_folded, rtol=1e-03, atol=1e-03)

  def test_none_sharding(self):
    self.xmap_pjit_equivalency(
        batch_size=4, attn_sharding=partitioning.AttnAllToAll.NONE)

  def test_attn_z_sharding(self):
    self.xmap_pjit_equivalency(
        batch_size=2, attn_sharding=partitioning.AttnAllToAll.AXIS_Z)

  def test_attn_yz_sharding(self):
    self.xmap_pjit_equivalency(
        batch_size=4, attn_sharding=partitioning.AttnAllToAll.AXES_YZ)

  def test_attn_yzx_sharding(self):
    self.xmap_pjit_equivalency(
        batch_size=8, attn_sharding=partitioning.AttnAllToAll.AXES_YZX)

  def shardmap_pjit_equivalency(self,
                                batch_size=4,
                                seq_len=32,
                                attn_sharding=partitioning.AttnAllToAll.NONE):
    """Tests equivalency of a pjit and xmap forward pass."""

    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding))
    with rules:
      (dtype, h, mesh, params_pjit, kv_caches, token_chunk, _, _, _, _, _,
       chunk_sharding, params_sharding, _) = setup(batch_size, seq_len)

      @functools.partial(pjit)
      def fwd_pjit(token_chunk, params):
        x, z, k, v = pjit_embed_and_layer(
            h,
            layers_parallel.pjit_transformer_layer,
            params,
            kv_caches,
            token_chunk,
            intermediate_dtype=dtype)
        return x, z, k, v

      with mesh:
        x_baseline, z_baseline, k_baseline, _ = fwd_pjit(
            token_chunk, params_pjit)

      x_sharding = partitioning.logical_to_physical(
          P('residual_batch', 'time', 'residual_embed'))
      z_sharding = partitioning.logical_to_physical(
          P('residual_batch', 'time', 'residual_embed'))
      kv_sharding = partitioning.logical_to_physical(
          P('attn_batch', 'length', 'qkv'))

      def fwd_xmap(token_chunk, params):
        x, z, k, v = xmap_embed_and_layer(
            h,
            layers_parallel.transformer_layer_weight_stationary,
            params,
            kv_caches,
            token_chunk,
            attn_all_to_all=attn_sharding,
            shard_seqlen_vs_batch=False,
            intermediate_dtype=dtype)

        return x, z, k, v

      with mesh:
        x_xm, z_xm, k_xm, _ = shard_map(
            fwd_xmap,
            mesh,
            in_pspecs=(chunk_sharding, params_sharding),
            out_pspecs=(x_sharding, z_sharding, kv_sharding, kv_sharding),
        )(token_chunk, params_pjit)

      np.testing.assert_allclose(x_xm, x_baseline, rtol=1e-03, atol=1e-03)
      np.testing.assert_allclose(k_xm, k_baseline, rtol=1e-03, atol=1e-03)
      np.testing.assert_allclose(z_xm, z_baseline, rtol=1e-03, atol=1e-03)

  def test_shardmap(self):
    self.shardmap_pjit_equivalency(
        batch_size=4, attn_sharding=partitioning.AttnAllToAll.NONE)


if __name__ == '__main__':
  absltest.main()
