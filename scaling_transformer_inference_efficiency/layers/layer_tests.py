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

from functools import partial  # pylint: disable = g-importing-member
from typing import Callable, Sequence

from absl.testing import absltest
import jax
from jax import lax
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.layers import layers_pjit
from scaling_transformer_inference_efficiency.layers import one_d_parallel_xmap
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap
from scaling_transformer_inference_efficiency.partitioning import _with_sharding_constraint

CheckpointSpec = checkpoint.CheckpointSpec
HParams = checkpoint.HParams
Weights = weights.Weights
QuantizedWeights = weights.QuantizedWeights
Layer = weights.Layer
QuantizedLayer = weights.QuantizedLayer


def setup(batch_size, seq_len, one_d):
  """Sets up necessary inputs."""
  X, Y, Z = 2, 2, 2  # slice sizes pylint: disable = invalid-name
  assert len(jax.devices()) == X * Y * Z

  mesh = partitioning.make_mesh(one_d=one_d)
  key = jax.random.PRNGKey(0)
  dtype = jnp.float32
  h = checkpoint.HParams(
      layers=8, embed=16, ff=32, heads=16, qkv=4, max_len=256, vocab=1024)
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
  params_sharding = jax.tree_util.tree_map(partitioning.logical_to_physical,
                                           weights.Weights.logical_axes())

  # create the token inputs
  token_chunk = chunk.Chunk(
      tokens=jnp.reshape(
          jnp.arange(batch_size * seq_len), (batch_size, seq_len)),
      lengths=jnp.array([seq_len] * batch_size))
  chunk_sharding = jax.tree_util.tree_map(partitioning.logical_to_physical,
                                          chunk.Chunk.logical_axes())

  result_sharding = jax.tree_util.tree_map(partitioning.logical_to_physical,
                                           chunk.FullChunkResult.logical_axes())

  def to_named_sharding(mesh, spec):
    return jax.sharding.NamedSharding(mesh, spec)

  to_named_sharding = partial(to_named_sharding, mesh)

  # pjit sharding
  chunk_spec = jax.tree_util.tree_map(to_named_sharding, chunk_sharding)
  param_spec = jax.tree_util.tree_map(to_named_sharding, params_sharding)

  token_chunk = jax.device_put(token_chunk, chunk_spec)
  params_pjit = jax.device_put(params_pjit, param_spec)

  kv_caches = []

  return (dtype, h, mesh, params_pjit, kv_caches, token_chunk, chunk_sharding,
          params_sharding, result_sharding)


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
  x = _with_sharding_constraint(
      x, ('residual_batch', 'residual_time', 'residual_embed'))

  z, layer_k, layer_v = _transformer_layer_fn(h, 0, params.layer, sin, cos,
                                              kv_caches, x)
  # TODO(sholto): Repro an interesting jax casting issue where not casting here
  # causes it to be very wrong!
  return intermediate_dtype(x), intermediate_dtype(z), intermediate_dtype(
      layer_k), intermediate_dtype(layer_v)


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
def embed_manual_and_layer(
    h,
    embed_fn,
    _transformer_layer_fn,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    token_chunk,
    intermediate_dtype = jnp.bfloat16,
):
  """Forward pass through xmap path, returning per-token logits."""
  x_axis = lax.psum(1, 'x')
  y_axis = lax.psum(1, 'y')
  z_axis = lax.psum(1, 'z')

  x, sin, cos = embed_fn(params, kv_caches, token_chunk)

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
      intermediate_dtype=intermediate_dtype)

  return intermediate_dtype(x), intermediate_dtype(z), intermediate_dtype(
      layer_k), intermediate_dtype(layer_v)


# pylint: disable = dangerous-default-value
class LayersTest(absltest.TestCase):
  """Tests for inference fwd pass."""

  def xmap_pjit_equivalency(
      self,
      batch_size=4,
      seq_len=48,
      rules = [],
      attn_sharding = partitioning.AttnAllToAll.NONE,
      latency_collectives = False,
      shard_seqlen_vs_batch = False,
      batch_unsharded = False,
      layer_fn = two_d_parallel_xmap.transformer_layer_weight_stationary,
  ):
    """Tests equivalency of a pjit and xmap forward pass."""

    one_d = layer_fn == one_d_parallel_xmap.weight_stationary_simple
    with rules:
      (
          dtype,
          h,
          mesh,
          params,
          kv_caches,
          token_chunk,
          chunk_sharding,
          params_sharding,
          _,
      ) = setup(batch_size, seq_len, one_d)

      @partial(pjit)
      def fwd_pjit(token_chunk, params):
        x, z, k, v = pjit_embed_and_layer(
            h,
            layers_pjit.pjit_transformer_layer,
            params,
            kv_caches,
            token_chunk,
            intermediate_dtype=dtype)
        return x, z, k, v

      with mesh:
        x_baseline, z_baseline, k_baseline, _ = fwd_pjit(token_chunk, params)

      z_sharding = partitioning.logical_to_physical(
          P('residual_batch', 'residual_time', 'residual_embed'))
      kv_sharding = partitioning.logical_to_physical(
          P('attn_batch', 'length', 'qkv'))

      embed_fn = partial(
          two_d_parallel_xmap.embed_manual,
          shard_seqlen_vs_batch=shard_seqlen_vs_batch,
          batch_unsharded=batch_unsharded,
          one_d=one_d,
      )

    if layer_fn == two_d_parallel_xmap.transformer_layer_weight_stationary:
      layer_fn = partial(
          layer_fn,
          attn_all_to_all=attn_sharding,
          latency_collectives=latency_collectives,
          shard_seqlen_vs_batch=shard_seqlen_vs_batch,
          batch_unsharded=batch_unsharded,
      )
    elif layer_fn == one_d_parallel_xmap.weight_stationary_simple:
      layer_fn = partial(layer_fn, latency_collectives=latency_collectives)
    elif layer_fn == two_d_parallel_xmap.transformer_layer_weight_gathered:
      raise NotImplementedError

    def fwd(token_chunk, params):
      x, z, k, v = embed_manual_and_layer(
          h,
          embed_fn,
          layer_fn,
          params,
          kv_caches,
          token_chunk,
          intermediate_dtype=dtype,
      )

      return x, z, k, v

    with mesh:
      x, z, k, _ = jax.jit(
          shard_map(
              fwd,
              mesh,
              in_specs=(chunk_sharding, params_sharding),
              out_specs=(z_sharding, z_sharding, kv_sharding, kv_sharding),
              check_rep=False,
          )
      )(token_chunk, params)

      np.testing.assert_allclose(x_baseline, x, rtol=1e-02, atol=1e-01)
      np.testing.assert_allclose(k_baseline, k, rtol=1e-02, atol=1e-01)
      np.testing.assert_allclose(z_baseline, z, rtol=1e-02, atol=1e-01)

  def test_batch_one(self):
    attn_sharding = partitioning.AttnAllToAll.NONE
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding, batch_unsharded=True))
    self.xmap_pjit_equivalency(
        batch_size=1,
        seq_len=1,
        attn_sharding=partitioning.AttnAllToAll.NONE,
        rules=rules,
        batch_unsharded=True)

  def test_none_sharding_2D(self):
    attn_sharding = partitioning.AttnAllToAll.NONE
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding))
    self.xmap_pjit_equivalency(
        batch_size=4, rules=rules, attn_sharding=attn_sharding)

  def test_seqlen_sharding_2D(self):
    attn_sharding = partitioning.AttnAllToAll.NONE
    shard_seqlen_vs_batch = True
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding, shard_seqlen_vs_batch))
    self.xmap_pjit_equivalency(
        batch_size=4,
        rules=rules,
        attn_sharding=attn_sharding,
        shard_seqlen_vs_batch=shard_seqlen_vs_batch)

  def test_attn_z_sharding(self):
    attn_sharding = partitioning.AttnAllToAll.AXIS_Z
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding))
    self.xmap_pjit_equivalency(
        batch_size=2, rules=rules, attn_sharding=attn_sharding)

  def test_attn_yz_sharding(self):
    attn_sharding = partitioning.AttnAllToAll.AXES_YZ
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding))
    self.xmap_pjit_equivalency(
        batch_size=4, rules=rules, attn_sharding=attn_sharding)

  def test_attn_yzx_sharding(self):
    attn_sharding = partitioning.AttnAllToAll.AXES_YZX
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding))
    self.xmap_pjit_equivalency(
        batch_size=8, rules=rules, attn_sharding=attn_sharding)

  def test_batch1_1D(self):
    rules = partitioning.PartitioningRules(partitioning.make_rules_one_d())
    self.xmap_pjit_equivalency(
        batch_size=1,
        rules=rules,
        layer_fn=one_d_parallel_xmap.weight_stationary_simple,
    )

  def test_batch8_1D(self):
    rules = partitioning.PartitioningRules(partitioning.make_rules_one_d())
    self.xmap_pjit_equivalency(
        batch_size=8,
        rules=rules,
        layer_fn=one_d_parallel_xmap.weight_stationary_simple,
    )


if __name__ == '__main__':
  jax.config.update('jax_threefry_partitionable', False)
  absltest.main()
