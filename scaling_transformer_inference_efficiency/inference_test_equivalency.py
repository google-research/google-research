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

"""Tests for inference."""

import functools

from absl.testing import absltest
import jax
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import collectives
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import layers_pjit
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap
from scaling_transformer_inference_efficiency.maps import shard_map

jax.config.update('jax_array', True)  # required for jax < 0.4.0

X, Y, Z = 2, 2, 2  # slice sizes pylint: disable = invalid-name


def setup(batch_size, seq_len, latency_collectives):
  """Sets up necessary inputs."""
  assert len(jax.devices()) == X * Y * Z

  mesh = partitioning.make_mesh()

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

  result_logical = chunk.FullChunkResult.logical_axes()
  result_sharding = jax.tree_util.tree_map(partitioning.logical_to_physical,
                                           result_logical)

  def to_named_sharding(mesh, spec):
    return jax.sharding.NamedSharding(mesh, spec)

  to_named_sharding = functools.partial(to_named_sharding, mesh)

  # pjit sharding
  chunk_spec = jax.tree_util.tree_map(to_named_sharding, chunk_sharding)
  param_spec = jax.tree_util.tree_map(to_named_sharding, params_sharding)
  # result_spec = jax.tree_util.tree_map(to_named_sharding, result_sharding)

  token_chunk = jax.device_put(token_chunk, chunk_spec)
  params_pjit = jax.device_put(params_pjit, param_spec)

  def rotate_weights(params):
    """Rotate the weights for the collectives.

    Assumed to occur in a per device form. Assumes 2D partitioning.
    q_wi: [layers, heads.YZ, dmodel.X, q_wi_per_head]
    o_wo: [layers, heads.YZ, owo_per_head, dmodel.X]

    Args:
      params: parameters

    Returns:
      params: rortated parameters
    """
    new_layer = params.layer
    new_layer = new_layer.replace(
        q_wi=collectives.preshuffle_for_reducescatter_latency(
            new_layer.q_wi, sharded_dim='x', scatter_dim=1))
    new_layer = new_layer.replace(
        o_wo=collectives.preshuffle_for_allgather_matmul_latency(
            new_layer.o_wo, shuffle_axis=1, shard_axis='x'))
    return params.replace(layer=new_layer)

  if latency_collectives:
    with mesh:
      rotated_params = shard_map.shard_map(
          rotate_weights,
          mesh,
          in_specs=(params_sharding,),
          out_specs=params_sharding)(
              params_pjit)
  else:
    rotated_params = params_pjit

  kv_caches = []

  return (dtype, h, mesh, params_pjit, rotated_params, kv_caches, token_chunk,
          chunk_sharding, params_sharding, result_sharding)


def xmap_pjit_equivalency(batch_size=4,
                          seq_len=32,
                          attn_sharding=partitioning.AttnAllToAll.NONE,
                          latency_collectives=False,
                          batch_unsharded=False,
                          atol=1e-03):
  """Tests shard map."""
  # Within this function, we device put the relevant arrays ahead of time

  rules = partitioning.PartitioningRules(
      partitioning.make_rules_two_d(
          attn_sharding, batch_unsharded=batch_unsharded))

  with rules:
    (dtype, h, mesh, params, rotated_params, kv_caches, token_chunk,
     chunk_sharding, param_sharding, result_sharding) = setup(
         batch_size=batch_size,
         seq_len=seq_len,
         latency_collectives=latency_collectives)

    @functools.partial(pjit)
    def fwd_pjit(token_chunk, params):
      return inference.infer(
          h,
          layers_pjit.pjit_transformer_layer,
          params,
          kv_caches,
          token_chunk,
          intermediate_dtype=dtype)

    with mesh:
      result_baseline = fwd_pjit(token_chunk, params)

    def fwd(params, token_chunk):
      """Wraps the inference fn to ease shardmap in pytree definition."""
      return inference.infer_xmap(
          h,
          two_d_parallel_xmap.transformer_layer_weight_stationary,
          params,
          kv_caches,
          token_chunk,
          attn_all_to_all=attn_sharding,
          latency_collectives=latency_collectives,
          shard_seqlen_vs_batch=False,
          batch_unsharded=batch_unsharded,
          intermediate_dtype=dtype)

    def wrapped_shardmap(params, token_chunk):
      """jit/pjit wrapping shardmap."""
      result = shard_map.shard_map(
          fwd,
          mesh,
          in_specs=(param_sharding, chunk_sharding),
          out_specs=result_sharding)(params, token_chunk)
      return result

    with mesh:
      result_shardmap = wrapped_shardmap(rotated_params, token_chunk)

    # np.testing.assert_allclose(
    #     result_baseline.kv_cache.k.astype(jnp.float32),
    #     result_shardmap.kv_cache.k.astype(jnp.float32),
    #     rtol=rtol)
    np.testing.assert_allclose(
        result_baseline.logits, result_shardmap.logits, atol=atol)


class InferenceTest(absltest.TestCase):
  """Tests for inference fwd pass."""

  def test_none_sharding_b1(self):
    xmap_pjit_equivalency(
        batch_size=1,
        seq_len=1,
        attn_sharding=partitioning.AttnAllToAll.NONE,
        batch_unsharded=True,
        atol=1e-01)  # TODO(sholto); Check if this is because it occurs on VPU like b/246436629 pylint: disable= line-too-long

  def test_none_sharding(self):
    xmap_pjit_equivalency(
        batch_size=2, attn_sharding=partitioning.AttnAllToAll.NONE)

  def test_attn_z_sharding(self):
    xmap_pjit_equivalency(
        batch_size=2, attn_sharding=partitioning.AttnAllToAll.AXIS_Z)

  def test_attn_yz_sharding(self):
    xmap_pjit_equivalency(
        batch_size=4, attn_sharding=partitioning.AttnAllToAll.AXES_YZ)

  def test_attn_yzx_sharding(self):
    xmap_pjit_equivalency(
        batch_size=8, attn_sharding=partitioning.AttnAllToAll.AXES_YZX)

  def test_none_sharding_with_latency(self):
    xmap_pjit_equivalency(
        batch_size=2,
        attn_sharding=partitioning.AttnAllToAll.NONE,
        latency_collectives=True)


if __name__ == '__main__':
  absltest.main()
