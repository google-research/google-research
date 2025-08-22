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

"""Tests for inference."""

from functools import partial  # pylint: disable = g-importing-member

from absl.testing import absltest
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import collectives
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import layers_pjit
from scaling_transformer_inference_efficiency.layers import one_d_parallel_xmap
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap

X, Y, Z = 2, 2, 2  # slice sizes pylint: disable = invalid-name


def setup(
    batch_size,
    seq_len,
    latency_collectives,
    one_d = False,
):
  """Sets up necessary inputs."""
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

  # create the token inputs
  token_chunk = chunk.Chunk(
      tokens=jnp.reshape(
          jnp.arange(batch_size * seq_len), (batch_size, seq_len)),
      lengths=jnp.array([seq_len] * batch_size))

  def to_named_sharding(mesh, spec):
    return jax.sharding.NamedSharding(mesh, spec)

  to_named_sharding = partial(to_named_sharding, mesh)

  # pjit sharding
  chunk_spec = jax.tree_util.tree_map(
      to_named_sharding, chunk.Chunk.physical_axes()
  )
  param_spec = jax.tree_util.tree_map(
      to_named_sharding, weights.Weights.physical_axes()
  )
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
      params: rotated parameters
    """
    new_layer = params.layer
    new_layer = new_layer.replace(
        q_wi=collectives.preshuffle_for_reducescatter_latency(
            new_layer.q_wi, scatter_axis=1, axis_name='x'))
    new_layer = new_layer.replace(
        o_wo=collectives.preshuffle_for_allgather_matmul_latency(
            new_layer.o_wo, shuffle_axis=1, axis_name='x'))
    return params.replace(layer=new_layer)

  if latency_collectives:
    with mesh:
      rotated_params = jax.jit(
          shard_map(
              rotate_weights,
              mesh,
              in_specs=(weights.Weights.physical_axes(),),
              out_specs=weights.Weights.physical_axes(),
              check_rep=False,
          )
      )(params_pjit)
  else:
    rotated_params = params_pjit

  kv_caches = []

  return (dtype, h, mesh, params_pjit, rotated_params, kv_caches, token_chunk)


# pylint: disable = dangerous-default-value
def xmap_pjit_equivalency(
    batch_size=4,
    seq_len=32,
    rules = [],
    attn_sharding=partitioning.AttnAllToAll.NONE,
    latency_collectives=False,
    batch_unsharded=False,
    shard_seqlen_vs_batch=False,
    layer_fn=two_d_parallel_xmap.transformer_layer_weight_stationary,
    atol=1e-03,
    rtol=1e-06,
):
  """Tests shard map."""
  # Within this function, we device put the relevant arrays ahead of time
  one_d = layer_fn == one_d_parallel_xmap.weight_stationary_simple

  with rules:
    (dtype, h, mesh, params, rotated_params, kv_caches, token_chunk) = setup(
        batch_size=batch_size,
        seq_len=seq_len,
        latency_collectives=latency_collectives,
        one_d=one_d,
    )

    def fwd_pjit(params, token_chunk):
      return inference.infer(
          h,
          layers_pjit.pjit_transformer_layer,
          params,
          kv_caches,
          token_chunk,
          intermediate_dtype=dtype)

    with mesh:
      result_baseline = jax.jit(fwd_pjit)(params, token_chunk)

    sharding_config = partitioning.ShardingConfig(
        mesh=mesh,
        attn_all_to_all=attn_sharding,
        latency_collectives=latency_collectives,
        shard_seqlen_vs_batch=shard_seqlen_vs_batch,
        batch_unsharded=batch_unsharded,
    )

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

    unembed_fn = partial(
        two_d_parallel_xmap.unembed_manual,
        batch_unsharded=batch_unsharded,
        one_d=one_d,
    )

    forward_pass = partial(
        inference.manual_fwd_pass,
        h,
        sharding_config,
        embed_fn,
        layer_fn,
        unembed_fn,
    )

    def fwd(params, token_chunk):
      """Wraps the inference fn to ease shardmap in pytree definition."""
      return inference.infer_template(
          h,
          sharding_config,
          forward_pass,
          params,
          kv_caches,
          token_chunk,
          intermediate_dtype=dtype,
      )

    with mesh:
      result_shardmap = jax.jit(fwd)(rotated_params, token_chunk)

    np.testing.assert_allclose(
        result_baseline.kv_cache.k.astype(jnp.float32),
        result_shardmap.kv_cache.k.astype(jnp.float32),
        rtol=1e-1,
    )  # none_b1 needs this tolerance - XLA? TODO(sholto): Check
    np.testing.assert_allclose(
        result_baseline.logits, result_shardmap.logits, rtol=rtol, atol=atol
    )
    # pylint: disable = unused-variable
    # TODO(sholto): The final grad(shard_map) bug
    # pylint: disable = protected-access
    def grads_pjit(params, token_chunk):
      def loss_fn(params, token_chunk):
        result = fwd_pjit(params, token_chunk)
        return result.logits.mean()

      loss, grads = jax.value_and_grad(loss_fn)(params, token_chunk)
      grads = jax.tree.map(
          partitioning._with_sharding_constraint,
          grads,
          weights.Weights.logical_axes(),
      )
      return loss, grads

    def grads(params, token_chunk):
      def loss_fn(params, token_chunk):
        result = fwd(params, token_chunk)
        return result.logits.mean()

      loss, grads = jax.value_and_grad(loss_fn)(params, token_chunk)
      grads = jax.tree.map(
          partitioning._with_sharding_constraint,
          grads,
          weights.Weights.logical_axes(),
      )
      return loss, grads

    if attn_sharding == partitioning.AttnAllToAll.NONE:
      with mesh:
        loss_pjit, grads_pjit = jax.jit(grads_pjit)(params, token_chunk)
        loss, grads = jax.jit(grads)(params, token_chunk)

      # jax.tree.map(
      #     partial(np.testing.assert_allclose, atol=atol),
      #     grads_pjit,
      #     grads,
      # )


class InferenceTest(absltest.TestCase):
  """Tests for inference fwd pass."""

  def test_none_sharding_b1(self):
    attn_sharding = partitioning.AttnAllToAll.NONE
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding, batch_unsharded=True)
    )
    xmap_pjit_equivalency(
        batch_size=1,
        seq_len=1,
        rules=rules,
        attn_sharding=partitioning.AttnAllToAll.NONE,
        batch_unsharded=True,
        atol=1e-01,
    )  # TODO(sholto); Check if this is because it occurs on VPU like b/246436629 pylint: disable= line-too-long

  def test_none_sharding(self):
    attn_sharding = partitioning.AttnAllToAll.NONE
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding, batch_unsharded=True)
    )
    xmap_pjit_equivalency(
        batch_size=2,
        rules=rules,
        attn_sharding=attn_sharding,
        batch_unsharded=True,
    )

  def test_one_d(self):
    rules = partitioning.PartitioningRules(partitioning.make_rules_one_d())
    xmap_pjit_equivalency(
        batch_size=2,
        rules=rules,
        layer_fn=one_d_parallel_xmap.weight_stationary_simple,
    )

  def test_attn_z_sharding(self):
    attn_sharding = partitioning.AttnAllToAll.AXIS_Z
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding)
    )
    xmap_pjit_equivalency(
        batch_size=2, rules=rules, attn_sharding=attn_sharding
    )

  def test_attn_yz_sharding(self):
    attn_sharding = partitioning.AttnAllToAll.AXES_YZ
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding)
    )
    xmap_pjit_equivalency(
        batch_size=4, rules=rules, attn_sharding=attn_sharding
    )

  def test_attn_yz_sharding_batch_unsharded(self):
    attn_sharding = partitioning.AttnAllToAll.AXES_YZ
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding, batch_unsharded=True)
    )
    xmap_pjit_equivalency(
        batch_size=4, rules=rules, attn_sharding=attn_sharding,
        batch_unsharded=True,
    )

  def test_attn_yzx_sharding(self):
    attn_sharding = partitioning.AttnAllToAll.AXES_YZX
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding)
    )
    xmap_pjit_equivalency(
        batch_size=8, rules=rules, attn_sharding=attn_sharding
    )

  def test_none_sharding_with_latency(self):
    attn_sharding = partitioning.AttnAllToAll.NONE
    rules = partitioning.PartitioningRules(
        partitioning.make_rules_two_d(attn_sharding, batch_unsharded=True)
    )
    xmap_pjit_equivalency(
        batch_size=2,
        rules=rules,
        attn_sharding=attn_sharding,
        latency_collectives=True,
        batch_unsharded=True,
    )


if __name__ == '__main__':
  absltest.main()
