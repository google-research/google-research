# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Exports xmap based transformer as an Australis library."""

from functools import partial  # pylint: disable = g-importing-member

import jax
from jax.experimental.australis import exporter
from jax.experimental.pjit import pjit
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap

X, Y, Z = 2, 2, 2  # slice sizes pylint: disable = invalid-name


def setup(mesh, batch_size, seq_len = 32):
  """Sets up necessary inputs."""
  # assert len(jax.devices()) == X * Y * Z

  dtype = jnp.float32
  h = checkpoint.HParams(
      layers=8, embed=16, ff=32, heads=16, qkv=4, max_len=256, vocab=1024
  )

  params_logical = weights.Weights.logical_axes()
  chunk_logical = chunk.Chunk(tokens=P(None, None), lengths=P(None))

  params_sharding = jax.tree_util.tree_map(
      partitioning.logical_to_physical, params_logical
  )

  chunk_sharding = jax.tree_util.tree_map(
      partitioning.logical_to_physical, chunk_logical
  )

  def model_init():
    key = jax.random.PRNGKey(0)
    key, k2, k3, k4, k5 = jax.random.split(key, 5)
    q_wi = jax.random.normal(
        k2, (h.layers, h.heads, h.embed, h.q_wi_per_head), dtype
    )
    kv = jax.random.normal(k3, (h.layers, h.embed, 1, 2 * h.qkv), dtype)
    o_wo = jax.random.normal(
        k4, (h.layers, h.heads, h.o_wo_per_head, h.embed), dtype
    )
    embedding = jax.random.normal(k5, (h.vocab, h.embed), dtype)
    sin = jnp.ones((h.max_len, h.qkv // 2), dtype)
    cos = jnp.ones((h.max_len, h.qkv // 2), dtype)

    # create the params
    params_pjit = weights.Weights(
        weights.Layer(q_wi, kv, o_wo), sin, cos, embedding
    )

    # create the token inputs
    token_chunk = chunk.Chunk(
        tokens=jnp.reshape(
            jnp.arange(batch_size * seq_len), (batch_size, seq_len)
        ),
        lengths=jnp.array([seq_len] * batch_size),
    )
    return params_pjit, token_chunk

  with mesh:
    model_init_lowered = pjit(
        model_init, out_shardings=(params_sharding, chunk_sharding)
    ).lower()
    params_pjit_shape, token_chunk_shape = jax.eval_shape(model_init)

  kv_caches = []

  return (
      model_init_lowered,
      dtype,
      h,
      mesh,
      params_pjit_shape,
      params_pjit_shape,
      kv_caches,
      token_chunk_shape,
  )


def lower():
  """Uses the jax staging API to lower the init and fwd functions."""
  device_mesh = np.array(exporter.fake_devices(8, 'tpu')).reshape((2, 2, 2))
  mesh_axis_names = ('x', 'y', 'z')
  mesh = jax.sharding.Mesh(device_mesh, mesh_axis_names)

  batch_unsharded = False
  latency_collectives = False
  shard_seqlen_vs_batch = False
  attn_sharding = partitioning.AttnAllToAll.NONE
  rules = partitioning.PartitioningRules(
      partitioning.make_rules_two_d(
          attn_sharding, batch_unsharded=batch_unsharded
      )
  )

  with rules:
    (
        model_init_lowered,
        dtype,
        h,
        mesh,
        _,
        rotated_params,
        kv_caches,
        token_chunk,
    ) = setup(mesh, batch_size=8)

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
    )

    layer_fn = partial(
        two_d_parallel_xmap.transformer_layer_weight_stationary,
        attn_all_to_all=attn_sharding,
        latency_collectives=latency_collectives,
        shard_seqlen_vs_batch=shard_seqlen_vs_batch,
        batch_unsharded=batch_unsharded,
    )

    unembed_fn = partial(
        two_d_parallel_xmap.unembed_manual, batch_unsharded=batch_unsharded
    )

    forward_pass = partial(
        inference.manual_fwd_pass,
        h,
        sharding_config,
        embed_fn,
        layer_fn,
        unembed_fn,
    )

    @jax.jit
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
      xmap_transformer_fwd_lowered = fwd.lower(rotated_params, token_chunk)

  return [
      ('xmap_transformer_init', model_init_lowered),
      ('xmap_transformer_fwd', xmap_transformer_fwd_lowered),
  ]


if __name__ == '__main__':
  exporter.run(lower)
