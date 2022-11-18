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
from jax.experimental.maps import Mesh
from jax.experimental.maps import xmap
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import global_to_per_device
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import layers_parallel
from scaling_transformer_inference_efficiency import weights

jax.config.update('jax_array', True)  # required for jax < 0.4.0


class InferenceTest(absltest.TestCase):
  """Tests for inference fwd pass."""

  def test_xmap_pjit_equivalency(self):
    """Tests equivalency of a pjit and xmap forward pass."""
    X, Y, Z = 2, 2, 2  # slice sizes pylint: disable = invalid-name
    assert len(jax.devices()) == X * Y * Z
    devices = np.array(jax.devices()[:X * Y * Z]).reshape((X, Y, Z))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    fold_out_for_mesh = functools.partial(global_to_per_device.fold_out, mesh)

    key = jax.random.PRNGKey(0)
    dtype = jnp.float32
    h = checkpoint.HParams(
        layers=8, embed=8, ff=32, heads=16, qkv=4, max_len=256, vocab=32)
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
    params_sharding = weights.Weights.logical_axes()
    folded_out = jax.tree_map(fold_out_for_mesh, params_pjit, params_sharding)
    params_xmap, params_layouts = global_to_per_device.unzip_tree(
        params_pjit, folded_out)

    # create the token inputs
    token_chunk = chunk.Chunk(
        tokens=jnp.reshape(jnp.arange(24), (4, 6)),
        lengths=jnp.array([6, 6, 6, 6]))
    chunk_sharding = chunk.Chunk(tokens=P(None, None), lengths=P(None))
    folded_out = jax.tree_map(fold_out_for_mesh, token_chunk, chunk_sharding)
    chunk_xmap, chunk_layout = global_to_per_device.unzip_tree(
        token_chunk, folded_out)

    kv_caches = []

    @functools.partial(pjit)
    def fwd_pjit(token_chunk, params):
      return inference.infer(
          h,
          layers_parallel.pjit_transformer_layer,
          params,
          kv_caches,
          token_chunk,
          intermediate_dtype=dtype)

    with mesh:
      result_baseline = fwd_pjit(token_chunk, params_pjit)

    # TODO(sholto): Make dynamic based on batch sharding of kv
    result_sharding = chunk.FullChunkResult(
        logits=P('batch.X', 'time', 'table_vocab.YZ'),
        kv_cache=attention.KVCache(
            lengths=P(None),
            k=P('length', 'layers', 'batch', 'query'),
            v=P('length', 'layers', 'batch', 'query')))

    def wrap_inner_specs(sharding):
      return global_to_per_device.Leaf(
          global_to_per_device.logical_to_layout(sharding))

    result_layout = jax.tree_map(wrap_inner_specs, result_sharding)

    @functools.partial(
        xmap,
        in_axes=(chunk_layout, params_layouts),
        out_axes=result_layout,
        axis_resources={
            'x': 'x',
            'y': 'y',
            'z': 'z'
        })
    def fwd_xmap(token_chunk, params):
      return inference.infer_xmap(
          h,
          layers_parallel.transformer_layer_weight_stationary,
          params,
          kv_caches,
          token_chunk,
          attn_all_to_all=layers_parallel.AttnAllToAll.NONE,
          shard_seqlen_vs_batch=False,
          intermediate_dtype=dtype)

    with mesh:
      result_xm = fwd_xmap(chunk_xmap, params_xmap)

    logits_xm_folded = global_to_per_device.fold_in(
        result_xm.logits, P('batch.X', 'time', 'table_vocab.YZ'))

    np.testing.assert_allclose(
        result_baseline.logits, logits_xm_folded, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
  absltest.main()
