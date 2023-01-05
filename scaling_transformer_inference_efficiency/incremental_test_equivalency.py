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
from jax.experimental.pjit import PartitionSpec as P
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import incremental
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import layers_pjit
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap

jax.config.update("jax_array", True)  # required for jax < 0.4.0

h = checkpoint.HParams(
    layers=8, embed=16, ff=32, heads=16, qkv=4, max_len=256, vocab=1024)
dtype = jnp.float32
eos_id = 0


def setup(batch_size, seq_len, mesh):
  """Sets up necessary inputs."""
  key = jax.random.PRNGKey(0)
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

  def to_named_sharding(mesh, spec):
    return jax.sharding.NamedSharding(mesh, spec)

  to_named_sharding = functools.partial(to_named_sharding, mesh)

  # pjit sharding
  chunk_spec = jax.tree_util.tree_map(to_named_sharding, chunk_sharding)
  param_spec = jax.tree_util.tree_map(to_named_sharding, params_sharding)

  token_chunk = jax.device_put(token_chunk, chunk_spec)
  params = jax.device_put(params_pjit, param_spec)

  return (params, token_chunk)


def xmap_pjit_equivalency(prompt_batch_size=4,
                          prompt_seq_len=32,
                          generate_batch_size=16,
                          attn_sharding=partitioning.AttnAllToAll.NONE,
                          latency_collectives=False,
                          batch_unsharded=False):
  """Tests shard map."""

  rules = partitioning.make_rules_two_d(
      attn_sharding, batch_unsharded=batch_unsharded)
  sample_ids = np.arange(generate_batch_size)
  temperature = 0.7
  steps = 32

  the_model = incremental.JittedModel(
      h, eos_id,
      functools.partial(
          inference.infer,
          h,
          layers_pjit.pjit_transformer_layer,
          intermediate_dtype=jnp.float32), weights.Weights.logical_axes(),
      rules)

  with the_model.rules:
    (params, prompt) = setup(
        batch_size=prompt_batch_size,
        seq_len=prompt_seq_len,
        mesh=the_model.mesh)

  # If you had a Seqio.Vocabulary object, you would typically tokenize using
  # prompt = chunk.Chunk.tokenize(the_vocab, ["[web] Which is worse out of"],
  # is_first_chunk=True)
  # however we are currently using the internal PaLM vocab, so spoof in setup.
  kv_cache = []
  prompt_result = the_model.prefill(params, [], prompt)
  kv_cache = [prompt_result]
  samples, _ = the_model.generate(steps, params,
                                  incremental.Sampling(temperature), kv_cache,
                                  sample_ids)

  ####### xmap version ############
  the_model_manual = incremental.XmapModel(
      h, eos_id,
      functools.partial(
          inference.infer_xmap,
          h,
          two_d_parallel_xmap.transformer_layer_weight_stationary,
          attn_all_to_all=attn_sharding,
          shard_seqlen_vs_batch=False,
          latency_collectives=latency_collectives,
          batch_unsharded=batch_unsharded,
          intermediate_dtype=jnp.float32), weights.Weights.logical_axes(),
      rules)
  params = the_model_manual.rotate_weights(
      params, latency_collectives) if latency_collectives else params
  params_xmap = the_model_manual.prepare_params(params)
  sample_ids_xmap = the_model_manual.prepare_sample_ids(sample_ids)

  kv_cache = []
  prefill_fn = the_model_manual.instantiate_prefill_fn()
  prompt_result2 = the_model_manual.prefill(params_xmap, prefill_fn, [], prompt)
  kv_cache = [prompt_result2]
  generating_fn = the_model_manual.instantiate_generating_fn(
      steps, incremental.Sampling(temperature), batch_unsharded)
  samples2, _ = the_model_manual.generate(params_xmap, generating_fn, kv_cache,
                                          sample_ids_xmap)

  assert (samples2.tokens == samples.tokens).all()


class InferenceTest(absltest.TestCase):
  """Tests for inference fwd pass."""

  def test_none_sharding_b1(self):
    xmap_pjit_equivalency(
        prompt_batch_size=1,
        prompt_seq_len=32,
        generate_batch_size=1,
        attn_sharding=partitioning.AttnAllToAll.NONE,
        batch_unsharded=True)

  # def test_none_sharding(self):
  #   xmap_pjit_equivalency(
  #       prompt_batch_size=2, attn_sharding=partitioning.AttnAllToAll.NONE)

  # def test_attn_z_sharding(self):
  #   xmap_pjit_equivalency(
  #       prompt_batch_size=2, attn_sharding=partitioning.AttnAllToAll.AXIS_Z)

  # def test_attn_yz_sharding(self):
  #   xmap_pjit_equivalency(
  #       prompt_batch_size=4, attn_sharding=partitioning.AttnAllToAll.AXES_YZ)

  # def test_attn_yzx_sharding(self):
  #   xmap_pjit_equivalency(
  #       prompt_batch_size=8, attn_sharding=partitioning.AttnAllToAll.AXES_YZX)

  # def test_none_sharding_with_latency(self):
  #   xmap_pjit_equivalency(
  #       prompt_batch_size=2,
  #       attn_sharding=partitioning.AttnAllToAll.NONE,
  #       latency_collectives=True)


if __name__ == "__main__":
  absltest.main()
