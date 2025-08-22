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
from typing import Any, Callable, List, Tuple

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import chunk
from scaling_transformer_inference_efficiency import incremental
from scaling_transformer_inference_efficiency import inference
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import sampling
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import layers_pjit
from scaling_transformer_inference_efficiency.layers import one_d_parallel_xmap
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap
from scaling_transformer_inference_efficiency.sampling import SamplingHyperParams


h = checkpoint.HParams(
    layers=4, embed=16, ff=32, heads=16, qkv=4, max_len=256, vocab=1024
)
dtype = jnp.float32
eos_id = 0


def setup(batch_size, seq_len, mesh):
  """Sets up necessary inputs."""
  key = jax.random.PRNGKey(4)
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

  token_chunk = jax.device_put(token_chunk, chunk_spec)
  params = jax.device_put(params_pjit, param_spec)

  return (params, token_chunk)


def create_model(
    mesh,
    attn_all_to_all,
    latency_collectives,
    shard_seqlen_vs_batch,
    batch_unsharded,
    rules,
    layer_fn,  # pylint: disable = g-bare-generic
):
  """Factory function for model class."""
  one_d = layer_fn == one_d_parallel_xmap.weight_stationary_simple
  sharding_config = partitioning.ShardingConfig(
      mesh=mesh,
      attn_all_to_all=attn_all_to_all,
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
        attn_all_to_all=attn_all_to_all,
        latency_collectives=latency_collectives,
        shard_seqlen_vs_batch=shard_seqlen_vs_batch,
        batch_unsharded=batch_unsharded,
    )
    sample_fn = sampling.sample
  elif layer_fn == one_d_parallel_xmap.weight_stationary_simple:
    layer_fn = partial(layer_fn, latency_collectives=latency_collectives)
    sample_fn = sampling.sample_manual_batch_unsharded
  elif layer_fn == two_d_parallel_xmap.transformer_layer_weight_gathered:
    sample_fn = sampling.sample
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

  shard_map_inference_fn = partial(
      inference.infer_template,
      h,
      sharding_config,
      forward_pass,
      intermediate_dtype=jnp.float32,
  )

  model = incremental.InferenceModel(
      h, eos_id, shard_map_inference_fn, sample_fn, mesh, rules
  )
  return model


def xmap_pjit_equivalency(
    prompt_batch_size=4,
    prompt_seq_len=96,
    generate_batch_size=16,
    attn_all_to_all=partitioning.AttnAllToAll.NONE,
    latency_collectives=False,
    batch_unsharded=False,
    shard_seqlen_vs_batch=False,
    layer_fn=two_d_parallel_xmap.transformer_layer_weight_stationary,
):
  """Tests equivalency between pjit and manual versions."""
  one_d = layer_fn == one_d_parallel_xmap.weight_stationary_simple
  weight_gathered = (
      layer_fn == two_d_parallel_xmap.transformer_layer_weight_gathered
  )  # pylint: disable = line-too-long
  # import re
  # import os
  # def set_host_device_count(n):
  #   xla_flags = os.getenv('XLA_FLAGS', '')
  #   xla_flags = re.sub(
  #       r'--xla_force_host_platform_device_count=\S+', '', xla_flags
  #   ).split()
  #   os.environ['XLA_FLAGS'] = ' '.join(
  #       ['--xla_force_host_platform_device_count={}'.format(n)] + xla_flags
  #   )
  # set_host_device_count(8)
  # devices = jax.devices()

  if one_d:
    rules = partitioning.make_rules_one_d()
  elif weight_gathered:
    raise NotImplementedError
  else:
    rules = partitioning.make_rules_two_d(
        attn_all_to_all, batch_unsharded=batch_unsharded
    )
  sample_ids = np.arange(generate_batch_size)

  steps = 32
  mesh = partitioning.make_mesh(one_d=one_d)

  pjit_inference_function = partial(
      inference.infer,
      h,
      layers_pjit.pjit_transformer_layer,
      intermediate_dtype=jnp.float32,
  )

  the_model = incremental.InferenceModel(
      h,
      eos_id,
      pjit_inference_function,
      sampling.sample,
      mesh,
      rules,
  )
  prefill_fn = the_model.instantiate_prefill_fn()
  generate_fn = the_model.instantiate_generating_fn(steps=steps)

  with the_model.rules:
    (params, prompt) = setup(
        batch_size=prompt_batch_size, seq_len=prompt_seq_len, mesh=mesh
    )

  # If you had a Seqio.Vocabulary object, you would typically tokenize using
  # prompt = chunk.Chunk.tokenize(the_vocab, ["[web] Which is worse out of"],
  # is_first_chunk=True)
  # however we are currently using the internal PaLM vocab, so spoof in setup.
  kv_cache = []
  prompt_result = the_model.prefill(params, prefill_fn, [], prompt)
  kv_cache = [prompt_result]
  samples, _ = the_model.generate(
      params,
      generate_fn,
      kv_cache,
      sample_ids,
      sample_params=SamplingHyperParams(temperature=0.7),
  )

  ####### Shard map version ############

  the_model_manual = create_model(
      mesh,
      attn_all_to_all,
      latency_collectives,
      shard_seqlen_vs_batch,
      batch_unsharded,
      rules,
      layer_fn,
  )

  params = (
      the_model_manual.rotate_weights(params, latency_collectives)
      if latency_collectives
      else params
  )

  kv_cache = []
  prefill_fn = the_model_manual.instantiate_prefill_fn()
  # prompt_result2 = the_model_manual.prefill(params, prefill_fn, [], prompt)
  generate_fn = the_model_manual.instantiate_generating_fn(steps=steps)
  # print(jax.tree.map(jnp.shape, prompt_result2))
  # samples2, _ = jax.jit(the_model_manual.generate).lower(
  #     params, generate_fn, [prompt_result], sample_ids,
  # )
  prefix = [prompt_result]
  with the_model_manual.mesh, the_model_manual.rules:
    prev_chunk_next_token_logits = prefix[-1].next_token_logits
    cache = [p.kv_cache for p in prefix]

    samples2, _ = jax.experimental.pjit.pjit(
        generate_fn, static_argnames=("sample_params",)
    )(
        params=params,
        prefix=cache,
        prev_chunk_next_token_logits=prev_chunk_next_token_logits,
        sample_ids=sample_ids,
        sample_params=SamplingHyperParams(temperature=0.7),
    )

  assert (samples2.tokens == samples.tokens).all()


def switch_sharding_patterns(
    prompt_batch_size=4,
    prompt_seq_len=32,
    generate_batch_size=16,
    attn_all_to_all=partitioning.AttnAllToAll.AXIS_Z,
    latency_collectives=False,
    batch_unsharded=False,
):
  """Tests weight gathered to stationary, and batch-unsharded to batch sharded."""

  rules_2d = partitioning.make_rules_two_d(
      attn_all_to_all, batch_unsharded=batch_unsharded
  )
  rules_2d_batch_unsharded = partitioning.make_rules_two_d(
      partitioning.AttnAllToAll.NONE, batch_unsharded=True
  )

  sample_ids = np.arange(generate_batch_size)
  steps = 32
  mesh = partitioning.make_mesh()

  pjit_inference_function = partial(
      inference.infer,
      h,
      layers_pjit.pjit_transformer_layer,
      intermediate_dtype=jnp.float32,
  )

  the_model = incremental.InferenceModel(
      h,
      eos_id,
      pjit_inference_function,
      sampling.sample,
      mesh,
      rules_2d_batch_unsharded,
  )
  prefill_fn = the_model.instantiate_prefill_fn()
  generate_fn = the_model.instantiate_generating_fn(steps=steps)

  with the_model.rules:
    (params, prompt) = setup(
        batch_size=prompt_batch_size, seq_len=prompt_seq_len, mesh=mesh
    )

  # If you had a Seqio.Vocabulary object, you would typically tokenize using
  # prompt = chunk.Chunk.tokenize(the_vocab, ["[web] Which is worse out of"],
  # is_first_chunk=True)
  # however we are currently using the internal PaLM vocab, so spoof in setup.
  kv_cache = []
  prompt_result = the_model.prefill(params, prefill_fn, [], prompt)
  kv_cache = [prompt_result]
  samples, _ = the_model.generate(
      params,
      generate_fn,
      kv_cache,
      sample_ids,
      sample_params=SamplingHyperParams(temperature=0.7),
  )

  ####################### xmap version ########################################

  prefill_model = create_model(
      mesh=mesh,
      attn_all_to_all=partitioning.AttnAllToAll.NONE,
      latency_collectives=latency_collectives,
      shard_seqlen_vs_batch=False,
      batch_unsharded=True,
      rules=rules_2d_batch_unsharded,
      layer_fn=two_d_parallel_xmap.transformer_layer_weight_stationary,
  )

  generate_model = create_model(
      mesh=mesh,
      attn_all_to_all=attn_all_to_all,
      latency_collectives=latency_collectives,
      shard_seqlen_vs_batch=False,
      batch_unsharded=batch_unsharded,
      rules=rules_2d,
      layer_fn=two_d_parallel_xmap.transformer_layer_weight_stationary,
  )

  kv_cache = []
  prefill_fn = prefill_model.instantiate_prefill_fn()
  prompt_result2 = prefill_model.prefill(params, prefill_fn, [], prompt)
  generate_fn = generate_model.instantiate_generating_fn(steps=steps)
  # print(jax.tree.map(jnp.shape, prompt_result2))
  samples2, _ = generate_model.generate(
      params,
      generate_fn,
      [prompt_result2],
      sample_ids,
      sample_params=SamplingHyperParams(temperature=0.7),
  )

  assert (samples2.tokens == samples.tokens).all()


def circular_buffer_equivalency(
    prompt_batch_size=4,
    prompt_seq_len=96,
    generate_batch_size=8,
    attn_all_to_all=partitioning.AttnAllToAll.NONE,
    latency_collectives=False,
    batch_unsharded=True,
    shard_seqlen_vs_batch=False,
    layer_fn=two_d_parallel_xmap.transformer_layer_weight_stationary,
):
  """Tests equivalency between pjit and manual versions."""

  rules = partitioning.make_rules_two_d(
      attn_all_to_all, batch_unsharded=batch_unsharded
  )
  sample_ids = np.arange(generate_batch_size)

  steps = 16
  mesh = partitioning.make_mesh()

  the_model = create_model(
      mesh,
      attn_all_to_all,
      latency_collectives,
      shard_seqlen_vs_batch,
      batch_unsharded,
      rules,
      layer_fn,
  )
  with the_model.rules:
    (params, prompt) = setup(
        batch_size=prompt_batch_size, seq_len=prompt_seq_len, mesh=mesh
    )

  params = (
      the_model.rotate_weights(params, latency_collectives)
      if latency_collectives
      else params
  )

  prefill_fn = the_model.instantiate_prefill_fn()
  prompt_result = the_model.prefill(params, prefill_fn, [], prompt)
  generate_fn = the_model.instantiate_generating_fn(steps=steps)
  sample_params = sampling.SamplingHyperParams(temperature=0.7, top_p=0.95)

  prefix = [prompt_result]
  with the_model.mesh, the_model.rules:
    cache = [p.kv_cache for p in prefix]

    # test typical generation function
    samples, _ = the_model.generate(
        params,
        generate_fn,
        [prompt_result],
        sample_ids,
        sample_params=sample_params,
    )

    # create a circular buffer to write to
    sample_rngs, token_indexes_start, generate_chunk, generate_chunk_result = (
        the_model.create_output_buffer(
            the_model._hparams,  # pylint: disable = protected-access
            sample_ids,
            cache,
            steps,
            prompt_result.next_token_logits,
            circular=True,
        )
    )
    loop_fn = jax.jit(
        the_model.sample_infer_write, static_argnames=("model", "sample_params")
    )

    # Go from the middle back around
    for i in range(steps // 2, steps + steps // 2):
      i = i % steps
      generate_chunk, generate_chunk_result = loop_fn(
          the_model,
          params,
          cache,
          sample_params,
          token_indexes_start,
          sample_rngs,
          i,
          (generate_chunk, generate_chunk_result),
      )

  recombined = jnp.concatenate(
      [
          generate_chunk.tokens[:, steps // 2 :],
          generate_chunk.tokens[:, : steps // 2],
      ],
      axis=-1,
  )

  assert (samples.tokens == recombined).all()


class InferenceTest(absltest.TestCase):
  """Tests for inference fwd pass."""

  def test_none_sharding_b1(self):
    xmap_pjit_equivalency(
        prompt_batch_size=1,
        prompt_seq_len=32,
        generate_batch_size=1,
        attn_all_to_all=partitioning.AttnAllToAll.NONE,
        batch_unsharded=True,
    )

  def test_none_sharding_b1_g16(self):
    xmap_pjit_equivalency(
        prompt_batch_size=1,
        prompt_seq_len=32,
        generate_batch_size=16,
        attn_all_to_all=partitioning.AttnAllToAll.NONE,
        batch_unsharded=True,
    )

  def test_none_sharding(self):
    xmap_pjit_equivalency(
        prompt_batch_size=2,
        attn_all_to_all=partitioning.AttnAllToAll.NONE,
        batch_unsharded=True,
    )

  def test_one_d(self):
    xmap_pjit_equivalency(
        prompt_batch_size=2,
        attn_all_to_all=partitioning.AttnAllToAll.NONE,
        layer_fn=one_d_parallel_xmap.weight_stationary_simple,
    )

  def test_attn_z_sharding(self):
    xmap_pjit_equivalency(
        prompt_batch_size=2, attn_all_to_all=partitioning.AttnAllToAll.AXIS_Z
    )

  def test_attn_yz_sharding(self):
    xmap_pjit_equivalency(
        prompt_batch_size=4, attn_all_to_all=partitioning.AttnAllToAll.AXES_YZ
    )

  def test_attn_yzx_sharding(self):
    xmap_pjit_equivalency(
        prompt_batch_size=8, attn_all_to_all=partitioning.AttnAllToAll.AXES_YZX,
        generate_batch_size=16,
    )

  def test_none_sharding_with_latency(self):
    xmap_pjit_equivalency(
        prompt_batch_size=2,
        attn_all_to_all=partitioning.AttnAllToAll.NONE,
        latency_collectives=True,
    )

  def test_batch_unsharded_to_batch_sharded(self):
    switch_sharding_patterns(
        prompt_batch_size=1,
        generate_batch_size=16,
        attn_all_to_all=partitioning.AttnAllToAll.AXES_YZX,
    )

  def test_circular_buffer(self):
    circular_buffer_equivalency(
        prompt_batch_size=2, attn_all_to_all=partitioning.AttnAllToAll.NONE
    )


if __name__ == "__main__":
  absltest.main()
