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

"""Tests for incremental."""

from typing import Any, Sequence

from absl.testing import absltest
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import incremental
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import sampling
from scaling_transformer_inference_efficiency import special2
from scaling_transformer_inference_efficiency.checkpoint import HParams
from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.chunk import FullChunkResult

jax.config.update('jax_threefry_partitionable', False)


################################################################################
# We pick a simple predictable behavior of the toy infer function:
# * probability 0.2 of input_token_id
# * probability 0.8 of (input_token_id + 1)
#
# For simplicity we disregard the input kv_caches, and we produce an all-zeros
# output kv cache.
def toy_infer_fn(weights, kv_caches,
                 chunk):
  del weights, kv_caches
  batch, seqlen = chunk.tokens.shape
  batch_iota = lax.broadcasted_iota(np.int32, (batch, seqlen), 0)
  seqlen_iota = lax.broadcasted_iota(np.int32, (batch, seqlen), 1)
  logits = np.full((batch, seqlen, _TOY_HPARAMS.vocab), -1e10, np.float32)
  logits = jnp.array(logits).at[batch_iota, seqlen_iota,
                                chunk.tokens].set(_LOGIT_SAME * special2.LOG2_E)
  logits = logits.at[batch_iota, seqlen_iota,
                     chunk.tokens + 1].set(_LOGIT_NEXT * special2.LOG2_E)
  kv_cache = attention.KVCache.zeros(_TOY_HPARAMS, batch, seqlen)
  return FullChunkResult(logits=logits, kv_cache=kv_cache)


_TOY_HPARAMS = HParams(
    layers=2,
    embed=-1,
    ff=-1,
    heads=-1,
    qkv=1,
    max_len=-1,
    vocab=32,
)


def make_toy_weights():
  return jnp.zeros((16, 32), jnp.bfloat16)


def make_jitted_model():
  rules = partitioning.make_rules_two_d(partitioning.AttnAllToAll.NONE)
  mesh = partitioning.make_mesh()
  return incremental.InferenceModel(
      _TOY_HPARAMS, 1, toy_infer_fn, sampling.sample, mesh, rules=rules
  )


_TOY_CHUNK = Chunk(
    tokens=np.array(
        [
            [0, 1, 1, 2, 3],
            [0, 1, 1, 1, 1],
        ],
        np.int32,
    ),
    lengths=np.array([5, 3], np.int32),
)

_LOGIT_SAME = np.log(0.2)  # score for input_token_id
_LOGIT_NEXT = np.log(0.8)  # score for (input_token_id + 1)

_TOY_PER_TOKEN_SCORES = np.array([
    [0.0, _LOGIT_NEXT, _LOGIT_SAME, _LOGIT_NEXT, _LOGIT_NEXT],
    [0.0, _LOGIT_NEXT, _LOGIT_SAME, _LOGIT_SAME, _LOGIT_SAME],
], np.float32)

_TOY_CHUNK_GREEDY_DECODE = Chunk(
    tokens=np.array([
        [4, 5, 6, 7],
        [2, 3, 4, 5],
    ], np.int32),
    lengths=np.array([4, 4], np.int32),
)


class IncrementalTest(absltest.TestCase):

  def test_prefill(self):
    jitted_model = make_jitted_model()
    prefill_fn = jitted_model.instantiate_prefill_fn()
    chunk_result = jitted_model.prefill(
        make_toy_weights(), prefill_fn, [], _TOY_CHUNK
    )
    np.testing.assert_allclose(
        _TOY_PER_TOKEN_SCORES, chunk_result.per_token_scores
    )

  def test_prefill_incremental(self):
    jitted_model = make_jitted_model()
    prefill_fn = jitted_model.instantiate_prefill_fn()
    weights = make_toy_weights()
    for split in [1, 3]:
      chunk_a, chunk_b = _TOY_CHUNK.split_at(split)

      chunk_a_result = jitted_model.prefill(weights, prefill_fn, [], chunk_a)
      chunk_b_result = jitted_model.prefill(
          weights, prefill_fn, [chunk_a_result], chunk_b
      )
      np.testing.assert_allclose(_TOY_PER_TOKEN_SCORES[:, :split],
                                 chunk_a_result.per_token_scores)
      np.testing.assert_allclose(_TOY_PER_TOKEN_SCORES[:, split:],
                                 chunk_b_result.per_token_scores)

  def test_generate(self):

    jitted_model = make_jitted_model()
    prefill_fn = jitted_model.instantiate_prefill_fn()
    generate_fn = jitted_model.instantiate_generating_fn(4)
    weights = make_toy_weights()
    chunk_result = jitted_model.prefill(weights, prefill_fn, [], _TOY_CHUNK)
    np.testing.assert_allclose(
        _TOY_PER_TOKEN_SCORES, chunk_result.per_token_scores
    )
    gen_chunk, _ = jitted_model.generate(
        weights,
        generate_fn,
        [chunk_result],
        np.arange(2),
        sampling.SamplingHyperParams(temperature=0.7),
    )
    np.testing.assert_equal(
        _TOY_CHUNK_GREEDY_DECODE.tokens, gen_chunk.copy_to_host().tokens
    )


if __name__ == '__main__':
  absltest.main()
