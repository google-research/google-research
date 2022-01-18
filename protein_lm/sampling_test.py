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

# Lint as: python3
"""Tests for sampling from models."""

import functools

from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v1 as tf
from protein_lm import domains
from protein_lm import models
from protein_lm import sampling

lm_cls = functools.partial(
    models.FlaxLM,
    num_layers=2,
    num_heads=1,
    emb_dim=64,
    mlp_dim=64,
    qkv_dim=64)


class SamplingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ([-1e10, 0, -1e10], 1),
      ([[-1e10, 0, -1e10]], [1]),
  )
  def test_multinomial(self, logits, expected_samples):
    """Tests that multinomial works as expected_samples."""
    rng = random.PRNGKey(0)
    samples = sampling.multinomial(rng=rng, logits=jnp.array(logits))
    self.assertAllEqual(samples, expected_samples)

  def _make_pretrained_transformer(self, **kwargs):
    """Trains a transformer to produce strings of alternating a's and b's."""
    seqs = ['abab', 'baba'] * 64
    domain = domains.VariableLengthDiscreteDomain(
        vocab=domains.Vocabulary(tokens=['a', 'b'],
                                 include_bos=True,
                                 include_eos=True),
        length=len(seqs[0]))
    enc_seqs = np.array(domain.encode(seqs, pad=False))
    lm = lm_cls(domain=domain, learning_rate=0.001, **kwargs)
    lm.fit(enc_seqs, batch_size=len(enc_seqs), epochs=20)
    return lm, domain

  def test_sampling(self):
    """Tests that samples match the training data: 'abab', 'baba'."""
    lm, domain = self._make_pretrained_transformer()
    batch_size = 128
    samples = lm.sample(batch_size)
    samples_str = domain.decode(samples)
    samples_hist = {elem: samples_str.count(elem) for elem in set(samples_str)}
    estimated = [samples_hist[elem] for elem in ['abab', 'baba']]
    estimated = np.array(estimated) / batch_size
    self.assertAllClose(estimated, [0.5, 0.5], atol=0.1)

  def test_sampling_with_prompt(self):
    """Tests that samples with prompt 'a' are almost all equal to 'abab'."""
    lm, domain = self._make_pretrained_transformer()
    batch_size = 10
    prompt_token = domain.vocab.tokens.index('a')
    prompt = jnp.concatenate(
        [jnp.ones((batch_size, 1)).astype(jnp.int32) * lm.bos_token,
         jnp.ones((batch_size, 1)).astype(jnp.int32) * prompt_token],
        axis=1)
    samples = lm.sample_with_prompt(prompt)
    samples_str = domain.decode(samples)
    samples_hist = {elem: samples_str.count(elem) for elem in set(samples_str)}
    self.assertAllClose(samples_hist['abab'] / batch_size, 1., atol=0.1)


if __name__ == '__main__':
  tf.test.main()
