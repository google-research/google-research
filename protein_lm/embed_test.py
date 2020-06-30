# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Tests for embedding functions."""

import functools

from absl.testing import absltest

import jax
from protein_lm import data
from protein_lm import embed
from protein_lm import models


class EmbedTest(absltest.TestCase):

  def test_embed_with_loaded_model(self):
    p1 = 'ACDEFHIKLNQP'
    p2 = 'ALNQP'
    encoded = data.protein_domain.encode([p1, p2])
    model = models.FlaxLM(
        domain=data.protein_domain,
        num_layers=1,
        num_heads=1,
        qkv_dim=32,
        mlp_dim=32)
    reduction = functools.partial(jax.numpy.sum, axis=-2)

    # Check we can embed int encoded sequences.
    embed_with_sum_fn = embed.ProteinLMEmbedder(
        model=model, output_head='output_emb', length=128, reduce_fn=reduction)
    int_embs = embed_with_sum_fn(encoded)
    self.assertEqual((2, 32), int_embs.shape)

    # Check we can embed strings
    embed_strings_fn = embed.get_embed_fn(model=model, reduce_fn=reduction)
    str_embs = embed_strings_fn([p1, p2])
    self.assertEqual((2, 32), str_embs.shape)


if __name__ == '__main__':
  absltest.main()
