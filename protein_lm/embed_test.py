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
"""Tests for embedding functions."""

from absl.testing import absltest
from absl.testing import parameterized
import mock
import numpy as np
import tensorflow.compat.v1 as tf

from protein_lm import data
from protein_lm import embed
from protein_lm import models


class EncodingTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (10, None, 4),
      (None, 10, 4),
      (None, 3, 3),
      (3, None, 3))
  def test_encode_string_sequences(self, domain_length, length,
                                   expected_output_length):
    seqs = ['ABCD', 'EFG']
    domain = data.make_protein_domain(
        length=domain_length) if domain_length else None

    output_batch = embed._encode_string_sequences(
        seqs, domain=domain, length=length)
    self.assertEqual(output_batch.shape, (2, expected_output_length))


def _get_model(domain):
  return models.FlaxLM(
      domain=domain,
      num_layers=1,
      num_heads=1,
      qkv_dim=32,
      emb_dim=32,
      mlp_dim=32)


class EmbedTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self._domain = data.make_protein_domain(length=12)
    self._model = _get_model(self._domain)
    self._embed_fn = embed.get_embed_fn(
        model=self._model,
        domain=self._domain,
        reduce_fn=embed.masked_reduce_fn)
    super().setUp()

  def test_get_embed_fn_int_sequences(self):
    p1 = 'ACDEFHIKLNQP'
    p2 = 'ALNQP'
    encoded = self._domain.encode([p1, p2])
    int_embs = self._embed_fn(encoded)
    self.assertEqual((2, 32), int_embs.shape)

  def test_embed_strings(self):
    p1 = 'ACDEFHIKLNQP'
    p2 = 'ALNQP'
    str_embs = self._embed_fn([p1, p2])
    self.assertEqual((2, 32), str_embs.shape)

  @parameterized.parameters(
      (embed.sum_reducer, [[5, 7, 9]]),
      (embed.mean_reducer, [[2.5, 3.5, 4.5]]),
      (embed.max_reducer, [[4, 5, 6]]),
  )
  def test_reducer(self, reduce_fn, expected):
    embedding = np.array([[[1, 2, 6], [4, 5, 3], [7, 10, 9]]])
    mask = np.array([[1, 1, 0]])
    reduced = reduce_fn(embedding, mask)
    self.assertAllClose(reduced, expected)

  @parameterized.parameters(
      (True, True, True, True, [[1, 2]]),
      (False, True, True, True, [[3, 4]]),
      (True, False, True, True, [[7, 8]]),
      (True, True, False, True, [[9, 10]]),
      (True, True, True, False, [[46, 11]]),
  )
  def test_masked_reduce_fn(self, ignore_eos, ignore_bos, ignore_pad,
                            ignore_mask, expected):
    embedding = np.array([[[1, 2], [13, 14], [5, 6], [17, 18], [91, 20]]])
    domain = self._domain
    inputs = np.array([[0, domain.vocab.bos, domain.vocab.eos, domain.vocab.pad,
                        domain.vocab.mask]])
    reduced = embed.masked_reduce_fn(embedding=embedding,
                                     inputs=inputs,
                                     ignore_eos=ignore_eos,
                                     ignore_bos=ignore_bos,
                                     ignore_pad=ignore_pad,
                                     ignore_mask=ignore_mask)
    self.assertAllClose(reduced, expected)

  def test_validate_input_int_sequences(self):
    with self.assertRaisesRegex(ValueError, 'Input int-encoded sequences'):
      self._embed_fn([np.ones(14)])


class ProteinLMEmbedderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    self._domain = data.make_protein_domain(length=12)
    self._model = _get_model(self._domain)
    super().setUp()

  # TODO(gandreea): get this test to by fixing domain.encode padding issues
  # def test_embedder_encoding(self):
  #   seqs = ['ACDEFHIKLNQP', 'ALNQP']
  #   str_embs = self._embed_fn(seqs)
  #   int_embs = self._embed_fn(self._domain.encode(seqs))
  #   self.assertAllClose(int_embs, str_embs)

  def test_embedder_batching(self):
    """Asserts that the model is always called with fixed-size batches."""
    batch_size = 4
    embedder = embed.ProteinLMEmbedder(
        model=self._model, output_head='output_emb', batch_size=batch_size)
    embedder._embed_fn = mock.Mock(wraps=embedder._embed_fn)
    # Call on various differently-sized batches.
    expected_call_shapes = []
    expected_batch_shape = (batch_size, self._domain.length)
    for num_seqs in [1, 3, 5, 10]:
      embedder(self._domain.sample_uniformly(num_seqs))
      expected_num_batches = int(np.ceil(num_seqs / batch_size))
      expected_call_shapes.extend([expected_batch_shape] * expected_num_batches)

    actual_call_shapes = [
        call[0][0].shape for call in embedder._embed_fn.call_args_list
    ]
    self.assertAllEqual(actual_call_shapes, expected_call_shapes)


if __name__ == '__main__':
  absltest.main()
