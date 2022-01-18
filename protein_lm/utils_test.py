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
"""Tests for utils."""

import functools
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
from protein_lm import domains
from protein_lm import models
from protein_lm import utils

lm_cfg = dict(
    batch_size=1, num_layers=2, num_heads=2, emb_dim=32, mlp_dim=32, qkv_dim=32)
lm_cls = functools.partial(models.FlaxLM, **lm_cfg)


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_count_params(self):
    domain = domains.FixedLengthDiscreteDomain(length=4, vocab_size=2)
    lm = lm_cls(domain=domain)
    count = utils.param_count(lm)
    self.assertEqual(13059, count)

    # Check these methods run.
    utils.param_pprint(lm)
    sizes = utils.param_reduce(lm, log=True)
    self.assertIsInstance(sizes, dict)

  @parameterized.parameters((5, 5), (5, 1), (5, 2), (5, 6), (5, 12))
  def test_batch_apply(self, batch_size, num_inputs):
    def fn(inputs):
      return np.power(inputs + 1, 2)

    def batch_fn(batched_inputs):
      if len(batched_inputs) != batch_size:
        raise ValueError('fn() called with a batch that is '
                         'the wrong size (%d vs. %d).' % (len(batched_inputs),
                                                          batch_size))
      return fn(batched_inputs)
    inputs = np.stack([np.arange(num_inputs), -np.arange(num_inputs)], axis=1)
    unbatched_output = fn(inputs)
    batched_output = utils.batch_apply(batch_fn, inputs, batch_size)
    np.testing.assert_array_equal(unbatched_output, batched_output)

  def test_get_normalized_matrix(self):
    """Tests that the normalized matrix is computed correctly."""
    domain = domains.FixedLengthDiscreteDomain(
        vocab=domains.Vocabulary(tokens=['A', 'B', 'C']),
        length=2)
    freq_dict = {'A': {'A': 5, 'B': 3, 'C': 1},
                 'B': {'A': 3, 'B': 5, 'C': 1},
                 'C': {'A': 1, 'B': 1, 'C': 1}}
    matrix = utils.get_normalized_matrix(domain, freq_dict)
    expected_matrix = [[1, 0.5, 0], [0.5, 1, 0,], [0, 0, 0]]
    self.assertAllEqual(matrix, expected_matrix)

  def test_soft_accuracy(self):
    """Tests that soft accuracy is computed correctly."""
    domain = domains.FixedLengthDiscreteDomain(
        vocab=domains.Vocabulary(tokens=['A', 'B', 'C']),
        length=2)
    targets = np.array([[0, 1]])
    logits = np.log([[[0.9, 0.1], [0.6, 0.4]]])
    freq_dict = {'A': {'A': 5, 'B': 3, 'C': 1},
                 'B': {'A': 3, 'B': 5, 'C': 1},
                 'C': {'A': 1, 'B': 1, 'C': 1}}
    accuracy, denominator = utils.compute_weighted_soft_accuracy(
        logits, targets,
        weights=None,
        matrix=utils.get_normalized_matrix(domain, freq_dict))
    self.assertEqual(accuracy / denominator, 0.75)


if __name__ == '__main__':
  tf.test.main()
