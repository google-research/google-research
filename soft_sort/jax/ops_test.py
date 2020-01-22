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
"""Tests for the ops module in Jax."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as np
import jax.test_util

from soft_sort.jax import ops


class OpsTestCase(jax.test_util.JaxTestCase):
  """Tests for the ops module in jax."""

  def setUp(self):
    super(OpsTestCase, self).setUp()
    self.x = np.array(
        [[[7.9, 1.2, 5.5, 9.8, 3.5],
          [7.9, 12.2, 45.5, 9.8, 3.5],
          [17.9, 14.2, 55.5, 9.8, 3.5]]
        ])

  def test_sort(self):
    s = ops.softsort(self.x, axis=-1)
    self.assertEqual(s.shape, self.x.shape)
    deltas = np.diff(s, axis=-1) > 0
    self.assertAllClose(deltas, np.ones(deltas.shape, dtype=bool), True)

  def test_sort_descending(self):
    x = self.x[0][0]
    s = ops.softsort(x, axis=-1, direction='DESCENDING')
    self.assertEqual(s.shape, x.shape)
    deltas = np.diff(s, axis=-1) < 0
    self.assertAllClose(deltas, np.ones(deltas.shape, dtype=bool), True)

  def test_ranks(self):
    ranks = ops.softranks(self.x, axis=-1)
    self.assertEqual(ranks.shape, self.x.shape)
    true_ranks = np.argsort(np.argsort(self.x, axis=-1), axis=-1)
    self.assertAllClose(ranks, true_ranks, False, atol=1e-3)

  def test_ranks_one_based(self):
    ranks = ops.softranks(self.x, axis=-1, zero_based=False)
    self.assertEqual(ranks.shape, self.x.shape)
    true_ranks = np.argsort(np.argsort(self.x, axis=-1), axis=-1) + 1
    self.assertAllClose(ranks, true_ranks, False, atol=1e-3)

  def test_ranks_descending(self):
    ranks = ops.softranks(
        self.x, axis=-1, zero_based=True, direction='DESCENDING')
    self.assertEqual(ranks.shape, self.x.shape)

    max_rank = self.x.shape[-1] - 1
    true_ranks = max_rank - np.argsort(np.argsort(self.x, axis=-1), axis=-1)
    self.assertAllClose(ranks, true_ranks, False, atol=1e-3)

  @parameterized.named_parameters(
      ('medians_-1', 0.5, -1),
      ('medians_1', 0.5, 1),
      ('percentile25_-1', 0.25, -1))
  def test_softquantile(self, quantile, axis):
    x = np.array([
        [[7.9, 1.2, 5.5, 9.8, 3.5],
         [7.9, 12.2, 45.5, 9.8, 3.5],
         [17.9, 14.2, 55.5, 9.8, 3.5]],
        [[4.9, 1.2, 15.5, 4.8, 3.5],
         [7.9, 1.2, 5.5, 7.8, 2.5],
         [1.9, 4.2, 55.5, 9.8, 1.5]]
    ])
    qs = ops.softquantile(x, quantile, axis=axis)
    s = list(x.shape)
    s.pop(axis)
    self.assertTupleEqual(qs.shape, tuple(s))
    self.assertAllClose(
        qs, np.quantile(x, quantile, axis=axis), True, rtol=1e-2)


if __name__ == '__main__':
  absltest.main()
