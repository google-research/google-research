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
"""Tests for the soft_quantilizer module."""

from absl.testing import absltest
import jax.numpy as np
import jax.test_util

from soft_sort.jax import soft_quantilizer


class SoftQuantilizerTestCase(jax.test_util.JaxTestCase):
  """Test case for the SoftQuantilizer class."""

  def setUp(self):
    super(SoftQuantilizerTestCase, self).setUp()
    self.x = [
        [7.9, 1.2, 5.5, 9.8, 3.5],
        [7.9, 12.2, 45.5, 9.8, 3.5],
        [17.9, 14.2, 55.5, 9.8, 3.5]]

  def test_sort(self):
    q = soft_quantilizer.SoftQuantilizer(self.x, threshold=1e-3, epsilon=1e-3)
    deltas = np.diff(q.softsort, axis=-1) > 0
    self.assertAllClose(
        deltas, np.ones(deltas.shape, dtype=bool), check_dtypes=True)

  def test_target_weights(self):
    q = soft_quantilizer.SoftQuantilizer(
        self.x, target_weights=[0.49, 0.02, 0.49], threshold=1e-3, epsilon=1e-3)
    self.assertTupleEqual(q.softsort.shape, (3, 3))

  def test_targets(self):
    q = soft_quantilizer.SoftQuantilizer(
        self.x, y=[0.1, 0.2, 0.3], threshold=1e-3, epsilon=1e-3)
    self.assertTupleEqual(q.softsort.shape, (3, 3))

    q = soft_quantilizer.SoftQuantilizer(
        self.x, y=[[0.1, 0.2, 0.3], [0.5, 0.7, 0.9], [-0.3, -0.2, -0.1]],
        threshold=1e-3, epsilon=1e-3)
    self.assertTupleEqual(q.softsort.shape, (3, 3))

  def test_ranks(self):
    q = soft_quantilizer.SoftQuantilizer(self.x, threshold=1e-3, epsilon=1e-3)
    soft_ranks = q._n * q.softcdf
    true_ranks = np.argsort(np.argsort(q.x, axis=-1), axis=-1) + 1
    self.assertAllClose(soft_ranks, true_ranks, check_dtypes=False, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
