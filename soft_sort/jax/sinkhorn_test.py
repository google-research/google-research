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
"""Tests for the sinkhorn module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import jax.numpy as np

from soft_sort.jax import sinkhorn


class SinkhornTestCase(absltest.TestCase):
  """Test case for Sinkhorn1D."""

  def setUp(self):
    super(SinkhornTestCase, self).setUp()
    self.no_decay = dict(
        epsilon_0=1e-3, epsilon_decay=1.00, max_iterations=5000)
    self.decay = dict(
        epsilon_0=1e-1, epsilon_decay=0.95, max_iterations=5000)
    self.x = np.array([[0.2, -0.5, 0.4, 0.44, -0.9, 0.9]])
    self.y = 1 / self.x.shape[1] * np.cumsum(np.ones(self.x.shape), axis=-1)
    self.a = 1.0 / self.x.shape[1] * np.ones(self.x.shape)
    self.b = 1.0 / self.y.shape[1] * np.ones(self.y.shape)

  def test_sinkhorn(self):
    """Tests that the __call__ methods returns transport maps."""
    f, g, eps, cost, _, iterations = sinkhorn.sinkhorn_iterations(
        self.x, self.y, self.a, self.b, **self.decay)
    p = sinkhorn.transport(cost, f, g, eps)
    self.assertTupleEqual(p.shape, self.x.shape + (self.y.shape[1],))
    self.assertLessEqual(iterations, self.decay['max_iterations'])

  def test_decay(self):
    """Tests that applying the epsilon decay scheme speeds up convergence."""
    decay_iterations = sinkhorn.sinkhorn_iterations(
        self.x, self.y, self.a, self.b, **self.decay)[-1]
    no_decay_iterations = sinkhorn.sinkhorn_iterations(
        self.x, self.y, self.a, self.b, **self.no_decay)[-1]
    self.assertLess(decay_iterations, no_decay_iterations)


if __name__ == '__main__':
  absltest.main()
