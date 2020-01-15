# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
import jax.test_util

from soft_sort.jax import sinkhorn


class Sinkhorn1DTestCase(jax.test_util.JaxTestCase):
  """Test case for Sinkhorn1D."""

  def setUp(self):
    super(Sinkhorn1DTestCase, self).setUp()
    self.no_decay = sinkhorn.Sinkhorn1D(
        epsilon_0=1e-3, epsilon_decay=1.00, max_iterations=5000)
    self.decay = sinkhorn.Sinkhorn1D(
        epsilon_0=1e-1, epsilon_decay=0.95, max_iterations=5000)
    self.x = np.array([[0.2, -0.5, 0.4, 0.44, -0.9, 0.9]])
    self.y = 1 / self.x.shape[1] * np.cumsum(np.ones(self.x.shape), axis=-1)
    self.a = 1.0 / self.x.shape[1] * np.ones(self.x.shape)
    self.b = 1.0 / self.y.shape[1] * np.ones(self.y.shape)

  def test_call(self):
    """Tests that the __call__ methods returns transport maps."""
    p = self.decay(self.x, self.y, self.a, self.b)
    self.assertTupleEqual(p.shape, self.x.shape + (self.y.shape[1],))
    self.assertLessEqual(self.decay.iterations, self.decay.max_iterations)

  def test_decay(self):
    """Tests that applying the epsilon decay scheme speeds up convergence."""
    self.decay(self.x, self.y, self.a, self.b)
    self.no_decay(self.x, self.y, self.a, self.b)
    self.assertLess(self.decay.iterations, self.no_decay.iterations)
    self.assertLessEqual(self.no_decay.iterations, self.no_decay.max_iterations)


if __name__ == '__main__':
  absltest.main()
