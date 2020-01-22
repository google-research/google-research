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
"""Tests for sinkhorn module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from soft_sort import sinkhorn


class SinkhornTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(SinkhornTest, self).setUp()
    tf.random.set_seed(0)
    np.random.seed(seed=0)

    self.x = tf.constant([[0.5, 0.2, -0.1, 0.4, 0.1, 0.3, -0.2, 0.0]])
    self.y = tf.constant([[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]])
    self.a = 1.0 / self.x.shape[1] * tf.ones(self.x.shape)
    self.b = 1.0 / self.y.shape[1] * tf.ones(self.y.shape)

  def test_routine(self):
    sinkhorn1d = sinkhorn.Sinkhorn1D()
    p = sinkhorn1d(self.x, self.y, self.a, self.b)
    self.assertEqual(p.shape.as_list(), [1, 8, 8])

  def test_decay(self):
    sinkhorn_no_decay = sinkhorn.Sinkhorn1D(
        epsilon=1e-3, epsilon_0=1e-3, epsilon_decay=1.0, power=2.0,
        threshold=1e-3)
    sinkhorn_decay = sinkhorn.Sinkhorn1D(
        epsilon=1e-3, epsilon_0=1e-1, epsilon_decay=0.95, power=2.0,
        threshold=1e-3)

    sinkhorn_no_decay(self.x, self.y, self.a, self.b)
    sinkhorn_decay(self.x, self.y, self.a, self.b)
    self.assertLess(sinkhorn_decay.iterations, sinkhorn_no_decay.iterations)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
