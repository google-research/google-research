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

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from soft_sort import sinkhorn


class SinkhornTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)
    np.random.seed(seed=0)

    self.x = tf.constant([[0.5, 0.2, -0.1, 0.4, 0.1, 0.3, -0.2, 0.0]])
    self.y = tf.constant([[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]])
    self.a = 1.0 / self.x.shape[1] * tf.ones(self.x.shape)
    self.b = 1.0 / self.y.shape[1] * tf.ones(self.y.shape)

  def test_routine(self):
    p = sinkhorn.sinkhorn(self.x, self.y, self.a, self.b)
    self.assertEqual(p.shape.as_list(), [1, 8, 8])

  def test_decay(self):
    result_nodecay = sinkhorn.sinkhorn_iterations(
        self.x, self.y, self.a, self.b, epsilon=1e-3, epsilon_0=1e-3,
        epsilon_decay=1.0, threshold=1e-3)

    result_decay = sinkhorn.sinkhorn_iterations(
        self.x, self.y, self.a, self.b, epsilon=1e-3, epsilon_0=1e-1,
        epsilon_decay=0.95, threshold=1e-3)
    self.assertLess(result_decay[-1], result_nodecay[-1])

  def test_sinkhorn_divergence(self):
    x = tf.convert_to_tensor([[[0, 0], [1, 1]], [[1, 1], [0, 0]]],
                             dtype=tf.float32)
    y = tf.convert_to_tensor([[[0, 1], [0, 1]], [[1, 0], [0, 1]]],
                             dtype=tf.float32)
    a = tf.convert_to_tensor([[.3, .7], [.7, .3]], dtype=tf.float32)
    b = tf.convert_to_tensor([[.4, .6], [.6, .4]], dtype=tf.float32)
    divergences = sinkhorn.sinkhorn_divergence(x, y, a, b, power=2)
    self.assertAllClose(divergences[0], divergences[1], rtol=1e-03, atol=1e-03)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
