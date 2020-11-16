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

"""Tests for tf3d.utils.angle_utils."""

import math
import numpy as np
import tensorflow as tf
from tf3d.utils import angle_utils


class AngleUtilsTest(tf.test.TestCase):

  def test_absolute_angular_distance(self):
    pi = math.pi
    radians_a = tf.constant([-pi, -pi / 2, 0, pi / 2, pi, 3 * pi, pi])
    radians_b = tf.constant([0, pi / 2, 2 * pi, 0, 0, 0, 0.1])
    distance = angle_utils.absolute_angular_distance(radians_a, radians_b)
    actual = distance.numpy()
    self.assertTrue(np.all(actual >= 0.0))
    self.assertTrue(np.all(actual <= pi))
    expected = np.array([pi, pi, 0, pi / 2, pi, pi, pi - 0.1])
    self.assertAllClose(actual, expected)

  def test_signed_angular_distance(self):
    pi = math.pi
    radians_a = tf.constant([-pi, -pi / 2, 0, pi / 2, pi, 3 * pi, pi])
    radians_b = tf.constant([0, pi / 2, 2 * pi, 0, 0, 0, 0.1])
    distance = angle_utils.signed_angular_distance(radians_a, radians_b)
    actual = distance.numpy()
    self.assertTrue(np.all(actual >= -pi))
    self.assertTrue(np.all(actual < pi))
    expected = np.array([-pi, -pi, 0, pi / 2, -pi, -pi, pi - 0.1])
    self.assertAllClose(actual, expected)

  def test_wrap_to_pi(self):
    pi = math.pi
    radians = tf.constant([-3 * pi, -pi, -pi / 2, pi / 2, pi, 2 * pi, pi + 0.1])
    wrapped = angle_utils.wrap_to_pi(radians)
    actual = wrapped.numpy()
    self.assertTrue(np.all(actual >= -pi))
    self.assertTrue(np.all(actual < pi))
    expected = np.array([-pi, -pi, -pi / 2, pi / 2, -pi, 0.0, -pi + 0.1])
    self.assertAllClose(actual, expected)

  def test_degrees_to_radians(self):
    degrees = tf.random.uniform([20, 10], minval=-500.0, maxval=500.0)
    radians = angle_utils.degrees_to_radians(degrees)
    self.assertAllClose(radians.numpy(), np.deg2rad(degrees.numpy()))

  def test_radians_to_degrees(self):
    radians = tf.random.uniform([20, 10], minval=-10.0, maxval=10.0)
    degrees = angle_utils.radians_to_degrees(radians)
    self.assertAllClose(degrees.numpy(), np.rad2deg(radians.numpy()))


if __name__ == '__main__':
  tf.test.main()
