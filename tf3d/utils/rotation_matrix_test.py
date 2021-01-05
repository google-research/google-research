# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for tf3d.utils.rotation_matrix."""

import numpy as np
import tensorflow as tf
from tf3d.utils import rotation_matrix


class RotationMatrixTest(tf.test.TestCase):

  def test_rotation_around_axis_with_scalar_input(self):
    for rad in [0.0, 0.5 * np.pi, -0.5 * np.pi, np.pi, -np.pi]:
      angle = tf.constant(rad)
      x = rotation_matrix.from_rotation_around_x(angle)
      y = rotation_matrix.from_rotation_around_y(angle)
      z = rotation_matrix.from_rotation_around_z(angle)

      self.assertEqual(x.shape, (3, 3))
      self.assertEqual(y.shape, (3, 3))
      self.assertEqual(z.shape, (3, 3))

      x = tf.matmul(x, rotation_matrix.from_rotation_around_x(-angle))
      y = tf.matmul(y, rotation_matrix.from_rotation_around_y(-angle))
      z = tf.matmul(z, rotation_matrix.from_rotation_around_z(-angle))

      self.assertAllClose(x.numpy(), np.eye(3))
      self.assertAllClose(y.numpy(), np.eye(3))
      self.assertAllClose(z.numpy(), np.eye(3))

  def test_rotation_around_axis_with_vector_input(self):
    angle = tf.constant([0.0, 0.5 * np.pi, -0.5 * np.pi, np.pi, -np.pi])
    x = rotation_matrix.from_rotation_around_x(angle)
    y = rotation_matrix.from_rotation_around_y(angle)
    z = rotation_matrix.from_rotation_around_z(angle)

    self.assertEqual(x.shape, (5, 3, 3))
    self.assertEqual(y.shape, (5, 3, 3))
    self.assertEqual(z.shape, (5, 3, 3))

    x = tf.matmul(x, rotation_matrix.from_rotation_around_x(-angle))
    y = tf.matmul(y, rotation_matrix.from_rotation_around_y(-angle))
    z = tf.matmul(z, rotation_matrix.from_rotation_around_z(-angle))

    expected = np.repeat(np.expand_dims(np.eye(3), axis=0), 5, axis=0)
    self.assertAllClose(x.numpy(), expected)
    self.assertAllClose(y.numpy(), expected)
    self.assertAllClose(z.numpy(), expected)

  def test_rotation_around_axis_with_tensor_input(self):
    angle = tf.random.uniform(shape=(5, 1))
    x = rotation_matrix.from_rotation_around_x(angle)
    y = rotation_matrix.from_rotation_around_y(angle)
    z = rotation_matrix.from_rotation_around_z(angle)

    self.assertEqual(x.shape, (5, 3, 3))
    self.assertEqual(y.shape, (5, 3, 3))
    self.assertEqual(z.shape, (5, 3, 3))

    x = tf.matmul(x, rotation_matrix.from_rotation_around_x(-angle))
    y = tf.matmul(y, rotation_matrix.from_rotation_around_y(-angle))
    z = tf.matmul(z, rotation_matrix.from_rotation_around_z(-angle))

    expected = np.repeat(np.expand_dims(np.eye(3), axis=0), 5, axis=0)
    self.assertAllClose(x.numpy(), expected)
    self.assertAllClose(y.numpy(), expected)
    self.assertAllClose(z.numpy(), expected)

  def test_rotation_around_y_cos_sin(self):
    angle = tf.random.uniform(shape=(5, 1))
    y = rotation_matrix.from_rotation_around_y_cos_sin(
        cos=tf.cos(angle), sin=tf.sin(angle))
    expected_y = rotation_matrix.from_rotation_around_y(angle)
    self.assertAllClose(y.numpy(), expected_y.numpy())

  def test_from_euler(self):
    angles = tf.constant([[np.pi, 0.0, 0.0],
                          [0.0, np.pi, 0.0],
                          [0.0, 0.0, np.pi],
                          [np.pi, np.pi, np.pi],
                          [0.0, 0.0, 0.0]])
    rotations = rotation_matrix.from_euler(angles)

    x = rotation_matrix.from_rotation_around_x(tf.constant(np.pi))
    y = rotation_matrix.from_rotation_around_y(tf.constant(np.pi))
    z = rotation_matrix.from_rotation_around_z(tf.constant(np.pi))

    self.assertEqual(rotations.shape, (5, 3, 3))

    np_rotations = rotations.numpy()
    self.assertAllClose(np_rotations[0, Ellipsis], x.numpy())
    self.assertAllClose(np_rotations[1, Ellipsis], y.numpy())
    self.assertAllClose(np_rotations[2, Ellipsis], z.numpy())
    self.assertAllClose(np_rotations[3, Ellipsis],
                        z.numpy().dot(y.numpy()).dot(x.numpy()))
    self.assertAllClose(np_rotations[4, Ellipsis], np.eye(3))


if __name__ == '__main__':
  tf.test.main()
