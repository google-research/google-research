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

"""Tests for factorize_city.libs.pano_transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from factorize_a_city.libs import pano_transformer


class PanoTransformerTest(tf.test.TestCase):

  def test_pano_shift_transform(self):
    random_state = np.random.RandomState(seed=0)
    with tf.name_scope("test_shift_pano_forward_pass"), self.session():
      pano_stack = tf.constant(random_state.uniform(size=[1, 128, 256, 3]),
                               dtype=tf.float32)
      # This is 90 degree rotation about the z-axis with the convention of
      # z-axis facing down in the world coordinate system.
      yaw_rotation_radians = 0.5 * np.pi * tf.ones(shape=[1], dtype=tf.float32)
      rotated_panorama = pano_transformer.rotate_pano_horizontally(
          pano_stack, yaw_rotation_radians)
      # A 90 degree rotation is a 256 / 64=64 pixel shift of the
      # panorama. Shift works in the opposite direction to rotation
      # to tf.roll's pixel shifting behavior.
      rolled_pano = tf.roll(pano_stack, -64, axis=2)
      self.assertAllClose(rotated_panorama.eval(), rolled_pano.eval())

      # Assert the opposite rotation as well.
      yaw_rotation_radians = -0.5 * np.pi * tf.ones(shape=[1], dtype=tf.float32)
      rotated_panorama = pano_transformer.rotate_pano_horizontally(
          pano_stack, yaw_rotation_radians)
      rolled_pano = tf.roll(pano_stack, 64, axis=2)
      self.assertAllClose(rotated_panorama.eval(), rolled_pano.eval())

  def test_full_rotation(self):
    # Performing multiple smaller rotation that sum to a complete rotation
    # and check that it returns the input tensor.
    random_state = np.random.RandomState(seed=0)
    with tf.name_scope("test_shift_pano_forward_pass"), self.session():
      pano_stack = tf.constant(random_state.uniform(size=[1, 128, 256, 3]),
                               dtype=tf.float32)

      yaw_rotation_radians = 0.5 * np.pi * tf.ones(shape=[1], dtype=tf.float32)
      rot_pano_stack = pano_stack
      for unused_i in range(4):
        rot_pano_stack = pano_transformer.rotate_pano_horizontally(
            rot_pano_stack, yaw_rotation_radians)
      # Rotated 90 degrees four times is a complete rotation.
      self.assertAllClose(rot_pano_stack.eval(), pano_stack.eval())

  def test_rotation_and_inverse_rotation(self):
    # Performing a rotation and then its negative should cancel each other
    # and the result should be the same as the input.
    random_state = np.random.RandomState(seed=0)
    with tf.name_scope("test_shift_pano_forward_pass"), self.session():
      pano_stack = tf.constant(random_state.uniform(size=[1, 128, 256, 3]),
                               dtype=tf.float32)
      for test_rotation in np.arange(-np.pi, np.pi, 10):
        # Note that rotations which shift by a non-integer pixel amount is
        # non-invertible. Here we only test that the inverse rotation returns
        # an interpolatation that is close to the original input.
        yaw_rotation_radians = test_rotation * tf.ones(
            shape=[1], dtype=tf.float32)
        rot_pano_stack = pano_transformer.rotate_pano_horizontally(
            pano_stack, yaw_rotation_radians)
        inv_rot_pano_stack = pano_transformer.rotate_pano_horizontally(
            rot_pano_stack, -yaw_rotation_radians)
        self.assertAllClose(inv_rot_pano_stack.eval(),
                            pano_stack.eval(), rtol=1e-3, atol=1e-3)

  def test_differentiable_gradients(self):
    # Verify that there are gradients through rotate_pano_horizontally.
    with tf.name_scope("test_shift_pano_forward_pass"), self.session():
      pano_stack = tf.get_variable("image", shape=[1, 128, 256, 3],
                                   trainable=True)
      learned_rotation = tf.get_variable("learned_rot", shape=[1],
                                         trainable=True)

      rot_pano_stack = pano_transformer.rotate_pano_horizontally(
          pano_stack, learned_rotation)

      loss = tf.reduce_mean(tf.abs(rot_pano_stack))

      computed_loss_gradients = tf.gradients(loss,
                                             [pano_stack, learned_rotation])
      # Verify we have two elements.
      self.assertAllEqual(len(computed_loss_gradients), 2)

      # Verify that none of the computed gradients is None (i.e. gradients
      # exist for all variables).
      self.assertTrue(np.all([v is not None for v in computed_loss_gradients]))


if __name__ == "__main__":
  tf.disable_eager_execution()
  tf.test.main()
