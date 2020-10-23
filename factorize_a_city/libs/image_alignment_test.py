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

"""Tests for factorize_city.libs.image_alignment."""


import numpy as np
import tensorflow as tf

from factorize_a_city.libs import image_alignment


class BSplineWarpTest(tf.test.TestCase):

  def test_spline_pano_spline_roll(self):
    # random image
    random_state = np.random.RandomState(seed=0)
    warp_input = random_state.uniform(size=[1, 40, 40, 3])

    with tf.name_scope("control_point_warp"):
      # Testing an x-shift of 0.1 which is 10% of the image.
      # This is equivalent to a horizontal roll when panorama is True.

      tf_warp_input = tf.constant(warp_input, dtype=tf.float32)

      x_control_points = tf.ones([1, 10, 10, 1]) * 0.1
      y_control_points = tf.zeros([1, 10, 10, 1])

      control_points = tf.concat([y_control_points, x_control_points], axis=-1)
      warped = image_alignment.bspline_warp(
          control_points, tf_warp_input, 2, pano_pad=True)
      roll10_tf_warp_input = tf.roll(tf_warp_input, -4, axis=2)
    with self.session():
      self.assertAllClose(roll10_tf_warp_input.eval(), warped.eval())


if __name__ == "__main__":
  tf.test.main()
