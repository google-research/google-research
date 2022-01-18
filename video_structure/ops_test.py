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

"""Tests for video_structure.ops."""

from absl.testing import absltest
import numpy as np
import tensorflow.compat.v1 as tf
from video_structure import ops


class OpsTest(tf.test.TestCase):

  def testAddCoordChannel(self):
    batch_size, height, width, channels = 2, 32, 32, 3
    image = tf.zeros((batch_size, height, width, channels))
    image_with_coords = ops.add_coord_channels(image)
    self.assertEqual(
        image_with_coords.shape.as_list(),
        [batch_size, height, width, channels + 2])


class MapsToKeypointsTest(tf.test.TestCase):

  def setUp(self):
    super(MapsToKeypointsTest, self).setUp()
    self.map_shape = 1, 33, 33, 1  # [batch_size, H, W, num_keypoints]

  def compute_coords(self, test_map):
    map_tensor = tf.convert_to_tensor(test_map, tf.float32)
    keypoints_op = tf.squeeze(ops.maps_to_keypoints(map_tensor))
    with self.session() as sess:
      return sess.run(keypoints_op)

  def testZeroMapIsZeroCoords(self):
    """Tests that an all-zero map defaults to zero (centered) coordinates."""
    test_map = np.zeros(self.map_shape)
    np.testing.assert_array_almost_equal(
        self.compute_coords(test_map), [0.0, 0.0, 0.0], decimal=2)

  def testObjectInTopLeft(self):
    test_map = np.zeros(self.map_shape)
    test_map[0, 0, 0, 0] = 1.0  # Set one pixel to 1 to simulate object.
    np.testing.assert_array_almost_equal(
        self.compute_coords(test_map), [-1.0, 1.0, 1.0], decimal=2)

  def testObjectInBottomRight(self):
    test_map = np.zeros(self.map_shape)
    test_map[0, -1, -1, 0] = 1.0  # Set one pixel to 1 to simulate object.
    np.testing.assert_array_almost_equal(
        self.compute_coords(test_map), [1.0, -1.0, 1.0], decimal=2)

  def testObjectInCenter(self):
    test_map = np.zeros(self.map_shape)
    test_map[0, self.map_shape[1]//2, self.map_shape[2]//2, 0] = 1.0
    np.testing.assert_array_almost_equal(
        self.compute_coords(test_map), [0.0, 0.0, 1.0], decimal=2)


class KeypointsToMapsTest(tf.test.TestCase):

  def setUp(self):
    super(KeypointsToMapsTest, self).setUp()
    self.heatmap_width = 17

  def compute_map(self, test_coords):
    test_coords = np.array(test_coords, dtype=np.float32)
    test_coords = test_coords[None, None, :]
    maps_op = ops.keypoints_to_maps(
        test_coords, sigma=2, heatmap_width=self.heatmap_width)
    with self.session() as sess:
      return sess.run(tf.squeeze(maps_op))

  def testZeroScaleIsZeroMap(self):
    """Tests that if scale==0.0, the output map is all zeros."""
    np.testing.assert_array_equal(self.compute_map([0.0, 0.0, 0.0]), 0.0)

  def testObjectInTopLeft(self):
    test_map = self.compute_map([-1.0, 1.0, 1.0])
    arg_max = np.concatenate((test_map == np.max(test_map)).nonzero())
    np.testing.assert_array_equal(arg_max, [0, 0])

  def testObjectInBottomRight(self):
    test_map = self.compute_map([1.0, -1.0, 1.0])
    arg_max = np.concatenate((test_map == np.max(test_map)).nonzero())
    np.testing.assert_array_equal(
        arg_max, [self.heatmap_width-1, self.heatmap_width-1])

  def testObjectInCenter(self):
    test_map = self.compute_map([0.0, 0.0, 1.0])
    arg_max = np.concatenate((test_map == np.max(test_map)).nonzero())
    np.testing.assert_array_equal(
        arg_max, [self.heatmap_width//2, self.heatmap_width//2])

if __name__ == '__main__':
  absltest.main()
