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

"""Tests for tf3d.utils.voxel_utils."""

import numpy as np
from six.moves import range
import tensorflow as tf

from tf3d.utils import voxel_utils


class VoxelUtilsTest(tf.test.TestCase):

  def get_sample_points(self):
    return tf.constant([[10.0, 12.0, 2.0],
                        [2.0, 10.0, 9.0],
                        [1.0, 11.0, 11.0],
                        [0.0, 1.0, 11.0],
                        [0.0, 0.0, 10.0],
                        [-1.0, 1.0, 11.0],
                        [11.0, 11.0, 1.0],
                        [11.0, 12.0, -1.0],
                        [0.0, 0.0, 11.0],
                        [0.01, 0.0, 11.0]], dtype=tf.float32)

  def test_crop_and_pad(self):
    voxels = tf.ones([100, 75, 50, 3], dtype=tf.float32)
    cropped_voxels_1 = voxel_utils.crop_and_pad_voxels(
        voxels=voxels,
        start_coordinates=[43, 40, 0, 0],
        end_coordinates=[58, 61, voxels.shape[2], voxels.shape[3]])
    cropped_voxels_2 = voxel_utils.crop_and_pad_voxels(
        voxels=voxels,
        start_coordinates=[-5, -5, 0, 0],
        end_coordinates=[16, 16, voxels.shape[2], voxels.shape[3]])
    cropped_voxels_3 = voxel_utils.crop_and_pad_voxels(
        voxels=voxels,
        start_coordinates=[84, 59, 0, 0],
        end_coordinates=[115, 90, voxels.shape[2], voxels.shape[3]])
    np_cropped_region_1 = cropped_voxels_1.numpy()
    np_cropped_region_2 = cropped_voxels_2.numpy()
    np_cropped_region_3 = cropped_voxels_3.numpy()
    self.assertAllEqual(np_cropped_region_1.shape, (15, 21, 50, 3))
    # Check that every value is a one
    self.assertEqual(np_cropped_region_1.mean(), 1)
    self.assertEqual(np_cropped_region_1.std(), 0)
    self.assertAllEqual(np_cropped_region_2.shape, (21, 21, 50, 3))
    # Check that the padded region is all zeros
    self.assertEqual(np_cropped_region_2[:5, :5, :, :].sum(), 0)
    # Check that for cropped regione very value is 1
    self.assertEqual(np_cropped_region_2[5:, 5:, :, :].mean(), 1)
    self.assertEqual(np_cropped_region_2[5:, 5:, :, :].std(), 0)
    self.assertAllEqual(np_cropped_region_3.shape, (31, 31, 50, 3))
    # Cropped region
    self.assertEqual(np_cropped_region_3[:16, :16, :, :].mean(), 1)
    # Padding region
    self.assertEqual(np_cropped_region_3[:16, :16, :, :].std(), 0)
    self.assertEqual(np_cropped_region_3[16:, 16:, :, :].sum(), 0)

  def test_pointcloud_to_voxel_grid_shapes(self):
    start_locations = [(-5, -5, -5),
                       (0, 0, 0),
                       (2.5, 2.5, 2.5)]
    end_locations = [(0, 0, 0),
                     (10, 10, 10),
                     (3, 3, 3)]
    grid_cell_sizes = [(0.5, 0.5, 0.5),
                       (0.1, 0.1, 0.1),
                       (0.5, 0.5, 0.5)]
    feature_dims = [3, 5, 10]

    expected_output_shapes = [(10, 10, 10, 3),
                              (100, 100, 100, 5),
                              (1, 1, 1, 10)]

    # For each test case we want to check if the output shape matches
    for test_case in range(3):
      points = tf.constant([[0.1, 0.1, 0.1]], tf.float32)
      features = tf.constant([list(range(feature_dims[test_case]))], tf.float32)

      voxel_grid, segment_ids, _ = voxel_utils.pointcloud_to_voxel_grid(
          points=points,
          features=features,
          grid_cell_size=grid_cell_sizes[test_case],
          start_location=start_locations[test_case],
          end_location=end_locations[test_case])

      self.assertEqual(voxel_grid.shape,
                       tuple(expected_output_shapes[test_case]))
      self.assertEqual(segment_ids.shape, (1,))

  def test_pointcloud_to_voxel_grid(self):
    points = self.get_sample_points()
    grid_cell_size = (20, 20, 20)
    start_location = (-20, -20, -20)
    end_location = (20, 20, 20)
    features = tf.constant([[10.0, 12.0, 2.0, 1.0],
                            [2.0, 10.0, 9.0, 0.0],
                            [1.0, 11.0, 11.0, 1.0],
                            [0.01, 1.01, 11.0, 0.0],
                            [0.01, 0.01, 10.0, 1.0],
                            [-1.0, 1.0, 11.0, 0.0],
                            [11.0, 11.0, 1.0, 1.0],
                            [11.0, 12.0, -1.0, 0.0],
                            [0.01, 0.01, 11.0, 1.0],
                            [0.01, 0.01, 11.0, 0.0]], dtype=tf.float32)
    voxel_features, _, _ = voxel_utils.pointcloud_to_voxel_grid(
        points=points,
        features=features,
        grid_cell_size=grid_cell_size,
        start_location=start_location,
        end_location=end_location)
    np_voxel_features = voxel_features.numpy()
    # [-20:0, -20:0, -20:0]
    self.assertAllClose(np_voxel_features[0, 0, 0, :], [0.0, 0.0, 0.0, 0.0])
    # [-20:0, -20:0, 0:20]
    self.assertAllClose(np_voxel_features[0, 0, 1, :], [0.0, 0.0, 0.0, 0.0])
    # [-20:0, 0:20, -20:0]
    self.assertAllClose(np_voxel_features[0, 1, 0, :], [0.0, 0.0, 0.0, 0.0])
    # [-20:0, 20:0, 0:20]
    self.assertAllClose(np_voxel_features[0, 1, 1, :], [-1.0, 1.0, 11.0, 0.0])
    # [0:20, -20:0, -20:0]
    self.assertAllClose(np_voxel_features[1, 0, 0, :], [0.0, 0.0, 0.0, 0.0])
    # [0:20, -20:0, 0:20]
    self.assertAllClose(np_voxel_features[1, 0, 1, :], [0.0, 0.0, 0.0, 0.0])
    # [0:20, 0:20, -20:0]
    self.assertAllClose(np_voxel_features[1, 1, 0, :], [11.0, 12.0, -1.0, 0.0])
    # [0:20, 20:0, 0:20]
    self.assertAllClose(np_voxel_features[1, 1, 1, :],
                        [24.04 / 8.0, 45.04 / 8.0, 66.0 / 8.0, 5.0 / 8.0])

  def test_pointcloud_to_voxel_grid_placement(self):
    points = tf.constant([[0.5, 0.5, 0.5],
                          [0.25, 0.25, 0.25],
                          [1.6, 1.6, 1.6],
                          [1.75, 1.75, 1.75],
                          [1.9, 1.9, 1.9],
                          [2.1, 2.1, 2.1],
                          [2.3, 2.35, 2.37]], dtype=tf.float32)
    features = tf.constant([[100, 110, 120],
                            [120, 130, 140],
                            [1, 2, 3],
                            [2, 3, 4],
                            [3, 4, 5],
                            [1000, 500, 250],
                            [200, 300, 150]], dtype=tf.float32)
    grid_cell_size = (1, 1, 1)
    start_location = (0, 0, 0)
    end_location = (10, 10, 10)
    voxel_features, segment_ids, _ = voxel_utils.pointcloud_to_voxel_grid(
        points=points,
        features=features,
        grid_cell_size=grid_cell_size,
        start_location=start_location,
        end_location=end_location)
    per_point_values = voxel_utils.voxels_to_points(voxel_features, segment_ids)
    np_voxel_features = voxel_features.numpy()
    np_segment_ids = segment_ids.numpy()
    np_per_point_values = per_point_values.numpy()
    # Check voxel grid values
    self.assertAllClose(np_voxel_features[0, 0, 0, :], [110, 120, 130])
    self.assertAllClose(np_voxel_features[1, 1, 1, :], [2, 3, 4])
    self.assertAllClose(np_voxel_features[2, 2, 2, :], [600, 400, 200])
    # Check values after mapping back to points
    self.assertAllClose(np_per_point_values[0, :], (110.0, 120.0, 130.0))
    self.assertAllClose(np_per_point_values[1, :], (110.0, 120.0, 130.0))
    self.assertAllClose(np_per_point_values[2, :], (2.0, 3.0, 4.0))
    self.assertAllClose(np_per_point_values[3, :], (2.0, 3.0, 4.0))
    self.assertAllClose(np_per_point_values[4, :], (2.0, 3.0, 4.0))
    self.assertAllClose(np_per_point_values[5, :], (600.0, 400.0, 200.0))
    self.assertAllClose(np_per_point_values[6, :], (600.0, 400.0, 200.0))
    # Check segment ids match what they should
    # Locations: [0, 0, 0] == 0, [1, 1, 1] == 111, [2, 2, 2] == 222
    self.assertAllEqual([0, 0, 111, 111, 111, 222, 222], np_segment_ids)

  def test_points_offset_in_voxels(self):
    points = tf.constant([[[0.5, 0.5, 0.5],
                           [0.25, 0.25, 0.25],
                           [1.6, 1.6, 1.6],
                           [1.75, 1.75, 1.75],
                           [1.9, 1.9, 1.9],
                           [2.1, 2.1, 2.1],
                           [2.3, 2.35, 2.37]]], dtype=tf.float32)
    point_offsets = voxel_utils.points_offset_in_voxels(
        points, grid_cell_size=(0.1, 0.1, 0.1))
    expected_points = np.array(
        [[[0.0, 0.0, 0.0],
          [-0.5, -0.5, -0.5],
          [0.0, 0.0, 0.0],
          [-0.5, -0.5, -0.5],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.5, -0.3]]], dtype=np.float32)
    self.assertAllClose(point_offsets.numpy(), expected_points, atol=1e-3)

  def test_pointcloud_to_sparse_voxel_grid_unbatched(self):
    points = tf.constant([[0.5, 0.5, 0.5],
                          [0.25, 0.25, 0.25],
                          [1.6, 1.6, 1.6],
                          [1.75, 1.75, 1.75],
                          [1.9, 1.9, 1.9],
                          [2.1, 2.1, 2.1],
                          [2.3, 2.35, 2.37]], dtype=tf.float32)
    features = tf.constant([[100, 110, 120],
                            [120, 130, 140],
                            [1, 2, 3],
                            [2, 3, 4],
                            [3, 4, 5],
                            [1000, 500, 250],
                            [200, 300, 150]], dtype=tf.float32)
    grid_cell_size = (0.5, 0.5, 0.5)
    (voxel_features_max, voxel_indices_max, segment_ids_max,
     voxel_start_location_max
    ) = voxel_utils.pointcloud_to_sparse_voxel_grid_unbatched(
        points=points,
        features=features,
        grid_cell_size=grid_cell_size,
        segment_func=tf.math.unsorted_segment_max)
    (voxel_features_mean, voxel_indices_mean, segment_ids_mean,
     voxel_start_location_mean
    ) = voxel_utils.pointcloud_to_sparse_voxel_grid_unbatched(
        points=points,
        features=features,
        grid_cell_size=grid_cell_size,
        segment_func=tf.math.unsorted_segment_mean)
    self.assertAllClose(voxel_features_max.numpy(),
                        np.array([[120., 130., 140.],
                                  [1., 2., 3.],
                                  [1000., 500., 250.],
                                  [200., 300., 150.]]))
    self.assertAllClose(voxel_features_mean.numpy(),
                        np.array([[110., 120., 130.],
                                  [1., 2., 3.],
                                  [335., 169., 259.0 / 3.0],
                                  [200., 300., 150.]]))
    self.assertAllEqual(voxel_indices_max.numpy(), np.array([[0, 0, 0],
                                                             [2, 2, 2],
                                                             [3, 3, 3],
                                                             [4, 4, 4]]))
    self.assertAllEqual(segment_ids_max.numpy(),
                        np.array([0, 0, 1, 2, 2, 2, 3]))
    self.assertAllEqual(voxel_indices_mean.numpy(), voxel_indices_max.numpy())
    self.assertAllEqual(segment_ids_mean.numpy(), segment_ids_max.numpy())
    self.assertAllClose(voxel_start_location_mean.numpy(),
                        np.array([0.25, 0.25, 0.25]))
    self.assertAllClose(voxel_start_location_max.numpy(),
                        np.array([0.25, 0.25, 0.25]))

  def test_pointcloud_to_sparse_voxel_grid(self):
    points = tf.constant([[[0.5, 0.5, 0.5],
                           [0.25, 0.25, 0.25],
                           [1.6, 1.6, 1.6],
                           [1.75, 1.75, 1.75],
                           [1.9, 1.9, 1.9],
                           [2.1, 2.1, 2.1],
                           [2.3, 2.35, 2.37],
                           [0.0, 0.0, 0.0]]], dtype=tf.float32)
    features = tf.constant([[[100, 110, 120],
                             [120, 130, 140],
                             [1, 2, 3],
                             [2, 3, 4],
                             [3, 4, 5],
                             [1000, 500, 250],
                             [200, 300, 150],
                             [0, 0, 0]]], dtype=tf.float32)
    num_valid_points = tf.constant([7], dtype=tf.int32)
    grid_cell_size = (0.5, 0.5, 0.5)
    (voxel_features, voxel_indices, num_valid_voxels, segment_ids,
     voxel_start_locations) = voxel_utils.pointcloud_to_sparse_voxel_grid(
         points=points,
         features=features,
         num_valid_points=num_valid_points,
         grid_cell_size=grid_cell_size,
         voxels_pad_or_clip_size=5,
         segment_func=tf.math.unsorted_segment_max)
    self.assertAllClose(voxel_features.numpy(), np.array([[[120., 130., 140.],
                                                           [1., 2., 3.],
                                                           [1000., 500., 250.],
                                                           [200., 300., 150.],
                                                           [0.0, 0.0, 0.0]]]))
    self.assertAllEqual(voxel_indices.numpy(), np.array([[[0, 0, 0],
                                                          [2, 2, 2],
                                                          [3, 3, 3],
                                                          [4, 4, 4],
                                                          [0, 0, 0]]]))
    self.assertAllEqual(segment_ids.numpy(),
                        np.array([[0, 0, 1, 2, 2, 2, 3, 0]]))
    self.assertAllEqual(num_valid_voxels.numpy(), np.array([4]))
    self.assertAllClose(voxel_start_locations.numpy(),
                        np.array([[0.25, 0.25, 0.25]]))

  def test_sparse_voxel_grid_to_pointcloud(self):
    voxel_features_0 = tf.constant([[0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], dtype=tf.float32)
    voxel_features_1 = tf.constant([[0.0, 0.0, 0.5],
                                    [0.0, 0.5, 0.0],
                                    [0.5, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.5, 0.5, 0.5]], dtype=tf.float32)
    voxel_features = tf.stack([voxel_features_0, voxel_features_1], axis=0)
    segment_ids = tf.constant([[0, 0, 1, 1, 2, 2, 0, 0, 0, 0],
                               [1, 3, 1, 2, 0, 4, 4, 0, 0, 0]], dtype=tf.int32)
    num_valid_voxels = tf.constant([3, 5], dtype=tf.int32)
    num_valid_points = tf.constant([7, 9], dtype=tf.int32)
    point_features = voxel_utils.sparse_voxel_grid_to_pointcloud(
        voxel_features=voxel_features,
        segment_ids=segment_ids,
        num_valid_voxels=num_valid_voxels,
        num_valid_points=num_valid_points)
    np_point_features = point_features.numpy()
    self.assertAllEqual(np_point_features.shape, [2, 10, 3])
    self.assertAllClose(np_point_features[0], np.array([[0.0, 0.0, 1.0],
                                                        [0.0, 0.0, 1.0],
                                                        [0.0, 1.0, 0.0],
                                                        [0.0, 1.0, 0.0],
                                                        [1.0, 0.0, 0.0],
                                                        [1.0, 0.0, 0.0],
                                                        [0.0, 0.0, 1.0],
                                                        [0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 0.0],
                                                        [0.0, 0.0, 0.0]]))
    self.assertAllClose(np_point_features[1], np.array([[0.0, 0.5, 0.0],
                                                        [0.0, 0.0, 0.0],
                                                        [0.0, 0.5, 0.0],
                                                        [0.5, 0.0, 0.0],
                                                        [0.0, 0.0, 0.5],
                                                        [0.5, 0.5, 0.5],
                                                        [0.5, 0.5, 0.5],
                                                        [0.0, 0.0, 0.5],
                                                        [0.0, 0.0, 0.5],
                                                        [0.0, 0.0, 0.0]]))

  def test_per_voxel_point_sample_segment_func(self):
    data = tf.constant(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
         [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=tf.float32)
    segment_ids = tf.constant([0, 3, 1, 0, 3, 0, 0], dtype=tf.int32)
    num_segments = 4
    num_samples_per_voxel = 2
    voxel_features = voxel_utils.per_voxel_point_sample_segment_func(
        data=data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        num_samples_per_voxel=num_samples_per_voxel)
    expected_voxel_features = tf.constant([[[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
                                           [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                                           [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                           [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    self.assertAllEqual(voxel_features.shape, np.array([4, 2, 3]))
    self.assertAllClose(voxel_features.numpy(), expected_voxel_features.numpy())

  def test_compute_pointcloud_weights_based_on_voxel_density(self):
    points = tf.constant([[-1.0, -1.0, -1.0],
                          [-1.1, -1.1, -1.1],
                          [5.0, 5.0, 5.0],
                          [5.1, 5.1, 5.1],
                          [5.2, 5.2, 5.2],
                          [10.0, 10.0, 10.0],
                          [15.0, 15.0, 15.0]], dtype=tf.float32)
    point_weights = (
        voxel_utils.compute_pointcloud_weights_based_on_voxel_density(
            points=points, grid_cell_size=(4.0, 4.0, 4.0)))
    self.assertAllClose(
        point_weights.numpy(),
        np.array([[0.875], [0.875], [0.5833334], [0.5833334], [0.5833334],
                  [1.75], [1.75]],
                 dtype=np.float32))


if __name__ == '__main__':
  tf.test.main()
