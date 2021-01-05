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

"""Tests for tf3d.utils.pointcloud_utils."""

import numpy as np
import tensorflow as tf
from tf3d.utils import pointcloud_utils


class PointcloudUtilsTest(tf.test.TestCase):

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

  def test_flip_normals_towards_viewpoint(self):
    points = tf.constant([[1.0, 1.0, 1.0],
                          [2.0, 2.0, 1.0],
                          [1.0, 2.0, 1.0],
                          [2.0, 1.0, 1.0]], dtype=tf.float32)
    normals = tf.constant([[0.1, 0.2, -1.0],
                           [0.1, 0.2, -1.0],
                           [0.0, 0.0, -1.0],
                           [0.1, -0.2, -1.0]], dtype=tf.float32)
    viewpoint = tf.constant([1.0, 1.0, 100.0])
    flipped_normals = pointcloud_utils.flip_normals_towards_viewpoint(
        points=points, normals=normals, viewpoint=viewpoint)
    self.assertAllClose(flipped_normals.numpy(),
                        np.array([[-0.1, -0.2, 1.0],
                                  [-0.1, -0.2, 1.0],
                                  [0.0, 0.0, 1.0],
                                  [-0.1, 0.2, 1.0]]))

  # def test_points_to_normals_unbatched_pca(self):
  #   points = tf.constant([[1.0, 1.0, 1.0],
  #                         [2.0, 2.0, 1.0],
  #                         [1.0, 2.0, 1.0],
  #                         [2.0, 1.0, 1.0]], dtype=tf.float32)
  #   normals = pointcloud_utils.points_to_normals_unbatched(
  #       points=points,
  #       k=4,
  #       distance_upper_bound=5.0,
  #       viewpoint=tf.constant([1.0, 1.0, 100.0]),
  #       method='pca')
  #   self.assertAllClose(normals.numpy(),
  #                       np.array([[0.0, 0.0, 1.0],
  #                                 [0.0, 0.0, 1.0],
  #                                 [0.0, 0.0, 1.0],
  #                                 [0.0, 0.0, 1.0]]),
  #                       atol=0.001)

#   def test_points_to_normals_unbatched_cross(self):
#     points = tf.constant([[1.0, 1.0, 1.0],
#                           [2.0, 2.0, 1.0],
#                           [1.0, 2.0, 1.0],
#                           [2.0, 1.0, 1.0]], dtype=tf.float32)
#     normals = pointcloud_utils.points_to_normals_unbatched(
#         points=points,
#         k=3,
#         distance_upper_bound=5.0,
#         viewpoint=tf.constant([1.0, 1.0, 100.0]),
#         method='cross')
#     normals_zero = pointcloud_utils.points_to_normals_unbatched(
#         points=points, k=3, distance_upper_bound=0.5, method='cross')
#     self.assertAllClose(normals.numpy(),
#                         np.array([[0.0, 0.0, 1.0],
#                                   [0.0, 0.0, 1.0],
#                                   [0.0, 0.0, 1.0],
#                                   [0.0, 0.0, 1.0]]),
#                         atol=0.001)
#     self.assertAllClose(normals_zero.numpy(),
#                         np.array([[0.0, 0.0, 0.0],
#                                   [0.0, 0.0, 0.0],
#                                   [0.0, 0.0, 0.0],
#                                   [0.0, 0.0, 0.0]]),
#                         atol=0.001)

#   def test_points_to_normals(self):
#     points = tf.constant([[[1.0, 1.0, 1.0],
#                            [2.0, 2.0, 1.0],
#                            [1.0, 2.0, 1.0],
#                            [2.0, 1.0, 1.0]]], dtype=tf.float32)
#     normals = pointcloud_utils.points_to_normals(
#         points=points,
#         num_valid_points=tf.constant([4], dtype=tf.int32),
#         k=4,
#         distance_upper_bound=5.0,
#         viewpoints=tf.constant([[1.0, 1.0, 100.0]]),
#         method='pca')
#     self.assertAllClose(normals.numpy(),
#                         np.array([[[0.0, 0.0, 1.0],
#                                    [0.0, 0.0, 1.0],
#                                    [0.0, 0.0, 1.0],
#                                    [0.0, 0.0, 1.0]]]),
#                         atol=0.001)

  def test_np_knn_graph_from_points_unbatched(self):
    points = np.array([[1.0, 1.0, 1.0],
                       [1.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 1.0, 1.0],
                       [0.0, 0.0, 0.0]])
    mask = np.array([1, 1, 0, 0, 1, 1, 0, 1], dtype=np.bool)
    distances, indices = pointcloud_utils.np_knn_graph_from_points_unbatched(
        points=points, k=3, distance_upper_bound=1.1, mask=mask)
    expected_distances = np.array([[0., 1., 1.],
                                   [0., 1., 1.],
                                   [0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 1., 1.],
                                   [0., 1., 0.],
                                   [0., 0., 0.],
                                   [0., 1., 0.]])
    expected_indices = np.array([[0, 5, 1],
                                 [1, 4, 0],
                                 [2, 2, 2],
                                 [3, 3, 3],
                                 [4, 7, 1],
                                 [5, 0, 5],
                                 [6, 6, 6],
                                 [7, 4, 7]])
    self.assertAllClose(distances, expected_distances)
    self.assertAllEqual(indices, expected_indices)

  def test_knn_graph_from_points_unbatched(self):
    points = self.get_sample_points()
    distances, indices = pointcloud_utils.knn_graph_from_points_unbatched(
        points, k=2, distance_upper_bound=2.0)
    expected_distances = tf.constant([[0, 1.73205081],
                                      [0, 0],
                                      [0, 0],
                                      [0, 1],
                                      [0, 1],
                                      [0, 1],
                                      [0, 1.73205081],
                                      [0, 0],
                                      [0, 0.01],
                                      [0, 0.01]], dtype=tf.float32)
    expected_indices = tf.constant([[0, 6],
                                    [1, 1],
                                    [2, 2],
                                    [3, 5],
                                    [4, 8],
                                    [5, 3],
                                    [6, 0],
                                    [7, 7],
                                    [8, 9],
                                    [9, 8]], dtype=tf.int32)
    self.assertAllClose(expected_distances.numpy(), distances.numpy())
    self.assertAllEqual(expected_indices.numpy(), indices.numpy())

#   def test_knn_graph_from_points_unbatched_less_than_k_dynamic_shape(self):
#     points = self.get_sample_points()
#     distances, indices = pointcloud_utils.knn_graph_from_points_unbatched(
#         points, k=20, distance_upper_bound=1000.0)
#     self.assertAllClose(distances.shape, [10, 20])
#     self.assertAllEqual(indices.shape, [10, 20])

#   def test_knn_graph_from_points_unbatched_dynamic_shape(self):
#     points = tf.random_uniform([500000, 3],
#                                minval=-10.0,
#                                maxval=10.0,
#                                dtype=tf.float32)
#     distances, indices = pointcloud_utils.knn_graph_from_points_unbatched(
#         points, k=5, distance_upper_bound=100.0)
#     self.assertAllEqual(distances.shape, [500000, 5])
#     self.assertAllEqual(indices.shape, [500000, 5])

#   def test_knn_graph_from_points_dynamic_shape(self):
#     points = tf.random_uniform([8, 500000, 3],
#                                minval=-10.0,
#                                maxval=10.0,
#                                dtype=tf.float32)
#     num_valid_points = tf.ones([8], dtype=tf.int32) * 500000
#     distances, indices = pointcloud_utils.knn_graph_from_points(
#         points,
#         num_valid_points=num_valid_points,
#         k=5,
#         distance_upper_bound=20.0)
#     self.assertAllEqual(distances.shape, [8, 500000, 5])
#     self.assertAllEqual(indices.shape, [8, 500000, 5])

#   def test_identity_knn_graph_unbatched(self):
#     points = self.get_sample_points()
#     distances, indices = pointcloud_utils.identity_knn_graph_unbatched(
#         points, 2)
#     expected_distances = tf.zeros([10, 2], dtype=tf.float32)
#     expected_indices = tf.constant([[0, 0],
#                                     [1, 1],
#                                     [2, 2],
#                                     [3, 3],
#                                     [4, 4],
#                                     [5, 5],
#                                     [6, 6],
#                                     [7, 7],
#                                     [8, 8],
#                                     [9, 9]], dtype=tf.int32)
#     self.assertAllClose(expected_distances.numpy(), distances.numpy())
#     self.assertAllEqual(expected_indices.numpy(), indices.numpy())

#   def test_identity_knn_graph_dynamic_shape(self):
#     points = tf.random_uniform([8, 500000, 3],
#                                minval=-10.0,
#                                maxval=10.0,
#                                dtype=tf.float32)
#     num_valid_points = tf.ones([8], dtype=tf.int32) * 500000
#     distances, indices = pointcloud_utils.identity_knn_graph(
#         points, num_valid_points=num_valid_points, k=5)
#     self.assertAllEqual(distances.shape, [8, 500000, 5])
#     self.assertAllEqual(indices.shape, [8, 500000, 5])

#   def test_identity_knn_graph(self):
#     points = self.get_sample_points()
#     points = tf.tile(tf.expand_dims(points, axis=0), [8, 1, 1])
#     num_valid_points = tf.ones([8], dtype=tf.int32) * 10
#     distances, indices = pointcloud_utils.identity_knn_graph(
#         points, num_valid_points=num_valid_points, k=2)
#     expected_distances = tf.zeros([8, 10, 2], dtype=tf.float32)
#     expected_indices = tf.constant([[0, 0],
#                                     [1, 1],
#                                     [2, 2],
#                                     [3, 3],
#                                     [4, 4],
#                                     [5, 5],
#                                     [6, 6],
#                                     [7, 7],
#                                     [8, 8],
#                                     [9, 9]], dtype=tf.int32)
#     expected_indices = tf.tile(
#         tf.expand_dims(expected_indices, axis=0), [8, 1, 1])
#     self.assertAllClose(expected_distances.numpy(), distances.numpy())
#     self.assertAllEqual(expected_indices.numpy(), indices.numpy())


if __name__ == '__main__':
  tf.test.main()
