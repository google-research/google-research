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

"""Tests for tf3d.utils.box_utils."""

import numpy as np
import tensorflow as tf
from tf3d.utils import box_utils


class BoxUtilsTest(tf.test.TestCase):

  def test_point_features_to_box_features(self):
    # pyformat: disable
    points = tf.constant([[0.0, 0.0, 0.0],
                          [1.5, 0.0, 0.0],
                          [1.0, 1.5, 0.0],
                          [2.5, 2.5, 2.5],
                          [2.5, 2.5, 0.0]])
    point_features = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)
    box_lengths = tf.constant([[4.0], [2.0]])
    box_heights = tf.constant([[1.0], [2.0]])
    box_widths = tf.constant([[2.0], [2.0]])
    box_rotations = tf.constant([[[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]],
                                 [[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]]])
    box_translations = tf.constant([[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]])
    # pyformat: enable

    box_features = box_utils.point_features_to_box_features(
        points=points,
        point_features=point_features,
        num_bins_per_axis=2,
        boxes_length=box_lengths,
        boxes_height=box_heights,
        boxes_width=box_widths,
        boxes_rotation_matrix=box_rotations,
        boxes_center=box_translations)
    expected_box_features1 = tf.constant(
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0.5, 0.5, 0., 0., 0.]],
        dtype=tf.float32)
    expected_box_features2 = tf.constant(
        [[1., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.]],
        dtype=tf.float32)
    expected_box_features = tf.reshape(
        tf.stack([expected_box_features1, expected_box_features2], axis=0),
        [2, -1])
    self.assertAllClose(box_features.numpy(), expected_box_features.numpy())

  def test_map_points_to_boxes_tf(self):
    # pyformat: disable
    points = tf.constant([[0.0, 0.0, 0.0],
                          [1.5, 0.0, 0.0],
                          [1.0, 1.5, 0.0],
                          [2.5, 2.5, 2.5],
                          [2.5, 2.5, 0.0]])
    box_lengths = tf.constant([[4.0], [1.0]])
    box_heights = tf.constant([[1.0], [1.0]])
    box_widths = tf.constant([[2.0], [1.0]])
    box_rotations = tf.constant([[[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]],
                                 [[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]]])
    box_translations = tf.constant([[0.0, 0.0, 0.0],
                                    [2.5, 2.5, 2.5]])
    # pyformat: enable
    box_indices = box_utils.map_points_to_boxes(
        points=points,
        boxes_length=box_lengths,
        boxes_height=box_heights,
        boxes_width=box_widths,
        boxes_rotation_matrix=box_rotations,
        boxes_center=box_translations,
        box_margin=0.0)
    self.assertAllEqual(box_indices.numpy(), np.array([0, 0, -1, 1, -1]))

  def test_get_box_corners_3d(self):
    # pyformat: disable
    box_lengths = tf.constant([[4.0], [1.0]])
    box_heights = tf.constant([[1.0], [1.0]])
    box_widths = tf.constant([[2.0], [1.0]])
    box_rotations = tf.constant([[[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]],
                                 [[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]]])
    box_translations = tf.constant([[0.0, 0.0, 0.0],
                                    [2.5, 2.5, 2.5]])
    # pyformat: enable
    box_corners = box_utils.get_box_corners_3d(
        boxes_length=box_lengths,
        boxes_height=box_heights,
        boxes_width=box_widths,
        boxes_rotation_matrix=box_rotations,
        boxes_center=box_translations)
    self.assertEqual(box_corners.shape, (2, 8, 3))
    expected_box_corners1 = np.array(
        [[2.0, 1.0, 0.5],
         [-2.0, 1.0, 0.5],
         [-2.0, -1.0, 0.5],
         [2.0, -1.0, 0.5],
         [2.0, 1.0, -0.5],
         [-2.0, 1.0, -0.5],
         [-2.0, -1.0, -0.5],
         [2.0, -1.0, -0.5]], np.float32)
    expected_box_corners2 = np.array(
        [[3.0, 3.0, 3.0],
         [2.0, 3.0, 3.0],
         [2.0, 2.0, 3.0],
         [3.0, 2.0, 3.0],
         [3.0, 3.0, 2.0],
         [2.0, 3.0, 2.0],
         [2.0, 2.0, 2.0],
         [3.0, 2.0, 2.0]], np.float32)
    self.assertAllClose(box_corners.numpy()[0], expected_box_corners1)
    self.assertAllClose(box_corners.numpy()[1], expected_box_corners2)

  def test_get_box_as_dotted_lines(self):
    corners = tf.repeat(
        tf.expand_dims(
            tf.expand_dims(tf.range(0, 8, dtype=tf.float32), axis=0), axis=2),
        3,
        axis=2)
    corners = tf.concat([corners, corners + 10.], axis=0)
    num_of_points_per_line = 100
    points = box_utils.get_box_as_dotted_lines(
        corners, num_of_points_per_line=num_of_points_per_line)
    expected_points = np.repeat(
        np.linspace(0., 1., num_of_points_per_line)[:, np.newaxis],
        repeats=3,
        axis=1)
    self.assertAllEqual(points.shape, [2, 12 * num_of_points_per_line, 3])
    self.assertAllClose(points[0, :num_of_points_per_line, :], expected_points)
    self.assertAllClose(points[1, :num_of_points_per_line, :],
                        expected_points + 10.)

  def test_ray_to_box_coordinate_frame(self):
    rays_start_point = tf.constant([[-1, -1, -1], [0, 0, 0]], dtype=tf.float32)
    rays_end_point = tf.constant([[1, 1, 1], [2, 0, 0]], dtype=tf.float32)
    box_center = tf.constant([1, -1, 0], dtype=tf.float32)
    box_rotation_matrix1 = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                       dtype=tf.float32)
    box_rotation_matrix2 = tf.constant([[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                                       dtype=tf.float32)
    transformed_rays_start_point1, transformed_rays_end_point1 = (
        box_utils.ray_to_box_coordinate_frame(
            box_center=box_center,
            box_rotation_matrix=box_rotation_matrix1,
            rays_start_point=rays_start_point,
            rays_end_point=rays_end_point))
    transformed_rays_start_point2, transformed_rays_end_point2 = (
        box_utils.ray_to_box_coordinate_frame(
            box_center=box_center,
            box_rotation_matrix=box_rotation_matrix2,
            rays_start_point=rays_start_point,
            rays_end_point=rays_end_point))
    self.assertAllClose(transformed_rays_start_point1.numpy(),
                        np.array([[-2, 0, -1], [-1, 1, 0]]))
    self.assertAllClose(transformed_rays_end_point1.numpy(),
                        np.array([[0, 2, 1], [1, 1, 0]]))
    self.assertAllClose(transformed_rays_start_point2.numpy(),
                        np.array([[-2, -1, 0], [-1, 0, -1]]))
    self.assertAllClose(transformed_rays_end_point2.numpy(),
                        np.array([[0, 1, -2], [1, 0, -1]]))

  def test_ray_to_box_coordinate_frame_pairwise(self):
    rays_start_point = tf.constant([[-1, -1, -1], [0, 0, 0]], dtype=tf.float32)
    rays_end_point = tf.constant([[1, 1, 1], [2, 0, 0]], dtype=tf.float32)
    box_center = tf.constant([[1, -1, 0], [1, -1, 0]], dtype=tf.float32)
    box_rotation_matrix1 = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                       dtype=tf.float32)
    box_rotation_matrix2 = tf.constant([[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                                       dtype=tf.float32)
    transformed_rays_start_point1, transformed_rays_end_point1 = (
        box_utils.ray_to_box_coordinate_frame(
            box_center=box_center,
            box_rotation_matrix=box_rotation_matrix1,
            rays_start_point=rays_start_point,
            rays_end_point=rays_end_point))
    transformed_rays_start_point2, transformed_rays_end_point2 = (
        box_utils.ray_to_box_coordinate_frame(
            box_center=box_center,
            box_rotation_matrix=box_rotation_matrix2,
            rays_start_point=rays_start_point,
            rays_end_point=rays_end_point))
    self.assertAllClose(transformed_rays_start_point1.numpy(),
                        np.array([[-2, 0, -1], [-1, 1, 0]]))
    self.assertAllClose(transformed_rays_end_point1.numpy(),
                        np.array([[0, 2, 1], [1, 1, 0]]))
    self.assertAllClose(transformed_rays_start_point2.numpy(),
                        np.array([[-2, -1, 0], [-1, 0, -1]]))
    self.assertAllClose(transformed_rays_end_point2.numpy(),
                        np.array([[0, 1, -2], [1, 0, -1]]))

  def test_ray_box_intersection(self):
    rays_start_point = tf.constant([[-1, 0, 0], [0, 0, 0], [-5, 0, 0]],
                                   dtype=tf.float32)
    rays_end_point = tf.constant([[1, 0, 0], [0, 1, 0], [-5, 1, 0]],
                                 dtype=tf.float32)
    box_center = tf.constant([0, 0, 0], dtype=tf.float32)
    box_rotation_matrix = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                      dtype=tf.float32)
    box_size = tf.constant([2.0, 4.0, 6.0], dtype=tf.float32)
    intersection_points, intersection_indices = box_utils.ray_box_intersection(
        box_center=box_center,
        box_rotation_matrix=box_rotation_matrix,
        box_length=box_size[0],
        box_width=box_size[1],
        box_height=box_size[2],
        rays_start_point=rays_start_point,
        rays_end_point=rays_end_point)
    self.assertAllClose(
        intersection_points.numpy(),
        np.array([[[-1, 0, 0], [1, 0, 0]], [[0, -2, 0], [0, 2, 0]]]))
    self.assertAllEqual(intersection_indices.numpy(), np.array([0, 1]))

  def test_ray_box_intersection_pairwise(self):
    rays_start_point = tf.constant([[-1, 0, 0], [0, 0, 0], [-5, 0, 0]],
                                   dtype=tf.float32)
    rays_end_point = tf.constant([[1, 0, 0], [0, 1, 0], [-5, 1, 0]],
                                 dtype=tf.float32)
    box_center = tf.zeros([3, 3], dtype=tf.float32)
    box_rotation_matrix = tf.tile(
        tf.constant([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=tf.float32),
        [3, 1, 1])
    box_size = tf.tile(tf.constant([[2.0, 4.0, 6.0]], dtype=tf.float32), [3, 1])
    intersection_points, intersection_indices = box_utils.ray_box_intersection(
        box_center=box_center,
        box_rotation_matrix=box_rotation_matrix,
        box_length=box_size[:, 0],
        box_width=box_size[:, 1],
        box_height=box_size[:, 2],
        rays_start_point=rays_start_point,
        rays_end_point=rays_end_point)
    self.assertAllClose(
        intersection_points.numpy(),
        np.array([[[-1, 0, 0], [1, 0, 0]], [[0, -2, 0], [0, 2, 0]]]))
    self.assertAllEqual(intersection_indices.numpy(), np.array([0, 1]))

  def test_ray_box_intersection_shape(self):
    box_center = tf.random.uniform([3],
                                   minval=-1000.0,
                                   maxval=1000.0,
                                   dtype=tf.float32)
    box_size = tf.random.uniform([3],
                                 minval=1.0,
                                 maxval=100.0,
                                 dtype=tf.float32)
    box_rotation_matrix = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                      dtype=tf.float32)
    rays_start_point = tf.random.uniform([10000, 3],
                                         minval=-1000.0,
                                         maxval=1000.0,
                                         dtype=tf.float32)
    rays_end_point = tf.random.uniform([10000, 3],
                                       minval=-1000.0,
                                       maxval=1000.0,
                                       dtype=tf.float32)
    intersection_points, intersection_indices = box_utils.ray_box_intersection(
        box_center=box_center,
        box_rotation_matrix=box_rotation_matrix,
        box_length=box_size[0],
        box_height=box_size[1],
        box_width=box_size[2],
        rays_start_point=rays_start_point,
        rays_end_point=rays_end_point)
    num_intersecting_rays = intersection_points.shape[0]
    self.assertAllEqual(intersection_points.shape,
                        np.array([num_intersecting_rays, 2, 3]))
    self.assertAllEqual(intersection_indices.shape,
                        np.array([num_intersecting_rays]))

  def test_ray_box_intersection_shape_pairwise(self):
    r = 10000
    box_center = tf.random.uniform([r, 3],
                                   minval=-1000.0,
                                   maxval=1000.0,
                                   dtype=tf.float32)
    box_size = tf.random.uniform([r, 3],
                                 minval=1.0,
                                 maxval=100.0,
                                 dtype=tf.float32)
    box_rotation_matrix = tf.tile(
        tf.expand_dims(
            tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32),
            axis=0), [r, 1, 1])
    rays_start_point = tf.random.uniform([r, 3],
                                         minval=-1000.0,
                                         maxval=1000.0,
                                         dtype=tf.float32)
    rays_end_point = tf.random.uniform([r, 3],
                                       minval=-1000.0,
                                       maxval=1000.0,
                                       dtype=tf.float32)
    intersection_points, intersection_indices = box_utils.ray_box_intersection(
        box_center=box_center,
        box_rotation_matrix=box_rotation_matrix,
        box_length=box_size[:, 0],
        box_height=box_size[:, 1],
        box_width=box_size[:, 2],
        rays_start_point=rays_start_point,
        rays_end_point=rays_end_point)
    num_intersecting_rays = intersection_points.shape[0]
    self.assertAllEqual(intersection_points.shape,
                        np.array([num_intersecting_rays, 2, 3]))
    self.assertAllEqual(intersection_indices.shape,
                        np.array([num_intersecting_rays]))


if __name__ == '__main__':
  tf.test.main()
