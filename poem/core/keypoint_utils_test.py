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

"""Tests for keypoint utility functions."""

import math

import tensorflow as tf

from poem.core import keypoint_profiles
from poem.core import keypoint_utils


class KeypointUtilsTest(tf.test.TestCase):

  def test_get_single_points(self):
    # Shape = [2, 1, 3, 2].
    points = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]],
              [[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]]]
    indices = [1]
    # Shape = [2, 1, 1, 2].
    points = keypoint_utils.get_points(points, indices)
    self.assertAllEqual(points, [[[[2.0, 3.0]]], [[[12.0, 13.0]]]])

  def test_get_center_points(self):
    # Shape = [2, 1, 3, 2].
    points = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]],
              [[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]]]
    indices = [0, 1]
    # Shape = [2, 1, 1, 2].
    points = keypoint_utils.get_points(points, indices)
    self.assertAllClose(points, [[[[1.0, 2.0]]], [[[11.0, 12.0]]]])

  def test_swap_x_y_2d(self):
    # Shape = [2, 3, 2]
    points = tf.constant([[[1.0, 2.0], [3.0, 6.0], [5.0, 6.0]],
                          [[11.0, 12.0], [13.0, 16.0], [15.0, 16.0]]])
    swapped_points = keypoint_utils.swap_x_y(points)
    self.assertAllClose(swapped_points,
                        [[[2.0, 1.0], [6.0, 3.0], [6.0, 5.0]],
                         [[12.0, 11.0], [16.0, 13.0], [16.0, 15.0]]])

  def test_swap_x_y_3d(self):
    # Shape = [2, 3, 3]
    points = tf.constant([[[1.0, 2.0, 3.0], [3.0, 6.0, 7.0], [5.0, 6.0, 7.0]],
                          [[11.0, 12.0, 13.0], [13.0, 16.0, 17.0],
                           [15.0, 16.0, 17.0]]])
    swapped_points = keypoint_utils.swap_x_y(points)
    self.assertAllClose(
        swapped_points,
        [[[2.0, 1.0, 3.0], [6.0, 3.0, 7.0], [6.0, 5.0, 7.0]],
         [[12.0, 11.0, 13.0], [16.0, 13.0, 17.0], [16.0, 15.0, 17.0]]])

  def test_override_points(self):
    # Shape = [2, 3, 3]
    points = tf.constant([[[1.0, 2.0, 3.0], [3.0, 6.0, 7.0], [5.0, 6.0, 7.0]],
                          [[11.0, 12.0, 13.0], [13.0, 16.0, 17.0],
                           [15.0, 16.0, 17.0]]])
    updated_points = keypoint_utils.override_points(
        points, from_indices_list=[[0, 2], [1]], to_indices=[0, 1])
    self.assertAllClose(
        updated_points,
        [[[3.0, 5.0, 6.0], [3.0, 5.0, 6.0], [5.0, 6.0, 7.0]],
         [[13.0, 15.0, 16.0], [13.0, 15.0, 16.0], [15.0, 16.0, 17.0]]])

  def test_naive_normalize_points(self):
    # Shape = [2, 3, 2]
    points = tf.constant([[[1.0, 2.0], [3.0, 6.0], [5.0, 6.0]],
                          [[11.0, 12.0], [13.0, 16.0], [15.0, 16.0]]])
    # Shape = [2, 3].
    point_masks = tf.constant([[True, True, False], [False, False, False]])
    # Shape = [2, 3, 2].
    normalized_points = keypoint_utils.naive_normalize_points(
        points, point_masks)
    self.assertAllClose(normalized_points,
                        [[[-0.25, -0.5], [0.25, 0.5], [0.0, 0.0]],
                         [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])

  def test_normalize_points(self):
    # Shape = [2, 1, 3, 2].
    points = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]],
              [[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]]]
    offset_point_indices = [0, 1]
    scale_distance_point_index_pairs = [([0, 1], [1]), ([0], [1, 2])]
    normalized_points, offset_points, scale_distances = (
        keypoint_utils.normalize_points(
            points,
            offset_point_indices=offset_point_indices,
            scale_distance_point_index_pairs=scale_distance_point_index_pairs,
            scale_distance_reduction_fn=tf.math.reduce_sum,
            scale_unit=1.0))

    sqrt_2 = 1.414213562
    self.assertAllClose(normalized_points, [
        [[
            [-0.25 / sqrt_2, -0.25 / sqrt_2],
            [0.25 / sqrt_2, 0.25 / sqrt_2],
            [0.75 / sqrt_2, 0.75 / sqrt_2],
        ]],
        [[
            [-0.25 / sqrt_2, -0.25 / sqrt_2],
            [0.25 / sqrt_2, 0.25 / sqrt_2],
            [0.75 / sqrt_2, 0.75 / sqrt_2],
        ]],
    ])
    self.assertAllClose(offset_points, [[[[1.0, 2.0]]], [[[11.0, 12.0]]]])
    self.assertAllClose(scale_distances,
                        [[[[4.0 * sqrt_2]]], [[[4.0 * sqrt_2]]]])

  def test_centralize_masked_points(self):
    # Shape = [2, 4, 2].
    points = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
              [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]]
    # Shape = [2, 4].
    point_masks = [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]

    # Shape = [2, 4, 2].
    centralized_points = keypoint_utils.centralize_masked_points(
        points, point_masks)

    self.assertAllClose(
        centralized_points,
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [3.0, 4.0]],
         [[9.0, 10.0], [11.0, 12.0], [10.0, 11.0], [10.0, 11.0]]])

  def test_standardize_points(self):
    # Shape = [2, 3, 2].
    x = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                     [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]])
    standardized_x, offsets, scales = keypoint_utils.standardize_points(x)
    self.assertAllClose(standardized_x,
                        [[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]],
                         [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]])
    self.assertAllClose(offsets, [[[3.0, 4.0]], [[6.0, 8.0]]])
    self.assertAllClose(scales, [[[4.0]], [[8.0]]])

  def test_compute_procrustes_alignment_params(self):
    # Shape = [3, 4, 3].
    target_points = tf.constant([[[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                  [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]],
                                 [[1.0, 0.0, 1.0], [5.0, 2.0, 2.0],
                                  [4.0, 0.0, 0.0], [-1.0, 3.0, -2.0]],
                                 [[2.0, 0.0, 2.0], [10.0, 4.0, 4.0],
                                  [8.0, 0.0, 0.0], [-2.0, 6.0, -4.0]]])
    source_points = tf.constant([[[3.0, 1.0, 5.0], [-2.0, 3.0, 0.0],
                                  [1.0, -1.0, 1.0], [8.0, 3.0, -2.0]],
                                 [[3.0, 5.0, 1.0], [-2.0, 0.0, 3.0],
                                  [1.0, 1.0, -1.0], [8.0, -2.0, 3.0]],
                                 [[6.0, 10.0, 2.0], [-4.0, 0.0, 6.0],
                                  [2.0, 2.0, -2.0], [16.0, -4.0, 6.0]]])
    rotations, scales, translations = (
        keypoint_utils.compute_procrustes_alignment_params(
            target_points, source_points))
    self.assertAllClose(rotations, [[[-0.87982, -0.47514731, 0.01232074],
                                     [-0.31623112, 0.60451691, 0.73113418],
                                     [-0.35484453, 0.63937027, -0.68212243]],
                                    [[-0.87982, 0.01232074, -0.47514731],
                                     [-0.35484453, -0.68212243, 0.63937027],
                                     [-0.31623112, 0.73113418, 0.60451691]],
                                    [[-0.87982, 0.01232074, -0.47514731],
                                     [-0.35484453, -0.68212243, 0.63937027],
                                     [-0.31623112, 0.73113418, 0.60451691]]])
    self.assertAllClose(
        scales, [[[0.63716284347]], [[0.63716284347]], [[0.63716284347]]])
    self.assertAllClose(translations, [[[4.17980137, 0.02171898, 0.96621997]],
                                       [[4.17980137, 0.96621997, 0.02171898]],
                                       [[8.35960274, 1.93243994, 0.04343796]]])

  def test_compute_procrustes_alignment_params_with_masks(self):
    # Shape = [3, 6, 3].
    target_points = tf.constant([[[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                  [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0],
                                  [100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
                                 [[1.0, 0.0, 1.0], [5.0, 2.0, 2.0],
                                  [700.0, 800.0, 900.0], [800.0, 700.0, 600.0],
                                  [4.0, 0.0, 0.0], [-1.0, 3.0, -2.0]],
                                 [[2.0, 0.0, 2.0], [500.0, 400.0, 300.0],
                                  [10.0, 4.0, 4.0], [200.0, 100.0, 200.0],
                                  [8.0, 0.0, 0.0], [-2.0, 6.0, -4.0]]])
    source_points = tf.constant([[[3.0, 1.0, 5.0], [-2.0, 3.0, 0.0],
                                  [1.0, -1.0, 1.0], [8.0, 3.0, -2.0],
                                  [300.0, 400.0, 500.0], [600.0, 700.0, 800.0]],
                                 [[3.0, 5.0, 1.0], [-2.0, 0.0, 3.0],
                                  [900.0, 800.0, 700.0], [600.0, 500.0, 400.0],
                                  [1.0, 1.0, -1.0], [8.0, -2.0, 3.0]],
                                 [[6.0, 10.0, 2.0], [300.0, 200.0, 100.0],
                                  [-4.0, 0.0, 6.0], [200.0, 300.0, 400.0],
                                  [2.0, 2.0, -2.0], [16.0, -4.0, 6.0]]])
    # Shape = [3, 6].
    point_masks = tf.constant([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                               [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]])

    rotations, scales, translations = (
        keypoint_utils.compute_procrustes_alignment_params(
            target_points, source_points, point_masks=point_masks))
    self.assertAllClose(rotations, [[[-0.87982, -0.47514731, 0.01232074],
                                     [-0.31623112, 0.60451691, 0.73113418],
                                     [-0.35484453, 0.63937027, -0.68212243]],
                                    [[-0.87982, 0.01232074, -0.47514731],
                                     [-0.35484453, -0.68212243, 0.63937027],
                                     [-0.31623112, 0.73113418, 0.60451691]],
                                    [[-0.87982, 0.01232074, -0.47514731],
                                     [-0.35484453, -0.68212243, 0.63937027],
                                     [-0.31623112, 0.73113418, 0.60451691]]])
    self.assertAllClose(
        scales, [[[0.63716284347]], [[0.63716284347]], [[0.63716284347]]])
    self.assertAllClose(translations, [[[4.17980137, 0.02171898, 0.96621997]],
                                       [[4.17980137, 0.96621997, 0.02171898]],
                                       [[8.35960274, 1.93243994, 0.04343796]]])

  def test_compute_mpjpes_case_1(self):
    # Shape = [2, 3, 2].
    lhs_points = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                              [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    rhs_points = tf.constant([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                              [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]]])
    mpjpes = keypoint_utils.compute_mpjpes(lhs_points, rhs_points)
    self.assertAllClose(mpjpes, [1.41421356237, 1.41421356237])

  def test_compute_mpjpes_case_2(self):
    lhs_points = tf.constant([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
    rhs_points = tf.constant([[1.0, 0.0, 2.0], [2.0, -1.0, 3.0]])
    mpjpes = keypoint_utils.compute_mpjpes(lhs_points, rhs_points)
    self.assertAlmostEqual(self.evaluate(mpjpes), 2.76866, places=5)

  def test_compute_mpjpes_with_point_masks(self):
    # Shape = [2, 3, 2].
    lhs_points = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                              [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    rhs_points = tf.constant([[[2.0, 3.0], [4.0, 5.0], [7.0, 8.0]],
                              [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]]])
    # Shape = [2, 3].
    point_masks = tf.constant([[1.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    mpjpes = keypoint_utils.compute_mpjpes(
        lhs_points, rhs_points, point_masks=point_masks)
    self.assertAllClose(mpjpes, [1.88561808316, 0.0])

  def test_compute_procrustes_aligned_mpjpes_case_1(self):
    target_points = tf.constant([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0]],
                                 [[2.0, 2.0, 2.0], [-1.5, -1.0, 0.0],
                                  [2.5, 1.3, 1.4]]])
    source_points = tf.constant([[[2.0, 2.0, 2.0], [-1.5, -1.0, 0.0],
                                  [2.5, 1.3, 1.4]],
                                 [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0]]])
    mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
        target_points, source_points)
    self.assertAllClose(mpjpes, [0.133016, 0.3496029])

  def test_compute_procrustes_aligned_mpjpes_case_2(self):
    target_points = tf.constant([[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                 [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]])
    source_points = tf.constant([[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                 [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]])
    mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
        target_points, source_points)
    self.assertAlmostEqual(self.evaluate(mpjpes), 0.0, places=5)

  def test_compute_procrustes_aligned_mpjpes_case_3(self):
    target_points = tf.constant([[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                 [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]])
    source_points = tf.constant([[1.5, 0.5, 0.0], [5.0, 2.0, 2.2],
                                 [4.1, 0.0, -1.0], [-1.0, -2.5, -2.0]])
    mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
        target_points, source_points)
    self.assertAlmostEqual(self.evaluate(mpjpes), 1.00227, places=5)

  def test_compute_procrustes_aligned_mpjpes_case_4(self):
    target_points = tf.constant([[1.0, 1.0, 0.0], [5.0, 2.0, 2.0],
                                 [4.0, 0.0, 0.0], [-1.0, -2.0, 3.0]])
    source_points = tf.constant([[-10.0, -24.5, -49.5], [-9.0, -22.5, -49.0],
                                 [-10.0, -23.0, -50.0], [-8.5, -25.5, -51.0]])
    mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
        target_points, source_points)
    self.assertAlmostEqual(self.evaluate(mpjpes), 0.0, places=5)

  def test_compute_procrustes_aligned_mpjpes_case_5(self):
    target_points = tf.constant([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                 [1.0, 1.0, 1.0]])
    source_points = tf.constant([[2.0, 2.0, 2.0], [-1.5, -1.0, 0.0],
                                 [2.5, 1.3, 1.4]])
    mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
        target_points, source_points)
    self.assertAlmostEqual(self.evaluate(mpjpes), 0.133016, places=5)

  def test_compute_procrustes_aligned_mpjpes_case_6(self):
    target_points = tf.constant([[2.0, 2.0, 2.0], [-1.5, -1.0, 0.0],
                                 [2.5, 1.3, 1.4]])
    source_points = tf.constant([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                 [1.0, 1.0, 1.0]])
    mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
        target_points, source_points)
    self.assertAlmostEqual(self.evaluate(mpjpes), 0.3496029, places=5)

  def test_compute_procrustes_aligned_mpjpes_with_masks(self):
    # Shape = [5, 3].
    target_points = tf.constant([[2.0, 2.0, 2.0], [-1.5, -1.0, 0.0],
                                 [100.0, 200.0, 300.0], [2.5, 1.3, 1.4],
                                 [400.0, 500.0, 600.0]])
    source_points = tf.constant([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                                 [700.0, 800.0, 900.0], [1.0, 1.0, 1.0],
                                 [800.0, 700.0, 600.0]])
    # Shape = [5].
    point_masks = tf.constant([1.0, 1.0, 0.0, 1.0, 0.0])
    mpjpes = keypoint_utils.compute_procrustes_aligned_mpjpes(
        target_points, source_points, point_masks=point_masks)
    self.assertAlmostEqual(self.evaluate(mpjpes), 0.3496029, places=5)

  def test_normalize_points_by_image_size(self):
    points = tf.constant([
        [[10.0, 40.0], [30.0, 80.0], [50.0, 120.0]],
        [[0.2, 0.2], [0.6, 0.4], [1.0, 0.6]],
    ])
    image_sizes = tf.constant([[100, 200], [20, 10]])
    normalized_points = keypoint_utils.normalize_points_by_image_size(
        points, image_sizes)
    self.assertAllClose(normalized_points, [
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
    ])

  def test_denormalize_points_by_image_size(self):
    points = tf.constant([
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
    ])
    image_sizes = tf.constant([[100, 200], [20, 10]])
    denormalized_points = keypoint_utils.denormalize_points_by_image_size(
        points, image_sizes)
    self.assertAllClose(denormalized_points, [
        [[10.0, 40.0], [30.0, 80.0], [50.0, 120.0]],
        [[0.2, 0.2], [0.6, 0.4], [1.0, 0.6]],
    ])


  def test_create_rotation_matrices_3d(self):
    # Shape = [3, 2].
    azimuths = tf.constant([[0.0, math.pi / 2.0], [math.pi / 2.0, 0.0],
                            [0.0, math.pi / 2.0]])
    elevations = tf.constant([[0.0, -math.pi / 2.0], [-math.pi / 2.0, 0.0],
                              [0.0, -math.pi / 2.0]])
    rolls = tf.constant([[0.0, math.pi], [math.pi, 0.0], [0.0, math.pi]])
    self.assertAllClose(
        keypoint_utils.create_rotation_matrices_3d(azimuths, elevations, rolls),
        [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
          [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]],
         [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
          [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
         [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
          [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]])

  def test_rotate_points(self):
    rotation_matrices = tf.constant([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]],
                                      [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                                       [1.0, 0.0, 0.0]]],
                                     [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                                       [1.0, 0.0, 0.0]],
                                      [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]]],
                                     [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0]],
                                      [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                                       [1.0, 0.0, 0.0]]]])
    points = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                          [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]])
    self.assertAllClose(
        keypoint_utils.rotate_points(rotation_matrices, points),
        [[[1.0, 2.0, 3.0], [5.0, 6.0, 4.0]],
         [[8.0, 9.0, 7.0], [10.0, 11.0, 12.0]],
         [[13.0, 14.0, 15.0], [17.0, 18.0, 16.0]]])

  def test_random_rotate_and_project_3d_to_2d_without_default_camera(self):
    keypoints_3d = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]])
    keypoints_2d = keypoint_utils.random_rotate_and_project_3d_to_2d(
        keypoints_3d,
        azimuth_range=(math.pi / 2.0, math.pi / 2.0),
        elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
        roll_range=(math.pi, math.pi),
        default_camera=False)
    self.assertAllClose(
        keypoints_2d,
        [[[2.0 / 1.0, 3.0 / 1.0], [5.0 / 4.0, 6.0 / 4.0]],
         [[8.0 / 7.0, 9.0 / 7.0], [11.0 / 10.0, 12.0 / 10.0]],
         [[14.0 / 13.0, 15.0 / 13.0], [17.0 / 16.0, 18.0 / 16.0]]])

  def test_random_rotate_and_project_3d_to_2d_with_default_camera(self):
    keypoints_3d = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]])
    keypoints_2d = keypoint_utils.random_rotate_and_project_3d_to_2d(
        keypoints_3d,
        azimuth_range=(math.pi / 2.0, math.pi / 2.0),
        elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
        roll_range=(math.pi, math.pi),
        default_camera=True)
    self.assertAllClose(
        keypoints_2d,
        [[[-1.0 / 5.0, 2.0 / 5.0], [-4.0 / 8.0, 5.0 / 8.0]],
         [[-7.0 / 11.0, 8.0 / 11.0], [-10.0 / 14.0, 11.0 / 14.0]],
         [[-13.0 / 17.0, 14.0 / 17.0], [-16.0 / 20.0, 17.0 / 20.0]]])

  def test_random_rotate_and_project_temporal_3d_to_2d_without_default_camera(
      self):
    keypoints_3d_t1 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                   [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                   [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]])
    keypoints_3d_t2 = tf.constant([[[1.2, 2.2, 3.2], [4.2, 5.2, 6.2]],
                                   [[7.4, 8.4, 9.4], [10.4, 11.4, 12.4]],
                                   [[13.8, 14.8, 15.8], [16.8, 17.8, 18.8]]])
    keypoints_3d_t3 = tf.constant([[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
                                   [[7.6, 8.6, 9.6], [10.6, 11.6, 12.6]],
                                   [[13.7, 14.7, 15.7], [16.7, 17.7, 18.7]]])
    keypoints_3d = tf.stack(
        [keypoints_3d_t1, keypoints_3d_t2, keypoints_3d_t3],
        axis=-3)
    keypoints_2d = keypoint_utils.random_rotate_and_project_3d_to_2d(
        keypoints_3d,
        azimuth_range=(math.pi / 2.0, math.pi / 2.0),
        elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
        roll_range=(math.pi, math.pi),
        default_camera=False,
        sequential_inputs=True)
    self.assertAllClose(
        keypoints_2d,
        [[[[2.0 / 1.0, 3.0 / 1.0], [5.0 / 4.0, 6.0 / 4.0]],
          [[2.2 / 1.2, 3.2 / 1.2], [5.2 / 4.2, 6.2 / 4.2]],
          [[2.5 / 1.5, 3.5 / 1.5], [5.5 / 4.5, 6.5 / 4.5]]],
         [[[8.0 / 7.0, 9.0 / 7.0], [11.0 / 10.0, 12.0 / 10.0]],
          [[8.4 / 7.4, 9.4 / 7.4], [11.4 / 10.4, 12.4 / 10.4]],
          [[8.6 / 7.6, 9.6 / 7.6], [11.6 / 10.6, 12.6 / 10.6]]],
         [[[14.0 / 13.0, 15.0 / 13.0], [17.0 / 16.0, 18.0 / 16.0]],
          [[14.8 / 13.8, 15.8 / 13.8], [17.8 / 16.8, 18.8 / 16.8]],
          [[14.7 / 13.7, 15.7 / 13.7], [17.7 / 16.7, 18.7 / 16.7]]]])

  def test_random_rotate_and_project_temporal_3d_to_2d_with_default_camera(
      self):
    keypoints_3d_t1 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                   [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                   [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]])
    keypoints_3d_t2 = tf.constant([[[1.2, 2.2, 3.2], [4.2, 5.2, 6.2]],
                                   [[7.4, 8.4, 9.4], [10.4, 11.4, 12.4]],
                                   [[13.8, 14.8, 15.8], [16.8, 17.8, 18.8]]])
    keypoints_3d_t3 = tf.constant([[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
                                   [[7.6, 8.6, 9.6], [10.6, 11.6, 12.6]],
                                   [[13.7, 14.7, 15.7], [16.7, 17.7, 18.7]]])
    keypoints_3d = tf.stack(
        [keypoints_3d_t1, keypoints_3d_t2, keypoints_3d_t3],
        axis=-3)
    keypoints_2d = keypoint_utils.random_rotate_and_project_3d_to_2d(
        keypoints_3d,
        azimuth_range=(math.pi / 2.0, math.pi / 2.0),
        elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
        roll_range=(math.pi, math.pi),
        default_camera=True,
        sequential_inputs=True)

    self.assertAllClose(
        keypoints_2d,
        [[[[-1.0 / 5.0, 2.0 / 5.0], [-4.0 / 8.0, 5.0 / 8.0]],
          [[-1.2 / 5.2, 2.2 / 5.2], [-4.2 / 8.2, 5.2 / 8.2]],
          [[-1.5 / 5.5, 2.5 / 5.5], [-4.5 / 8.5, 5.5 / 8.5]]],
         [[[-7.0 / 11.0, 8.0 / 11.0], [-10.0 / 14.0, 11.0 / 14.0]],
          [[-7.4 / 11.4, 8.4 / 11.4], [-10.4 / 14.4, 11.4 / 14.4]],
          [[-7.6 / 11.6, 8.6 / 11.6], [-10.6 / 14.6, 11.6 / 14.6]]],
         [[[-13.0 / 17.0, 14.0 / 17.0], [-16.0 / 20.0, 17.0 / 20.0]],
          [[-13.8 / 17.8, 14.8 / 17.8], [-16.8 / 20.8, 17.8 / 20.8]],
          [[-13.7 / 17.7, 14.7 / 17.7], [-16.7 / 20.7, 17.7 / 20.7]]]])

  def test_create_smooth_rotation_matrices(self):
    start_euler_angles = (-math.pi, -math.pi / 6.0, -math.pi / 6.0)
    end_euler_angles = (math.pi, math.pi / 6.0, math.pi / 6.0)
    metrics = keypoint_utils.create_smooth_rotation_matrices(
        start_euler_angles, end_euler_angles, num_views=3)
    self.assertAllClose(
        metrics,
        [[[-0.866, -0.25, 0.433],
          [0., -0.866, -0.5],
          [0.5, -0.433, 0.75]],
         [[1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]],
         [[-0.866, -0.25, -0.433],
          [-0.0, -0.866, 0.5],
          [-0.5, 0.433, 0.75]]], atol=1e-04)

  def test_select_keypoints_by_name(self):
    input_keypoints = tf.constant([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
        [7.0, 7.0, 7.0],
        [8.0, 8.0, 8.0],
        [9.0, 9.0, 9.0],
        [10.0, 10.0, 10.0],
        [11.0, 11.0, 11.0],
        [12.0, 12.0, 12.0],
        [13.0, 13.0, 13.0],
        [14.0, 14.0, 14.0],
        [15.0, 15.0, 15.0],
        [16.0, 16.0, 16.0],
    ])
    keypoint_profile_3d = (
        keypoint_profiles.create_keypoint_profile_or_die('LEGACY_3DH36M17'))
    keypoint_profile_2d = (
        keypoint_profiles.create_keypoint_profile_or_die('LEGACY_2DCOCO13'))
    output_keypoints, _ = keypoint_utils.select_keypoints_by_name(
        input_keypoints,
        input_keypoint_names=keypoint_profile_3d.keypoint_names,
        output_keypoint_names=(
            keypoint_profile_2d.compatible_keypoint_name_dict['LEGACY_3DH36M17']
        ))
    self.assertAllClose(output_keypoints, [
        [1.0, 1.0, 1.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
        [7.0, 7.0, 7.0],
        [8.0, 8.0, 8.0],
        [9.0, 9.0, 9.0],
        [11.0, 11.0, 11.0],
        [12.0, 12.0, 12.0],
        [13.0, 13.0, 13.0],
        [14.0, 14.0, 14.0],
        [15.0, 15.0, 15.0],
        [16.0, 16.0, 16.0],
    ])

  def test_random_project_and_select_keypoints(self):
    keypoints_3d = tf.constant([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
        [7.0, 7.0, 7.0],
        [8.0, 8.0, 8.0],
        [9.0, 9.0, 9.0],
        [10.0, 10.0, 10.0],
        [11.0, 11.0, 11.0],
        [12.0, 12.0, 12.0],
        [13.0, 13.0, 13.0],
        [14.0, 14.0, 14.0],
        [15.0, 15.0, 15.0],
        [16.0, 16.0, 16.0],
    ])
    keypoint_profile_3d = (
        keypoint_profiles.create_keypoint_profile_or_die('LEGACY_3DH36M17'))
    keypoint_profile_2d = (
        keypoint_profiles.create_keypoint_profile_or_die('LEGACY_2DCOCO13'))
    keypoints_2d, _ = keypoint_utils.random_project_and_select_keypoints(
        keypoints_3d,
        keypoint_profile_3d=keypoint_profile_3d,
        output_keypoint_names=(
            keypoint_profile_2d.compatible_keypoint_name_dict['LEGACY_3DH36M17']
        ),
        azimuth_range=(math.pi / 2.0, math.pi / 2.0),
        elevation_range=(math.pi / 2.0, math.pi / 2.0),
        roll_range=(-math.pi / 2.0, -math.pi / 2.0))
    keypoints_2d, _, _ = keypoint_profile_2d.normalize(keypoints_2d)
    self.assertAllClose(keypoints_2d, [
        [-0.4356161, 0.4356161],
        [-0.32822642, 0.32822642],
        [-0.2897728, 0.28977284],
        [-0.24986516, 0.24986516],
        [-0.2084193, 0.2084193],
        [-0.16534455, 0.16534461],
        [-0.12054307, 0.1205431],
        [-0.025327, 0.025327],
        [0.025327, -0.025327],
        [0.07818867, -0.07818867],
        [0.13340548, -0.13340548],
        [0.19113854, -0.19113848],
        [0.2515637, -0.25156358],
    ])

  def test_remove_at_indices(self):
    keypoints = tf.constant([[[1.0, 2.0], [7.0, 8.0], [3.0, 4.0], [5.0, 6.0],
                              [9.0, 10.0]]])
    indices = [1, 4]
    keypoints = keypoint_utils.remove_at_indices(keypoints, indices)
    self.assertAllClose(keypoints, [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

  def test_insert_at_indices(self):
    keypoints = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    indices = [1, 3, 3]
    insert_keypoints = tf.constant([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    keypoints = keypoint_utils.insert_at_indices(
        keypoints, indices, insert_keypoints=insert_keypoints)
    self.assertAllClose(keypoints, [[[1.0, 2.0], [7.0, 8.0], [3.0, 4.0],
                                     [5.0, 6.0], [9.0, 10.0], [11.0, 12.0]]])

  def test_insert_zeros_at_indices(self):
    keypoints = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    indices = [1, 1, 3]
    keypoints = keypoint_utils.insert_at_indices(
        keypoints, indices, insert_keypoints=None)
    self.assertAllClose(keypoints, [[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0],
                                     [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]]])

  def test_transfer_keypoint_masks_case_1(self):
    # Shape = [2, 13].
    input_keypoint_masks = tf.constant([
        [
            1.0,  # NOSE_TIP
            1.0,  # LEFT_SHOULDER
            1.0,  # RIGHT_SHOULDER
            0.0,  # LEFT_ELBOW
            1.0,  # RIGHT_ELBOW
            1.0,  # LEFT_WRIST
            0.0,  # RIGHT_WRIST
            1.0,  # LEFT_HIP
            0.0,  # RIGHT_HIP
            1.0,  # LEFT_KNEE
            1.0,  # RIGHT_KNEE
            0.0,  # LEFT_ANKLE
            0.0,  # RIGHT_ANKLE
        ],
        [
            0.0,  # NOSE_TIP
            0.0,  # LEFT_SHOULDER
            0.0,  # RIGHT_SHOULDER
            1.0,  # LEFT_ELBOW
            0.0,  # RIGHT_ELBOW
            0.0,  # LEFT_WRIST
            1.0,  # RIGHT_WRIST
            0.0,  # LEFT_HIP
            1.0,  # RIGHT_HIP
            0.0,  # LEFT_KNEE
            0.0,  # RIGHT_KNEE
            1.0,  # LEFT_ANKLE
            1.0,  # RIGHT_ANKLE
        ]
    ])
    input_keypoint_profile = keypoint_profiles.create_keypoint_profile_or_die(
        '2DSTD13')
    output_keypoint_profile = keypoint_profiles.create_keypoint_profile_or_die(
        '3DSTD16')
    # Shape = [2, 16].
    output_keypoint_masks = keypoint_utils.transfer_keypoint_masks(
        input_keypoint_masks, input_keypoint_profile, output_keypoint_profile)
    self.assertAllClose(
        output_keypoint_masks,
        [
            [
                1.0,  # NOSE
                1.0,  # NECK
                1.0,  # LEFT_SHOULDER
                1.0,  # RIGHT_SHOULDER
                0.0,  # LEFT_ELBOW
                1.0,  # RIGHT_ELBOW
                1.0,  # LEFT_WRIST
                0.0,  # RIGHT_WRIST
                0.0,  # SPINE
                0.0,  # PELVIS
                1.0,  # LEFT_HIP
                0.0,  # RIGHT_HIP
                1.0,  # LEFT_KNEE
                1.0,  # RIGHT_KNEE
                0.0,  # LEFT_ANKLE
                0.0,  # RIGHT_ANKLE
            ],
            [
                0.0,  # NOSE
                0.0,  # NECK
                0.0,  # LEFT_SHOULDER
                0.0,  # RIGHT_SHOULDER
                1.0,  # LEFT_ELBOW
                0.0,  # RIGHT_ELBOW
                0.0,  # LEFT_WRIST
                1.0,  # RIGHT_WRIST
                0.0,  # SPINE
                0.0,  # PELVIS
                0.0,  # LEFT_HIP
                1.0,  # RIGHT_HIP
                0.0,  # LEFT_KNEE
                0.0,  # RIGHT_KNEE
                1.0,  # LEFT_ANKLE
                1.0,  # RIGHT_ANKLE
            ]
        ])


def test_transfer_keypoint_masks_case_2(self):
  # Shape = [2, 16].
  input_keypoint_masks = tf.constant([
      [
          1.0,  # NOSE
          1.0,  # NECK
          1.0,  # LEFT_SHOULDER
          1.0,  # RIGHT_SHOULDER
          0.0,  # LEFT_ELBOW
          1.0,  # RIGHT_ELBOW
          1.0,  # LEFT_WRIST
          0.0,  # RIGHT_WRIST
          0.0,  # SPINE
          0.0,  # PELVIS
          1.0,  # LEFT_HIP
          0.0,  # RIGHT_HIP
          1.0,  # LEFT_KNEE
          1.0,  # RIGHT_KNEE
          0.0,  # LEFT_ANKLE
          0.0,  # RIGHT_ANKLE
      ],
      [
          0.0,  # NOSE
          0.0,  # NECK
          0.0,  # LEFT_SHOULDER
          0.0,  # RIGHT_SHOULDER
          1.0,  # LEFT_ELBOW
          0.0,  # RIGHT_ELBOW
          0.0,  # LEFT_WRIST
          1.0,  # RIGHT_WRIST
          1.0,  # SPINE
          1.0,  # PELVIS
          0.0,  # LEFT_HIP
          1.0,  # RIGHT_HIP
          0.0,  # LEFT_KNEE
          0.0,  # RIGHT_KNEE
          1.0,  # LEFT_ANKLE
          1.0,  # RIGHT_ANKLE
      ]
  ])
  input_keypoint_profile = keypoint_profiles.create_keypoint_profile_or_die(
      '3DSTD16')
  output_keypoint_profile = keypoint_profiles.create_keypoint_profile_or_die(
      '2DSTD13')
  # Shape = [2, 13].
  output_keypoint_masks = keypoint_utils.transfer_keypoint_masks(
      input_keypoint_masks, input_keypoint_profile, output_keypoint_profile)
  self.assertAllClose(
      output_keypoint_masks,
      [
          [
              1.0,  # NOSE_TIP
              1.0,  # LEFT_SHOULDER
              1.0,  # RIGHT_SHOULDER
              0.0,  # LEFT_ELBOW
              1.0,  # RIGHT_ELBOW
              1.0,  # LEFT_WRIST
              0.0,  # RIGHT_WRIST
              1.0,  # LEFT_HIP
              0.0,  # RIGHT_HIP
              1.0,  # LEFT_KNEE
              1.0,  # RIGHT_KNEE
              0.0,  # LEFT_ANKLE
              0.0,  # RIGHT_ANKLE
          ],
          [
              0.0,  # NOSE_TIP
              0.0,  # LEFT_SHOULDER
              0.0,  # RIGHT_SHOULDER
              1.0,  # LEFT_ELBOW
              0.0,  # RIGHT_ELBOW
              0.0,  # LEFT_WRIST
              1.0,  # RIGHT_WRIST
              0.0,  # LEFT_HIP
              1.0,  # RIGHT_HIP
              0.0,  # LEFT_KNEE
              0.0,  # RIGHT_KNEE
              1.0,  # LEFT_ANKLE
              1.0,  # RIGHT_ANKLE
          ]
      ])


if __name__ == '__main__':
  tf.test.main()
