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

"""Tests for input generator functions."""

import math

import numpy as np
import tensorflow as tf

from poem.core import common
from poem.core import input_generator
from poem.core import keypoint_profiles


class InputGeneratorTest(tf.test.TestCase):

  def test_preprocess_keypoints_3d(self):
    profile = keypoint_profiles.KeypointProfile3D(
        name='Dummy',
        keypoint_names=[('A', keypoint_profiles.LeftRightType.UNKNOWN),
                        ('B', keypoint_profiles.LeftRightType.UNKNOWN),
                        ('C', keypoint_profiles.LeftRightType.UNKNOWN)],
        offset_keypoint_names=['A'],
        scale_keypoint_name_pairs=[(['A'], ['B'])],
        segment_name_pairs=[])
    keypoints_3d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    preprocessed_keypoints_3d, side_outputs = (
        input_generator.preprocess_keypoints_3d(keypoints_3d, profile))

    sqrt_3 = 1.73205080757
    self.assertAllClose(
        preprocessed_keypoints_3d,
        [[0.0, 0.0, 0.0], [1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
         [2.0 / sqrt_3, 2.0 / sqrt_3, 2.0 / sqrt_3]])
    self.assertCountEqual(
        side_outputs,
        ['offset_points_3d', 'scale_distances_3d', 'preprocessed_keypoints_3d'])
    self.assertAllClose(side_outputs['offset_points_3d'], [[1.0, 2.0, 3.0]])
    self.assertAllClose(side_outputs['scale_distances_3d'], [[3.0 * sqrt_3]])
    self.assertAllClose(
        side_outputs['preprocessed_keypoints_3d'],
        [[0.0, 0.0, 0.0], [1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
         [2.0 / sqrt_3, 2.0 / sqrt_3, 2.0 / sqrt_3]])

  def test_preprocess_keypoints_2d_with_projection(self):
    # Shape = [4, 2, 17, 3].
    keypoints_3d = tf.constant([
        [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [7.0, 8.0, 9.0]],
         [[11.0, 12.0, 13.0], [13.0, 14.0, 15.0], [15.0, 16.0, 17.0],
          [17.0, 18.0, 19.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]],
        [[[31.0, 32.0, 33.0], [33.0, 34.0, 35.0], [35.0, 36.0, 37.0],
          [37.0, 38.0, 39.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]],
         [[41.0, 42.0, 43.0], [43.0, 44.0, 35.0], [45.0, 46.0, 47.0],
          [47.0, 48.0, 49.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]],
        [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [7.0, 8.0, 9.0]],
         [[11.0, 12.0, 13.0], [13.0, 14.0, 15.0], [15.0, 16.0, 17.0],
          [17.0, 18.0, 19.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]],
        [[[31.0, 32.0, 33.0], [33.0, 34.0, 35.0], [35.0, 36.0, 37.0],
          [37.0, 38.0, 39.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]],
         [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [7.0, 8.0, 9.0]]],
    ])

    keypoint_profile_3d = (
        keypoint_profiles.create_keypoint_profile_or_die('LEGACY_3DH36M17'))
    keypoint_profile_2d = (
        keypoint_profiles.create_keypoint_profile_or_die('LEGACY_2DCOCO13'))
    keypoints_2d, _ = input_generator.preprocess_keypoints_2d(
        keypoints_2d=None,
        keypoint_masks_2d=None,
        keypoints_3d=keypoints_3d,
        model_input_keypoint_type=common
        .MODEL_INPUT_KEYPOINT_TYPE_3D_PROJECTION,
        keypoint_profile_2d=keypoint_profile_2d,
        keypoint_profile_3d=keypoint_profile_3d,
        azimuth_range=(math.pi / 2.0, math.pi / 2.0),
        elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
        roll_range=(math.pi, math.pi))

    # Note that the results here were copied from test output; this test is
    # mainly meant for protecting the executability testing batch mixing. The
    # actual projection accuracy is tested separately.
    expected_keypoints_2d = [
        [[[-0.08777856, -0.08777856], [0., 0.], [-0.08777856, -0.08777856],
          [-0.1613905, -0.1613905], [-0.22400928, -0.22400929], [0., 0.],
          [-0.08777856, -0.08777856], [-0.22400928, -0.22400929], [0., 0.],
          [-0.08777856, -0.08777856], [-0.1613905, -0.1613905],
          [-0.22400928, -0.22400929], [-0.22400928, -0.22400929]],
         [[-0.03107818, -0.03107818], [0.19100799, 0.19100802],
          [0.14718375, 0.14718376], [0.10647015, 0.10647015],
          [0.06854735, 0.06854735], [0.19100799, 0.19100802],
          [0.14718375, 0.14718376], [0.06854735, 0.06854735],
          [0.19100799, 0.19100802], [0.14718375, 0.14718376],
          [0.10647015, 0.10647015], [0.06854735, 0.06854735],
          [0.06854735, 0.06854735]]],
        [[[-0.0098562, -0.0098562], [0.17552288, 0.1755229],
          [0.16192658, 0.1619266], [0.14864118, 0.1486412],
          [0.13565616, 0.13565616], [0.17552288, 0.1755229],
          [0.16192658, 0.1619266], [0.13565616, 0.13565616],
          [0.17552288, 0.1755229], [0.16192658, 0.1619266],
          [0.14864118, 0.1486412], [0.13565616, 0.13565616],
          [0.13565616, 0.13565616]],
         [[-0.00734754, 0.02939016], [0.17376201, 0.17376202],
          [0.16365208, 0.16365209], [0.15371482, 0.15371484],
          [0.14394586, 0.14394587], [0.17376201, 0.17376202],
          [0.16365208, 0.16365209], [0.14394586, 0.14394587],
          [0.17376201, 0.17376202], [0.16365208, 0.16365209],
          [0.15371482, 0.15371484], [0.14394586, 0.14394587],
          [0.14394586, 0.14394587]]],
        [[[-0.08777856, -0.08777856], [0., 0.], [-0.08777856, -0.08777856],
          [-0.1613905, -0.1613905], [-0.22400928, -0.22400929], [0., 0.],
          [-0.08777856, -0.08777856], [-0.22400928, -0.22400929], [0., 0.],
          [-0.08777856, -0.08777856], [-0.1613905, -0.1613905],
          [-0.22400928, -0.22400929], [-0.22400928, -0.22400929]],
         [[-0.03107818, -0.03107818], [0.19100799, 0.19100802],
          [0.14718375, 0.14718376], [0.10647015, 0.10647015],
          [0.06854735, 0.06854735], [0.19100799, 0.19100802],
          [0.14718375, 0.14718376], [0.06854735, 0.06854735],
          [0.19100799, 0.19100802], [0.14718375, 0.14718376],
          [0.10647015, 0.10647015], [0.06854735, 0.06854735],
          [0.06854735, 0.06854735]]],
        [[[-0.0098562, -0.0098562], [0.17552288, 0.1755229],
          [0.16192658, 0.1619266], [0.14864118, 0.1486412],
          [0.13565616, 0.13565616], [0.17552288, 0.1755229],
          [0.16192658, 0.1619266], [0.13565616, 0.13565616],
          [0.17552288, 0.1755229], [0.16192658, 0.1619266],
          [0.14864118, 0.1486412], [0.13565616, 0.13565616],
          [0.13565616, 0.13565616]],
         [[-0.08777856, -0.08777856], [0., 0.], [-0.08777856, -0.08777856],
          [-0.1613905, -0.1613905], [-0.22400928, -0.22400929], [0., 0.],
          [-0.08777856, -0.08777856], [-0.22400928, -0.22400929], [0., 0.],
          [-0.08777856, -0.08777856], [-0.1613905, -0.1613905],
          [-0.22400928, -0.22400929], [-0.22400928, -0.22400929]]]
    ]

    self.assertAllClose(keypoints_2d, expected_keypoints_2d)

  def test_preprocess_keypoints_2d_with_input_and_projection(self):
    # Shape = [4, 2, 13, 2].
    keypoints_2d = tf.constant([
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
            [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
        ],
        [
            [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
            [[41.0, 42.0], [43.0, 44.0], [45.0, 46.0], [47.0, 48.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
        ],
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
            [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
        ],
        [
            [[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
            [[41.0, 42.0], [43.0, 44.0], [45.0, 46.0], [47.0, 48.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
        ],
    ])
    # Shape = [4, 2, 13].
    keypoint_masks_2d = tf.constant([
        [[
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11,
            0.12, 0.13
        ],
         [
             0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24,
             0.25, 0.26
         ]],
        [[
            0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
            0.38, 0.39
        ],
         [
             0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50,
             0.51, 0.52
         ]],
        [[
            0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63,
            0.64, 0.65
        ],
         [
             0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
             0.77, 0.78
         ]],
        [[
            0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
            0.90, 0.91
        ],
         [
             0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 0.99, 0.98,
             0.97, 0.96
         ]],
    ])
    # Shape = [4, 2, 17, 3].
    keypoints_3d = tf.constant([
        [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [7.0, 8.0, 9.0]],
         [[11.0, 12.0, 13.0], [13.0, 14.0, 15.0], [15.0, 16.0, 17.0],
          [17.0, 18.0, 19.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]],
        [[[31.0, 32.0, 33.0], [33.0, 34.0, 35.0], [35.0, 36.0, 37.0],
          [37.0, 38.0, 39.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]],
         [[41.0, 42.0, 43.0], [43.0, 44.0, 35.0], [45.0, 46.0, 47.0],
          [47.0, 48.0, 49.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]],
        [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [7.0, 8.0, 9.0]],
         [[11.0, 12.0, 13.0], [13.0, 14.0, 15.0], [15.0, 16.0, 17.0],
          [17.0, 18.0, 19.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]],
        [[[31.0, 32.0, 33.0], [33.0, 34.0, 35.0], [35.0, 36.0, 37.0],
          [37.0, 38.0, 39.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0],
          [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]],
         [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0],
          [7.0, 8.0, 9.0]]],
    ])
    # Shape = [4, 2].
    assignment = tf.constant([[True, True], [False, False], [True, False],
                              [False, True]])

    keypoint_profile_3d = (
        keypoint_profiles.create_keypoint_profile_or_die('LEGACY_3DH36M17'))
    keypoint_profile_2d = (
        keypoint_profiles.create_keypoint_profile_or_die('LEGACY_2DCOCO13'))
    keypoints_2d, _ = input_generator.preprocess_keypoints_2d(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d,
        model_input_keypoint_type=common
        .MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT_AND_3D_PROJECTION,
        keypoint_profile_2d=keypoint_profile_2d,
        keypoint_profile_3d=keypoint_profile_3d,
        azimuth_range=(math.pi / 2.0, math.pi / 2.0),
        elevation_range=(-math.pi / 2.0, -math.pi / 2.0),
        roll_range=(math.pi, math.pi),
        projection_mix_batch_assignment=assignment)

    # Note that the results here were copied from test output; this test is
    # mainly meant for protecting the executability testing batch mixing. The
    # actual projection accuracy is tested separately.
    expected_keypoints_2d = [
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
            [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
        ],
        [
            [[-0.0098562, -0.0098562], [0.17552288, 0.1755229],
             [0.16192658, 0.1619266], [0.14864118, 0.1486412],
             [0.13565616, 0.13565616], [0.17552288, 0.1755229],
             [0.16192658, 0.1619266], [0.13565616, 0.13565616],
             [0.17552288, 0.1755229], [0.16192658, 0.1619266],
             [0.14864118, 0.1486412], [0.13565616, 0.13565616],
             [0.13565616, 0.13565616]],
            [[-0.00734754, 0.02939016], [0.17376201, 0.17376202],
             [0.16365208, 0.16365209], [0.15371482, 0.15371484],
             [0.14394586, 0.14394587], [0.17376201, 0.17376202],
             [0.16365208, 0.16365209], [0.14394586, 0.14394587],
             [0.17376201, 0.17376202], [0.16365208, 0.16365209],
             [0.15371482, 0.15371484], [0.14394586, 0.14394587],
             [0.14394586, 0.14394587]],
        ],
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
            [[-0.03107818, -0.03107818], [0.19100799, 0.19100802],
             [0.14718375, 0.14718376], [0.10647015, 0.10647015],
             [0.06854735, 0.06854735], [0.19100799, 0.19100802],
             [0.14718375, 0.14718376], [0.06854735, 0.06854735],
             [0.19100799, 0.19100802], [0.14718375, 0.14718376],
             [0.10647015, 0.10647015], [0.06854735, 0.06854735],
             [0.06854735, 0.06854735]],
        ],
        [
            [[-0.0098562, -0.0098562], [0.17552288, 0.1755229],
             [0.16192658, 0.1619266], [0.14864118, 0.1486412],
             [0.13565616, 0.13565616], [0.17552288, 0.1755229],
             [0.16192658, 0.1619266], [0.13565616, 0.13565616],
             [0.17552288, 0.1755229], [0.16192658, 0.1619266],
             [0.14864118, 0.1486412], [0.13565616, 0.13565616],
             [0.13565616, 0.13565616]],
            [[41.0, 42.0], [43.0, 44.0], [45.0, 46.0], [47.0, 48.0], [1.0, 2.0],
             [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0],
             [5.0, 6.0], [7.0, 8.0], [1.0, 2.0]],
        ],
    ]

    self.assertAllClose(keypoints_2d, expected_keypoints_2d)

  def test_create_model_keypoints_2d_input(self):
    keypoint_profile_2d = keypoint_profiles.KeypointProfile2D(
        name='Dummy',
        keypoint_names=[('A', keypoint_profiles.LeftRightType.UNKNOWN),
                        ('B', keypoint_profiles.LeftRightType.UNKNOWN),
                        ('C', keypoint_profiles.LeftRightType.UNKNOWN)],
        offset_keypoint_names=['A', 'B'],
        scale_keypoint_name_pairs=[(['A', 'B'], ['B']), (['A'], ['B', 'C'])],
        segment_name_pairs=[],
        scale_distance_reduction_fn=tf.math.reduce_sum,
        scale_unit=1.0)

    # Shape = [2, 3, 2].
    keypoints_2d = tf.constant([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                                [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]])
    keypoint_masks_2d = tf.ones([2, 3], dtype=tf.float32)
    # Shape = [2, 6].
    features, side_outputs = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        normalize_keypoints_2d=True,
        keypoint_profile_2d=keypoint_profile_2d)

    sqrt_2 = 1.414213562
    self.assertAllClose(features,
                        [[
                            -0.25 / sqrt_2, -0.25 / sqrt_2, 0.25 / sqrt_2,
                            0.25 / sqrt_2, 0.75 / sqrt_2, 0.75 / sqrt_2
                        ],
                         [
                             -0.25 / sqrt_2, -0.25 / sqrt_2, 0.25 / sqrt_2,
                             0.25 / sqrt_2, 0.75 / sqrt_2, 0.75 / sqrt_2
                         ]])
    self.assertCountEqual(side_outputs.keys(), [
        'preprocessed_keypoints_2d', 'preprocessed_keypoint_masks_2d',
        'offset_points_2d', 'scale_distances_2d'
    ])
    self.assertAllClose(
        side_outputs['preprocessed_keypoints_2d'],
        [[[-0.25 / sqrt_2, -0.25 / sqrt_2], [0.25 / sqrt_2, 0.25 / sqrt_2],
          [0.75 / sqrt_2, 0.75 / sqrt_2]],
         [[-0.25 / sqrt_2, -0.25 / sqrt_2], [0.25 / sqrt_2, 0.25 / sqrt_2],
          [0.75 / sqrt_2, 0.75 / sqrt_2]]])
    self.assertAllClose(side_outputs['preprocessed_keypoint_masks_2d'],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    self.assertAllClose(side_outputs['offset_points_2d'],
                        [[[1.0, 2.0]], [[11.0, 12.0]]])
    self.assertAllClose(side_outputs['scale_distances_2d'],
                        [[[4.0 * sqrt_2]], [[4.0 * sqrt_2]]])

  def test_create_model_masked_keypoints_2d_input(self):
    keypoints_2d = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    keypoint_masks_2d = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    features, side_outputs = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        model_input_keypoint_mask_type=(
            common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS),
        normalize_keypoints_2d=False,
        rescale_features=True)

    expected_features = np.array([[1.0, 2.0, 0.0, 0.0, 5.0, 6.0],
                                  [0.0, 0.0, 0.0, 0.0, 11.0, 12.0]])
    expected_features *= np.array([[3.0 / 2.0], [3.0 / 1.0]])
    self.assertAllClose(features, expected_features)
    self.assertCountEqual(
        side_outputs.keys(),
        {'preprocessed_keypoints_2d', 'preprocessed_keypoint_masks_2d'})
    self.assertAllClose(side_outputs['preprocessed_keypoints_2d'],
                        [[[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]],
                         [[0.0, 0.0], [0.0, 0.0], [11.0, 12.0]]])
    self.assertAllClose(side_outputs['preprocessed_keypoint_masks_2d'],
                        [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

  def test_create_model_keypoints_with_masks_2d_input(self):
    keypoints_2d = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    keypoint_masks_2d = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    features, side_outputs = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        model_input_keypoint_mask_type=(
            common.MODEL_INPUT_KEYPOINT_MASK_TYPE_AS_INPUT),
        normalize_keypoints_2d=False,
        rescale_features=True)

    expected_features = np.array(
        [[1.0, 2.0, 1.0, 3.0, 4.0, 0.0, 5.0, 6.0, 1.0],
         [7.0, 8.0, 0.0, 9.0, 10.0, 0.0, 11.0, 12.0, 1.0]])
    expected_features *= np.array([[3.0 / 2.0], [3.0 / 1.0]])
    self.assertAllClose(features, expected_features)
    self.assertCountEqual(
        side_outputs.keys(),
        {'preprocessed_keypoints_2d', 'preprocessed_keypoint_masks_2d'})
    self.assertAllClose(side_outputs['preprocessed_keypoints_2d'],
                        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                         [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    self.assertAllClose(side_outputs['preprocessed_keypoint_masks_2d'],
                        [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

  def test_create_model_masked_keypoints_with_masks_2d_input(self):
    keypoints_2d = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    keypoint_masks_2d = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    features, side_outputs = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        model_input_keypoint_mask_type=(
            common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT),
        normalize_keypoints_2d=False,
        rescale_features=True)

    expected_features = np.array(
        [[1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 5.0, 6.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 1.0]])
    expected_features *= np.array([[3.0 / 2.0], [3.0 / 1.0]])
    self.assertAllClose(features, expected_features)
    self.assertCountEqual(
        side_outputs.keys(),
        {'preprocessed_keypoints_2d', 'preprocessed_keypoint_masks_2d'})
    self.assertAllClose(side_outputs['preprocessed_keypoints_2d'],
                        [[[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]],
                         [[0.0, 0.0], [0.0, 0.0], [11.0, 12.0]]])
    self.assertAllClose(side_outputs['preprocessed_keypoint_masks_2d'],
                        [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

  def test_create_model_input_with_instance_keypoint_dropout(self):
    # Shape = [2, 3, 4, 2].
    keypoints_2d = tf.constant([
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
         [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
         [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]]],
        [[[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0]],
         [[41.0, 42.0], [43.0, 44.0], [45.0, 46.0], [47.0, 48.0]],
         [[51.0, 52.0], [53.0, 54.0], [55.0, 56.0], [57.0, 58.0]]],
    ])
    # Shape = [2, 3, 4].
    keypoint_masks_2d = tf.constant([[
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ], [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]])
    # Shape = [2, 3, 12].
    features, side_outputs = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        model_input_keypoint_mask_type=(
            common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT),
        normalize_keypoints_2d=False,
        keypoint_dropout_probs=(0.1, 0.8),
        rescale_features=True,
        seed=0)
    self.assertCountEqual(
        side_outputs.keys(),
        {'preprocessed_keypoints_2d', 'preprocessed_keypoint_masks_2d'})

    expected_features = np.array([
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 1.0, 7.0, 8.0, 1.0],
         [11.0, 12.0, 1.0, 13.0, 14.0, 1.0, 15.0, 16.0, 1.0, 17.0, 18.0, 1.0],
         [21.0, 22.0, 1.0, 23.0, 24.0, 1.0, 25.0, 26.0, 1.0, 27.0, 28.0, 1.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.0, 36.0, 1.0, 0.0, 0.0, 0.0],
         [41.0, 42.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 55.0, 56.0, 1.0, 57.0, 58.0, 1.0]],
    ])
    expected_features *= np.array([[[4.0 / 2.0], [4.0 / 4.0], [4.0 / 4.0]],
                                   [[4.0 / 1.0], [4.0 / 1.0], [4.0 / 2.0]]])

    self.assertAllClose(features, expected_features)
    self.assertAllClose(
        side_outputs['preprocessed_keypoints_2d'],
        [[[[0.0, 0.0], [0.0, 0.0], [5.0, 6.0], [7.0, 8.0]],
          [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
          [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]]],
         [[[0.0, 0.0], [0.0, 0.0], [35.0, 36.0], [0.0, 0.0]],
          [[41.0, 42.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
          [[0.0, 0.0], [0.0, 0.0], [55.0, 56.0], [57.0, 58.0]]]])
    self.assertAllClose(
        side_outputs['preprocessed_keypoint_masks_2d'],
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
         [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])

  def test_create_model_sequence_input_with_instance_keypoint_dropout(self):
    # Shape = [2, 3, 4, 2].
    keypoints_2d = tf.constant([
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
         [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
         [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]]],
        [[[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0]],
         [[41.0, 42.0], [43.0, 44.0], [45.0, 46.0], [47.0, 48.0]],
         [[51.0, 52.0], [53.0, 54.0], [55.0, 56.0], [57.0, 58.0]]],
    ])
    # Shape = [2, 3, 4].
    keypoint_masks_2d = tf.constant([[
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ], [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]])
    # Shape = [2, 3, 12].
    features, side_outputs = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        model_input_keypoint_mask_type=(
            common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT),
        normalize_keypoints_2d=False,
        keypoint_dropout_probs=(0.1, 0.8, 0.5),
        rescale_features=True,
        sequential_inputs=True,
        seed=0)
    self.assertCountEqual(
        side_outputs.keys(),
        {'preprocessed_keypoints_2d', 'preprocessed_keypoint_masks_2d'})
    self.assertAllEqual(features.shape.as_list(), [2, 3, 12])
    self.assertAllEqual(
        side_outputs['preprocessed_keypoints_2d'].shape.as_list(), [2, 3, 4, 2])
    self.assertAllEqual(
        side_outputs['preprocessed_keypoint_masks_2d'].shape.as_list(),
        [2, 3, 4])
    self.assertAllClose(side_outputs['preprocessed_keypoints_2d'],
                        [[[[0., 0.], [0., 0.], [5., 6.], [7., 8.]],
                          [[11., 12.], [13., 14.], [15., 16.], [17., 18.]],
                          [[21., 22.], [23., 24.], [25., 26.], [27., 28.]]],
                         [[[31., 32.], [33., 34.], [35., 36.], [0., 0.]],
                          [[41., 42.], [43., 44.], [45., 46.], [0., 0.]],
                          [[0., 0.], [0., 0.], [55., 56.], [57., 58.]]]])
    self.assertAllClose(
        side_outputs['preprocessed_keypoint_masks_2d'],
        [[[0., 0., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
         [[1., 1., 1., 0.], [1., 1., 1., 0.], [0., 0., 1., 1.]]])

  def test_create_model_input_with_on_mask_for_non_anchors(self):
    # Shape = [2, 3, 4, 2].
    keypoints_2d = tf.constant([
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
         [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
         [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]]],
        [[[31.0, 32.0], [33.0, 34.0], [35.0, 36.0], [37.0, 38.0]],
         [[41.0, 42.0], [43.0, 44.0], [45.0, 46.0], [47.0, 48.0]],
         [[51.0, 52.0], [53.0, 54.0], [55.0, 56.0], [57.0, 58.0]]],
    ])
    # Shape = [2, 3, 4].
    keypoint_masks_2d = tf.constant([[
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ], [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]])
    # Shape = [2, 3, 12].
    features, side_outputs = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        model_input_keypoint_mask_type=(
            common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT),
        normalize_keypoints_2d=False,
        keypoint_dropout_probs=(0.1, 0.8),
        set_on_mask_for_non_anchors=True,
        seed=0)
    self.assertCountEqual(
        side_outputs.keys(),
        {'preprocessed_keypoints_2d', 'preprocessed_keypoint_masks_2d'})

    expected_features = np.array([
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 1.0, 7.0, 8.0, 1.0],
         [11.0, 12.0, 1.0, 13.0, 14.0, 1.0, 15.0, 16.0, 1.0, 17.0, 18.0, 1.0],
         [21.0, 22.0, 1.0, 23.0, 24.0, 1.0, 25.0, 26.0, 1.0, 27.0, 28.0, 1.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.0, 36.0, 1.0, 0.0, 0.0, 0.0],
         [41.0, 42.0, 1.0, 43.0, 44.0, 1.0, 45.0, 46.0, 1.0, 47.0, 48.0, 1.0],
         [51.0, 52.0, 1.0, 53.0, 54.0, 1.0, 55.0, 56.0, 1.0, 57.0, 58.0, 1.0]],
    ])

    self.assertAllClose(features, expected_features)
    self.assertAllClose(
        side_outputs['preprocessed_keypoints_2d'],
        [[[[0.0, 0.0], [0.0, 0.0], [5.0, 6.0], [7.0, 8.0]],
          [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
          [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]]],
         [[[0.0, 0.0], [0.0, 0.0], [35.0, 36.0], [0.0, 0.0]],
          [[41.0, 42.0], [43.0, 44.0], [45.0, 46.0], [47.0, 48.0]],
          [[51.0, 52.0], [53.0, 54.0], [55.0, 56.0], [57.0, 58.0]]]])
    self.assertAllClose(
        side_outputs['preprocessed_keypoint_masks_2d'],
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
         [[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])

  def test_create_model_input_with_forced_on_masks(self):
    # Shape = [2, 13, 2].
    keypoints_2d = tf.ones_like([2, 13, 2], dtype=tf.float32)
    # Shape = [2, 13].
    keypoint_masks_2d = tf.constant(
        [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    # Shape = [2, 39].
    _, side_outputs = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        model_input_keypoint_mask_type=(
            common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT),
        normalize_keypoints_2d=False,
        keypoint_profile_2d=keypoint_profiles.Std13KeypointProfile2D(),
        keypoint_dropout_probs=(0.5, 0.5),
        forced_mask_on_part_names=['HEAD', 'LEFT_SHOULDER', 'LEFT_ANKLE'],
        seed=0)
    self.assertCountEqual(
        side_outputs.keys(),
        {'preprocessed_keypoints_2d', 'preprocessed_keypoint_masks_2d'})
    expected_preprocessed_keypoint_masks_2d = np.array(
        [[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
    self.assertAllClose(side_outputs['preprocessed_keypoint_masks_2d'],
                        expected_preprocessed_keypoint_masks_2d)

  def test_create_model_input_with_forced_off_masks(self):
    # Shape = [2, 13, 2].
    keypoints_2d = tf.ones_like([2, 13, 2], dtype=tf.float32)
    # Shape = [2, 13].
    keypoint_masks_2d = tf.constant(
        [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    # Shape = [2, 39].
    _, side_outputs = input_generator.create_model_input(
        keypoints_2d,
        keypoint_masks_2d,
        keypoints_3d=None,
        model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
        model_input_keypoint_mask_type=(
            common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT),
        normalize_keypoints_2d=False,
        keypoint_profile_2d=keypoint_profiles.Std13KeypointProfile2D(),
        keypoint_dropout_probs=(0.5, 0.5),
        forced_mask_off_part_names=['HEAD', 'LEFT_SHOULDER', 'LEFT_ANKLE'],
        seed=0)
    self.assertCountEqual(
        side_outputs.keys(),
        {'preprocessed_keypoints_2d', 'preprocessed_keypoint_masks_2d'})
    expected_preprocessed_keypoint_masks_2d = np.array(
        [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    self.assertAllClose(side_outputs['preprocessed_keypoint_masks_2d'],
                        expected_preprocessed_keypoint_masks_2d)


if __name__ == '__main__':
  tf.test.main()
