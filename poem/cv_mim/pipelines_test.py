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

"""Tests for pipeline utility functions."""

import os

from absl import flags
import tensorflow as tf

from poem.core import common
from poem.core import keypoint_profiles
from poem.cv_mim import pipelines

FLAGS = flags.FLAGS


class PipelinesTest(tf.test.TestCase):

  def test_create_dataset_from_tables(self):
    testdata_dir = 'poem/testdata'  # Assume $PWD == "google_research/".
    dataset = pipelines.create_dataset_from_tables(
        [os.path.join(FLAGS.test_srcdir, testdata_dir, 'tfe-2.tfrecords')],
        batch_sizes=[4],
        num_instances_per_record=2,
        shuffle=True,
        num_epochs=None,
        keypoint_names_3d=keypoint_profiles.create_keypoint_profile_or_die(
            'LEGACY_3DH36M17').keypoint_names,
        keypoint_names_2d=keypoint_profiles.create_keypoint_profile_or_die(
            'LEGACY_2DCOCO13').keypoint_names,
        seed=0)

    inputs = list(dataset.take(1))[0]
    self.assertCountEqual(inputs.keys(), [
        'image_sizes', 'keypoints_2d', 'keypoint_scores_2d', 'keypoints_3d'
    ])
    self.assertEqual(inputs['image_sizes'].shape, [4, 2, 2])
    self.assertEqual(inputs['keypoints_2d'].shape, [4, 2, 13, 2])
    self.assertEqual(inputs['keypoint_scores_2d'].shape, [4, 2, 13])
    self.assertEqual(inputs['keypoints_3d'].shape, [4, 2, 17, 3])

  def test_create_model_input(self):
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
    keypoint_scores_2d = tf.ones(keypoints_2d.shape, dtype=tf.float32)

    inputs = {
        common.KEY_KEYPOINTS_2D: keypoints_2d,
        common.KEY_KEYPOINTS_3D: None,
        common.KEY_KEYPOINT_SCORES_2D: keypoint_scores_2d,
        common.KEY_IMAGE_SIZES: tf.ones((1, 2))
    }
    features, side_outputs = pipelines.create_model_input(
        inputs,
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
        'offset_points_2d', 'scale_distances_2d', 'keypoints_2d',
        'keypoint_masks_2d'
    ])
    self.assertAllClose(side_outputs['keypoints_2d'],
                        [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                         [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]])
    self.assertAllClose(
        side_outputs['preprocessed_keypoints_2d'],
        [[[-0.25 / sqrt_2, -0.25 / sqrt_2], [0.25 / sqrt_2, 0.25 / sqrt_2],
          [0.75 / sqrt_2, 0.75 / sqrt_2]],
         [[-0.25 / sqrt_2, -0.25 / sqrt_2], [0.25 / sqrt_2, 0.25 / sqrt_2],
          [0.75 / sqrt_2, 0.75 / sqrt_2]]])
    self.assertAllClose(side_outputs['keypoint_masks_2d'],
                        [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                         [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
    self.assertAllClose(side_outputs['preprocessed_keypoint_masks_2d'],
                        [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                         [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
    self.assertAllClose(side_outputs['offset_points_2d'],
                        [[[1.0, 2.0]], [[11.0, 12.0]]])
    self.assertAllClose(side_outputs['scale_distances_2d'],
                        [[[4.0 * sqrt_2]], [[4.0 * sqrt_2]]])


if __name__ == '__main__':
  tf.test.main()
