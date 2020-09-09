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

"""Tests for keypoint profiles.

Note: Most of the tests in this file are in one-to-one correspondence with tests
in:
//photos/vision/human_sensing/pose_estimation/e3d/utils/keypoint_profiles_test.cc.

Updates should be synced between the two files.

"""

import tensorflow as tf

from poem.core import keypoint_profiles


class KeypointProfileTest(tf.test.TestCase):

  def test_std16_keypoint_profile_3d_is_correct(self):
    profile = keypoint_profiles.create_keypoint_profile_or_die('3DSTD16')
    self.assertEqual(profile.name, '3DSTD16')
    self.assertEqual(profile.keypoint_dim, 3)
    self.assertEqual(profile.keypoint_num, 16)
    self.assertEqual(profile.keypoint_names, [
        'HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'SPINE', 'PELVIS',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
        'RIGHT_ANKLE'
    ])
    self.assertEqual(
        profile.keypoint_left_right_type(1),
        keypoint_profiles.LeftRightType.CENTRAL)
    self.assertEqual(
        profile.segment_left_right_type(1, 2),
        keypoint_profiles.LeftRightType.LEFT)
    self.assertEqual(profile.offset_keypoint_index, [9])
    self.assertEqual(profile.scale_keypoint_index_pairs, [([1], [8]),
                                                          ([8], [9])])
    self.assertEqual(profile.keypoint_index('LEFT_SHOULDER'), 2)
    self.assertEqual(profile.keypoint_index('dummy'), -1)
    self.assertEqual(profile.segment_index_pairs, [([0], [1]), ([1], [2]),
                                                   ([1], [3]), ([1], [8]),
                                                   ([2], [4]), ([3], [5]),
                                                   ([4], [6]), ([5], [7]),
                                                   ([8], [9]), ([9], [10]),
                                                   ([9], [11]), ([10], [12]),
                                                   ([11], [13]), ([12], [14]),
                                                   ([13], [15])])
    self.assertAllEqual(profile.keypoint_affinity_matrix, [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    ])
    self.assertEqual(profile.head_keypoint_index, [0])
    self.assertEqual(profile.neck_keypoint_index, [1])
    self.assertEqual(profile.left_shoulder_keypoint_index, [2])
    self.assertEqual(profile.right_shoulder_keypoint_index, [3])
    self.assertEqual(profile.left_elbow_keypoint_index, [4])
    self.assertEqual(profile.right_elbow_keypoint_index, [5])
    self.assertEqual(profile.left_wrist_keypoint_index, [6])
    self.assertEqual(profile.right_wrist_keypoint_index, [7])
    self.assertEqual(profile.spine_keypoint_index, [8])
    self.assertEqual(profile.pelvis_keypoint_index, [9])
    self.assertEqual(profile.left_hip_keypoint_index, [10])
    self.assertEqual(profile.right_hip_keypoint_index, [11])
    self.assertEqual(profile.left_knee_keypoint_index, [12])
    self.assertEqual(profile.right_knee_keypoint_index, [13])
    self.assertEqual(profile.left_ankle_keypoint_index, [14])
    self.assertEqual(profile.right_ankle_keypoint_index, [15])

  def test_std13_keypoint_profile_3d_is_correct(self):
    profile = keypoint_profiles.create_keypoint_profile_or_die('3DSTD13')
    self.assertEqual(profile.name, '3DSTD13')
    self.assertEqual(profile.keypoint_dim, 3)
    self.assertEqual(profile.keypoint_num, 13)
    self.assertEqual(profile.keypoint_names, [
        'HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE',
        'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'
    ])
    self.assertEqual(
        profile.keypoint_left_right_type(1),
        keypoint_profiles.LeftRightType.LEFT)
    self.assertEqual(
        profile.segment_left_right_type(1, 2),
        keypoint_profiles.LeftRightType.CENTRAL)
    self.assertEqual(profile.offset_keypoint_index, [7, 8])
    self.assertEqual(profile.scale_keypoint_index_pairs, [([1, 2], [7, 8])])
    self.assertEqual(profile.keypoint_index('LEFT_SHOULDER'), 1)
    self.assertEqual(profile.keypoint_index('dummy'), -1)
    self.assertEqual(profile.segment_index_pairs, [([0], [1, 2]), ([1, 2], [1]),
                                                   ([1, 2], [2]),
                                                   ([1, 2], [1, 2, 7, 8]),
                                                   ([1], [3]), ([2], [4]),
                                                   ([3], [5]), ([4], [6]),
                                                   ([1, 2, 7, 8], [7, 8]),
                                                   ([7, 8], [7]), ([7, 8], [8]),
                                                   ([7], [9]), ([8], [10]),
                                                   ([9], [11]), ([10], [12])])
    self.assertAllEqual(profile.keypoint_affinity_matrix, [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    ])
    self.assertEqual(profile.head_keypoint_index, [0])
    self.assertEqual(profile.neck_keypoint_index, [1, 2])
    self.assertEqual(profile.left_shoulder_keypoint_index, [1])
    self.assertEqual(profile.right_shoulder_keypoint_index, [2])
    self.assertEqual(profile.left_elbow_keypoint_index, [3])
    self.assertEqual(profile.right_elbow_keypoint_index, [4])
    self.assertEqual(profile.left_wrist_keypoint_index, [5])
    self.assertEqual(profile.right_wrist_keypoint_index, [6])
    self.assertEqual(profile.spine_keypoint_index, [1, 2, 7, 8])
    self.assertEqual(profile.pelvis_keypoint_index, [7, 8])
    self.assertEqual(profile.left_hip_keypoint_index, [7])
    self.assertEqual(profile.right_hip_keypoint_index, [8])
    self.assertEqual(profile.left_knee_keypoint_index, [9])
    self.assertEqual(profile.right_knee_keypoint_index, [10])
    self.assertEqual(profile.left_ankle_keypoint_index, [11])
    self.assertEqual(profile.right_ankle_keypoint_index, [12])
    self.assertEqual(profile.standard_part_names, [
        'HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'SPINE', 'PELVIS',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
        'RIGHT_ANKLE'
    ])
    self.assertEqual(profile.get_standard_part_index('HEAD'), [0])
    self.assertEqual(profile.get_standard_part_index('NECK'), [1, 2])
    self.assertEqual(profile.get_standard_part_index('LEFT_SHOULDER'), [1])
    self.assertEqual(profile.get_standard_part_index('RIGHT_SHOULDER'), [2])
    self.assertEqual(profile.get_standard_part_index('LEFT_ELBOW'), [3])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ELBOW'), [4])
    self.assertEqual(profile.get_standard_part_index('LEFT_WRIST'), [5])
    self.assertEqual(profile.get_standard_part_index('RIGHT_WRIST'), [6])
    self.assertEqual(profile.get_standard_part_index('SPINE'), [1, 2, 7, 8])
    self.assertEqual(profile.get_standard_part_index('PELVIS'), [7, 8])
    self.assertEqual(profile.get_standard_part_index('LEFT_HIP'), [7])
    self.assertEqual(profile.get_standard_part_index('RIGHT_HIP'), [8])
    self.assertEqual(profile.get_standard_part_index('LEFT_KNEE'), [9])
    self.assertEqual(profile.get_standard_part_index('RIGHT_KNEE'), [10])
    self.assertEqual(profile.get_standard_part_index('LEFT_ANKLE'), [11])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ANKLE'), [12])

  def test_legacy_h36m17_keypoint_profile_3d_is_correct(self):
    profile = keypoint_profiles.create_keypoint_profile_or_die(
        'LEGACY_3DH36M17')
    self.assertEqual(profile.name, 'LEGACY_3DH36M17')
    self.assertEqual(profile.keypoint_dim, 3)
    self.assertEqual(profile.keypoint_num, 17)
    self.assertEqual(profile.keypoint_names, [
        'Hip', 'Head', 'Neck/Nose', 'Thorax', 'LShoulder', 'RShoulder',
        'LElbow', 'RElbow', 'LWrist', 'RWrist', 'Spine', 'LHip', 'RHip',
        'LKnee', 'RKnee', 'LFoot', 'RFoot'
    ])
    self.assertEqual(
        profile.keypoint_left_right_type(1),
        keypoint_profiles.LeftRightType.CENTRAL)
    self.assertEqual(
        profile.segment_left_right_type(1, 4),
        keypoint_profiles.LeftRightType.LEFT)
    self.assertEqual(profile.offset_keypoint_index, [0])
    self.assertEqual(profile.scale_keypoint_index_pairs, [([0], [10]),
                                                          ([10], [3])])
    self.assertEqual(profile.keypoint_index('Thorax'), 3)
    self.assertEqual(profile.keypoint_index('dummy'), -1)
    self.assertEqual(profile.segment_index_pairs, [([0], [10]), ([0], [11]),
                                                   ([0], [12]), ([10], [3]),
                                                   ([11], [13]), ([12], [14]),
                                                   ([13], [15]), ([14], [16]),
                                                   ([3], [2]), ([3], [4]),
                                                   ([3], [5]), ([2], [1]),
                                                   ([4], [6]), ([5], [7]),
                                                   ([6], [8]), ([7], [9])])
    self.assertAllEqual(profile.keypoint_affinity_matrix, [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    ])
    self.assertEqual(profile.head_keypoint_index, [1])
    self.assertEqual(profile.neck_keypoint_index, [3])
    self.assertEqual(profile.left_shoulder_keypoint_index, [4])
    self.assertEqual(profile.right_shoulder_keypoint_index, [5])
    self.assertEqual(profile.left_elbow_keypoint_index, [6])
    self.assertEqual(profile.right_elbow_keypoint_index, [7])
    self.assertEqual(profile.left_wrist_keypoint_index, [8])
    self.assertEqual(profile.right_wrist_keypoint_index, [9])
    self.assertEqual(profile.spine_keypoint_index, [10])
    self.assertEqual(profile.pelvis_keypoint_index, [0])
    self.assertEqual(profile.left_hip_keypoint_index, [11])
    self.assertEqual(profile.right_hip_keypoint_index, [12])
    self.assertEqual(profile.left_knee_keypoint_index, [13])
    self.assertEqual(profile.right_knee_keypoint_index, [14])
    self.assertEqual(profile.left_ankle_keypoint_index, [15])
    self.assertEqual(profile.right_ankle_keypoint_index, [16])
    self.assertEqual(profile.standard_part_names, [
        'HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'SPINE', 'PELVIS',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
        'RIGHT_ANKLE'
    ])
    self.assertEqual(profile.get_standard_part_index('HEAD'), [1])
    self.assertEqual(profile.get_standard_part_index('NECK'), [3])
    self.assertEqual(profile.get_standard_part_index('LEFT_SHOULDER'), [4])
    self.assertEqual(profile.get_standard_part_index('RIGHT_SHOULDER'), [5])
    self.assertEqual(profile.get_standard_part_index('LEFT_ELBOW'), [6])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ELBOW'), [7])
    self.assertEqual(profile.get_standard_part_index('LEFT_WRIST'), [8])
    self.assertEqual(profile.get_standard_part_index('RIGHT_WRIST'), [9])
    self.assertEqual(profile.get_standard_part_index('SPINE'), [10])
    self.assertEqual(profile.get_standard_part_index('PELVIS'), [0])
    self.assertEqual(profile.get_standard_part_index('LEFT_HIP'), [11])
    self.assertEqual(profile.get_standard_part_index('RIGHT_HIP'), [12])
    self.assertEqual(profile.get_standard_part_index('LEFT_KNEE'), [13])
    self.assertEqual(profile.get_standard_part_index('RIGHT_KNEE'), [14])
    self.assertEqual(profile.get_standard_part_index('LEFT_ANKLE'), [15])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ANKLE'), [16])

  def test_legacy_h36m13_keypoint_profile_3d_is_correct(self):
    profile = keypoint_profiles.create_keypoint_profile_or_die(
        'LEGACY_3DH36M13')
    self.assertEqual(profile.name, 'LEGACY_3DH36M13')
    self.assertEqual(profile.keypoint_dim, 3)
    self.assertEqual(profile.keypoint_num, 13)
    self.assertEqual(profile.keypoint_names, [
        'Head', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist',
        'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LFoot', 'RFoot'
    ])
    self.assertEqual(
        profile.keypoint_left_right_type(1),
        keypoint_profiles.LeftRightType.LEFT)
    self.assertEqual(
        profile.segment_left_right_type(1, 4),
        keypoint_profiles.LeftRightType.CENTRAL)
    self.assertEqual(profile.offset_keypoint_index, [7])
    self.assertEqual(profile.scale_keypoint_index_pairs, [([7, 8], [1, 2])])
    self.assertEqual(profile.keypoint_index('LShoulder'), 1)
    self.assertEqual(profile.keypoint_index('dummy'), -1)
    self.assertEqual(profile.segment_index_pairs, [([7, 8], [1, 2]),
                                                   ([7, 8], [7]), ([7, 8], [8]),
                                                   ([7], [9]), ([8], [10]),
                                                   ([9], [11]), ([10], [12]),
                                                   ([1, 2], [0]), ([1, 2], [1]),
                                                   ([1, 2], [2]), ([1], [3]),
                                                   ([2], [4]), ([3], [5]),
                                                   ([4], [6])])
    self.assertAllEqual(profile.keypoint_affinity_matrix, [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    ])
    self.assertEqual(profile.head_keypoint_index, [0])
    self.assertEqual(profile.neck_keypoint_index, [1, 2])
    self.assertEqual(profile.left_shoulder_keypoint_index, [1])
    self.assertEqual(profile.right_shoulder_keypoint_index, [2])
    self.assertEqual(profile.left_elbow_keypoint_index, [3])
    self.assertEqual(profile.right_elbow_keypoint_index, [4])
    self.assertEqual(profile.left_wrist_keypoint_index, [5])
    self.assertEqual(profile.right_wrist_keypoint_index, [6])
    self.assertEqual(profile.spine_keypoint_index, [1, 2, 7, 8])
    self.assertEqual(profile.pelvis_keypoint_index, [7, 8])
    self.assertEqual(profile.left_hip_keypoint_index, [7])
    self.assertEqual(profile.right_hip_keypoint_index, [8])
    self.assertEqual(profile.left_knee_keypoint_index, [9])
    self.assertEqual(profile.right_knee_keypoint_index, [10])
    self.assertEqual(profile.left_ankle_keypoint_index, [11])
    self.assertEqual(profile.right_ankle_keypoint_index, [12])
    self.assertEqual(profile.standard_part_names, [
        'HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'SPINE', 'PELVIS',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
        'RIGHT_ANKLE'
    ])
    self.assertEqual(profile.get_standard_part_index('HEAD'), [0])
    self.assertEqual(profile.get_standard_part_index('NECK'), [1, 2])
    self.assertEqual(profile.get_standard_part_index('LEFT_SHOULDER'), [1])
    self.assertEqual(profile.get_standard_part_index('RIGHT_SHOULDER'), [2])
    self.assertEqual(profile.get_standard_part_index('LEFT_ELBOW'), [3])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ELBOW'), [4])
    self.assertEqual(profile.get_standard_part_index('LEFT_WRIST'), [5])
    self.assertEqual(profile.get_standard_part_index('RIGHT_WRIST'), [6])
    self.assertEqual(profile.get_standard_part_index('SPINE'), [1, 2, 7, 8])
    self.assertEqual(profile.get_standard_part_index('PELVIS'), [7, 8])
    self.assertEqual(profile.get_standard_part_index('LEFT_HIP'), [7])
    self.assertEqual(profile.get_standard_part_index('RIGHT_HIP'), [8])
    self.assertEqual(profile.get_standard_part_index('LEFT_KNEE'), [9])
    self.assertEqual(profile.get_standard_part_index('RIGHT_KNEE'), [10])
    self.assertEqual(profile.get_standard_part_index('LEFT_ANKLE'), [11])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ANKLE'), [12])

  def test_legacy_mpii3dhp17_keypoint_profile_3d_is_correct(self):
    profile = keypoint_profiles.create_keypoint_profile_or_die(
        'LEGACY_3DMPII3DHP17')
    self.assertEqual(profile.name, 'LEGACY_3DMPII3DHP17')
    self.assertEqual(profile.keypoint_dim, 3)
    self.assertEqual(profile.keypoint_num, 17)
    self.assertEqual(profile.keypoint_names, [
        'pelvis', 'head', 'neck', 'head_top', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'spine',
        'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
        'right_ankle'
    ])
    self.assertEqual(
        profile.keypoint_left_right_type(1),
        keypoint_profiles.LeftRightType.CENTRAL)
    self.assertEqual(
        profile.segment_left_right_type(1, 4),
        keypoint_profiles.LeftRightType.LEFT)
    self.assertEqual(profile.offset_keypoint_index, [0])
    self.assertEqual(profile.scale_keypoint_index_pairs, [([0], [10]),
                                                          ([10], [2])])
    self.assertEqual(profile.keypoint_index('left_shoulder'), 4)
    self.assertEqual(profile.keypoint_index('dummy'), -1)
    self.assertEqual(profile.segment_index_pairs, [([0], [10]), ([0], [11]),
                                                   ([0], [12]), ([10], [2]),
                                                   ([11], [13]), ([12], [14]),
                                                   ([13], [15]), ([14], [16]),
                                                   ([2], [1]), ([2], [4]),
                                                   ([2], [5]), ([1], [3]),
                                                   ([4], [6]), ([5], [7]),
                                                   ([6], [8]), ([7], [9])])
    self.assertAllEqual(profile.keypoint_affinity_matrix, [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    ])
    self.assertEqual(profile.head_keypoint_index, [1])
    self.assertEqual(profile.neck_keypoint_index, [2])
    self.assertEqual(profile.left_shoulder_keypoint_index, [4])
    self.assertEqual(profile.right_shoulder_keypoint_index, [5])
    self.assertEqual(profile.left_elbow_keypoint_index, [6])
    self.assertEqual(profile.right_elbow_keypoint_index, [7])
    self.assertEqual(profile.left_wrist_keypoint_index, [8])
    self.assertEqual(profile.right_wrist_keypoint_index, [9])
    self.assertEqual(profile.spine_keypoint_index, [10])
    self.assertEqual(profile.pelvis_keypoint_index, [0])
    self.assertEqual(profile.left_hip_keypoint_index, [11])
    self.assertEqual(profile.right_hip_keypoint_index, [12])
    self.assertEqual(profile.left_knee_keypoint_index, [13])
    self.assertEqual(profile.right_knee_keypoint_index, [14])
    self.assertEqual(profile.left_ankle_keypoint_index, [15])
    self.assertEqual(profile.right_ankle_keypoint_index, [16])
    self.assertEqual(profile.standard_part_names, [
        'HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'SPINE', 'PELVIS',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
        'RIGHT_ANKLE'
    ])
    self.assertEqual(profile.get_standard_part_index('HEAD'), [1])
    self.assertEqual(profile.get_standard_part_index('NECK'), [2])
    self.assertEqual(profile.get_standard_part_index('LEFT_SHOULDER'), [4])
    self.assertEqual(profile.get_standard_part_index('RIGHT_SHOULDER'), [5])
    self.assertEqual(profile.get_standard_part_index('LEFT_ELBOW'), [6])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ELBOW'), [7])
    self.assertEqual(profile.get_standard_part_index('LEFT_WRIST'), [8])
    self.assertEqual(profile.get_standard_part_index('RIGHT_WRIST'), [9])
    self.assertEqual(profile.get_standard_part_index('SPINE'), [10])
    self.assertEqual(profile.get_standard_part_index('PELVIS'), [0])
    self.assertEqual(profile.get_standard_part_index('LEFT_HIP'), [11])
    self.assertEqual(profile.get_standard_part_index('RIGHT_HIP'), [12])
    self.assertEqual(profile.get_standard_part_index('LEFT_KNEE'), [13])
    self.assertEqual(profile.get_standard_part_index('RIGHT_KNEE'), [14])
    self.assertEqual(profile.get_standard_part_index('LEFT_ANKLE'), [15])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ANKLE'), [16])

  def _test_std13_keypoint_profile_2d_is_correct(self, name):
    profile = keypoint_profiles.create_keypoint_profile_or_die(name)
    self.assertEqual(profile.name, name)
    self.assertEqual(profile.keypoint_dim, 2)
    self.assertEqual(profile.keypoint_num, 13)
    self.assertEqual(profile.keypoint_names, [
        'NOSE_TIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'
    ])
    self.assertEqual(
        profile.keypoint_left_right_type(1),
        keypoint_profiles.LeftRightType.LEFT)
    self.assertEqual(
        profile.segment_left_right_type(1, 2),
        keypoint_profiles.LeftRightType.CENTRAL)
    self.assertEqual(profile.offset_keypoint_index, [7, 8])
    self.assertEqual(profile.scale_keypoint_index_pairs,
                     [([1], [2]), ([1], [7]), ([1], [8]), ([2], [7]),
                      ([2], [8]), ([7], [8])])
    self.assertEqual(profile.keypoint_index('LEFT_SHOULDER'), 1)
    self.assertEqual(profile.keypoint_index('dummy'), -1)
    self.assertEqual(profile.segment_index_pairs, [([0], [1]), ([0], [2]),
                                                   ([1], [2]), ([1], [3]),
                                                   ([2], [4]), ([3], [5]),
                                                   ([4], [6]), ([1], [7]),
                                                   ([2], [8]), ([7], [8]),
                                                   ([7], [9]), ([8], [10]),
                                                   ([9], [11]), ([10], [12])])
    self.assertAllEqual(profile.keypoint_affinity_matrix, [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    ])
    self.assertEqual(
        profile.compatible_keypoint_name_dict, {
            '3DSTD16': [
                'HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP',
                'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                'RIGHT_ANKLE'
            ],
            '3DSTD13': [
                'HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP',
                'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                'RIGHT_ANKLE'
            ],
            'LEGACY_3DH36M17': [
                'Head', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist',
                'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LFoot', 'RFoot'
            ],
            'LEGACY_3DMPII3DHP17': [
                'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
                'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
                'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                'right_ankle'
            ],
        })
    self.assertEqual(profile.head_keypoint_index, [0])
    self.assertEqual(profile.neck_keypoint_index, [1, 2])
    self.assertEqual(profile.left_shoulder_keypoint_index, [1])
    self.assertEqual(profile.right_shoulder_keypoint_index, [2])
    self.assertEqual(profile.left_elbow_keypoint_index, [3])
    self.assertEqual(profile.right_elbow_keypoint_index, [4])
    self.assertEqual(profile.left_wrist_keypoint_index, [5])
    self.assertEqual(profile.right_wrist_keypoint_index, [6])
    self.assertEqual(profile.spine_keypoint_index, [1, 2, 7, 8])
    self.assertEqual(profile.pelvis_keypoint_index, [7, 8])
    self.assertEqual(profile.left_hip_keypoint_index, [7])
    self.assertEqual(profile.right_hip_keypoint_index, [8])
    self.assertEqual(profile.left_knee_keypoint_index, [9])
    self.assertEqual(profile.right_knee_keypoint_index, [10])
    self.assertEqual(profile.left_ankle_keypoint_index, [11])
    self.assertEqual(profile.right_ankle_keypoint_index, [12])
    self.assertEqual(profile.standard_part_names, [
        'HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'SPINE', 'PELVIS',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
        'RIGHT_ANKLE'
    ])
    self.assertEqual(profile.get_standard_part_index('HEAD'), [0])
    self.assertEqual(profile.get_standard_part_index('NECK'), [1, 2])
    self.assertEqual(profile.get_standard_part_index('LEFT_SHOULDER'), [1])
    self.assertEqual(profile.get_standard_part_index('RIGHT_SHOULDER'), [2])
    self.assertEqual(profile.get_standard_part_index('LEFT_ELBOW'), [3])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ELBOW'), [4])
    self.assertEqual(profile.get_standard_part_index('LEFT_WRIST'), [5])
    self.assertEqual(profile.get_standard_part_index('RIGHT_WRIST'), [6])
    self.assertEqual(profile.get_standard_part_index('SPINE'), [1, 2, 7, 8])
    self.assertEqual(profile.get_standard_part_index('PELVIS'), [7, 8])
    self.assertEqual(profile.get_standard_part_index('LEFT_HIP'), [7])
    self.assertEqual(profile.get_standard_part_index('RIGHT_HIP'), [8])
    self.assertEqual(profile.get_standard_part_index('LEFT_KNEE'), [9])
    self.assertEqual(profile.get_standard_part_index('RIGHT_KNEE'), [10])
    self.assertEqual(profile.get_standard_part_index('LEFT_ANKLE'), [11])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ANKLE'), [12])

  def test_std13_keypoint_profile_2d_is_correct(self):
    self._test_std13_keypoint_profile_2d_is_correct('2DSTD13')

  def test_legacy_coco13_keypoint_profile_2d_is_correct(self):
    self._test_std13_keypoint_profile_2d_is_correct('LEGACY_2DCOCO13')

  def test_legacy_h36m13_keypoint_profile_2d_is_correct(self):
    profile = keypoint_profiles.create_keypoint_profile_or_die(
        'LEGACY_2DH36M13')
    self.assertEqual(profile.name, 'LEGACY_2DH36M13')
    self.assertEqual(profile.keypoint_dim, 2)
    self.assertEqual(profile.keypoint_num, 13)
    self.assertEqual(profile.keypoint_names, [
        'Head', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist',
        'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LFoot', 'RFoot'
    ])
    self.assertEqual(
        profile.keypoint_left_right_type(0),
        keypoint_profiles.LeftRightType.CENTRAL)
    self.assertEqual(
        profile.segment_left_right_type(0, 1),
        keypoint_profiles.LeftRightType.LEFT)
    self.assertEqual(profile.offset_keypoint_index, [7, 8])
    self.assertEqual(profile.scale_keypoint_index_pairs,
                     [([1], [2]), ([1], [7]), ([1], [8]), ([2], [7]),
                      ([2], [8]), ([7], [8])])
    self.assertEqual(profile.keypoint_index('LShoulder'), 1)
    self.assertEqual(profile.keypoint_index('dummy'), -1)
    self.assertEqual(profile.segment_index_pairs, [([0], [1]), ([0], [2]),
                                                   ([1], [3]), ([3], [5]),
                                                   ([2], [4]), ([4], [6]),
                                                   ([1], [7]), ([2], [8]),
                                                   ([7], [9]), ([9], [11]),
                                                   ([8], [10]), ([10], [12]),
                                                   ([1], [2]), ([7], [8])])
    self.assertAllEqual(profile.keypoint_affinity_matrix, [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    ])
    self.assertEqual(
        profile.compatible_keypoint_name_dict, {
            '3DSTD16': [
                'HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP',
                'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                'RIGHT_ANKLE'
            ],
            '3DSTD13': [
                'HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP',
                'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                'RIGHT_ANKLE'
            ],
            'LEGACY_3DH36M17': [
                'Head', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist',
                'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LFoot', 'RFoot'
            ],
            'LEGACY_3DMPII3DHP17': [
                'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
                'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
                'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                'right_ankle'
            ]
        })
    self.assertEqual(profile.head_keypoint_index, [0])
    self.assertEqual(profile.neck_keypoint_index, [1, 2])
    self.assertEqual(profile.left_shoulder_keypoint_index, [1])
    self.assertEqual(profile.right_shoulder_keypoint_index, [2])
    self.assertEqual(profile.left_elbow_keypoint_index, [3])
    self.assertEqual(profile.right_elbow_keypoint_index, [4])
    self.assertEqual(profile.left_wrist_keypoint_index, [5])
    self.assertEqual(profile.right_wrist_keypoint_index, [6])
    self.assertEqual(profile.spine_keypoint_index, [1, 2, 7, 8])
    self.assertEqual(profile.pelvis_keypoint_index, [7, 8])
    self.assertEqual(profile.left_hip_keypoint_index, [7])
    self.assertEqual(profile.right_hip_keypoint_index, [8])
    self.assertEqual(profile.left_knee_keypoint_index, [9])
    self.assertEqual(profile.right_knee_keypoint_index, [10])
    self.assertEqual(profile.left_ankle_keypoint_index, [11])
    self.assertEqual(profile.right_ankle_keypoint_index, [12])
    self.assertEqual(profile.standard_part_names, [
        'HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'SPINE', 'PELVIS',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
        'RIGHT_ANKLE'
    ])
    self.assertEqual(profile.get_standard_part_index('HEAD'), [0])
    self.assertEqual(profile.get_standard_part_index('NECK'), [1, 2])
    self.assertEqual(profile.get_standard_part_index('LEFT_SHOULDER'), [1])
    self.assertEqual(profile.get_standard_part_index('RIGHT_SHOULDER'), [2])
    self.assertEqual(profile.get_standard_part_index('LEFT_ELBOW'), [3])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ELBOW'), [4])
    self.assertEqual(profile.get_standard_part_index('LEFT_WRIST'), [5])
    self.assertEqual(profile.get_standard_part_index('RIGHT_WRIST'), [6])
    self.assertEqual(profile.get_standard_part_index('SPINE'), [1, 2, 7, 8])
    self.assertEqual(profile.get_standard_part_index('PELVIS'), [7, 8])
    self.assertEqual(profile.get_standard_part_index('LEFT_HIP'), [7])
    self.assertEqual(profile.get_standard_part_index('RIGHT_HIP'), [8])
    self.assertEqual(profile.get_standard_part_index('LEFT_KNEE'), [9])
    self.assertEqual(profile.get_standard_part_index('RIGHT_KNEE'), [10])
    self.assertEqual(profile.get_standard_part_index('LEFT_ANKLE'), [11])
    self.assertEqual(profile.get_standard_part_index('RIGHT_ANKLE'), [12])

  def test_normalize_keypoints(self):
    profile = keypoint_profiles.KeypointProfile2D(
        name='Dummy',
        keypoint_names=[('A', keypoint_profiles.LeftRightType.UNKNOWN),
                        ('B', keypoint_profiles.LeftRightType.UNKNOWN),
                        ('C', keypoint_profiles.LeftRightType.UNKNOWN)],
        offset_keypoint_names=['A', 'B'],
        scale_keypoint_name_pairs=[(['A', 'B'], ['B']), (['A'], ['B', 'C'])],
        segment_name_pairs=[],
        scale_distance_reduction_fn=tf.math.reduce_sum,
        scale_unit=1.0)
    points = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]],
              [[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]]]
    normalized_points, offset_points, scale_distances = profile.normalize(
        points)

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

  def test_denormalize_keypoints(self):
    profile = keypoint_profiles.KeypointProfile2D(
        name='Dummy',
        keypoint_names=[('A', keypoint_profiles.LeftRightType.UNKNOWN),
                        ('B', keypoint_profiles.LeftRightType.UNKNOWN),
                        ('C', keypoint_profiles.LeftRightType.UNKNOWN)],
        offset_keypoint_names=['A', 'B'],
        scale_keypoint_name_pairs=[(['A', 'B'], ['B']), (['A'], ['B', 'C'])],
        segment_name_pairs=[],
        scale_distance_reduction_fn=tf.math.reduce_sum,
        scale_unit=1.0)
    sqrt_2 = 1.414213562
    normalized_points = tf.constant([
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
    offset_points = [[[[1.0, 2.0]]], [[[11.0, 12.0]]]]
    scale_distances = [[[[4.0 * sqrt_2]]], [[[4.0 * sqrt_2]]]]
    denormalized_points = profile.denormalize(normalized_points, offset_points,
                                              scale_distances)
    self.assertAllClose(denormalized_points,
                        [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]],
                         [[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]]])


class MiscUtilsTest(tf.test.TestCase):

  def test_infer_keypoint_left_right_type(self):
    self.assertEqual(
        keypoint_profiles.infer_keypoint_left_right_type(
            [keypoint_profiles.LeftRightType.LEFT], []),
        keypoint_profiles.LeftRightType.UNKNOWN)

    self.assertEqual(
        keypoint_profiles.infer_keypoint_left_right_type([
            keypoint_profiles.LeftRightType.RIGHT,
            keypoint_profiles.LeftRightType.LEFT
        ], [1]), keypoint_profiles.LeftRightType.LEFT)

    self.assertEqual(
        keypoint_profiles.infer_keypoint_left_right_type([
            keypoint_profiles.LeftRightType.LEFT,
            keypoint_profiles.LeftRightType.LEFT
        ], [0, 1]), keypoint_profiles.LeftRightType.LEFT)

    self.assertEqual(
        keypoint_profiles.infer_keypoint_left_right_type([
            keypoint_profiles.LeftRightType.LEFT,
            keypoint_profiles.LeftRightType.RIGHT
        ], [0, 1]), keypoint_profiles.LeftRightType.CENTRAL)

    self.assertEqual(
        keypoint_profiles.infer_keypoint_left_right_type([
            keypoint_profiles.LeftRightType.LEFT,
            keypoint_profiles.LeftRightType.CENTRAL
        ], [0, 1]), keypoint_profiles.LeftRightType.LEFT)

    self.assertEqual(
        keypoint_profiles.infer_keypoint_left_right_type([
            keypoint_profiles.LeftRightType.UNKNOWN,
            keypoint_profiles.LeftRightType.CENTRAL
        ], [0, 1]), keypoint_profiles.LeftRightType.UNKNOWN)

  def test_infer_segment_left_right_type(self):
    self.assertEqual(
        keypoint_profiles.infer_segment_left_right_type([
            keypoint_profiles.LeftRightType.UNKNOWN,
            keypoint_profiles.LeftRightType.CENTRAL
        ], [0, 1], [1]), keypoint_profiles.LeftRightType.UNKNOWN)

    self.assertEqual(
        keypoint_profiles.infer_segment_left_right_type([
            keypoint_profiles.LeftRightType.UNKNOWN,
            keypoint_profiles.LeftRightType.CENTRAL
        ], [1], [0, 1]), keypoint_profiles.LeftRightType.UNKNOWN)

    self.assertEqual(
        keypoint_profiles.infer_segment_left_right_type([
            keypoint_profiles.LeftRightType.LEFT,
            keypoint_profiles.LeftRightType.CENTRAL
        ], [0], [1]), keypoint_profiles.LeftRightType.LEFT)

    self.assertEqual(
        keypoint_profiles.infer_segment_left_right_type([
            keypoint_profiles.LeftRightType.LEFT,
            keypoint_profiles.LeftRightType.RIGHT
        ], [0], [1]), keypoint_profiles.LeftRightType.CENTRAL)

    self.assertEqual(
        keypoint_profiles.infer_segment_left_right_type([
            keypoint_profiles.LeftRightType.LEFT,
            keypoint_profiles.LeftRightType.LEFT
        ], [0], [1]), keypoint_profiles.LeftRightType.LEFT)


if __name__ == '__main__':
  tf.test.main()
