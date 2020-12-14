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

"""Model input generators."""

import math

import tensorflow as tf

from poem.core import common
from poem.core import data_utils
from poem.core import keypoint_utils


def preprocess_keypoints_3d(keypoints_3d,
                            keypoint_profile_3d,
                            normalize_keypoints_3d=True):
  """Preprocesses 3D keypoints.

  Args:
    keypoints_3d: A tensor for input 3D keyopints. Shape = [..., num_keypoints,
      3].
    keypoint_profile_3d: A KeypointProfile3D object.
    normalize_keypoints_3d: A boolean for whether to normalize 3D keypoints at
      the end.

  Returns:
    keypoints_3d: A tensor for preprocessed 3D keypoints.
    side_outputs: A dictionary for side outputs, which includes
      `offset_points_3d` (shape = [..., 1, 3]) and `scale_distances_3d` (shape =
      [..., 1, 1]) if `normalize_keypoints_3d` is True.
  """
  side_outputs = {}
  if normalize_keypoints_3d:
    (keypoints_3d, side_outputs[common.KEY_OFFSET_POINTS_3D],
     side_outputs[common.KEY_SCALE_DISTANCES_3D]) = (
         keypoint_profile_3d.normalize(keypoints_3d))
  return keypoints_3d, side_outputs


def preprocess_keypoints_2d(keypoints_2d,
                            keypoint_masks_2d,
                            keypoints_3d,
                            model_input_keypoint_type,
                            keypoint_profile_2d=None,
                            keypoint_profile_3d=None,
                            azimuth_range=(-math.pi, math.pi),
                            elevation_range=(-math.pi / 6.0, math.pi / 6.0),
                            roll_range=(-math.pi / 6.0, math.pi / 6.0),
                            projection_mix_batch_assignment=None,
                            sequential_inputs=False,
                            seed=None):
  """Preprocesses input 2D keypoints.

  Note that this function does not normalize 2D keypoints at the end.

  Args:
    keypoints_2d: A tensor for input 2D keyopints. Shape = [..., num_keypoints,
      2]. Use None if irrelevant.
    keypoint_masks_2d: A tensor for input 2D keypoint masks. Shape = [...,
      num_keypoints]. Use None if irrelevant.
    keypoints_3d: A tensor for input 3D keyopints. Shape = [..., num_keypoints,
      3]. Use None if irrelevant.
    model_input_keypoint_type: An enum string for model input type. See
      `MODEL_INPUT_KEYPOINT_TYPE_*` for supported values.
    keypoint_profile_2d: A KeypointProfile2D object for input 2D keypoints. Only
      used when 3D-to-2D projection is involved.
    keypoint_profile_3d: A KeypointProfile3D object for input 3D keypoints. Only
      used when 3D-to-2D projection is involved.
    azimuth_range: A tuple for minimum and maximum azimuth angles to randomly
      rotate 3D keypoints with.
    elevation_range: A tuple for minimum and maximum elevation angles to
      randomly rotate 3D keypoints with.
    roll_range: A tuple for minimum and maximum roll angles to randomly rotate
      3D keypoints with.
    projection_mix_batch_assignment: A tensor for assignment indicator matrix
      for mixing batches of input/projection keypoints. Shape = [batch_size,
      ..., num_instances]. If None, input/projection keypoints are mixed roughly
      evenly following a uniform distribution.
    sequential_inputs: A boolean flag indicating whether the inputs are
      sequential. If True, the input keypoints are supposed to be in shape
      [..., sequence_length, num_keypoints, keypoint_dim].
    seed: An integer for random seed.

  Returns:
    keypoints_2d: A tensor for preprocessed 2D keypoints. Shape = [...,
      num_keypoints_2d, 2].
    keypoint_masks_2d: A tensor for preprecessed 2D keypoint masks. Shape =
      [..., num_keypoints_2d].

  Raises:
    ValueError: If `model_input_keypoint_type` is not supported.
    ValueError: If `keypoints_3d` is required but not specified.
  """
  if (model_input_keypoint_type ==
      common.MODEL_INPUT_KEYPOINT_TYPE_3D_PROJECTION):
    if keypoints_3d is None:
      raise ValueError('3D keypoints are not specified for random projection.')
    keypoints_2d, _ = keypoint_utils.random_project_and_select_keypoints(
        keypoints_3d,
        keypoint_profile_3d=keypoint_profile_3d,
        output_keypoint_names=(
            keypoint_profile_2d.compatible_keypoint_name_dict[
                keypoint_profile_3d.name]),
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        roll_range=roll_range,
        default_camera_z=1.0 / keypoint_profile_2d.scale_unit,
        sequential_inputs=sequential_inputs,
        seed=seed)
    keypoint_masks_2d = tf.ones(tf.shape(keypoints_2d)[:-1], dtype=tf.float32)

  elif (model_input_keypoint_type ==
        common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT_AND_3D_PROJECTION):
    if keypoints_3d is None:
      raise ValueError('3D keypoints are not specified for random projection.')
    projected_keypoints_2d, _ = (
        keypoint_utils.random_project_and_select_keypoints(
            keypoints_3d,
            keypoint_profile_3d=keypoint_profile_3d,
            output_keypoint_names=(
                keypoint_profile_2d.compatible_keypoint_name_dict[
                    keypoint_profile_3d.name]),
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            roll_range=roll_range,
            default_camera_z=1.0 / keypoint_profile_2d.scale_unit,
            sequential_inputs=sequential_inputs,
            seed=seed))
    projected_keypoint_masks_2d = tf.ones(
        tf.shape(projected_keypoints_2d)[:-1], dtype=tf.float32)
    keypoints_2d, keypoint_masks_2d = data_utils.mix_batch(
        [keypoints_2d, keypoint_masks_2d],
        [projected_keypoints_2d, projected_keypoint_masks_2d],
        axis=1,
        assignment=projection_mix_batch_assignment,
        seed=seed)

  elif model_input_keypoint_type != common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT:
    raise ValueError('Unsupported model input type: `%s`.' %
                     str(model_input_keypoint_type))

  return keypoints_2d, keypoint_masks_2d


def create_model_input(keypoints_2d,
                       keypoint_masks_2d,
                       keypoints_3d,
                       model_input_keypoint_type,
                       normalize_keypoints_2d=True,
                       keypoint_profile_2d=None,
                       keypoint_profile_3d=None,
                       azimuth_range=(-math.pi, math.pi),
                       elevation_range=(-math.pi / 6.0, math.pi / 6.0),
                       roll_range=(-math.pi / 6.0, math.pi / 6.0),
                       sequential_inputs=False,
                       seed=None):
  """Creates model input features from input keypoints.

  Args:
    keypoints_2d: A tensor for input 2D keyopints. Shape = [..., num_keypoints,
      2]. Use None if irrelevant.
    keypoint_masks_2d: A tensor for input 2D keypoint masks. Shape = [...,
      num_keypoints].
    keypoints_3d: A tensor for input 3D keyopints. Shape = [..., num_keypoints,
      3]. Use None if irrelevant.
    model_input_keypoint_type: An enum string for model input keypoint type. See
      `MODEL_INPUT_KEYPOINT_TYPE_*` for supported values.
    normalize_keypoints_2d: A boolean for whether to normalize 2D keypoints at
      the end.
    keypoint_profile_2d: A KeypointProfile2D object for input 2D keypoints.
      Required for normalizing 2D keypoints. Also required when 3D-to-2D
      projection is involved.
    keypoint_profile_3d: A KeypointProfile3D object for input 3D keypoints. Only
      used when 3D-to-2D projection is involved.
    azimuth_range: A tuple for minimum and maximum azimuth angles to randomly
      rotate 3D keypoints with.
    elevation_range: A tuple for minimum and maximum elevation angles to
      randomly rotate 3D keypoints with.
    roll_range: A tuple for minimum and maximum roll angles to randomly rotate
      3D keypoints with.
    sequential_inputs: A boolean flag indicating whether the inputs are
      sequential. If True, the input keypoints are supposed to be in shape
      [..., sequence_length, num_keypoints, keypoint_dim].
    seed: An integer for random seed.

  Returns:
    features: A tensor for input features. Shape = [..., feature_dim].
    side_outputs: A dictionary for side outputs, which includes
      `offset_points_2d` (shape = [..., 1, 2]) and `scale_distances_2d` (shape =
      [..., 1, 1]) if `normalize_keypoints_2d` is True.

  Raises:
    ValueError: If `normalize_keypoints_2d` is True, but `keypoint_profile_2d`
      is not specified.
  """
  keypoints_2d, keypoint_masks_2d = (
      preprocess_keypoints_2d(
          keypoints_2d,
          keypoint_masks_2d,
          keypoints_3d,
          model_input_keypoint_type,
          keypoint_profile_2d=keypoint_profile_2d,
          keypoint_profile_3d=keypoint_profile_3d,
          azimuth_range=azimuth_range,
          elevation_range=elevation_range,
          roll_range=roll_range,
          sequential_inputs=sequential_inputs,
          seed=seed))

  side_outputs = {}

  if normalize_keypoints_2d:
    if keypoint_profile_2d is None:
      raise ValueError(
          'Must specify 2D keypoint profile to normalize keypoints.')
    keypoints_2d, offset_points, scale_distances = (
        keypoint_profile_2d.normalize(keypoints_2d))
    side_outputs.update({
        common.KEY_OFFSET_POINTS_2D: offset_points,
        common.KEY_SCALE_DISTANCES_2D: scale_distances
    })

  side_outputs.update({
      common.KEY_PREPROCESSED_KEYPOINTS_2D: keypoints_2d,
      common.KEY_PREPROCESSED_KEYPOINT_MASKS_2D: keypoint_masks_2d,
  })

  features = data_utils.flatten_last_dims(keypoints_2d, num_last_dims=2)
  return features, side_outputs
