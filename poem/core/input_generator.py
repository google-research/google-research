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

  IMPORTANT: The returned `keypoints_3d` is meant for groundtruth computation.
  Tensor `side_outputs[common.KEY_PREPROCESSED_KEYPOINTS_3D]` is meant for
  generating model input.

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

  side_outputs[common.KEY_PREPROCESSED_KEYPOINTS_3D] = keypoints_3d
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
                            normalized_camera_depth_range=(),
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
      rotate 3D keypoints with. For non-sequential inputs, a 2-tuple for
      (minimum angle, maximum angle) is expected. For sequence inputs, uses
      2-tuple to independently sample starting and ending camera angles, or
      uses 4-tuple for (minimum starting angle, maximum starting angle,
      minimum angle increment, maximum angle increment) to first sample starting
      angles and add random delta angles to them as ending angles.
    elevation_range: A tuple for minimum and maximum elevation angles to
      randomly rotate 3D keypoints with. For non-sequential inputs, a 2-tuple
      for (minimum angle, maximum angle) is expected. For sequence inputs, uses
      2-tuple to independently sample starting and ending camera angles, or
      uses 4-tuple for (minimum starting angle, maximum starting angle,
      minimum angle increment, maximum angle increment) to first sample starting
      angles and add random delta angles to them as ending angles.
    roll_range: A tuple for minimum and maximum roll angles to randomly rotate
      3D keypoints with. For non-sequential inputs, a 2-tuple for (minimum
      angle, maximum angle) is expected. For sequence inputs, uses
      2-tuple to independently sample starting and ending camera angles, or
      uses 4-tuple for (minimum starting angle, maximum starting angle,
      minimum angle increment, maximum angle increment) to first sample starting
      angles and add random delta angles to them as ending angles.
    normalized_camera_depth_range: A tuple for minimum and maximum normalized
      camera depth for random camera augmentation. If empty, uses constant depth
      as 1 over the 2D pose normalization scale unit.
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

    if not normalized_camera_depth_range:
      normalized_camera_depth_range = (1.0 / keypoint_profile_2d.scale_unit,
                                       1.0 / keypoint_profile_2d.scale_unit)
    keypoints_2d, _ = keypoint_utils.randomly_project_and_select_keypoints(
        keypoints_3d,
        keypoint_profile_3d=keypoint_profile_3d,
        output_keypoint_names=(
            keypoint_profile_2d.compatible_keypoint_name_dict[
                keypoint_profile_3d.name]),
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        roll_range=roll_range,
        normalized_camera_depth_range=normalized_camera_depth_range,
        sequential_inputs=sequential_inputs,
        seed=seed)
    keypoint_masks_2d = tf.ones(tf.shape(keypoints_2d)[:-1], dtype=tf.float32)

  elif (model_input_keypoint_type ==
        common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT_AND_3D_PROJECTION):
    if keypoints_3d is None:
      raise ValueError('3D keypoints are not specified for random projection.')

    if not normalized_camera_depth_range:
      normalized_camera_depth_range = (1.0 / keypoint_profile_2d.scale_unit,
                                       1.0 / keypoint_profile_2d.scale_unit)
    projected_keypoints_2d, _ = (
        keypoint_utils.randomly_project_and_select_keypoints(
            keypoints_3d,
            keypoint_profile_3d=keypoint_profile_3d,
            output_keypoint_names=(
                keypoint_profile_2d.compatible_keypoint_name_dict[
                    keypoint_profile_3d.name]),
            azimuth_range=azimuth_range,
            elevation_range=elevation_range,
            roll_range=roll_range,
            normalized_camera_depth_range=normalized_camera_depth_range,
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


def _add_uniform_keypoint_jittering(keypoints, max_jittering_offset, seed=None):
  """Randomly jitters keypoints following uniform distribution.

  Args:
    keypoints: A tensor for keypoints. Shape = [..., keypoint_dim].
    max_jittering_offset: A float for maximum keypoint jittering offset. Random
      jittering offset within [-max_jittering_offset, max_jittering_offset] is
      to be added to each keypoint.
    seed: An integer for random seed.

  Returns:
    A tensor for jittered keypoints.
  """
  if max_jittering_offset <= 0.0:
    raise ValueError('Maximum jittering offset must be positive: %s.' %
                     str(max_jittering_offset))
  offsets = tf.random.uniform(
      tf.shape(keypoints),
      minval=-max_jittering_offset,
      maxval=max_jittering_offset,
      seed=seed)
  return keypoints + offsets


def _add_gaussian_keypoint_jittering(keypoints,
                                     jittering_offset_stddev,
                                     seed=None):
  """Randomly jitters keypoints following Gaussian distribution.

  Args:
    keypoints: A tensor for keypoints. Shape = [..., keypoint_dim].
    jittering_offset_stddev: A float for standard deviation of Gaussian keypoint
      jittering offset. Random jittering offset sampled from N(0,
      jittering_offset_stddev) is to be added to each keypoint.
    seed: An integer for random seed.

  Returns:
    A tensor for jittered keypoints.
  """
  if jittering_offset_stddev <= 0.0:
    raise ValueError(
        'Jittering offset standard deviation must be positive: %s.' %
        str(jittering_offset_stddev))
  offsets = tf.random.normal(
      tf.shape(keypoints), mean=0.0, stddev=jittering_offset_stddev, seed=seed)
  return keypoints + offsets


def apply_stratified_instance_keypoint_dropout(keypoint_masks,
                                               probability_to_apply,
                                               probability_to_drop,
                                               seed=None):
  """Applies stratified keypoint dropout on each instance.

  We perform stratified dropout as first select instances with
  `probability_to_apply` and then drop their keypoints with
  `probability_to_drop`.

  Args:
    keypoint_masks: A tensor for input keypoint masks. Shape = [...,
      num_keypoints].
    probability_to_apply: A float for the probability to perform dropout on an
      instance.
    probability_to_drop: A float for the probability to perform dropout on a
      keypoint.
    seed: An integer for random seed.

  Returns:
    A tensor for output 2D keypoint masks.

  Raises:
    ValueError: If any dropout probability is non-positive.
  """
  if probability_to_apply <= 0.0 or probability_to_drop <= 0.0:
    raise ValueError('Invalid dropout probabilities: (%f, %f)' %
                     (probability_to_apply, probability_to_drop))

  # Shape = [...].
  keep_instance_chances = tf.random.uniform(
      tf.shape(keypoint_masks)[:-1], minval=0.0, maxval=1.0, seed=seed)
  # Shape = [..., 1].
  drop_instance_masks = tf.expand_dims(
      keep_instance_chances < probability_to_apply, axis=-1)

  # Shape = [..., num_keypoints].
  keep_keypoint_chances = tf.random.uniform(
      tf.shape(keypoint_masks), minval=0.0, maxval=1.0, seed=seed)
  drop_keypoint_masks = keep_keypoint_chances < probability_to_drop

  keep_masks = tf.math.logical_not(
      tf.math.logical_and(drop_instance_masks, drop_keypoint_masks))

  return tf.where(keep_masks, keypoint_masks, tf.zeros_like(keypoint_masks))


def apply_stratified_sequence_keypoint_dropout(keypoint_masks,
                                               probability_to_apply,
                                               probability_to_drop,
                                               seed=None):
  """Applies stratified keypoint dropout on each sequence.

  We perform stratified dropout as first select instances with
  `probability_to_apply` and then drop their keypoints with
  `probability_to_drop`.

  Args:
    keypoint_masks: A tensor for input keypoint masks. Shape = [...,
      sequence_length, num_keypoints].
    probability_to_apply: A float for the probability to perform dropout on a
      sequence.
    probability_to_drop: A float for the probability to perform dropout on a
      keypoint.
    seed: An integer for random seed.

  Returns:
    A tensor for output 2D keypoint masks.

  Raises:
    ValueError: If any dropout probability is non-positive.
  """
  if probability_to_apply <= 0.0 or probability_to_drop <= 0.0:
    raise ValueError('Invalid dropout probabilities: (%f, %f)' %
                     (probability_to_apply, probability_to_drop))

  # Shape = [...].
  keep_sequence_chances = tf.random.uniform(
      tf.shape(keypoint_masks)[:-2], minval=0.0, maxval=1.0, seed=seed)
  # Shape = [..., 1, 1].
  drop_sequence_masks = data_utils.recursively_expand_dims(
      keep_sequence_chances < probability_to_apply, [-1, -1])

  # Shape = [..., 1, num_keypoints].
  shape = tf.concat(
      [tf.shape(keypoint_masks)[:-2], [1], [tf.shape(keypoint_masks)[-1]]],
      axis=-1)
  keep_keypoint_chances = tf.random.uniform(
      shape, minval=0.0, maxval=1.0, seed=seed)
  drop_keypoint_masks = keep_keypoint_chances < probability_to_drop

  keep_masks = tf.math.logical_not(
      tf.math.logical_and(drop_sequence_masks, drop_keypoint_masks))

  return tf.where(keep_masks, keypoint_masks, tf.zeros_like(keypoint_masks))


def _override_keypoint_masks(keypoint_masks, keypoint_profile, part_names,
                             overriding_func):
  """Overrides keypoint masks by part.

  Args:
    keypoint_masks: A tensor for input keypoint masks.
    keypoint_profile: A KeypointProfile object for keypoints.
    part_names: A list of standard names of parts of which the masks are
      overridden. See `KeypointProfile.get_standard_part_index` for standard
      part names.
    overriding_func: A function that returns overriding tensors.

  Returns:
    keypoint_masks: A tensor for output keypoint masks.

  """
  part_indices = []
  for name in part_names:
    part_indices.extend(keypoint_profile.get_standard_part_index(name))
  part_indices = list(set(part_indices))
  keypoint_masks = data_utils.update_sub_tensor(
      keypoint_masks,
      indices=part_indices,
      axis=-1,
      update_func=overriding_func)
  return keypoint_masks


def create_model_input(keypoints_2d,
                       keypoint_masks_2d,
                       keypoints_3d,
                       model_input_keypoint_type,
                       model_input_keypoint_mask_type=(
                           common.MODEL_INPUT_KEYPOINT_MASK_TYPE_NO_USE),
                       normalize_keypoints_2d=True,
                       keypoint_profile_2d=None,
                       uniform_keypoint_jittering_max_offset_2d=0.0,
                       gaussian_keypoint_jittering_offset_stddev_2d=0.0,
                       keypoint_dropout_probs=(0.0, 0.0),
                       structured_keypoint_mask_processor=None,
                       set_on_mask_for_non_anchors=False,
                       mix_mask_sub_batches=False,
                       rescale_features=False,
                       forced_mask_on_part_names=None,
                       forced_mask_off_part_names=None,
                       keypoint_profile_3d=None,
                       azimuth_range=(-math.pi, math.pi),
                       elevation_range=(-math.pi / 6.0, math.pi / 6.0),
                       roll_range=(-math.pi / 6.0, math.pi / 6.0),
                       normalized_camera_depth_range=(),
                       sequential_inputs=False,
                       seed=None):
  """Creates model input features from input data.

  Args:
    keypoints_2d: A tensor for input 2D keyopints. Shape = [..., num_keypoints,
      2]. Use None if irrelevant.
    keypoint_masks_2d: A tensor for input 2D keypoint masks. Shape = [...,
      num_keypoints]. Use None if irrelevant.
    keypoints_3d: A tensor for input 3D keyopints. Shape = [..., num_keypoints,
      3]. Use None if irrelevant.
    model_input_keypoint_type: An enum string for model input type. See
      `MODEL_INPUT_TYPE_*` for supported values.
    model_input_keypoint_mask_type: An enum string for model input keypoint mask
      type. See `MODEL_INPUT_KEYPOINT_MASK_TYPE_*` for supported values.
    normalize_keypoints_2d: A boolean for whether to normalize 2D keypoints at
      the end.
    keypoint_profile_2d: A KeypointProfile2D object for input 2D keypoints.
      Required for normalizing 2D keypoints, 3D-to-2D projection, or forcing
      masks on/off.
    uniform_keypoint_jittering_max_offset_2d: A float for maximum 2D keypoint
      jittering offset. Random jittering offset within
      [-uniform_keypoint_jittering_max_offset_2d,
      uniform_keypoint_jittering_max_offset_2d] is to be added to each keypoint
      2D. Note that the jittering happens after the 2D normalization. Ignored if
      non-positive.
    gaussian_keypoint_jittering_offset_stddev_2d: A float for standard deviation
      of Gaussian 2D keypoint jittering offset. Random jittering offset sampled
      from N(0, gaussian_keypoint_jittering_offset_stddev_2d) is to be added to
      each keypoint. Note that the jittering happens after the 2D normalization.
      Ignored if non-positive.
    keypoint_dropout_probs: A tuple of floats for the keypoint random dropout
      probabilities in the format (probability_to_apply, probability_to_drop).
      We perform stratified dropout as first select instances with
      `probability_to_apply` and then drop their keypoints with
      `probability_to_drop`. When sequential_input is True, there might be a
      third element indicating the probability of using sequence-level dropout.
      Only used when keypoint scores are relevant.
    structured_keypoint_mask_processor: A Python function for generating
      keypoint masks with structured dropout. Ignored if None.
    set_on_mask_for_non_anchors: A boolean for whether to always use on (1)
      masks for non-anchor samples. We assume the second from the left tensor
      dimension is for anchor/non-anchor, and the non-anchor samples start at
      the second element along that dimension.
    mix_mask_sub_batches: A boolean for whether to apply sub-batch mixing to
      processed masks and all-one masks.
    rescale_features: A boolean for whether to rescale features by the ratio
      between total number of mask elements and kept mask elements.
    forced_mask_on_part_names: A list of standard names of parts of which the
      masks are forced on (by setting value to 1.0). See
      `KeypointProfile.get_standard_part_index` for standard part names.
    forced_mask_off_part_names: A list of standard names of parts of which the
      masks are forced off (by setting value to 0.0). See
      `KeypointProfile.get_standard_part_index` for standard part names.
    keypoint_profile_3d: A KeypointProfile3D object for input 3D keypoints. Only
      used when 3D-to-2D projection is involved.
    azimuth_range: A tuple for minimum and maximum azimuth angles to randomly
      rotate 3D keypoints with. For non-sequential inputs, a 2-tuple for
      (minimum angle, maximum angle) is expected. For sequence inputs, uses
      2-tuple to independently sample starting and ending camera angles, or uses
      4-tuple for (minimum starting angle, maximum starting angle, minimum angle
      increment, maximum angle increment) to first sample starting angles and
      add random delta angles to them as ending angles.
    elevation_range: A tuple for minimum and maximum elevation angles to
      randomly rotate 3D keypoints with. For non-sequential inputs, a 2-tuple
      for (minimum angle, maximum angle) is expected. For sequence inputs, uses
      2-tuple to independently sample starting and ending camera angles, or uses
      4-tuple for (minimum starting angle, maximum starting angle, minimum angle
      increment, maximum angle increment) to first sample starting angles and
      add random delta angles to them as ending angles.
    roll_range: A tuple for minimum and maximum roll angles to randomly rotate
      3D keypoints with. For non-sequential inputs, a 2-tuple for (minimum
      angle, maximum angle) is expected. For sequence inputs, uses 2-tuple to
      independently sample starting and ending camera angles, or uses 4-tuple
      for (minimum starting angle, maximum starting angle, minimum angle
      increment, maximum angle increment) to first sample starting angles and
      add random delta angles to them as ending angles.
    normalized_camera_depth_range: A tuple for minimum and maximum normalized
      camera depth for random camera augmentation. If empty, uses constant depth
      as 1 over the 2D pose normalization scale unit.
    sequential_inputs: A boolean flag indicating whether the inputs are
      sequential. If True, the input keypoints are supposed to be in shape [...,
      sequence_length, num_keypoints, keypoint_dim].
    seed: An integer for random seed.

  Returns:
    features: A tensor for input features. Shape = [..., feature_dim].
    side_outputs: A dictionary for side outputs, which includes
      `offset_points_2d` (shape = [..., 1, 2]) and `scale_distances_2d` (shape =
      [..., 1, 1]) if `normalize_keypoints_2d` is True.

  Raises:
    ValueError: If `model_input_keypoint_type` is not supported.
    ValueError: If `keypoint_dropout_probs` is not of length 2 or 3.
    ValueError: If `keypoint_profile_2d` is not specified when normalizing 2D
      keypoints.
    ValueError: If keypoint profile name is not 'LEGACY_2DCOCO13', '2DSTD13',
      or 'INTERNAL_2DSTD13' when applying structured keypoint dropout.
    ValueError: If number of instances is not 1 or 2.
    ValueError: If `keypoint_profile_2d` is not specified when forcing keypoint
      masks on.
  """
  keypoints_2d, keypoint_masks_2d = preprocess_keypoints_2d(
      keypoints_2d,
      keypoint_masks_2d,
      keypoints_3d,
      model_input_keypoint_type,
      keypoint_profile_2d=keypoint_profile_2d,
      keypoint_profile_3d=keypoint_profile_3d,
      azimuth_range=azimuth_range,
      elevation_range=elevation_range,
      roll_range=roll_range,
      normalized_camera_depth_range=normalized_camera_depth_range,
      sequential_inputs=sequential_inputs,
      seed=seed)

  side_outputs = {}

  if len(keypoint_dropout_probs) not in [2, 3]:
    raise ValueError('Invalid keypoint dropout probability tuple: `%s`.' %
                     str(keypoint_dropout_probs))

  if keypoint_dropout_probs[0] > 0.0 and keypoint_dropout_probs[1] > 0.0:
    instance_keypoint_masks_2d = apply_stratified_instance_keypoint_dropout(
        keypoint_masks_2d,
        probability_to_apply=keypoint_dropout_probs[0],
        probability_to_drop=keypoint_dropout_probs[1],
        seed=seed)

    if (sequential_inputs and len(keypoint_dropout_probs) == 3 and
        keypoint_dropout_probs[2] > 0.0):
      sequence_keypoint_masks_2d = apply_stratified_sequence_keypoint_dropout(
          keypoint_masks_2d,
          probability_to_apply=keypoint_dropout_probs[0],
          probability_to_drop=keypoint_dropout_probs[1],
          seed=seed)
      sequence_axis = sequence_keypoint_masks_2d.shape.ndims - 1
      keypoint_masks_2d = data_utils.mix_batch(
          [sequence_keypoint_masks_2d], [instance_keypoint_masks_2d],
          axis=sequence_axis,
          keep_lhs_prob=keypoint_dropout_probs[2],
          seed=seed)[0]
    else:
      keypoint_masks_2d = instance_keypoint_masks_2d

  if structured_keypoint_mask_processor is not None:
    keypoint_masks_2d = structured_keypoint_mask_processor(
        keypoint_masks=keypoint_masks_2d,
        keypoint_profile=keypoint_profile_2d,
        seed=seed)

  if normalize_keypoints_2d:
    if keypoint_profile_2d is None:
      raise ValueError('Failed to normalize 2D keypoints due to unspecified '
                       'keypoint profile.')
    keypoints_2d, offset_points, scale_distances = (
        keypoint_profile_2d.normalize(keypoints_2d, keypoint_masks_2d))
    side_outputs.update({
        common.KEY_OFFSET_POINTS_2D: offset_points,
        common.KEY_SCALE_DISTANCES_2D: scale_distances
    })

  if uniform_keypoint_jittering_max_offset_2d > 0.0:
    keypoints_2d = _add_uniform_keypoint_jittering(
        keypoints_2d,
        max_jittering_offset=uniform_keypoint_jittering_max_offset_2d,
        seed=seed)

  if gaussian_keypoint_jittering_offset_stddev_2d > 0.0:
    keypoints_2d = _add_gaussian_keypoint_jittering(
        keypoints_2d,
        jittering_offset_stddev=gaussian_keypoint_jittering_offset_stddev_2d,
        seed=seed)

  if set_on_mask_for_non_anchors:
    non_anchor_indices = list(range(1, keypoint_masks_2d.shape.as_list()[1]))
    if non_anchor_indices:
      keypoint_masks_2d = data_utils.update_sub_tensor(
          keypoint_masks_2d,
          indices=non_anchor_indices,
          axis=1,
          update_func=tf.ones_like)

  if mix_mask_sub_batches:
    keypoint_masks_2d = data_utils.mix_batch([tf.ones_like(keypoint_masks_2d)],
                                             [keypoint_masks_2d],
                                             axis=1)[0]

  if forced_mask_on_part_names:
    keypoint_masks_2d = _override_keypoint_masks(
        keypoint_masks_2d,
        keypoint_profile=keypoint_profile_2d,
        part_names=forced_mask_on_part_names,
        overriding_func=tf.ones_like)

  if forced_mask_off_part_names:
    keypoint_masks_2d = _override_keypoint_masks(
        keypoint_masks_2d,
        keypoint_profile=keypoint_profile_2d,
        part_names=forced_mask_off_part_names,
        overriding_func=tf.zeros_like)

  if model_input_keypoint_mask_type in [
      common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS,
      common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT
  ]:
    # Mask out invalid keypoints.
    keypoints_2d = tf.where(
        data_utils.tile_last_dims(
            tf.expand_dims(tf.math.equal(keypoint_masks_2d, 1.0), axis=-1),
            last_dim_multiples=[tf.shape(keypoints_2d)[-1]]), keypoints_2d,
        tf.zeros_like(keypoints_2d))

  side_outputs.update({
      common.KEY_PREPROCESSED_KEYPOINTS_2D: keypoints_2d,
      common.KEY_PREPROCESSED_KEYPOINT_MASKS_2D: keypoint_masks_2d,
  })

  features = keypoints_2d
  if model_input_keypoint_mask_type in [
      common.MODEL_INPUT_KEYPOINT_MASK_TYPE_AS_INPUT,
      common.MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT
  ]:
    features = tf.concat(
        [keypoints_2d, tf.expand_dims(keypoint_masks_2d, axis=-1)], axis=-1)

  if rescale_features:
    # Scale up features to compensate for any keypoint masking.
    feature_rescales = keypoint_masks_2d.shape.as_list()[-1] / (
        tf.math.maximum(
            1e-12, tf.math.reduce_sum(
                keypoint_masks_2d, axis=-1, keepdims=True)))
    features *= tf.expand_dims(feature_rescales, axis=-1)

  features = data_utils.flatten_last_dims(features, num_last_dims=2)
  return features, side_outputs
