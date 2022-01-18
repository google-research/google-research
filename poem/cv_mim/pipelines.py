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

"""Implements pipeline utility functions."""

import math

import tensorflow as tf

from poem.core import common
from poem.core import input_generator
from poem.core import keypoint_utils
from poem.core import tfe_input_layer


def create_dataset_from_tables(
    input_table_patterns,
    batch_sizes,
    num_instances_per_record,
    shuffle=False,
    num_epochs=None,
    drop_remainder=False,
    keypoint_names_3d=None,
    keypoint_names_2d=None,
    feature_dim=None,
    num_classes=None,
    num_frames=None,
    shuffle_buffer_size=4096,
    num_shards=1,
    shard_index=None,
    common_module=common,
    dataset_class=tf.data.TFRecordDataset,
    input_example_parser_creator=tfe_input_layer.create_tfe_parser,
    seed=None):
  """Reads data from tf.Example table.

  Note that this function mainly duplicates `read_batch_from_tfe_tables` in
  `v1.pipeline_utils.py` for compatible with tf2.

  IMPORTANT: We assume that 2D keypoints from the input have been normalized by
  image size. No normalization is expected and no denormalization will be
  performed for both 2D and 3D keypoints.

  Args:
    input_table_patterns: A list of strings for the paths or pattern to input
      tables.
    batch_sizes: A list of integers for the batch sizes to read from each table.
    num_instances_per_record: An integer for the number of instances per
      tf.Example record.
    shuffle: A boolean for whether to shuffle batch.
    num_epochs: An integer for the number of epochs to read. Use `None` to read
      indefinitely.
    drop_remainder: A boolean for whether to drop remainder batch.
    keypoint_names_3d: A list of strings for 3D keypoint names to read
      (coordinates). Use None to skip reading 2D keypoints.
    keypoint_names_2d: A list of strings for 2D keypoint names to read
      (coordinates and scores). Use None to skip reading 2D keypoints.
    feature_dim: An integer for size of pre-computed feature vectors. Use None
      to skip reading feature vectors.
    num_classes: An integer for total number of classification label classes to
      read labels for. Use None to skip reading class labels.
    num_frames: An integer for the number of frames per object each example has.
      Use None to skip adding the frame dimension.
    shuffle_buffer_size: An integer for the buffer size used for shuffling. A
      large buffer size benefits shuffling quality.
    num_shards: An integer for the number of shards to divide the dataset. This
      is useful to distributed training. See `tf.data.Dataset.shard` for
      details.
    shard_index: An integer for the shard index to use. This is useful to
      distributed training, and should usually be set to the id of a
      synchronized worker. See `tf.data.Dataset.shard` for details. Note this
      must be specified if `num_shards` is greater than 1.
    common_module: A Python module that defines common constants.
    dataset_class: A dataset class to use. Must match input table type.
    input_example_parser_creator: A function handle for creating parser
      function.
    seed: An integer for random seed.

  Returns:
    A tf.data.Dataset object.
  """
  parser_kwargs = {
      'num_objects': num_instances_per_record,
  }

  if keypoint_names_3d:
    parser_kwargs.update({
        'keypoint_names_3d': keypoint_names_3d,
        'include_keypoint_scores_3d': False,
    })

  if keypoint_names_2d:
    parser_kwargs.update({
        'keypoint_names_2d': keypoint_names_2d,
        'include_keypoint_scores_2d': True,
    })

  if feature_dim:
    parser_kwargs.update({
        'feature_dim': feature_dim,
    })

  if num_classes:
    parser_kwargs.update({
        'num_classes': num_classes,
    })

  if num_frames:
    parser_kwargs.update({
        'sequence_length': num_frames,
    })

  parser_fn = input_example_parser_creator(
      common_module=common_module, **parser_kwargs)
  dataset = tfe_input_layer.read_batch_from_tables(
      input_table_patterns,
      batch_sizes=batch_sizes,
      drop_remainder=drop_remainder,
      num_epochs=num_epochs,
      num_shards=num_shards,
      shard_index=shard_index,
      shuffle=shuffle,
      shuffle_buffer_size=shuffle_buffer_size,
      seed=seed,
      dataset_class=dataset_class,
      parser_fn=parser_fn)

  return dataset


def create_model_input(inputs,
                       model_input_keypoint_type,
                       keypoint_profile_2d=None,
                       keypoint_profile_3d=None,
                       normalize_keypoints_2d=True,
                       min_keypoint_score_2d=-1.0,
                       azimuth_range=(-math.pi, math.pi),
                       elevation_range=(-math.pi / 6.0, math.pi / 6.0),
                       roll_range=(-math.pi / 6.0, math.pi / 6.0),
                       seed=None):
  """Creates model input features (2D keypoints) from input keypoints.

  Note that this function mainly duplicates `create_model_input` in
  `v1.input_generator.py` for compatible with tf2.

  IMPORTANT: We assume that 2D keypoints from the inputs have been normalized by
  image size. This function will reads image sizes from the input and
  denormalize the 2D keypoints with them. No normalization is expected and no
  denormalization will be performed for 3D keypoints.

  Args:
    inputs: A dictionary for tensor inputs.
    model_input_keypoint_type: An enum string for model input keypoint type. See
      `MODEL_INPUT_KEYPOINT_TYPE_*` for supported values.
    keypoint_profile_2d: A KeypointProfile2D object for input 2D keypoints.
      Required for normalizing 2D keypoints. Also required when 3D-to-2D
      projection is involved.
    keypoint_profile_3d: A KeypointProfile3D object for input 3D keypoints. Only
      used when 3D-to-2D projection is involved.
    normalize_keypoints_2d: A boolean for whether to normalize 2D keypoints at
      the end.
    min_keypoint_score_2d: A float for the minimum score to consider a 2D
      keypoint as invalid.
    azimuth_range: A tuple for minimum and maximum azimuth angles to randomly
      rotate 3D keypoints with.
    elevation_range: A tuple for minimum and maximum elevation angles to
      randomly rotate 3D keypoints with.
    roll_range: A tuple for minimum and maximum roll angles to randomly rotate
      3D keypoints with.
    seed: An integer for random seed.

  Returns:
    features: A tensor for input features. Shape = [..., feature_dim].
    side_outputs: A dictionary for side outputs, which includes
      `offset_points_2d` (shape = [..., 1, 2]) and `scale_distances_2d` (shape =
      [..., 1, 1]) if `normalize_keypoints_2d` is True.
  """
  keypoints_2d = keypoint_utils.denormalize_points_by_image_size(
      inputs[common.KEY_KEYPOINTS_2D],
      image_sizes=inputs[common.KEY_IMAGE_SIZES])

  keypoint_scores_2d = inputs[common.KEY_KEYPOINT_SCORES_2D]
  if min_keypoint_score_2d < 0.0:
    keypoint_masks_2d = tf.ones_like(keypoint_scores_2d, dtype=tf.float32)
  else:
    keypoint_masks_2d = tf.cast(
        tf.math.greater_equal(keypoint_scores_2d, min_keypoint_score_2d),
        dtype=tf.float32)

  keypoints_3d = inputs.get(common.KEY_KEYPOINTS_3D, None)
  features, side_outputs = input_generator.create_model_input(
      keypoints_2d,
      keypoint_masks_2d,
      keypoints_3d,
      model_input_keypoint_type,
      normalize_keypoints_2d=normalize_keypoints_2d,
      keypoint_profile_2d=keypoint_profile_2d,
      keypoint_profile_3d=keypoint_profile_3d,
      azimuth_range=azimuth_range,
      elevation_range=elevation_range,
      roll_range=roll_range,
      seed=seed)

  # IMPORTANT: It is better not to modify `inputs` in TF2. Instead, we save
  # results in the `side_outputs` for further computation.
  side_outputs.update({
      common.KEY_KEYPOINTS_2D: keypoints_2d,
      common.KEY_KEYPOINT_MASKS_2D: keypoint_masks_2d
  })

  return features, side_outputs
