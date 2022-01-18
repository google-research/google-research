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

"""Internal Tensorflow SequenceExample input utility functions."""

import tensorflow as tf

from poem.core import common
from poem.core import data_utils
from poem.core import tfe_input_layer


read_from_table = tfe_input_layer.read_from_table


def _set_and_permute_time_axis(tensor, sequence_length, axis=-3):
  """Sets a tensor into static shape and permutes its time axis."""
  shape = tensor.shape.as_list()
  shape[0] = sequence_length
  tensor.set_shape(shape)

  # in the original tensor the first axis is always the time axis.
  permutation = list(range(tensor.shape.ndims))
  permutation[0] = permutation[axis]
  permutation[axis] = 0
  return tf.transpose(tensor, permutation)


def add_decoder_image_sizes(instance_shape, common_module=common):
  """Adds decoders for image sizes.

  Args:
    instance_shape: A list of integers for the shape (layout) of instances for
      each record.
    common_module: A Python module that defines common constants.

  Returns:
    A dictionary for decoders.
  """
  return {
      common_module.TFSE_KEY_IMAGE_HEIGHT:
          tf.io.FixedLenFeature(instance_shape, dtype=tf.int64),
      common_module.TFSE_KEY_IMAGE_WIDTH:
          tf.io.FixedLenFeature(instance_shape, dtype=tf.int64),
  }


def add_decoder_keypoints_2d(keypoint_names_2d,
                             include_keypoint_scores_2d,
                             instance_shape,
                             common_module=common):
  """Adds decoders for 2D keypoints.

  Args:
    keypoint_names_2d: A list of strings for 2D keypoint names.
    include_keypoint_scores_2d: A boolean for whether to include 2D keypoint
      scores.
    instance_shape: A list of integers for the shape (layout) of instances for
      each record.
    common_module: A Python module that defines common constants.

  Returns:
    decoders: A dictionary for decoders.
  """
  decoders = {}

  for name in keypoint_names_2d:
    for keypoint_suffix in common_module.TFSE_KEY_SUFFIX_KEYPOINT_2D:
      key = name + keypoint_suffix
      decoders[key] = tf.io.FixedLenSequenceFeature(
          instance_shape, dtype=tf.float32)

    if include_keypoint_scores_2d:
      key = name + common_module.TFSE_KEY_SUFFIX_KEYPOINT_2D_SCORE
      decoders[key] = tf.io.FixedLenSequenceFeature(
          instance_shape, dtype=tf.float32)

  return decoders


def add_decoder_keypoints_3d(keypoint_names_3d,
                             include_keypoint_scores_3d,
                             instance_shape,
                             common_module=common):
  """Adds decoders for 3D keypoints.

  Args:
    keypoint_names_3d: A list of strings for 3D keypoint names.
    include_keypoint_scores_3d: A boolean for whether to include 3D keypoint
      scores.
    instance_shape: A list of integers for the shape (layout) of instances for
      each record.
    common_module: A Python module that defines common constants.

  Returns:
    decoders: A dictionary for decoders.
  """
  decoders = {}

  for name in keypoint_names_3d:
    for keypoint_suffix in common_module.TFSE_KEY_SUFFIX_KEYPOINT_3D:
      key = name + keypoint_suffix
      decoders[key] = tf.io.FixedLenSequenceFeature(
          instance_shape, dtype=tf.float32)

    if include_keypoint_scores_3d:
      key = name + common_module.TFSE_KEY_SUFFIX_KEYPOINT_3D_SCORE
      decoders[key] = tf.io.FixedLenSequenceFeature(
          instance_shape, dtype=tf.float32)

  return decoders


def process_decoded_image_sizes(decoded_tensors,
                                sequence_length,
                                common_module=common):
  """Processes decoded image sizes.

  Args:
    decoded_tensors: A dictionary for decoded tensors.
    sequence_length: An integer for input sequence length.
    common_module: A Python module that defines common constants.

  Returns:
    A dictionary for processed 2D keypoint tensors.
  """
  image_heights = decoded_tensors[common_module.TFSE_KEY_IMAGE_HEIGHT]
  image_widths = decoded_tensors[common_module.TFSE_KEY_IMAGE_WIDTH]
  image_sizes = tf.stack([image_heights, image_widths], axis=-1)
  image_sizes = data_utils.tile_last_dims(
      tf.expand_dims(image_sizes, axis=-2),
      last_dim_multiples=[sequence_length, 1])
  return {
      common_module.KEY_IMAGE_SIZES: image_sizes,
  }


def process_decoded_keypoints_2d(decoded_tensors,
                                 keypoint_names_2d,
                                 include_keypoint_scores_2d,
                                 sequence_length,
                                 common_module=common):
  """Processes decoded 2D keypoint tensors.

  Args:
    decoded_tensors: A dictionary for decoded tensors.
    keypoint_names_2d: A list of strings for 2D keypoint names.
    include_keypoint_scores_2d: A boolean for whether to include 2D keypoint
      scores.
    sequence_length: An integer for the length of input sequences.
    common_module: A Python module that defines common constants.

  Returns:
    outputs: A dictionary for processed 2D keypoint tensors.
  """
  outputs = {}

  keypoints_2d = []
  for name in keypoint_names_2d:
    sub_keypoints_2d = []
    for keypoint_suffix in common_module.TFSE_KEY_SUFFIX_KEYPOINT_2D:
      key = name + keypoint_suffix
      sub_keypoints_2d.append(decoded_tensors[key])
    keypoints_2d.append(tf.stack(sub_keypoints_2d, axis=-1))

  keypoints_2d = tf.stack(keypoints_2d, axis=-2)
  keypoints_2d = _set_and_permute_time_axis(keypoints_2d, sequence_length)
  outputs[common_module.KEY_KEYPOINTS_2D] = keypoints_2d

  if include_keypoint_scores_2d:
    keypoint_scores_2d = []
    for name in keypoint_names_2d:
      key = name + common_module.TFSE_KEY_SUFFIX_KEYPOINT_2D_SCORE
      keypoint_scores_2d.append(decoded_tensors[key])

    keypoint_scores_2d = tf.stack(keypoint_scores_2d, axis=-1)
    keypoint_scores_2d = _set_and_permute_time_axis(
        keypoint_scores_2d, sequence_length, axis=-2)
    outputs[common_module.KEY_KEYPOINT_SCORES_2D] = keypoint_scores_2d

  return outputs


def process_decoded_keypoints_3d(decoded_tensors,
                                 keypoint_names_3d,
                                 include_keypoint_scores_3d,
                                 sequence_length,
                                 common_module=common):
  """Processes decoded 3D keypoint tensors.

  Args:
    decoded_tensors: A dictionary for decoded tensors.
    keypoint_names_3d: A list of strings for 3D keypoint names.
    include_keypoint_scores_3d: A boolean for whether to include 2D keypoint
      scores.
    sequence_length: An integer for the length of input sequences.
    common_module: A Python module that defines common constants.

  Returns:
    outputs: A dictionary for processed 2D keypoint tensors.
  """
  outputs = {}

  keypoints_3d = []
  for name in keypoint_names_3d:
    sub_keypoints_3d = []
    for keypoint_suffix in common_module.TFSE_KEY_SUFFIX_KEYPOINT_3D:
      key = name + keypoint_suffix
      sub_keypoints_3d.append(decoded_tensors[key])
    keypoints_3d.append(tf.stack(sub_keypoints_3d, axis=-1))

  keypoints_3d = tf.stack(keypoints_3d, axis=-2)
  keypoints_3d = _set_and_permute_time_axis(keypoints_3d, sequence_length)
  outputs[common_module.KEY_KEYPOINTS_3D] = keypoints_3d

  if include_keypoint_scores_3d:
    keypoint_scores_3d = []
    for name in keypoint_names_3d:
      key = name + common_module.TFSE_KEY_SUFFIX_KEYPOINT_3D_SCORE
      keypoint_scores_3d.append(decoded_tensors[key])

    keypoint_scores_3d = tf.stack(keypoint_scores_3d, axis=-1)
    keypoint_scores_3d = _set_and_permute_time_axis(
        keypoint_scores_3d, sequence_length, axis=-2)
    outputs[common_module.KEY_KEYPOINT_SCORES_3D] = keypoint_scores_3d

  return outputs


def get_tfse_parser_fn(context_features_decoders, sequence_features_decoders,
                       post_process_fn):
  """Creates a tf.SequenceExample parser function.

  Args:
    context_features_decoders: A dictionary for keyed tf.SequenceExample context
      features decoders.
    sequence_features_decoders: A dictionary for keyed tf.SequenceExample
      sequence features decoders.
    post_process_fn: A function handle for postprocessing decoded tensors.

  Returns:
    parser_fn: A function handle for the parser function.
  """

  def parser_fn(*inputs):
    """Decoder function."""
    # Here `inputs` can be either just a serialized example or a (key,
    # serialized example) tuple (in which we ignore the key), and we would like
    # to handle both cases.
    serialized_example = inputs[-1]
    decoded_tensors = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=context_features_decoders,
        sequence_features=sequence_features_decoders)
    return post_process_fn(decoded_tensors)

  return parser_fn


def create_tfse_parser(keypoint_names_2d=None,
                       keypoint_names_3d=None,
                       include_keypoint_scores_2d=True,
                       include_keypoint_scores_3d=False,
                       num_objects=1,
                       sequence_length=None,
                       common_module=common):
  """Creates default tf.SequenceExample parser function.

  Args:
    keypoint_names_2d: A list of strings for 2D keypoint names. Use None to skip
      reading 2D keypoints.
    keypoint_names_3d: A list of strings for 3D keypoint names. Use None to skip
      reading 3D keypoints.
    include_keypoint_scores_2d: A boolean for whether to read 2D keypoint
      scores. Only used if `keypoint_names_2d` is specified.
    include_keypoint_scores_3d: A boolean for whether to read 3D keypoint
      scores. Only used if `keypoint_names_3d` is specified.
    num_objects: An integer for the number of objects each example has.
    sequence_length: An integer for the length of input sequences.
    common_module: A Python module that defines common constants.

  Returns:
    parser_fn: A function handle for the parser.
  """
  # Prepare context features
  context_features_decoders = add_decoder_image_sizes(
      instance_shape=[num_objects],
      common_module=common_module)

  # Prepare sequence features
  sequence_features_decoders = {}
  if keypoint_names_2d:
    sequence_features_decoders.update(
        add_decoder_keypoints_2d(
            keypoint_names_2d,
            include_keypoint_scores_2d=include_keypoint_scores_2d,
            instance_shape=[num_objects],
            common_module=common_module))

  if keypoint_names_3d:
    sequence_features_decoders.update(
        add_decoder_keypoints_3d(
            keypoint_names_3d,
            include_keypoint_scores_3d=include_keypoint_scores_3d,
            instance_shape=[num_objects],
            common_module=common_module))

  def post_process_decoded_tensors(decoded_tensors):
    """Postprocesses decoded tensors."""
    # Placeholder for postprocessing including static padding, temporal sampling
    # augmentation, etc.
    outputs = process_decoded_image_sizes(
        decoded_tensors[0], sequence_length, common_module)

    if keypoint_names_2d:
      outputs.update(
          process_decoded_keypoints_2d(
              decoded_tensors[1],
              keypoint_names_2d=keypoint_names_2d,
              include_keypoint_scores_2d=include_keypoint_scores_2d,
              sequence_length=sequence_length,
              common_module=common_module))

    if keypoint_names_3d:
      outputs.update(
          process_decoded_keypoints_3d(
              decoded_tensors[1],
              keypoint_names_3d=keypoint_names_3d,
              include_keypoint_scores_3d=include_keypoint_scores_3d,
              sequence_length=sequence_length,
              common_module=common_module))
    return outputs

  return get_tfse_parser_fn(context_features_decoders,
                            sequence_features_decoders,
                            post_process_decoded_tensors)
