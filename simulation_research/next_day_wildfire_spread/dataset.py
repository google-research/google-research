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

"""Library of utility functions for reading TF Example datasets."""

import re
from typing import Dict, Text, Sequence, Optional, Tuple

import tensorflow as tf
from tensorflow import estimator as tf_estimator

from simulation_research.next_day_wildfire_spread import constants
from simulation_research.next_day_wildfire_spread import image_utils
from tensorflow.contrib import training as contrib_training


def get_features_dict(
    sample_size,
    features,
):
  """Creates a features dictionary for TensorFlow IO.

  Args:
    sample_size: Size of the input tiles in pixels (square).
    features: List of feature names.

  Returns:
    A features dictionary for TensorFlow IO.
  """
  sample_shape = [sample_size, sample_size]
  features = set(features)
  columns = [tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32)
            ] * len(features)
  return dict(zip(features, columns))


def map_fire_labels(labels):
  """Remaps the raw MODIS fire labels to fire, non-fire, and uncertain.

  The raw fire labels have values spanning from `1` to `9`, inclusive.
  https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MYD14A1
  `1`, `2`, `4`, `6` correspond to uncertain labels.
  `3`, `5` correspond to non-fire labels.
  `7`, `8`, `9` correspond to fire labels.

  Args:
    labels: Raw fire labels.

  Returns:
    Labels with values `1` for fire, `0` for non-fire, and `-1` for uncertain.
  """
  non_fire = tf.where(
      tf.logical_or(tf.equal(labels, 3), tf.equal(labels, 5)),
      tf.zeros_like(labels), -1 * tf.ones_like(labels))
  fire = tf.where(tf.greater_equal(labels, 7), tf.ones_like(labels), non_fire)
  return tf.cast(fire, dtype=tf.float32)


def get_num_channels(features, sequence_length = 1):
  """Returns the number of channels."""
  if sequence_length == 1:
    return len(features)
  return len(features) // sequence_length


def _get_base_key(key):
  """Extracts the base key from the provided key.

  Earth Engine exports `TFRecords` containing each data variable with its
  corresponding variable name. In the case of time sequences, the name of the
  data variable is of the form `variable_1, variable_2, ..., variable_n`,
  where `variable` is the name of the variable, and n the number of elements
  in the time sequence. Extracting the base key ensures that each step of the
  time sequence goes through the same normalization steps.
  The base key obeys the following naming pattern: `([a-zA-Z]+)`
  For instance, for an input key `variable_1`, this function returns `variable`.
  For an input key `variable`, this function simply returns `variable`.

  Args:
    key: Input key.

  Returns:
    The corresponding base key.

  Raises:
    ValueError when `key` does not match the expected pattern.
  """
  match = re.fullmatch(r'([a-zA-Z]+)', key)
  if match:
    return match.group(1)
  raise ValueError(
      f'The provided key does not match the expected pattern: {key}')


def _clip_and_rescale(inputs, key):
  """Clips and rescales inputs with the stats corresponding to `key`.

  Args:
    inputs: Inputs to clip and rescale.
    key: Key describing the inputs.

  Returns:
    Clipped and rescaled input.

  Raises:
    `ValueError` if there are no data statistics available for `key`.
  """
  base_key = _get_base_key(key)
  if base_key not in constants.DATA_STATS:
    raise ValueError(
        f'No data statistics available for the requested key: {key}.')
  min_val, max_val, _, _ = constants.DATA_STATS[base_key]
  inputs = tf.clip_by_value(inputs, min_val, max_val)
  return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))


def _clip_and_normalize(inputs, key):
  """Clips and normalizes inputs with the stats corresponding to `key`.

  Args:
    inputs: Inputs to clip and normalize.
    key: Key describing the inputs.

  Returns:
    Clipped and normalized input.

  Raises:
    `ValueError` if there are no data statistics available for `key`.
  """
  base_key = _get_base_key(key)
  if base_key not in constants.DATA_STATS:
    raise ValueError(
        f'No data statistics available for the requested key: {key}.')
  min_val, max_val, mean, std = constants.DATA_STATS[base_key]
  inputs = tf.clip_by_value(inputs, min_val, max_val)
  inputs = inputs - mean
  return tf.math.divide_no_nan(inputs, std)


def _validate_input_features(
    input_features = constants.INPUT_FEATURES):
  """Validates values for `input_features`."""
  if not all(x in constants.INPUT_FEATURES for x in input_features):
    raise ValueError(f'input_features=[{input_features}] should be present in '
                     f'[{constants.INPUT_FEATURES}]')


def _validate_output_features(
    output_features = constants.OUTPUT_FEATURES):
  """Validates values for `output_features`."""
  if not all(x in constants.OUTPUT_FEATURES for x in output_features):
    raise ValueError(
        f'output_features=[{output_features}] should be present in '
        f'[{constants.OUTPUT_FEATURES}]')


def _parse_journal2021_dataset(
    example_proto, input_sequence_length,
    output_sequence_length, data_size, input_features,
    output_features, clip_and_normalize,
    clip_and_rescale):
  """Parses the 2021 journal dataset.

  Args:
    example_proto: A TensorFlow example protobuf.
    input_sequence_length: Number of samples in the input sequence (`>1` only
      for LSTM).
    output_sequence_length: Number of samples in the output sequence (`>1` only
      for LSTM).
    data_size: Size of tiles (square) as read from input files.
    input_features: Input features to the model.
    output_features: Output features to the model.
    clip_and_normalize: True if the data should be clipped and normalized.
    clip_and_rescale: True if the data should be clipped and rescaled.

  Returns:
    `(input_img, output_img)` tuple of inputs and outputs to the ML model.

  Raises:
    `ValueError` if outputs are incorrect shapes.
  """
  feature_names = list(input_features) + list(output_features)
  features_dict = get_features_dict(data_size, feature_names)
  features = tf.io.parse_single_example(example_proto, features_dict)

  if clip_and_normalize:
    inputs_list = [
        _clip_and_normalize(features.get(key), key) for key in input_features
    ]
  elif clip_and_rescale:
    inputs_list = [
        _clip_and_rescale(features.get(key), key) for key in input_features
    ]
  else:
    inputs_list = [features.get(key) for key in input_features]

  num_in_channels = get_num_channels(input_features, input_sequence_length)
  inputs_stacked = tf.stack(inputs_list, axis=0)
  if input_sequence_length > 1:
    inputs_stacked = tf.reshape(inputs_stacked,
                                [-1, num_in_channels, data_size, data_size])
    input_img = tf.transpose(inputs_stacked, [0, 2, 3, 1])
  else:
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])

  outputs_list = [features.get(key) for key in output_features]
  if not outputs_list:
    raise ValueError('outputs_list should not be empty.')
  outputs_stacked = tf.stack(outputs_list, axis=0)
  if output_sequence_length > 1:
    outputs_stacked = tf.reshape(outputs_stacked, [-1, 1, data_size, data_size])
    outputs_stacked_shape = outputs_stacked.get_shape().as_list()
    if len(outputs_stacked_shape) != 4:
      raise ValueError('outputs_stacked should be rank 4 but dimensions of '
                       f'outputs_stacked are {outputs_stacked_shape}')
    output_img = tf.transpose(outputs_stacked, [0, 2, 3, 1])
  else:
    outputs_stacked_shape = outputs_stacked.get_shape().as_list()
    if len(outputs_stacked_shape) != 3:
      raise ValueError('outputs_stacked should be rank 3 but dimensions of '
                       f'outputs_stacked are {outputs_stacked_shape}')
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])
  # In the `ongoing_64` and `onset_64` datasets, the fire labels are already
  # mapped.
  # If the labels are not already remapped, then call the following:
  # `output_img = _map_fire_labels(output_img)`
  return input_img, output_img


def _parse_fn(
    example_proto, input_sequence_length,
    output_sequence_length, data_size, sample_size,
    output_sample_size, downsample_threshold, binarize_output,
    input_features, output_features,
    clip_and_normalize, clip_and_rescale, random_flip,
    random_rotate, random_crop, center_crop,
    azimuth_in_channel,
    azimuth_out_channel):
  """Reads a serialized example.

  Args:
    example_proto: A TensorFlow example protobuf.
    input_sequence_length: Number of samples in the input sequence (`> 1` only
      for LSTM).
    output_sequence_length: Number of samples in the output sequence (`> 1` only
      for LSTM).
    data_size: Size of tiles (square) as read from input files.
    sample_size: Size the tiles (square) when input into the model.
    output_sample_size: Tile size used by the model (square) as output. Usually
      same as sample_size, but can be smaller for coarse segmentation.
    downsample_threshold: Threshold to determine the downsampled label from a
      square of `1` (positive), `0` (negative), `-1` (uncertain) labels.
      Ignoring uncertain labels, if the ratio of positive/negative labels is
      higher than this threshold, then downsampled label is `1`, otherwise `0`.
      Value is ignored if `output_sample_size == sample_size`.
    binarize_output: Whether to binarize the output values. If True, then output
      values are in `{1, 0, -1}`, else in `{[0, 1], -1}`.
    input_features: Input features to the model.
    output_features: Output features to the model.
    clip_and_normalize: `True` if the data should be clipped and normalized.
    clip_and_rescale: `True` if the data should be clipped and rescaled.
    random_flip: `True` if the data should be randomly flipped.
    random_rotate: `True` if the data should be randomly rotated.
    random_crop: `True` if the data should be randomly cropped.
    center_crop: `True` if the data should be cropped in the center.
    azimuth_in_channel: Channel index of the azimuth channel in the input.
      `None` means no azimuth.
    azimuth_out_channel: Channel index of the azimuth channel in the output.
      `None` means no azimuth.

  Returns:
    `(input_img, output_img)` tuple of inputs and outputs to the ML model.
  """
  if random_crop and center_crop:
    raise ValueError('Cannot have both random_crop and center_crop be True')

  # Read the tf.Examples.
  input_img, output_img = _parse_journal2021_dataset(
      example_proto, input_sequence_length, output_sequence_length, data_size,
      input_features, output_features, clip_and_normalize, clip_and_rescale)
  num_in_channels = get_num_channels(input_features, input_sequence_length)
  num_out_channels = get_num_channels(output_features, output_sequence_length)

  # Apply transformations.
  if random_flip:
    input_img, output_img = image_utils.random_flip_input_and_output_images(
        input_img, output_img, azimuth_in_channel, azimuth_out_channel)
  if random_rotate:
    # We would like arbitrary rotations between [0, 2*pi] and nearest neighbor
    # interpolation. However, currently the function tfa_image.rotate has a
    # NotFoundError.
    # rotation_angle = tf.random.uniform(()) * 2 * math.pi
    # input_img = tfa_image.rotate(input_img, rotation_angle)
    # output_img = tfa_image.rotate(output_img, rotation_angle)
    # central_fraction = 1 / (
    #    abs(math.cos(rotation_angle)) + abs(math.sin(rotation_angle)))
    # input_image = tf.image.central_crop(input_image, central_fraction)
    # output_image = tf.image.central_crop(output_image, central_fraction)
    input_img, output_img = image_utils.random_rotate90_input_and_output_images(
        input_img, output_img, azimuth_in_channel, azimuth_out_channel)
  if random_crop:
    input_img, output_img = image_utils.random_crop_input_and_output_images(
        input_img, output_img, sample_size, num_in_channels, num_out_channels)
  if center_crop:
    input_img, output_img = image_utils.center_crop_input_and_output_images(
        input_img, output_img, sample_size)
  output_img = image_utils.downsample_output_image(output_img,
                                                   output_sample_size,
                                                   downsample_threshold,
                                                   binarize_output)
  return input_img, output_img


def get_dataset(file_pattern,
                data_size,
                sample_size,
                output_sample_size,
                batch_size,
                input_features,
                output_features,
                shuffle,
                shuffle_buffer_size,
                compression_type,
                input_sequence_length,
                output_sequence_length,
                repeat,
                clip_and_normalize,
                clip_and_rescale,
                random_flip,
                random_rotate,
                random_crop,
                center_crop,
                azimuth_in_channel,
                azimuth_out_channel,
                downsample_threshold = 0.0,
                binarize_output = True):
  """Gets the dataset from the file pattern.

  Args:
    file_pattern: Input file pattern.
    data_size: Size of tiles (square) as read from input files.
    sample_size: Size the tiles (square) when input into the model.
    output_sample_size: Tile size used by the model (square) as output. Usually
      same as sample_size, but can be smaller for coarse segmentation.
    batch_size: Batch size.
    input_features: Input features to the model.
    output_features: Output features to the model.
    shuffle: If `True`, the data should be shuffled.
    shuffle_buffer_size: Buffer size for data shuffling (unused if `shuffle` is
      set to False).
    compression_type: Type of compression used for the input files.
    input_sequence_length: Number of samples in the input sequence (`> 1` only
      for LSTM).
    output_sequence_length: Number of samples in the output sequence (`> 1` only
      for LSTM).
    repeat: `True` if the data should be repeated indefinitely, `False`
      otherwise which only iterates through the dataset once.
    clip_and_normalize: If `True`, the data should be clipped and normalized.
    clip_and_rescale: If `True`, the data should be clipped and rescaled.
    random_flip: If `True`, the data should be randomly flipped.
    random_rotate: If `True`, the data should be randomly rotated.
    random_crop: If `True`, the data should be randomly cropped.
    center_crop: If `True`, the data should be cropped in the center.
    azimuth_in_channel: Channel index of the azimuth channel in the input. None
      if no azimuth.
    azimuth_out_channel: Channel index of the azimuth channel in the output.
      None if no azimuth.
    downsample_threshold: Threshold to determine the downsampled label from a
      square of `1` (positive), `0` (negative), `-1` (uncertain) labels.
      Ignoring uncertain labels, if the ratio of positive/negative labels is
      higher than this threshold, then downsampled label is `1`, otherwise `0`.
      E.g., if a 4x4 km region has 6 uncertain labels (and thus 10 certain
      labels), and downsample_threshold is `0.4`, then the downsampled region is
      labeled 'positive' if it has 4 or more positive labels. Value is ignored
      if `output_sample_size == sample_size`.
    binarize_output: Whether to binarize the output values. If `True`, then
      output values are in `{1, 0, -1}`, else in `{[0, 1], -1}`.

  Returns:
    A TensorFlow dataset loaded from the input file pattern, with features
    described in the constants, and with the shapes determined from the input
    parameters to this function.
  """
  if (clip_and_normalize and clip_and_rescale):
    raise ValueError('Cannot have both normalize and rescale.')
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  dataset = dataset.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(
      lambda x: _parse_fn(  # pylint: disable=g-long-lambda
          x, input_sequence_length, output_sequence_length, data_size,
          sample_size, output_sample_size, downsample_threshold,
          binarize_output, input_features, output_features, clip_and_normalize,
          clip_and_rescale, random_flip, random_rotate, random_crop,
          center_crop, azimuth_in_channel, azimuth_out_channel),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.batch(batch_size)
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


def make_dataset(
    hparams,
    mode = tf_estimator.ModeKeys.TRAIN
):
  """Creates a dataset.

  Args:
    hparams: Model hyper-parameters.
    mode: Estimator mode (TRAIN, EVAL, or PREDICT).

  Returns:
    A TensorFlow dataset.

  Raises:
    NotImplementedError if the mode is not supported.
  """
  input_features = list(hparams.input_features)
  output_features = list(hparams.output_features)
  # Validate features.
  _validate_input_features(input_features)
  _validate_output_features(output_features)

  if (not hparams.azimuth_in_channel or
      hparams.azimuth_in_channel not in input_features):
    azimuth_in_channel = None
  else:
    azimuth_in_channel = input_features.index(hparams.azimuth_in_channel)

  if (not hparams.azimuth_out_channel or
      hparams.azimuth_out_channel not in output_features):
    azimuth_out_channel = None
  else:
    azimuth_out_channel = output_features.index(hparams.azimuth_out_channel)

  if mode == tf_estimator.ModeKeys.TRAIN:
    file_pattern = hparams.train_path
    shuffle = True
    repeat = True
    random_flip = hparams.random_flip
    random_rotate = hparams.random_rotate
    random_crop = hparams.random_crop
    center_crop = False
  elif mode == tf_estimator.ModeKeys.EVAL:
    file_pattern = hparams.eval_path
    shuffle = True
    repeat = True
    random_flip = False
    random_rotate = False
    random_crop = hparams.random_crop
    center_crop = False
  elif mode == tf_estimator.ModeKeys.PREDICT:
    file_pattern = hparams.test_path
    shuffle = False
    repeat = False
    random_flip = False
    random_rotate = False
    random_crop = hparams.random_crop
    center_crop = False
  else:
    raise NotImplementedError(f'Unsupported mode {mode}.')

  return get_dataset(
      file_pattern,
      data_size=hparams.data_sample_size,
      sample_size=hparams.sample_size,
      output_sample_size=hparams.output_sample_size,
      batch_size=hparams.batch_size,
      input_features=hparams.input_features,
      output_features=hparams.output_features,
      shuffle=shuffle,
      shuffle_buffer_size=hparams.shuffle_buffer_size,
      compression_type=hparams.compression_type,
      input_sequence_length=hparams.input_sequence_length,
      output_sequence_length=hparams.output_sequence_length,
      repeat=repeat,
      clip_and_normalize=True,
      clip_and_rescale=False,
      random_flip=random_flip,
      random_rotate=random_rotate,
      random_crop=random_crop,
      center_crop=center_crop,
      azimuth_in_channel=azimuth_in_channel,
      azimuth_out_channel=azimuth_out_channel,
      downsample_threshold=hparams.downsample_threshold,
      binarize_output=hparams.binarize_output)
