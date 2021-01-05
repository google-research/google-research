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

"""Utility to preprocess seqeuences consistently."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import enum

import tensorflow.compat.v2 as tf
from tcc.preprocessors import sequence_preprocessor


class SequenceType(enum.Enum):
  """Sequence data types we know how to preprocess."""
  IMAGE = 1


IMAGE = SequenceType.IMAGE


class PreprocessorType(enum.Enum):
  """Preprocessors we know how to run."""
  BRIGHTNESS = 1
  CONTRAST = 2
  GAUSSIAN_NOISE = 3
  HUE = 4
  SATURATION = 5
  CLIP = 6
  RANDOM_CROP = 7
  CENTRAL_CROP = 8
  RESIZE = 9
  FLIP = 10
  NORMALIZE_MEAN_STDDEV = 11
  IMAGE_TO_FLOAT = 12


BRIGHTNESS = PreprocessorType.BRIGHTNESS
CONTRAST = PreprocessorType.CONTRAST
GAUSSIAN_NOISE = PreprocessorType.GAUSSIAN_NOISE
HUE = PreprocessorType.HUE
SATURATION = PreprocessorType.SATURATION
CLIP = PreprocessorType.CLIP
RANDOM_CROP = PreprocessorType.RANDOM_CROP
CENTRAL_CROP = PreprocessorType.CENTRAL_CROP
RESIZE = PreprocessorType.RESIZE
FLIP = PreprocessorType.FLIP
NORMALIZE_MEAN_STDDEV = PreprocessorType.NORMALIZE_MEAN_STDDEV
IMAGE_TO_FLOAT = PreprocessorType.IMAGE_TO_FLOAT


def preprocess_sequence(tensor_type_tuples, preprocess_ranges, seed=None):
  """Preprocesses a set of tensors consistently across tensors and across time.

  Args:
    tensor_type_tuples: Tuples of (tensorflow.Tensor, SequenceType.TYPE) in a
      list or tuple.
    preprocess_ranges: The bounds for generating random values during
      augmentation. These bounds are different for each PreprocessorType. See
      random_preprocessing_args for details.
    seed: if provided, passes a seed value to the random generators. If no seed
      is provided, then the color augmentation subgraph is randomized.

  Returns:
    The augmented tensors in the same order as tensor_type_tuples.
  Raises:
    ValueError: if the SequenceType of a tensor is unknown.
  """
  preprocess_args = random_preprocessing_args(preprocess_ranges, seed)
  output = []
  for tensor, sequence_type in tensor_type_tuples:
    if not isinstance(sequence_type, SequenceType):
      raise ValueError('sequence type "%s" is unknown.' % sequence_type)
    output.append(
        apply_sequence_preprocessing(
            tensor,
            preprocess_args,
            randomize_color_distortion=seed is not None,
            sequence_type=sequence_type))
  return output


def random_preprocessing_args(preprocess_ranges, seed=None):
  """Generates specific preprocess parameters within given ranges.

  The values in preprocess_ranges generally follow the corresponding
  preprocessor function in sequence_preprocessor.py and tf.image. All arguments
  must be supplied for each function. Some values specify a range to randomly
  generate a number in, other values are constants.

  Args:
    preprocess_ranges: a dict of PreprocessorType: {dict of parameter values}
      For each PreprocessorType value range parameters are below. IMAGE_TO_FLOAT
      - {}
        RESIZE - {'new_size': [height, width],
                  'method': one of tf.image.ResizeMethod,
                  'align_corners': bool} See tf.image.resize for a full
                    description.
        CENTRAL_CROP - {'image_size': [height, width],}
        RANDOM_CROP - {'image_size': [height, width],
                       'min_scale': ratio,}
        FLIP - {'dim': which dimension to flip,
                'probability': probability of flipping} The preprocessors below
                  only apply to images.
        BRIGHTNESS - {'max_delta': float, the maximum change in brightness}
        CONSTRAST - {'lower': float, minimum change,
                     'upper': float, maximum change}
        GAUSSIAN_NOISE - {'max_stddev': the maximum standard deviation of noise}
        HUE - {'max_delta': float, the maximum change in hue}
        SATURATION - {'lower': float, minimum change,
                      'upper': float, maximum change}
        CLIP - {'lower_limit': float, values will be greater than this,
                'upper_limit': float, values will be less than this}
        NORMALIZE_MEAN_STDDEV - {"mean": float, the value to subtract,
                                 "stddev": float, the value to divide by}
    seed: if present, sets the random seed for random generation.

  Returns:
    A random set of parameters within these ranges with the same keys.
  """
  random_args = {}
  if BRIGHTNESS in preprocess_ranges:
    max_delta = preprocess_ranges[BRIGHTNESS]['max_delta']
    delta = tf.random.uniform([], -max_delta, max_delta, seed=seed)
    random_args[BRIGHTNESS] = (delta,)
  if CONTRAST in preprocess_ranges:
    lower = preprocess_ranges[CONTRAST]['lower']
    upper = preprocess_ranges[CONTRAST]['upper']
    factor = tf.random.uniform([], lower, upper, seed=seed)
    random_args[CONTRAST] = (factor,)
  if GAUSSIAN_NOISE in preprocess_ranges:
    max_stddev = preprocess_ranges[GAUSSIAN_NOISE]['max_stddev']
    delta = tf.random.uniform([], 0, max_stddev, seed=seed)
    random_args[GAUSSIAN_NOISE] = (delta,)
  if HUE in preprocess_ranges:
    max_delta = preprocess_ranges[HUE]['max_delta']
    delta = tf.random.uniform([], -max_delta, max_delta, seed=seed)
    random_args[HUE] = (delta,)
  if SATURATION in preprocess_ranges:
    lower = preprocess_ranges[SATURATION]['lower']
    upper = preprocess_ranges[SATURATION]['upper']
    factor = tf.random.uniform([], lower, upper, seed=seed)
    random_args[SATURATION] = (factor,)
  if CLIP in preprocess_ranges:
    random_args[CLIP] = preprocess_ranges[CLIP]
  if CENTRAL_CROP in preprocess_ranges:
    random_args[CENTRAL_CROP] = sequence_preprocessor.largest_square_crop(
        preprocess_ranges[CENTRAL_CROP]['image_size'])
  if RANDOM_CROP in preprocess_ranges:
    random_args[RANDOM_CROP] = sequence_preprocessor.random_square_crop(
        **preprocess_ranges[RANDOM_CROP])
  if RESIZE in preprocess_ranges:
    random_args[RESIZE] = preprocess_ranges[RESIZE]
  if FLIP in preprocess_ranges:
    dim = preprocess_ranges[FLIP]['dim']
    probability = preprocess_ranges[FLIP]['probability']
    do_flip = tf.less(tf.random.uniform([], seed=seed), probability)
    random_args[FLIP] = (
        dim,
        do_flip,
    )
  if NORMALIZE_MEAN_STDDEV in preprocess_ranges:
    random_args[NORMALIZE_MEAN_STDDEV] = preprocess_ranges[
        NORMALIZE_MEAN_STDDEV]
  if IMAGE_TO_FLOAT in preprocess_ranges:
    random_args[IMAGE_TO_FLOAT] = {}
  return random_args


def apply_sequence_preprocessing(tensor,
                                 preprocess_args,
                                 randomize_color_distortion=True,
                                 sequence_type=False):
  """Applies consistent preprocessing to a tensor with the first dim as time.

  This function applies to sequences of unknown lengths.

  Args:
    tensor: the tensor to process.
    preprocess_args: a dict with keys as the desired preprocessing steps and
      values to arguments to apply to each frame. Supported preprocessing steps
      are any PreprocessorType. Values can be generated by
      random_preprocessing_args.
    randomize_color_distortion: if true, randomize the color preprocessing
      subgraph independently on each worker.
    sequence_type: a SequenceType value to ensure edge cases are handled
      correctly.

  Returns:
    The tensor with all of the modifications in preprocess_args.
  """

  if IMAGE_TO_FLOAT in preprocess_args and sequence_type == IMAGE:
    tensor = sequence_preprocessor.convert_image_sequence_dtype(tensor)

  if RANDOM_CROP in preprocess_args:
    tensor = sequence_preprocessor.crop_sequence(tensor,
                                                 *preprocess_args[RANDOM_CROP])

  if CENTRAL_CROP in preprocess_args:
    tensor = sequence_preprocessor.crop_sequence(tensor,
                                                 *preprocess_args[CENTRAL_CROP])
  if RESIZE in preprocess_args:
    tensor = sequence_preprocessor.resize_sequence(tensor,
                                                   **preprocess_args[RESIZE])

  if FLIP in preprocess_args:
    tensor = sequence_preprocessor.optionally_flip_sequence(
        tensor, *preprocess_args[FLIP])

  if sequence_type == IMAGE:
    # shuffle potential color manipulations so the same ops don't occur in the
    # same order on all workers. Will be consistent within a worker over time
    # though.
    # pylint: disable=g-long-lambda
    functions = []
    if BRIGHTNESS in preprocess_args:
      functions.append(
          lambda x: sequence_preprocessor.adjust_sequence_brightness(
              x, *preprocess_args[BRIGHTNESS]))
    if CONTRAST in preprocess_args:
      functions.append(lambda x: sequence_preprocessor.adjust_sequence_contrast(
          x, *preprocess_args[CONTRAST]))
    if GAUSSIAN_NOISE in preprocess_args:
      functions.append(
          lambda x: sequence_preprocessor.add_additive_noise_to_sequence(
              x, *preprocess_args[GAUSSIAN_NOISE]))
    if HUE in preprocess_args:
      functions.append(lambda x: sequence_preprocessor.adjust_sequence_hue(
          x, *preprocess_args[HUE]))
    if SATURATION in preprocess_args:
      functions.append(
          lambda x: sequence_preprocessor.adjust_sequence_saturation(
              x, *preprocess_args[SATURATION]))
    # pylint: enable=g-long-lambda
    if randomize_color_distortion:
      random.shuffle(functions)
    for function in functions:
      tensor = function(tensor)

    if CLIP in preprocess_args:
      tensor = sequence_preprocessor.clip_sequence_value(
          tensor, **preprocess_args[CLIP])

    if NORMALIZE_MEAN_STDDEV in preprocess_args:
      tensor = ((tensor - preprocess_args[NORMALIZE_MEAN_STDDEV]['mean']) /
                preprocess_args[NORMALIZE_MEAN_STDDEV]['stddev'])

  return tensor
