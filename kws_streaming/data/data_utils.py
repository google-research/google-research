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

# Lint as: python3
"""Utility functions for processing audio data."""

import tensorflow.compat.v1 as tf


def resample(input_samples, time_resample, desired_samples):
  """Resizes input audio samples in time dimension.

  Arguments:
    input_samples: input 2d tensor [batch, time].
    time_resample: resampling coefficient (can be any positive number),
      for example:
        if 1: no resampling
        if 0.5: image will be reduced by 2 times
          and padded by 0 to get desired_samples in time dim
        if 1.5: image will be increased by 1.5 times
          and cropped to get desired_samples in time dim
    desired_samples: desired size of time dim

  Returns:
    Resampled audio samples in time dim
  """
  if time_resample == 1.0:
    return input_samples

  shape = tf.shape(input_samples)
  samples = tf.expand_dims(input_samples, 0)
  samples = tf.expand_dims(samples, 2)
  samples_resized = tf.image.resize(
      images=samples,
      size=(tf.cast((tf.cast(shape[0], tf.float32) * time_resample),
                    tf.int32), 1),
      preserve_aspect_ratio=False)
  samples_resized_cropped = tf.image.resize_with_crop_or_pad(
      samples_resized,
      target_height=desired_samples,
      target_width=1,
  )
  samples_resized_cropped = tf.squeeze(samples_resized_cropped, axis=[0, 3])
  return samples_resized_cropped


def shift_in_time(input_samples, time_shift_padding, time_shift_offset,
                  desired_samples):
  """Shift the sample's start position, and pad any gaps with zeros.

  Arguments:
    input_samples: 2d input audio samples: [batch, time]
    time_shift_padding: paddings - defined by get_time_shift_pad_offset()
    time_shift_offset: offset array - defined by get_time_shift_pad_offset()
    desired_samples: desired size of time dim

  Returns:
    audio samples, shifted in time
  """
  padded_foreground = tf.pad(
      tensor=input_samples,
      paddings=time_shift_padding,
      mode='CONSTANT')
  # TODO(rybakov) use numpy pattern and add unit test:
  # sliced_foreground = padded_foreground[:desired_samples]
  sliced_foreground = tf.slice(padded_foreground,
                               time_shift_offset,
                               [desired_samples, -1])
  return sliced_foreground


def get_time_shift_pad_offset(time_shift_amount):
  """Gets time shift paddings and offsets.

  Arguments:
    time_shift_amount: time shift in samples, for example -100...100

  Returns:
    Shifting parameters which will be used by shift_in_time() function above
  """
  if time_shift_amount > 0:
    time_shift_padding = [[time_shift_amount, 0], [0, 0]]
    time_shift_offset = [0, 0]
  else:
    time_shift_padding = [[0, -time_shift_amount], [0, 0]]
    time_shift_offset = [-time_shift_amount, 0]
  return time_shift_padding, time_shift_offset
