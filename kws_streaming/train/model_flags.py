# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Model/data settings manipulation."""

import math
import os
import kws_streaming.data.input_data as input_data
from kws_streaming.layers.modes import Modes
from kws_streaming.models import utils

MS_PER_SECOND = 1000  # milliseconds in 1 second


def update_flags(flags):
  """Update flags with new parameters.

  Args:
    flags: All model and data parameters

  Returns:
    Updated flags

  Raises:
    ValueError: If the preprocessing mode isn't recognized.
  """

  label_count = len(
      input_data.prepare_words_list(flags.wanted_words.split(',')))
  desired_samples = int(flags.sample_rate * flags.clip_duration_ms /
                        MS_PER_SECOND)
  window_size_samples = int(flags.sample_rate * flags.window_size_ms /
                            MS_PER_SECOND)
  window_stride_samples = int(flags.sample_rate * flags.window_stride_ms /
                              MS_PER_SECOND)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if flags.preprocess == 'raw':
    average_window_width = -1
    fingerprint_width = desired_samples
    spectrogram_length = 1
  elif flags.preprocess == 'average':
    fft_bin_count = 1 + (utils.next_power_of_two(window_size_samples) / 2)
    average_window_width = int(
        math.floor(fft_bin_count / flags.feature_bin_count))
    fingerprint_width = int(
        math.ceil(float(fft_bin_count) / average_window_width))
  elif flags.preprocess == 'mfcc':
    average_window_width = -1
    fingerprint_width = flags.feature_bin_count
  elif flags.preprocess == 'micro':
    average_window_width = -1
    fingerprint_width = flags.feature_bin_count
  else:
    raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                     ' "average", or "micro")' % (flags.preprocess))

  fingerprint_size = fingerprint_width * spectrogram_length

  upd_flags = flags
  upd_flags.mode = Modes.TRAINING
  upd_flags.label_count = label_count
  upd_flags.desired_samples = desired_samples
  upd_flags.window_size_samples = window_size_samples
  upd_flags.window_stride_samples = window_stride_samples
  upd_flags.spectrogram_length = spectrogram_length
  upd_flags.fingerprint_width = fingerprint_width
  upd_flags.fingerprint_size = fingerprint_size
  upd_flags.average_window_width = average_window_width

  # summary logs for TensorBoard
  upd_flags.summaries_dir = os.path.join(flags.train_dir, 'logs/')
  return upd_flags
