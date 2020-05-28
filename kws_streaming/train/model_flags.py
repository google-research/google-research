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

"""Model/data settings manipulation."""

import os
from kws_streaming.data import input_data

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
      input_data.prepare_words_list(
          flags.wanted_words.split(','), flags.split_data))
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

  upd_flags = flags
  upd_flags.label_count = label_count
  upd_flags.desired_samples = desired_samples
  upd_flags.window_size_samples = window_size_samples
  upd_flags.window_stride_samples = window_stride_samples
  upd_flags.spectrogram_length = spectrogram_length
  if upd_flags.fft_magnitude_squared in (0, 1):
    upd_flags.fft_magnitude_squared = bool(upd_flags.fft_magnitude_squared)
  else:
    raise ValueError('Non boolean value %d' % upd_flags.fft_magnitude_squared)

  # summary logs for TensorBoard
  upd_flags.summaries_dir = os.path.join(flags.train_dir, 'logs/')
  return upd_flags
