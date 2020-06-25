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

"""Modes the model can be in and its input data shape."""


class Modes(object):
  """Definition of the mode the model is functioning in."""

  # Model is in a training state. No streaming is done.
  TRAINING = 'TRAINING'

  # Below are three options for inference:

  # Model is in inference mode and has state for efficient
  # computation/streaming, where state is kept inside of the model
  STREAM_INTERNAL_STATE_INFERENCE = 'STREAM_INTERNAL_STATE_INFERENCE'

  # Model is in inference mode and has state for efficient
  # computation/streaming, where state is received from outside of the model
  STREAM_EXTERNAL_STATE_INFERENCE = 'STREAM_EXTERNAL_STATE_INFERENCE'

  # Model its in inference mode and it's topology is the same with training
  # mode (with removed droputs etc)
  NON_STREAM_INFERENCE = 'NON_STREAM_INFERENCE'


def get_input_data_shape(flags, mode):
  """Gets data shape for a neural net input layer.

  Args:
    flags: command line flags, descibed at base_parser.py
    mode: inference mode described above at Modes

  Returns:
    data_shape for input layer
  """

  if mode not in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE,
                  Modes.STREAM_INTERNAL_STATE_INFERENCE,
                  Modes.STREAM_EXTERNAL_STATE_INFERENCE):
    raise ValueError('Unknown mode "%s" ' % flags.mode)

  if flags.preprocess == 'custom':
    # it is a special case to customize input data shape
    # and use model on its own (for debugging only)
    data_shape = flags.data_shape
  elif flags.preprocess == 'raw':
    if mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
      data_shape = (flags.desired_samples,)
    else:
      data_shape = (flags.window_stride_samples,)  # streaming
  elif flags.preprocess == 'mfcc':
    if mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
      data_shape = (flags.spectrogram_length, flags.dct_num_features,)
    else:
      data_shape = (1, flags.dct_num_features,)  # streaming
  elif flags.preprocess == 'micro':
    if mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
      data_shape = (flags.spectrogram_length, flags.mel_num_bins,)
    else:
      data_shape = (1, flags.mel_num_bins,)  # streaming
  else:
    raise ValueError('Unknown preprocess mode "%s"' % flags.preprocess)
  return data_shape
