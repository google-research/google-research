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

"""A layer which applies windowing on input data."""
import numpy as np
from kws_streaming.layers.compat import tf


def _hann_greco_window_generator(window_length, dtype):
  """Computes a greco-style hanning window.

  Note that the Hanning window in Wikipedia is not the same as the Hanning
  window in Greco.  The Greco3 Hanning window at 0 is NOT 0, as the wikipedia
  page would indicate. Talkin's explanation was that it was like wasting two
  samples to have the values at the edge of the window to be 0.0 exactly.

  Args:
    window_length: The length of the window (typically frame size).
    dtype: TF data type

  Returns:
    Tensor of size frame_size with the window to apply.
  """
  arg = np.pi * 2.0 / (window_length)
  hann = 0.5 - (0.5 * np.cos(arg * (np.arange(window_length) + 0.5)))
  return hann.astype(dtype)


def _hann_window_generator(window_length, dtype):
  """Computes a standard version of Hann window.

  More details at https://en.wikipedia.org/wiki/Hann_function
  Args:
    window_length: The length of the window (typically frame size).
    dtype: TF data type

  Returns:
    Tensor of size frame_size with the window to apply.
  """
  arg = 2 * np.pi / window_length
  hann = 0.5 - 0.5 * np.cos(arg * np.arange(window_length))
  return hann.astype(dtype)


class Windowing(tf.keras.layers.Layer):
  """Apply window function on input data.

  This is useful to enhance the ability of an FFT to extract spectral data
  from signal. It is applied on the last dim of input data
  """

  def __init__(self, window_size=400, window_type='hann', **kwargs):
    super(Windowing, self).__init__(**kwargs)
    self.window_size = window_size
    self.window_type = window_type

  def build(self, input_shape):
    super(Windowing, self).build(input_shape)
    self.window_size = int(input_shape[-1])
    if self.window_type == 'hann_greco':
      self.window = _hann_greco_window_generator(self.window_size, np.float32)
    elif self.window_type == 'hann':
      self.window = _hann_window_generator(self.window_size, np.float32)
    else:
      raise ValueError('unsupported window_type:%s' % self.window_type)

  def call(self, inputs):
    # last dim has to be the same with window_size
    if inputs.shape[-1] != self.window_size:
      raise ValueError('inputs.shape[-1]:%d must = self.window_size:%d' %
                       (inputs.shape[-1], self.window_size))

    return inputs * self.window

  def get_config(self):
    config = {'window_size': self.window_size, 'window_type': self.window_type}
    base_config = super(Windowing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
