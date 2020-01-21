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

"""A layer which computes Magnitude of RDFT."""
import math
import numpy as np
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes


class MagnitudeRDFT(tf.keras.layers.Layer):
  """Computes Real DFT Spectrogram and then returns its Magnitude.

  It is useful for speech feature extraction.
  It has two implementations one is based on direct DFT - works with TFLite
  and another is based on TF FFT
  """

  def __init__(self,
               mode=Modes.TRAINING,
               use_tf_fft=False,
               magnitude_squared=False,
               **kwargs):
    super(MagnitudeRDFT, self).__init__(**kwargs)
    self.use_tf_fft = use_tf_fft
    self.mode = mode
    self.magnitude_squared = magnitude_squared

  def build(self, input_shape, fft_mel_size=None):
    super(MagnitudeRDFT, self).build(input_shape)
    frame_size = int(input_shape[-1])
    self.fft_size = self._compute_fft_size(frame_size)

    if (self.use_tf_fft and fft_mel_size):
      raise ValueError('TF FFT(True) is not compatible with fft_mel_size')

    if not self.use_tf_fft:
      # it is a real DFT with cos and sin functions only
      # for real and imaginary components accordingly:
      dft_real = np.asarray(
          np.cos(2.0 * np.pi *
                 np.outer(np.arange(self.fft_size), np.arange(self.fft_size)) /
                 self.fft_size),
          dtype=np.float32)
      dft_imag = np.asarray(
          -np.sin(2.0 * np.pi *
                  np.outer(np.arange(self.fft_size), np.arange(self.fft_size)) /
                  self.fft_size),
          dtype=np.float32)

      if fft_mel_size is None:
        dft_real_half = dft_real[:self.fft_size // 2 + 1, :]
        dft_imag_half = dft_imag[:self.fft_size // 2 + 1, :]
      else:
        dft_real_half = dft_real[:fft_mel_size, :]
        dft_imag_half = dft_imag[:fft_mel_size, :]

      dft_real = dft_real_half.transpose()
      dft_imag = dft_imag_half.transpose()

      # extract only array with size of input signal, so that
      # there will be no need to do padding of input signal (it is not FFT)
      # and there will be no multiplications with padded zeros
      self.real_dft_tensor = tf.constant(dft_real[:frame_size, :])
      self.imag_dft_tensor = tf.constant(dft_imag[:frame_size, :])

  def call(self, inputs):
    # compute magnitude of fourier spectrum
    if self.use_tf_fft:
      return self._fft_magnitude(inputs)
    else:
      return self._dft_magnitude(inputs)

  def get_config(self):
    config = {
        'use_tf_fft': self.use_tf_fft,
        'mode': self.mode,
        'fft_size': self.fft_size,
        'magnitude_squared': self.magnitude_squared,
    }
    base_config = super(MagnitudeRDFT, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return []

  def get_output_state(self):
    return []

  def _dft_magnitude(self, signal):
    """Compute DFT and then its magnitude.

    It is avoiding tflite incompatible ops.
    Args:
      signal: has dims [..., frame_size]

    Returns:
      magnitude_spectrogram: with dims [..., fft_size]
    """
    real_spectrum = tf.matmul(signal, self.real_dft_tensor)
    imag_spectrum = tf.matmul(signal, self.imag_dft_tensor)

    magnitude_spectrum = tf.add(real_spectrum * real_spectrum,
                                imag_spectrum * imag_spectrum)
    if self.magnitude_squared:
      return magnitude_spectrum
    else:
      return tf.sqrt(magnitude_spectrum)

  def _fft_magnitude(self, inputs):
    """Compute FFT and returns its magnitude."""
    real_spectrum = tf.signal.rfft(inputs, [self.fft_size])
    magnitude_spectrum = tf.abs(real_spectrum)
    if self.magnitude_squared:
      return tf.square(magnitude_spectrum)
    else:
      return magnitude_spectrum

  def _compute_fft_size(self, frame_size):
    return 2**int(math.ceil(math.log(frame_size) / math.log(2.0)))
