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

"""A layer which computes mel spectrum of input speech signal."""
from kws_streaming.layers.compat import tf
import kws_streaming.layers.mel_table as mel_table
from kws_streaming.layers.modes import Modes


class MelSpectrogram(tf.keras.layers.Layer):
  """Compute Mel Spectrogram.

  This is useful for speech feature extraction.
  It computes mel spectrogram of input spectrum
  """

  def __init__(self,
               mode=Modes.TRAINING,
               use_tf=True,
               num_mel_bins=40,
               lower_edge_hertz=20.0,
               upper_edge_hertz=4000.0,
               sample_rate=16000.0,
               **kwargs):
    super(MelSpectrogram, self).__init__(**kwargs)
    self.num_mel_bins = num_mel_bins
    self.lower_edge_hertz = lower_edge_hertz
    self.upper_edge_hertz = upper_edge_hertz
    self.sample_rate = sample_rate
    self.use_tf = use_tf
    self.mode = mode

  def build(self, input_shape):
    super(MelSpectrogram, self).build(input_shape)
    feature_size = int(input_shape[-1])

    if self.use_tf and self.mode in (Modes.TRAINING,
                                     Modes.NON_STREAM_INFERENCE):
      # precompute mel matrix using tf
      self.mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins=self.num_mel_bins,
          num_spectrogram_bins=feature_size,
          sample_rate=self.sample_rate,
          lower_edge_hertz=self.lower_edge_hertz,
          upper_edge_hertz=self.upper_edge_hertz,
          dtype=tf.float32)
    else:
      # precompute mel matrix using np
      self.mel_weight_matrix = tf.constant(
          mel_table.SpectrogramToMelMatrix(
              num_mel_bins=self.num_mel_bins,
              num_spectrogram_bins=feature_size,
              audio_sample_rate=self.sample_rate,
              lower_edge_hertz=self.lower_edge_hertz,
              upper_edge_hertz=self.upper_edge_hertz),
          dtype=tf.float32)

  def call(self, inputs):
    # Multiply by mel weight matrix (num_spectrogram_bins, num_mel_bins)
    return tf.matmul(inputs, self.mel_weight_matrix)

  def get_config(self):
    config = {
        'num_mel_bins': self.num_mel_bins,
        'lower_edge_hertz': self.lower_edge_hertz,
        'upper_edge_hertz': self.upper_edge_hertz,
        'sample_rate': self.sample_rate,
        'use_tf': self.use_tf,
        'mode': self.mode,
    }
    base_config = super(MelSpectrogram, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return []

  def get_output_state(self):
    return []
