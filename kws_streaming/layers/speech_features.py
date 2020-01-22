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

"""A layer for extracting features from speech data."""
from kws_streaming.layers import magnitude_rdft_mel
from kws_streaming.layers.compat import tf
from kws_streaming.layers.dataframe import DataFrame
from kws_streaming.layers.dct import DCT
from kws_streaming.layers.modes import Modes
from kws_streaming.layers.normalizer import Normalizer
from kws_streaming.layers.preemphasis import Preemphasis
from kws_streaming.layers.windowing import Windowing


class SpeechFeatures(tf.keras.layers.Layer):
  """Compute speech features.

  This is useful for speech feature extraction.
  It is stateful: all internal states are managed by this class
  """

  def __init__(self,
               mode=Modes.TRAINING,
               use_tf_fft=False,
               inference_batch_size=1,
               preemph=0.0,
               window_type='hann',
               frame_size_ms=40.0,
               frame_step_ms=20.0,
               mel_num_bins=40,
               mel_lower_edge_hertz=20.0,
               mel_upper_edge_hertz=4000.0,
               mel_non_zero_only=1,
               log_epsilon=1e-12,
               sample_rate=16000.0,
               noise_scale=0.0,
               fft_magnitude_squared=False,
               dct_num_features=10,
               mean=None,
               stddev=None,
               **kwargs):
    super(SpeechFeatures, self).__init__(**kwargs)

    self.mode = mode
    self.use_tf_fft = use_tf_fft
    self.inference_batch_size = inference_batch_size
    self.preemph = preemph
    self.window_type = window_type
    self.frame_size_ms = frame_size_ms
    self.frame_step_ms = frame_step_ms
    self.mel_num_bins = mel_num_bins
    self.mel_lower_edge_hertz = mel_lower_edge_hertz
    self.mel_upper_edge_hertz = mel_upper_edge_hertz
    self.mel_non_zero_only = mel_non_zero_only
    self.log_epsilon = log_epsilon
    self.sample_rate = sample_rate
    self.noise_scale = noise_scale
    self.fft_magnitude_squared = fft_magnitude_squared
    self.dct_num_features = dct_num_features
    self.mean = mean
    self.stddev = stddev

    # convert milliseconds to discrete samples
    self.frame_size = int(round(sample_rate * frame_size_ms / 1000.0))
    self.frame_step = int(round(sample_rate * frame_step_ms / 1000.0))

  def build(self, input_shape):
    super(SpeechFeatures, self).build(input_shape)

    self.data_frame = DataFrame(
        mode=self.mode,
        inference_batch_size=self.inference_batch_size,
        frame_size=self.frame_size,
        frame_step=self.frame_step)

    if self.noise_scale != 0.0 and self.mode == Modes.TRAINING:
      self.add_noise = tf.keras.layers.GaussianNoise(stddev=self.noise_scale)
    else:
      self.add_noise = tf.keras.layers.Lambda(lambda x: x)

    if self.preemph != 0.0:
      self.preemphasis = Preemphasis(preemph=self.preemph)
    else:
      self.preemphasis = tf.keras.layers.Lambda(lambda x: x)

    if self.window_type is not None:
      self.windowing = Windowing(
          window_size=self.frame_size, window_type=self.window_type)
    else:
      self.windowing = tf.keras.layers.Lambda(lambda x: x)

    # If use_tf_fft is False, we will use
    # Real Discrete Fourier Transformation(RDFT), which is slower than RFFT
    # To increase RDFT efficiency we use properties of mel spectrum.
    # We find a range of non zero values in mel spectrum
    # and use it to compute RDFT: it will speed up computations.
    # If use_tf_fft is True, then we use TF RFFT which require
    # signal length alignment, so we disable mel_non_zero_only.
    self.mag_rdft_mel = magnitude_rdft_mel.MagnitudeRDFTmel(
        use_tf_fft=self.use_tf_fft,
        magnitude_squared=self.fft_magnitude_squared,
        num_mel_bins=self.mel_num_bins,
        lower_edge_hertz=self.mel_lower_edge_hertz,
        upper_edge_hertz=self.mel_upper_edge_hertz,
        sample_rate=self.sample_rate,
        mel_non_zero_only=self.mel_non_zero_only)

    self.log_max = tf.keras.layers.Lambda(
        lambda x: tf.math.log(tf.math.maximum(x, self.log_epsilon)))

    if self.dct_num_features != 0:
      self.dct = DCT(num_features=self.dct_num_features)
    else:
      self.dct = tf.keras.layers.Lambda(lambda x: x)

    self.normalizer = Normalizer(mean=self.mean, stddev=self.stddev)

  def call(self, inputs):
    outputs = self.data_frame(inputs)
    outputs = self.add_noise(outputs)
    outputs = self.preemphasis(outputs)
    outputs = self.windowing(outputs)
    outputs = self.mag_rdft_mel(outputs)
    outputs = self.log_max(outputs)
    outputs = self.dct(outputs)
    outputs = self.normalizer(outputs)
    return outputs

  def get_config(self):
    config = {
        'mode': self.mode,
        'use_tf_fft': self.use_tf_fft,
        'inference_batch_size': self.inference_batch_size,
        'preemph': self.preemph,
        'window_type': self.window_type,
        'frame_size_ms': self.frame_size_ms,
        'frame_step_ms': self.frame_step_ms,
        'mel_num_bins': self.mel_num_bins,
        'mel_lower_edge_hertz': self.mel_lower_edge_hertz,
        'mel_upper_edge_hertz': self.mel_upper_edge_hertz,
        'log_epsilon': self.log_epsilon,
        'sample_rate': self.sample_rate,
        'noise_scale': self.noise_scale,
        'fft_magnitude_squared': self.fft_magnitude_squared,
        'dct_num_features': self.dct_num_features,
        'mean': self.mean,
        'stddev': self.stddev,
    }
    base_config = super(SpeechFeatures, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return self.data_frame.get_input_state()

  def get_output_state(self):
    return self.data_frame.get_output_state()
