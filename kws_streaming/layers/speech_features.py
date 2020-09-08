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
import frozendict
from kws_streaming.layers import data_frame
from kws_streaming.layers import dct
from kws_streaming.layers import magnitude_rdft_mel
from kws_streaming.layers import modes
from kws_streaming.layers import normalizer
from kws_streaming.layers import preemphasis
from kws_streaming.layers import spectrogram_augment
from kws_streaming.layers import windowing
from kws_streaming.layers.compat import tf
from tensorflow.python.ops import gen_audio_ops as audio_ops  # pylint: disable=g-direct-tensorflow-import


class SpeechFeatures(tf.keras.layers.Layer):
  """Compute speech features.

  This is useful for speech feature extraction.
  It is stateful: all internal states are managed by this class
  """

  def __init__(
      self,
      params,
      mode=modes.Modes.TRAINING,  # layer with state has to have it
      inference_batch_size=1,
      noise_scale=0.0,
      mean=None,
      stddev=None,
      **kwargs):
    """Inits SpeechFeatures.

    Args:
      params: dict with parameters of speech feature extractor.
        It is definend in above function get_params().
        All parameters are command line arguments described at base_parser.py
      mode: mode can be training, non streaming inference, streaming inference
        with external or internal state
      inference_batch_size: batch size for inference mode
      noise_scale: parameter of noise added to input audio
        can be used during training for regularization
      mean: mean value of input features
        used for input audio normalization
      stddev: standard deviation of input features
        used for input audio normalization
      **kwargs: optional arguments for keras layer
    """
    super(SpeechFeatures, self).__init__(**kwargs)

    self.params = params
    self.mode = mode
    self.inference_batch_size = inference_batch_size
    self.noise_scale = noise_scale
    self.mean = mean
    self.stddev = stddev

    # convert milliseconds to discrete samples
    self.frame_size = int(
        round(self.params['sample_rate'] * self.params['window_size_ms'] /
              1000.0))
    self.frame_step = int(
        round(self.params['sample_rate'] * self.params['window_stride_ms'] /
              1000.0))

  def build(self, input_shape):
    super(SpeechFeatures, self).build(input_shape)

    self.data_frame = data_frame.DataFrame(
        mode=self.mode,
        inference_batch_size=self.inference_batch_size,
        frame_size=self.frame_size,
        frame_step=self.frame_step)

    if self.noise_scale != 0.0 and self.mode == modes.Modes.TRAINING:
      self.add_noise = tf.keras.layers.GaussianNoise(stddev=self.noise_scale)
    else:
      self.add_noise = tf.keras.layers.Lambda(lambda x: x)

    if self.params['preemph'] != 0.0:
      self.preemphasis = preemphasis.Preemphasis(
          preemph=self.params['preemph'])
    else:
      self.preemphasis = tf.keras.layers.Lambda(lambda x: x)

    if self.params['window_type'] is not None:
      self.windowing = windowing.Windowing(
          window_size=self.frame_size, window_type=self.params['window_type'])
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
        use_tf_fft=self.params['use_tf_fft'],
        magnitude_squared=self.params['fft_magnitude_squared'],
        num_mel_bins=self.params['mel_num_bins'],
        lower_edge_hertz=self.params['mel_lower_edge_hertz'],
        upper_edge_hertz=self.params['mel_upper_edge_hertz'],
        sample_rate=self.params['sample_rate'],
        mel_non_zero_only=self.params['mel_non_zero_only'])

    self.log_max = tf.keras.layers.Lambda(
        lambda x: tf.math.log(tf.math.maximum(x, self.params['log_epsilon'])))

    if self.params['dct_num_features'] != 0:
      self.dct = dct.DCT(num_features=self.params['dct_num_features'])
    else:
      self.dct = tf.keras.layers.Lambda(lambda x: x)

    self.normalizer = normalizer.Normalizer(
        mean=self.mean, stddev=self.stddev)

    # in any inference mode there is no need to add dynamic logic in tf graph
    if self.params['use_spec_augment'] and self.mode == modes.Modes.TRAINING:
      self.spec_augment = spectrogram_augment.SpecAugment(
          time_masks_number=self.params['time_masks_number'],
          time_mask_max_size=self.params['time_mask_max_size'],
          frequency_masks_number=self.params['frequency_masks_number'],
          frequency_mask_max_size=self.params['frequency_mask_max_size'])
    else:
      self.spec_augment = tf.keras.layers.Lambda(lambda x: x)

  def _mfcc_tf(self, inputs):
    # MFCC implementation based on TF.
    # It is based on DFT which is computed using matmul with const weights.
    # Where const weights are the part of the model, it increases model size.
    outputs = self.data_frame(inputs)
    outputs = self.add_noise(outputs)
    outputs = self.preemphasis(outputs)
    outputs = self.windowing(outputs)
    outputs = self.mag_rdft_mel(outputs)
    outputs = self.log_max(outputs)
    outputs = self.dct(outputs)
    outputs = self.normalizer(outputs)
    outputs = self.spec_augment(outputs)
    return outputs

  def _mfcc_op(self, inputs):
    # MFCC implementation based on TF custom op (supported by TFLite)
    # It reduces model size in comparison to _mfcc_tf
    if (self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE or
        self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE):
      outputs = self.data_frame(inputs)
      # in streaming mode there is only one frame for FFT calculation
      # dims will be [batch=1, time=1, frame],
      # but audio_spectrogram requre 2D input data, so we remove time dim
      outputs = tf.squeeze(outputs, axis=1)
    else:
      outputs = inputs

    # outputs has dims [batch, time]
    # but audio_spectrogram expects [time, channels/batch] so transpose it
    outputs = tf.transpose(outputs, [1, 0])

    # outputs: [time, channels/batch]
    outputs = audio_ops.audio_spectrogram(
        outputs,
        window_size=self.frame_size,
        stride=self.frame_step,
        magnitude_squared=self.params['fft_magnitude_squared'])
    # outputs: [channels/batch, frames, fft_feature]

    outputs = audio_ops.mfcc(
        outputs,
        self.params['sample_rate'],
        upper_frequency_limit=self.params['mel_upper_edge_hertz'],
        lower_frequency_limit=self.params['mel_lower_edge_hertz'],
        filterbank_channel_count=self.params['mel_num_bins'],
        dct_coefficient_count=self.params['dct_num_features'])
    # outputs: [channels/batch, frames, dct_coefficient_count]
    outputs = self.spec_augment(outputs)
    return outputs

  def call(self, inputs):
    if self.params['feature_type'] == 'mfcc_tf':
      outputs = self._mfcc_tf(inputs)
    elif self.params['feature_type'] == 'mfcc_op':
      outputs = self._mfcc_op(inputs)
    else:
      raise ValueError('unsupported feature_type', self.params['feature_type'])

    return outputs

  def get_config(self):
    config = {
        'inference_batch_size': self.inference_batch_size,
        'params': self.params,
        'mode': self.mode,
        'noise_scale': self.noise_scale,
        'mean': self.mean,
        'stddev': self.stddev,
    }
    base_config = super(SpeechFeatures, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return self.data_frame.get_input_state()

  def get_output_state(self):
    return self.data_frame.get_output_state()

  @staticmethod
  def get_params(flags):
    """Gets parameters for speech feature extractor.

    Args:
      flags: flags from command line
    Returns:
      dict with parameters
    """
    params = {
        'sample_rate': flags.sample_rate,
        'window_size_ms': flags.window_size_ms,
        'window_stride_ms': flags.window_stride_ms,
        'feature_type': flags.feature_type,
        'preemph': flags.preemph,
        'mel_lower_edge_hertz': flags.mel_lower_edge_hertz,
        'mel_upper_edge_hertz': flags.mel_upper_edge_hertz,
        'log_epsilon': flags.log_epsilon,
        'dct_num_features': flags.dct_num_features,
        'mel_non_zero_only': flags.mel_non_zero_only,
        'fft_magnitude_squared': flags.fft_magnitude_squared,
        'mel_num_bins': flags.mel_num_bins,
        'window_type': flags.window_type,
        'use_spec_augment': flags.use_spec_augment,
        'time_masks_number': flags.time_masks_number,
        'time_mask_max_size': flags.time_mask_max_size,
        'frequency_masks_number': flags.frequency_masks_number,
        'frequency_mask_max_size': flags.frequency_mask_max_size,
        'use_tf_fft': flags.use_tf_fft,
    }
    return frozendict.frozendict(params)
