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

# Lint as: python3
"""Tensorflow code for frontend functions in frontend.py."""

from absl import flags
import numpy as np
import tensorflow as tf

# Add flags here even though it's not a binary, for convenience.
flags.DEFINE_integer('frame_hop', 17, 'Frontend fn arg: frame_hop.')
flags.DEFINE_integer('n_required', 16000, 'Frontend fn arg: n_require.')
flags.DEFINE_integer('num_mel_bins', 64, 'Frontend fn arg: num_mel_bins.')
flags.DEFINE_integer('frame_width', 96, 'Frontend fn arg: frame_width.')


def stabilized_log(data, additive_offset, floor):
  """TF version of mfcc_mel.StabilizedLog."""
  return tf.math.log(tf.math.maximum(data, floor) + additive_offset)


def log_mel_spectrogram(data,
                        audio_sample_rate,
                        num_mel_bins=64,
                        log_additive_offset=0.001,
                        log_floor=1e-12,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        fft_length=None):
  """TF version of mfcc_mel.LogMelSpectrogram."""
  window_length_samples = int(round(audio_sample_rate * window_length_secs))
  hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
  if not fft_length:
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

  spectrogram = tf.abs(
      tf.signal.stft(
          tf.cast(data, tf.dtypes.float64),
          frame_length=window_length_samples,
          frame_step=hop_length_samples,
          fft_length=fft_length,
          window_fn=tf.signal.hann_window,
      )
  )

  to_mel = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins=num_mel_bins,
      num_spectrogram_bins=fft_length // 2 + 1,
      sample_rate=audio_sample_rate,
      lower_edge_hertz=125.0,
      upper_edge_hertz=7500.0,
      dtype=tf.dtypes.float64
  )

  mel = spectrogram @ to_mel
  log_mel = stabilized_log(mel, log_additive_offset, log_floor)
  return log_mel


def compute_frontend_features(samples,
                              sr,
                              frame_hop,
                              tflite=False,
                              n_required=16000,
                              num_mel_bins=64,
                              frame_width=96):
  """Compute features."""
  if tflite:
    raise ValueError('TFLite frontend unsupported.')
  if samples.dtype == np.int16:
    samples = tf.cast(samples, np.float32) / np.iinfo(np.int16).max
  if samples.dtype == np.float64:
    samples = tf.cast(samples, np.float32)
  assert samples.dtype == np.float32, samples.dtype
  n = tf.size(samples)
  samples = tf.cond(
      n < n_required,
      lambda: tf.pad(samples, [(0, n_required - n)]),
      lambda: samples
  )
  mel = log_mel_spectrogram(samples, sr, num_mel_bins=num_mel_bins)
  mel = tf.signal.frame(
      mel, frame_length=frame_width, frame_step=frame_hop, axis=0)
  return mel


def frontend_args_from_flags():
  """Return a dictionary of frontend fn args from, parsed from flags."""
  return {
      'frame_hop': flags.FLAGS.frame_hop,
      'n_required': flags.FLAGS.n_required,
      'num_mel_bins': flags.FLAGS.num_mel_bins,
      'frame_width': flags.FLAGS.frame_width,
  }
