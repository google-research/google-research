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
"""Tensorflow code for frontend functions in frontend.py."""

import numpy as np
import tensorflow as tf


def stabilized_log(data, additive_offset, floor):
  """TF version of mfcc_mel.StabilizedLog."""
  return tf.math.log(tf.math.maximum(data, floor) + additive_offset)


def log_mel_spectrogram(data,
                        audio_sample_rate,
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
      num_mel_bins=64,
      num_spectrogram_bins=fft_length // 2 + 1,
      sample_rate=audio_sample_rate,
      lower_edge_hertz=125.0,
      upper_edge_hertz=7500.0,
      dtype=tf.dtypes.float64
  )

  mel = spectrogram @ to_mel
  log_mel = stabilized_log(mel, log_additive_offset, log_floor)
  return log_mel


def compute_frontend_features(samples, sr, overlap_seconds, tflite=False):
  """Compute features."""
  if tflite:
    raise ValueError("TFLite frontend unsupported")
  if samples.dtype == np.int16:
    samples = tf.cast(samples, np.float32) / np.iinfo(np.int16).max
  if samples.dtype == np.float64:
    samples = tf.cast(samples, np.float32)
  assert samples.dtype == np.float32, samples.dtype
  n = tf.size(samples)
  n_required = 16000
  samples = tf.cond(
      n < n_required,
      lambda: tf.pad(samples, [(0, n_required - n)]),
      lambda: samples
  )
  mel = log_mel_spectrogram(samples, sr)
  # Frame to ~.96 seconds per chunk (96 frames) with ~.0.793 second overlap.
  step = 96 - overlap_seconds
  mel = tf.signal.frame(mel, frame_length=96, frame_step=step, axis=0)
  return mel
