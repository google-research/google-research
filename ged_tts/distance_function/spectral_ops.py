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

"""Library of spectral operations."""

import numpy as np
import tensorflow.compat.v2 as tf

EPSILON = 1e-8  # Small constant to avoid division by zero.

# Mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def aligned_random_crop(waves, frame_length):
  """Get aligned random crops from batches of input waves."""
  n, t = waves[0].shape
  crop_t = frame_length * (t//frame_length - 1)
  offsets = [tf.random.uniform(shape=(), minval=0,
                               maxval=t-crop_t, dtype=tf.int32)
             for _ in range(n)]
  waves_unbatched = [tf.split(w, n, axis=0) for w in waves]
  wave_crops = [[tf.slice(w, begin=[0, o], size=[1, crop_t])
                 for w, o in zip(ws, offsets)] for ws in waves_unbatched]
  wave_crops = [tf.concat(wc, axis=0) for wc in wave_crops]
  return wave_crops


def mel_to_hertz(frequencies_mel):
  """Converts frequencies in `frequencies_mel` from mel to Hertz scale."""
  return _MEL_BREAK_FREQUENCY_HERTZ * (
      tf.math.exp(frequencies_mel / _MEL_HIGH_FREQUENCY_Q) - 1.)


def hertz_to_mel(frequencies_hertz):
  """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
  return _MEL_HIGH_FREQUENCY_Q * tf.math.log(
      1. + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def get_spectral_matrix(n, num_spec_bins=256, use_mel_scale=True,
                        sample_rate=24000):
  """DFT matrix in overcomplete basis returned as a TF tensor.

  Args:
    n: Int. Frame length for the spectral matrix.
    num_spec_bins: Int. Number of bins to use in the spectrogram
    use_mel_scale: Bool. Equally spaced on Mel-scale or Hertz-scale?
    sample_rate: Int. Sample rate of the waveform audio.

  Returns:
    Constructed spectral matrix.
  """
  sample_rate = float(sample_rate)
  upper_edge_hertz = sample_rate / 2.
  lower_edge_hertz = sample_rate / n

  if use_mel_scale:
    upper_edge_mel = hertz_to_mel(upper_edge_hertz)
    lower_edge_mel = hertz_to_mel(lower_edge_hertz)
    mel_frequencies = tf.linspace(lower_edge_mel, upper_edge_mel, num_spec_bins)
    hertz_frequencies = mel_to_hertz(mel_frequencies)
  else:
    hertz_frequencies = tf.linspace(lower_edge_hertz, upper_edge_hertz,
                                    num_spec_bins)

  time_col_vec = (tf.reshape(tf.range(n, dtype=tf.float32), [n, 1])
                  * np.cast[np.float32](2. * np.pi / sample_rate))
  tmat = tf.reshape(hertz_frequencies, [1, num_spec_bins]) * time_col_vec
  dct_mat = tf.math.cos(tmat)
  dst_mat = tf.math.sin(tmat)
  dft_mat = tf.complex(real=dct_mat, imag=-dst_mat)
  return dft_mat


def matmul_real_with_complex(real_input, complex_matrix):
  real_part = tf.matmul(real_input, tf.math.real(complex_matrix))
  imag_part = tf.matmul(real_input, tf.math.imag(complex_matrix))
  return tf.complex(real_part, imag_part)


def calc_spectrograms(waves, window_lengths, spectral_diffs=(0, 1),
                      window_name='hann', use_mel_scale=True,
                      proj_method='matmul', num_spec_bins=256,
                      random_crop=True):
  """Calculate spectrograms with multiple window sizes for list of input waves.

  Args:
    waves: List of float tensors of shape [batch, length] or [batch, length, 1].
    window_lengths: List of Int. Window sizes (frame lengths) to use for
      computing the spectrograms.
    spectral_diffs: Int. order of finite diff. to take before computing specs.
    window_name: Str. Name of the window to use when computing the spectrograms.
      Supports 'hann' and None.
    use_mel_scale: Bool. Whether or not to project to mel-scale frequencies.
    proj_method: Str. Spectral projection method implementation to use.
      Supported are 'fft' and 'matmul'.
    num_spec_bins: Int. Number of bins in the spectrogram.
    random_crop: Bool. Take random crop or not.

  Returns:
    Tuple of lists of magnitude spectrograms, with output[i][j] being the
      spectrogram for input wave i, computed for window length j.
  """
  waves = [tf.squeeze(w, axis=-1) for w in waves]

  if window_name == 'hann':
    windows = [tf.reshape(tf.signal.hann_window(wl, periodic=False), [1, 1, -1])
               for wl in window_lengths]
  elif window_name is None:
    windows = [None] * len(window_lengths)
  else:
    raise ValueError('Unknown window function (%s).' % window_name)

  spec_len_wave = []
  for d in spectral_diffs:
    for length, window in zip(window_lengths, windows):

      wave_crops = waves
      for _ in range(d):
        wave_crops = [w[:, 1:] - w[:, :-1] for w in wave_crops]

      if random_crop:
        wave_crops = aligned_random_crop(wave_crops, length)

      frames = [tf.signal.frame(wc, length, length // 2) for wc in wave_crops]
      if window is not None:
        frames = [f * window for f in frames]

      if proj_method == 'fft':
        ffts = [tf.signal.rfft(f)[:, :, 1:] for f in frames]

      elif proj_method == 'matmul':
        mat = get_spectral_matrix(length, num_spec_bins=num_spec_bins,
                                  use_mel_scale=use_mel_scale)
        ffts = [matmul_real_with_complex(f, mat) for f in frames]

      sq_mag = lambda x: tf.square(tf.math.real(x)) + tf.square(tf.math.imag(x))
      specs_sq = [sq_mag(f) for f in ffts]

      if use_mel_scale and proj_method == 'fft':
        sample_rate = 24000
        upper_edge_hertz = sample_rate / 2.
        lower_edge_hertz = sample_rate / length
        lin_to_mel = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_spec_bins,
            num_spectrogram_bins=length // 2 + 1,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
            dtype=tf.dtypes.float32)[1:]
        specs_sq = [tf.matmul(s, lin_to_mel) for s in specs_sq]

      specs = [tf.sqrt(s+EPSILON) for s in specs_sq]
      spec_len_wave.append(specs)

  spec_wave_len = zip(*spec_len_wave)
  return spec_wave_len


def sum_spectral_dist(specs1, specs2, add_log_l2=True):
  """Sum over distances in frequency space for different window sizes.

  Args:
    specs1: List of float tensors of shape [batch, frames, frequencies].
      Spectrograms of the first wave to compute the distance for.
    specs2: List of float tensors of shape [batch, frames, frequencies].
      Spectrograms of the second wave to compute the distance for.
    add_log_l2: Bool. Whether or not to add L2 in log space to L1 distances.

  Returns:
    Tensor of shape [batch] with sum of L1 distances over input spectrograms.
  """

  l1_distances = [tf.reduce_mean(abs(s1 - s2), axis=[1, 2])
                  for s1, s2 in zip(specs1, specs2)]
  sum_dist = tf.math.accumulate_n(l1_distances)

  if add_log_l2:
    log_deltas = [tf.math.squared_difference(
                      tf.math.log(s1 + EPSILON), tf.math.log(s2 + EPSILON))  # pylint: disable=bad-continuation
                  for s1, s2 in zip(specs1, specs2)]
    log_l2_norms = [tf.reduce_mean(
        tf.sqrt(tf.reduce_mean(ld, axis=-1) + EPSILON), axis=-1)
                    for ld in log_deltas]
    sum_log_l2 = tf.math.accumulate_n(log_l2_norms)
    sum_dist += sum_log_l2

  return sum_dist


def ged(wav_fake1, wav_fake2, wav_real):
  """Multi-scale spectrogram-based generalized energy distance.

  Args:
    wav_fake1: Float tensors of shape [batch, time, 1].
      Generated audio samples conditional on a set of linguistic features.
    wav_fake2: Float tensors of shape [batch, time, 1].
      Second set of samples conditional on same features, but using new noise.
    wav_real: Float tensors of shape [batch, time, 1].
      Real (data) audio samples corresponding to the same features.

  Returns:
    Tensor of shape [batch] with the GED values.
  """

  specs_fake1, specs_fake2, specs_real = calc_spectrograms(
      waves=[wav_fake1, wav_fake2, wav_real],
      window_lengths=[2**i for i in range(6, 12)])

  dist_real_fake1 = sum_spectral_dist(specs_real, specs_fake1)
  dist_real_fake2 = sum_spectral_dist(specs_real, specs_fake2)
  dist_fake_fake = sum_spectral_dist(specs_fake1, specs_fake2)

  return dist_real_fake1 + dist_real_fake2 - dist_fake_fake
