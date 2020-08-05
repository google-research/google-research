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
"""Re-implementation of the DS2 encoder (Amodei et al., 2016).

Made to be compatible with Open Seq2Seq implementation (Kuchaev et al., 2018).

The architecture is compatible with the original weights and architecture, but
had to be re-implemented because the original checkpoint metagraph relied on
CuDNN for GRUs and was impossible to run on other accelerators.
"""

import numpy as np
import scipy
import scipy.linalg

import tensorflow.compat.v2 as tf
import tensorflow.compat.v2.keras as K


class DS2Encoder(tf.Module):
  """Re-implementation of the DeepSpeech2 ASR model."""

  def __init__(self,
               conv_desc=(((11, 41), (2, 2), 32),
                          ((11, 21), (1, 2), 32)),
               gru_num_layers=5,
               gru_num_units=800,
               num_hidden=1600,
               dropout_rate=0.5,
               vocab_size=29,
               num_features=160,
               sample_freq=24000,
               window_size=0.02,
               window_step=0.01):
    """Model constructor.

    Populated with default values to match Open Seq2Seq ds2_large_8gpus_mp.

    Args:
      conv_desc: Iterable of (kernel_size, stride, channels) tuples. Description
        of the convolutions.
      gru_num_layers: Int. GRU layers.
      gru_num_units: Int. Hidden state dimensionality of the GRU layers.
      num_hidden: Int. Number of channels to use for the pre-logits layer.
      dropout_rate: Float. Droupout rate. Dropout is applied after every GRU
        layer, and after the pre-logits.
      vocab_size: Int. Size of the vocabulary (i.e. size of the logits fed to
        the CTCDecoder).

      The following features are only used if computing representations in TF.

      num_features: Int. Number of frequencies to select from the STFT.
      sample_freq: Int. Waveform sampling frequency.
      window_size: Float. Size of the STFT window in seconds.
      window_step: Float. Size of the STFT step in seconds.
    """
    super(DS2Encoder, self).__init__()
    self.conv_desc = conv_desc
    self.gru_num_layers = gru_num_layers
    self.gru_num_units = gru_num_units
    self.num_hidden = num_hidden
    self.dropout_rate = dropout_rate
    self.vocab_size = vocab_size

    self.num_features = num_features
    self.sample_freq = sample_freq
    self.window_size = window_size
    self.window_step = window_step

    # Convolutions.
    self.conv, self.bn = [], []
    for ks, stride, channels in self.conv_desc:
      conv = K.layers.Conv2D(
          channels, ks, stride, padding='SAME', use_bias=False)
      bn = K.layers.BatchNormalization()
      self.conv.append(conv)
      self.bn.append(bn)

    # GRUs.
    self.gru_stack = []
    for _ in range(self.gru_num_layers):
      gru = K.layers.Bidirectional(
          K.layers.GRU(self.gru_num_units,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       recurrent_dropout=0.,
                       dropout=self.dropout_rate,
                       unroll=False,
                       use_bias=True,
                       reset_after=True,
                       time_major=False,
                       implementation=2,
                       return_sequences=True),
          merge_mode='concat')
      self.gru_stack.append(gru)

    # Fully-connected.
    # Should be float16, but it is not supported on TPU.
    self.fully_connected = K.layers.Dense(self.num_hidden,
                                          activation=tf.nn.relu)

    # Dropout.
    self.dropout = K.layers.Dropout(self.dropout_rate)

    # Fully-connected for CTC.
    # Should be float16, but it is not supported on TPU.
    self.fully_connected_ctc = K.layers.Dense(self.vocab_size)

  def represent(self, waves):
    """Transform waves into a representation suited for the DS2 encoder."""
    waves = tf.squeeze(waves, -1)

    # Re-scale.
    waves = waves / (tf.reduce_max(tf.abs(waves), axis=1, keepdims=True) + 1e-5)
    waves *= 32767
    # To match PSF the following line should be uncommented. But it's not
    # supported by TPUs.
    # waves = tf.cast(tf.cast(waves, tf.int16), waves.dtype)  # Matching PSF.

    # Determine frame and step sizes.
    window_size = int(self.sample_freq * self.window_size)
    window_step = int(self.sample_freq * self.window_step)

    # Compute STFT.
    fft_window = tf.signal.hann_window(
        window_size, periodic=False, dtype=waves.dtype)
    fft_window = tf.reshape(fft_window, [1, 1, window_size])

    frames = tf.signal.frame(waves, window_size, window_step, True)
    # Do the slow DFT matmul because window size generally will not be a power
    # of 2.
    dft_w = scipy.linalg.dft(window_size).astype(np.complex64)
    stft = tf.matmul(tf.cast(fft_window * frames, dft_w.dtype), dft_w)
    mag = tf.abs(stft) / float(window_size)
    mag = tf.where(tf.less_equal(mag, 1e-30), tf.ones_like(mag) * 1e-30, mag)
    log_mag = 10. * tf.math.log(mag) / tf.math.log(10.)

    # Select features and standardize.
    features = log_mag[Ellipsis, :self.num_features]

    counts, means_ss, variance_ss, _ = tf.nn.sufficient_statistics(
        features, axes=[1, 2], keepdims=True)
    mean, variance = tf.nn.normalize_moments(
        counts, means_ss, variance_ss, None)
    features = (features - mean) / tf.sqrt(variance)

    return features

  @tf.Module.with_name_scope
  def __call__(self, waves=None, inputs=None, training=False):
    endpoints = {}

    if (inputs is not None) and (waves is not None):
      raise ValueError('Either inputs or waves must be provided, but not both.')

    if waves is not None:
      inputs = self.represent(waves)

    # Reshape for 2D conv.
    inputs = tf.expand_dims(inputs, -1)

    # Convs.
    for conv, bn in zip(self.conv, self.bn):
      inputs = conv(inputs)
      inputs = bn(inputs, training=training)
      inputs = tf.nn.relu(inputs)

    # GRUs.
    inputs = tf.reshape(inputs, inputs.shape[:2] + [np.prod(inputs.shape[2:])])
    for gru in self.gru_stack:
      inputs = gru(inputs, training=training)

    # Reshape + fully-connected (i.e. 1x1 convolution).
    # inputs = tf.cast(inputs, tf.float16)
    # We should be using float16, but it is not supported on TPUs.
    batch_size, time_dim, channels = inputs.shape
    inputs = tf.reshape(inputs, [batch_size * time_dim, channels])
    inputs = self.fully_connected(inputs)

    # Dropout.
    inputs = self.dropout(inputs, training=training)
    inputs = tf.reshape(inputs, [batch_size, time_dim, self.num_hidden])
    endpoints['activations'] = inputs

    # Pooled activations are used by Frechet DeepSpeech distance. Adding them
    # to the model outputs.
    # TODO(agritsenko): Keep the model clean by finding a way to not litter it
    #  with metric-related quantities.
    endpoints['pooled_activations'] = tf.reduce_mean(inputs, axis=1)

    # Reshape + fully-connected (i.e. 1x1 convolution).
    batch_size, time_dim, channels = inputs.shape
    inputs = tf.reshape(inputs, [batch_size * time_dim, channels])
    inputs = self.fully_connected_ctc(inputs)
    inputs = tf.reshape(inputs, [batch_size, time_dim, self.vocab_size])
    endpoints['logits'] = inputs
    return endpoints
