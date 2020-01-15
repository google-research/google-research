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

"""Wrapper for the AudioSet VGGish model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf  # tf

from tensorflow_models.audioset import mel_features
from tensorflow_models.audioset import vggish_params
from tensorflow_models.audioset import vggish_slim


class AudioSetModel(object):
  """Wrapper class for the AudioSet VGGish model."""

  def __init__(self, checkpoint, step_size=None, normalize=True):
    """Initializes AudioSetModel.

    Args:
      checkpoint: path to the model checkpoint that should be loaded.
      step_size: Number of samples to shift for each input feature. If
        unspecified, step size will be set to the window size.
      normalize: Normalizes the sample loudness prior the feature extraction.
    """
    with tf.Graph().as_default():
      self._sess = tf.Session()
      vggish_slim.define_vggish_slim()
      vggish_slim.load_vggish_slim_checkpoint(self._sess, checkpoint)
      self._features_tensor = self._sess.graph.get_tensor_by_name(
          vggish_params.INPUT_TENSOR_NAME)
      self._embedding_tensor = self._sess.graph.get_tensor_by_name(
          vggish_params.OUTPUT_TENSOR_NAME)

    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    self._example_window_length = int(
        round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    self._example_hop_length = int(
        round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    self._step_size = step_size
    self._normalize = normalize

  def process_batch(self, data):
    """Computes the embedding from a batched input.

    Args:
      data: Intup to the model. 2d numpy array of shape: (batch_size,
        feature_size).

    Returns:
      Embeddings as a 2d numpy array of shape:  (batch_size, embedding_size).
    """
    [embedding_batch] = self._sess.run([self._embedding_tensor],
                                       feed_dict={self._features_tensor: data})
    return embedding_batch

  def extract_features(self, np_samples):
    """Converts audio samples into an array of examples for VGGish.

    Args:
      np_samples: 1d np.array with shape (#number_of_samples). Each sample is
        generally expected to lie in the range [-1.0, +1.0].

    Returns:
      List of numpy arrays that can be used as inputs to the model.
    """
    log_mel_examples = []
    samples = np_samples.shape[0]
    if self._normalize:
      min_ratio = 0.1  # = 10^(max_db/-20) with max_db = 20
      np_samples /= np.maximum(min_ratio, np.amax(np_samples))
    if self._step_size is not None:
      samples_splits = []
      for i in xrange(0, samples - vggish_params.SAMPLE_RATE + 1,
                      self._step_size):
        samples_splits.append(np_samples[i:i + vggish_params.SAMPLE_RATE])
    else:
      samples_splits = np.split(np_samples, samples / vggish_params.SAMPLE_RATE)
    # Compute log mel spectrogram features.
    for samples_window in samples_splits:
      log_mel = mel_features.log_mel_spectrogram(
          samples_window,
          audio_sample_rate=vggish_params.SAMPLE_RATE,
          log_offset=vggish_params.LOG_OFFSET,
          window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
          hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
          num_mel_bins=vggish_params.NUM_MEL_BINS,
          lower_edge_hertz=vggish_params.MEL_MIN_HZ,
          upper_edge_hertz=vggish_params.MEL_MAX_HZ)

      log_mel_examples.append(
          mel_features.frame(
              log_mel,
              window_length=self._example_window_length,
              hop_length=self._example_window_length))
    return log_mel_examples
