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
"""Tests for kws_streaming.layers.temporal_padding."""

import itertools
from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import modes
from kws_streaming.layers import temporal_padding
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
tf1.disable_eager_execution()


class TemporalPaddingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(['causal', 'same', 'future'], [5, 0, -5]))
  def test_padding_and_cropping(self, padding, padding_size):
    batch_size = 1
    time_dim = 10
    feature_dim = 3
    inputs = tf.keras.layers.Input(
        shape=(time_dim, feature_dim), batch_size=batch_size)
    net = temporal_padding.TemporalPadding(
        padding=padding, padding_size=padding_size)(
            inputs)
    model = tf.keras.Model(inputs, net)

    np.random.seed(1)
    input_signal = np.random.rand(batch_size, time_dim, feature_dim)
    output_signal = model.predict(input_signal)
    if padding_size >= 0:
      reference_padding = {
          'causal': (padding_size, 0),
          'same': (padding_size // 2, padding_size - padding_size // 2),
          'future': (0, padding_size),
      }[padding]
      output_reference = tf.keras.backend.temporal_padding(
          input_signal, padding=reference_padding)
    else:
      reference_cropping = {
          'causal': (-padding_size, 0),
          'same': ((-padding_size) // 2, -padding_size - (-padding_size) // 2),
          'future': (0, -padding_size),
      }[padding]
      output_reference = tf.keras.layers.Cropping1D(reference_cropping)(
          input_signal)
    self.assertAllClose(output_signal, output_reference)
    self.assertAllEqual(output_signal.shape,
                        [batch_size, time_dim + padding_size, feature_dim])

  @parameterized.parameters(
      itertools.product(['causal', 'same', 'future'], [5, 0, -5]))
  def test_no_padding_or_cropping_in_streaming(self, padding, padding_size):
    batch_size = 1
    feature_dim = 3
    mode = modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
    inputs = tf.keras.layers.Input(
        shape=(1, feature_dim), batch_size=batch_size)
    net = temporal_padding.TemporalPadding(
        padding=padding, padding_size=padding_size, mode=mode)(
            inputs)
    self.assertAllEqual(
        tf.keras.backend.int_shape(net), [batch_size, 1, feature_dim])


if __name__ == '__main__':
  tf.test.main()
