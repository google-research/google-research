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

"""Tests for kws_streaming.layers.gru."""

import numpy as np
from kws_streaming.layers import gru
from kws_streaming.layers import modes
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
tf1.disable_eager_execution()


class GRUTest(tf.test.TestCase):

  def setUp(self):
    super(GRUTest, self).setUp()
    test_utils.set_seed(123)

    # generate input signal
    self.inference_batch_size = 1
    self.data_size = 32
    self.feature_size = 4
    self.signal = np.random.rand(self.inference_batch_size, self.data_size,
                                 self.feature_size)
    # create non streamable model
    inputs = tf.keras.layers.Input(
        shape=(self.data_size, self.feature_size),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    self.units = 3
    outputs = gru.GRU(units=self.units, return_sequences=True)(inputs)
    self.model_non_streamable = tf.keras.Model(inputs, outputs)
    self.output_gru = self.model_non_streamable.predict(self.signal)

  def test_streaming_inference_internal_state(self):
    # create streaming inference model with internal state
    mode = modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
    inputs = tf.keras.layers.Input(
        shape=(1, self.feature_size),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    outputs = gru.GRU(units=self.units, mode=mode)(inputs)
    model_stream = tf.keras.Model(inputs, outputs)

    # set weights + states
    weights_states = self.model_non_streamable.get_weights() + [
        np.zeros((self.inference_batch_size, self.units))
    ]
    model_stream.set_weights(weights_states)

    # compare streamable vs non streamable models
    for i in range(self.data_size):  # loop over time samples
      input_stream = self.signal[:, i, :]
      input_stream = np.expand_dims(input_stream, 1)
      output_stream = model_stream.predict(input_stream)
      self.assertAllClose(output_stream[0][0], self.output_gru[0][i])

  def test__streaming_inference_external_state(self):
    # create streaming inference model with external state
    mode = modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
    inputs = tf.keras.layers.Input(
        shape=(1, self.feature_size),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    gru_layer = gru.GRU(units=self.units, mode=mode)
    outputs = gru_layer(inputs)
    model_stream = tf.keras.Model([inputs] + gru_layer.get_input_state(),
                                  [outputs] + gru_layer.get_output_state())
    # set weights only
    model_stream.set_weights(self.model_non_streamable.get_weights())

    # input states
    input_state1 = np.zeros((self.inference_batch_size, self.units))

    # compare stateless streamable vs non streamable models
    for i in range(self.data_size):  # loop over time samples
      input_stream = self.signal[:, i, :]
      input_stream = np.expand_dims(input_stream, 1)
      output_streams = model_stream.predict([input_stream, input_state1])

      # update input states
      input_state1 = output_streams[1]

      # compare streaming and non streaming outputs
      self.assertAllClose(output_streams[0][0][0], self.output_gru[0][i])


if __name__ == '__main__':
  tf.test.main()
