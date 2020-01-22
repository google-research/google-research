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
"""Tests for kws_streaming.layers.conv2d."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes
from kws_streaming.layers.stream import Stream
from kws_streaming.models import utils


class Conv2DTest(tf.test.TestCase):

  def setUp(self):
    super(Conv2DTest, self).setUp()
    self.filters = 2
    self.kernel_size = (3, 3)
    self.batch_size = 2
    self.time_dim = 7
    self.feature_dim = 8
    self.input_shape = [self.batch_size, self.time_dim, self.feature_dim, 1]

    self.inputs = np.arange(
        self.batch_size * self.time_dim * self.feature_dim,
        dtype=np.float32).reshape(self.input_shape)
    self.expected_base_output = self._get_expected_output()

  def test_non_streaming(self):
    # Standard convolution - used for training.
    mode = Modes.NON_STREAM_INFERENCE

    layer = self._get_conv2d_layer(mode)
    inputs = tf.keras.layers.Input(
        shape=(self.time_dim, self.feature_dim, 1), batch_size=self.batch_size)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)
    model_output = model.predict(self.inputs)

    self.assertAllClose(model_output, self.expected_base_output)

  def test_streaming_internal_state(self):
    # Streaming convolution with internal state - used for inference.
    mode = Modes.STREAM_INTERNAL_STATE_INFERENCE

    layer = self._get_conv2d_layer(mode)
    inputs = tf.keras.layers.Input(
        shape=(1, self.feature_dim, 1), batch_size=self.batch_size)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)

    # Simulating streaming using a loop
    for i in range(self.time_dim):
      input_feature = self.inputs[:, i, :, :]

      # Creating a fake "time" dimension
      input_feature = np.expand_dims(input_feature, 1)
      model_output = model.predict(input_feature)

      for b in range(self.batch_size):
        self.assertAllClose(model_output[b][0], self.expected_base_output[b][i])

  def test_streaming_external_state(self):
    # Streaming convolution with external state - used for inference.
    # Create a non-streaming model first
    mode = Modes.TRAINING
    layer = self._get_conv2d_layer(mode)
    inputs = tf.keras.layers.Input(
        shape=(self.time_dim, self.feature_dim, 1), batch_size=self.batch_size)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)

    # Swap to streaming mode
    mode = Modes.STREAM_EXTERNAL_STATE_INFERENCE
    input_tensors = [
        tf.keras.Input(
            shape=(
                1,
                self.feature_dim,
                1,
            ),
            batch_size=self.batch_size,
            name='inp1')
    ]

    # Initialize the first state with zeros.
    input_states = np.zeros(
        [self.batch_size, self.kernel_size[0], self.feature_dim, 1])

    # Use the pipeline to convert the model into streaming external state
    model_stream = utils.convert_to_inference_model(model, input_tensors, mode)

    # Simulating streaming using a for loop
    for i in range(self.time_dim):
      input_feature = self.inputs[:, i, :, :]
      input_feature = np.expand_dims(input_feature, 1)
      output_np, output_states = model_stream.predict(
          [input_feature, input_states])

      # Propagate the output state as the input state of the next iteration.
      input_states = output_states
      for b in range(self.batch_size):  # loop over batch
        self.assertAllClose(output_np[b][0], self.expected_base_output[b][i])

  def test_dilation(self):
    # Test the logic with dilation rate larger than (1, 1)
    mode = Modes.STREAM_INTERNAL_STATE_INFERENCE
    dilation_rate = (2, 2)
    output_with_dilation = self._get_expected_output(
        dilation_rate=dilation_rate)

    layer = self._get_conv2d_layer(mode, dilation_rate=dilation_rate)
    inputs = tf.keras.layers.Input(
        shape=(1, self.feature_dim, 1), batch_size=self.batch_size)
    outputs = layer(inputs)
    model = tf.keras.Model(inputs, outputs)

    for i in range(self.time_dim):
      input_feature = self.inputs[:, i, :, :]
      input_feature = np.expand_dims(input_feature, 1)
      model_output = model.predict(input_feature)
      for b in range(self.batch_size):
        self.assertAllClose(model_output[b][0], output_with_dilation[b][i])

  def test_stacked_layers(self):
    # Test that the layers play nice with each other
    mode = Modes.STREAM_INTERNAL_STATE_INFERENCE
    stacked_output = self._get_expected_output(stacked=True)
    layer = self._get_conv2d_layer(mode)
    layer2 = self._get_conv2d_layer(mode)
    inputs = tf.keras.layers.Input(
        shape=(1, self.feature_dim, 1), batch_size=self.batch_size)
    outputs = layer(inputs)
    outputs = layer2(outputs)
    model = tf.keras.Model(inputs, outputs)

    for i in range(self.time_dim):
      input_feature = self.inputs[:, i, :, :]
      input_feature = np.expand_dims(input_feature, 1)
      model_output = model.predict(input_feature)
      for b in range(self.batch_size):
        self.assertAllClose(model_output[b][0], stacked_output[b][i])

  def _get_conv2d_layer(self, mode, dilation_rate=(1, 1)):
    cell = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=self.kernel_size,
        dilation_rate=dilation_rate,
        kernel_initializer='ones')
    return Stream(
        cell,
        mode=mode,
        inference_batch_size=self.batch_size,
        pad_time_dim=True,
    )

  def _get_expected_output(self, dilation_rate=(1, 1), stacked=False):
    # Pad the front to match the padding of the streamed version
    dilated_kernel_size = dilation_rate[0] * (self.kernel_size[0] - 1) + 1
    inputs_conv = np.pad(self.inputs,
                         ((0, 0), (dilated_kernel_size - 1, 0), (0, 0), (0, 0)),
                         'constant')

    # Put through basic convolution layer
    layer = tf.keras.layers.Conv2D(
        self.filters,
        self.kernel_size,
        dilation_rate=dilation_rate,
        kernel_initializer='ones')
    inputs = tf.keras.layers.Input(
        shape=(self.time_dim + dilated_kernel_size - 1, self.feature_dim, 1),
        batch_size=self.batch_size)
    outputs = layer(inputs)

    # Stacking 2 convolutional layers on top of each other.
    if stacked:
      padded_outputs = tf.pad(outputs, ((0, 0), (dilated_kernel_size - 1, 0),
                                        (0, 0), (0, 0)), 'constant')

      layer = tf.keras.layers.Conv2D(
          self.filters,
          self.kernel_size,
          dilation_rate=dilation_rate,
          kernel_initializer='ones')
      outputs = layer(padded_outputs)

    model = tf.keras.Model(inputs, outputs)
    model_output = model.predict(inputs_conv)
    return model_output


if __name__ == '__main__':
  tf.test.main()
