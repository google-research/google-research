# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for kws_streaming.layers.counter."""

from absl.testing import parameterized
import numpy as np

from kws_streaming.layers import counter
from kws_streaming.layers import modes
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import utils


class CounterTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(CounterTest, self).setUp()
    test_utils.set_seed(123)
    self.time_size = 30
    self.feature_size = 3
    self.max_counter = 11
    self.input_non_stream_np = np.random.randn(1, self.time_size,
                                               self.feature_size)

    inputs = tf.keras.layers.Input(
        shape=(
            self.time_size,
            self.feature_size,
        ), batch_size=1)
    net = counter.Counter(max_counter=self.max_counter)(inputs)
    self.model = tf.keras.Model(inputs, net)

  def test_inference_non_stream(self):
    # confirm that in non streaming mode it returns input always
    for _ in range(self.max_counter * 2):
      self.assertAllClose(
          self.input_non_stream_np,
          self.model.predict(self.input_non_stream_np))

  def test_inference_internal_state(self):
    mode = modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
    input_tensors = [
        tf.keras.layers.Input(
            shape=(
                1,
                self.feature_size,
            ), batch_size=1, name="inp_stream")
    ]
    # convert non streaming model to streaming one
    model_stream = utils.convert_to_inference_model(self.model,
                                                    input_tensors, mode)
    model_stream.summary()
    # confirm that it returns zeros
    for i in range(self.max_counter):
      i = i % self.input_non_stream_np.shape[1]
      input_stream_np = self.input_non_stream_np[:, i, :]
      input_stream_np = np.expand_dims(input_stream_np, 1)
      self.assertAllEqual(
          np.zeros_like(input_stream_np), model_stream.predict(input_stream_np))
    # confirm that after self.max_counter iterations it returns input
    for _ in range(self.max_counter):
      self.assertAllClose(
          input_stream_np, model_stream.predict(input_stream_np))

  @parameterized.parameters(8, 11)
  def test_inference_external_state(self, max_counter):
    inputs = tf.keras.layers.Input(
        shape=(
            self.time_size,
            self.feature_size,
        ), batch_size=1)
    net = counter.Counter(max_counter=max_counter)(inputs)
    model = tf.keras.Model(inputs, net)

    mode = modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
    input_tensors = [
        tf.keras.layers.Input(
            shape=(
                1,
                self.feature_size,
            ), batch_size=1, name="inp_stream")
    ]
    # convert non streaming model to streaming one
    model_stream = utils.convert_to_inference_model(model,
                                                    input_tensors, mode)
    model_stream.summary()

    # second input to stream model is a state, so we can use its shape
    input_state_np = np.zeros(model_stream.inputs[1].shape, dtype=np.float32)

    # confirm that it returns zeros in the first max_counter iterations
    for i in range(max_counter):
      i = i % self.input_non_stream_np.shape[1]
      input_stream_np = self.input_non_stream_np[:, i, :]
      input_stream_np = np.expand_dims(input_stream_np, 1)
      output_stream_np, output_state_np = model_stream.predict(
          [input_stream_np, input_state_np])
      input_state_np = output_state_np  # update input state
      self.assertAllEqual(
          np.zeros_like(input_stream_np), output_stream_np)

    # confirm that after self.max_counter iterations it returns input tensor
    for i in range(max_counter):
      i = i % self.input_non_stream_np.shape[1]
      input_stream_np = self.input_non_stream_np[:, i, :]
      input_stream_np = np.expand_dims(input_stream_np, 1)
      output_stream_np, output_state_np = model_stream.predict(
          [input_stream_np, input_state_np])
      input_state_np = output_state_np  # update input state
      self.assertAllClose(
          input_stream_np, output_stream_np)


if __name__ == "__main__":
  tf1.disable_eager_execution()
  tf.test.main()
