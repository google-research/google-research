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

"""Tests for kws_streaming.layers.svdf."""

from absl import logging
import numpy as np
from kws_streaming.layers import modes
from kws_streaming.layers import svdf
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
import kws_streaming.layers.test_utils as tu
from kws_streaming.models import utils
tf1.disable_eager_execution()


class SvdfTest(tu.TestBase):

  def _run_non_stream_model(self):
    # below model expects that input_data are already initialized in tu.TestBase
    # in setUp, by default input_data should have 3 dimensions.
    # size of each dimesnion is constant and is defiend by self.weights
    mode = modes.Modes.TRAINING
    input_tf = tf.keras.layers.Input(shape=(
        None,
        self.input_data.shape[2],
    ))
    svdf_layer = svdf.Svdf(
        units1=self.weights[0].shape[1],
        memory_size=self.memory_size,
        units2=self.weights[3].shape[1],
        activation="linear",
        inference_batch_size=self.batch_size,
        mode=mode)
    output_tf = svdf_layer(inputs=input_tf)
    svdf_layer.dense1.set_weights([self.weights[0]])
    depth_cnn_weight = self.weights[1]
    depth_cnn_weight = np.expand_dims(depth_cnn_weight, 1)
    depth_cnn_weight = np.expand_dims(depth_cnn_weight, 3)
    svdf_layer.depth_cnn1.cell.set_weights([depth_cnn_weight, self.weights[2]])
    svdf_layer.dense2.set_weights([self.weights[3], self.weights[4]])

    model_tf = tf.keras.models.Model(input_tf, output_tf)

    # run inference in non streaming mode
    output_non_stream_np = model_tf.predict(self.input_data)
    return output_non_stream_np, model_tf

  def test_streaming_inference_internal_state(self):
    output_non_stream_np, _ = self._run_non_stream_model()

    mode = modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
    input_tf = tf.keras.layers.Input(shape=(
        1,
        self.input_data.shape[2],
    ), batch_size=None)

    svdf_layer = svdf.Svdf(
        units1=self.weights[0].shape[1],
        memory_size=self.memory_size,
        units2=self.weights[3].shape[1],
        activation="linear",
        inference_batch_size=self.batch_size,
        mode=mode)
    output_tf = svdf_layer(inputs=input_tf)

    svdf_layer.dense1.set_weights([self.weights[0]])
    depth_cnn_weight = self.weights[1]
    depth_cnn_weight = np.expand_dims(depth_cnn_weight, 1)
    depth_cnn_weight = np.expand_dims(depth_cnn_weight, 3)

    input_states_np = np.zeros(svdf_layer.depth_cnn1.get_weights()[2].shape)

    svdf_layer.depth_cnn1.set_weights(
        [depth_cnn_weight, self.weights[2], input_states_np])
    svdf_layer.dense2.set_weights([self.weights[3], self.weights[4]])
    model = tf.keras.models.Model(input_tf, output_tf)

    for i in range(self.input_data.shape[1]):  # loop over every element in time
      input_batch_np = self.input_data[:, i, :]
      input_batch_np = np.expand_dims(input_batch_np, 1)
      output_np = model.predict(input_batch_np)
      for b in range(self.input_data.shape[0]):  # loop over batch
        self.assertAllClose(output_np[b][0], output_non_stream_np[b][i])

  def test_streaming_inference_external_state(self):

    with tf1.Session() as sess:
      output_non_stream_np, model_tf = self._run_non_stream_model()

      # input data for streaming stateless model
      input_tensors = [
          tf.keras.layers.Input(
              shape=(
                  1,
                  self.input_data.shape[2],
              ),
              batch_size=self.batch_size,
              dtype=tf.float32)
      ]

      # convert non streaming model to streaming one with external state
      mode = modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
      model_stream = utils.convert_to_inference_model(model_tf, input_tensors,
                                                      mode)

      # validate that model is convertable to tflite
      converter = tf1.lite.TFLiteConverter.from_session(
          sess, model_stream.inputs, model_stream.outputs)
      self.assertTrue(converter.convert())

      inputs = []
      for s in range(len(model_stream.inputs)):
        inputs.append(np.zeros(model_stream.inputs[s].shape, dtype=np.float32))

      # streaming emulation: loop over every element in time
      for i in range(self.input_data.shape[1]):
        input_batch_np = self.input_data[:, i, :]
        input_batch_np = np.expand_dims(input_batch_np, 1)
        inputs[0] = input_batch_np
        outputs = model_stream.predict(inputs)
        # input_states_np = output_states_np
        for s in range(1, len(model_stream.inputs)):
          inputs[s] = outputs[s]
        for b in range(self.input_data.shape[0]):  # loop over batch
          self.assertAllClose(outputs[0][b][0], output_non_stream_np[b][i])

  def test_training(self):
    # Test stateful svdf layer in training mode.

    # create training model and run inference
    output_np, model = self._run_non_stream_model()

    # compile and train model
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            lr=0.001, rho=0.9, epsilon=None, decay=0.0),
        loss="mse")
    model.summary()
    res = model.fit(self.input_data, output_np)
    logging.info("%f", res.history["loss"][0])
    self.assertLess(res.history["loss"][0], 0.1)

if __name__ == "__main__":
  tf.test.main()
