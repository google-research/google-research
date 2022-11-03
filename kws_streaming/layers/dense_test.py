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

"""Tests for kws_streaming.layers.dense."""

from absl import logging
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
import kws_streaming.layers.test_utils as tu
tf1.disable_eager_execution()


class DenseTest(tu.TestBase):

  def _run_non_stream_model(self):
    # below model expects that input_data are already initialized in tu.TestBase
    # in setUp, by default input_data should have 3 dimensions
    # size of each dimesnion is constant and is defiend by self.weights
    input_tf = tf.keras.layers.Input(shape=(
        None,
        self.input_data.shape[2],
    ))
    # first dense layer
    dense1 = tf.keras.layers.Dense(
        units=self.weights[0].shape[1], use_bias=False)
    output_tf = dense1(input_tf)
    dense1.set_weights([self.weights[0]])
    model = tf.keras.Model(input_tf, output_tf)

    # run non streaming inference
    output_np = model.predict(self.input_data)

    return output_np, model

  def test_streaming_inference(self):
    output_non_stream_np, _ = self._run_non_stream_model()

    input_tf = tf.keras.layers.Input(shape=(self.input_data.shape[2],))
    dense1 = tf.keras.layers.Dense(
        units=self.weights[0].shape[1], use_bias=False)
    output_tf = dense1(input_tf)
    dense1.set_weights([self.weights[0]])
    model = tf.keras.Model(input_tf, output_tf)
    # streaming emulation: loop over time dim
    for i in range(self.input_data.shape[1]):
      input_batch_np = self.input_data[:, i, :]
      output_np = model.predict(input_batch_np)
      for b in range(self.input_data.shape[0]):  # loop over batch
        self.assertAllClose(output_np[b], output_non_stream_np[b][i])

  def test_training(self):
    output_np, model = self._run_non_stream_model()

    # compile and train model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.RMSprop(
            lr=0.001, rho=0.9, epsilon=None, decay=0.0),
        loss="mse")
    model.summary()
    res = model.fit(self.input_data, output_np)
    logging.info("%f", res.history["loss"][0])
    self.assertLess(res.history["loss"][0], 0.1)


if __name__ == "__main__":
  # Uncomment line below to get more debug information for failing tests.
  # tf.enable_eager_execution()
  tf.test.main()
