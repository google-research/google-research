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

"""Test for ds_tc_resnet model in session mode."""
import numpy as np
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
from kws_streaming.models import utils
import kws_streaming.models.ds_tc_resnet as ds_tc_resnet
from kws_streaming.train import inference


class DsTcResnetTest(tf.test.TestCase):
  """Test ds_tc_resnet model in non streaming and streaming modes."""

  def setUp(self):
    super(DsTcResnetTest, self).setUp()

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(self.sess)
    tf.keras.backend.set_learning_phase(0)

    test_utils.set_seed(123)
    self.params = utils.ds_tc_resnet_model_params(True)

    self.model = ds_tc_resnet.model(self.params)
    self.model.summary()

    self.input_data = np.random.rand(self.params.batch_size,
                                     self.params.desired_samples)

    # run non streaming inference
    self.non_stream_out = self.model.predict(self.input_data)

  def test_ds_tc_resnet_stream(self):
    """Test for tf streaming with internal state."""
    # prepare tf streaming model
    model_stream = utils.to_streaming_inference(
        self.model, self.params, Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run streaming inference
    stream_out = inference.run_stream_inference_classification(
        self.params, model_stream, self.input_data)
    self.assertAllClose(stream_out, self.non_stream_out, atol=1e-5)

  def test_ds_tc_resnet_stream_tflite(self):
    """Test for tflite streaming with external state."""
    tflite_streaming_model = utils.model_to_tflite(
        self.sess, self.model, self.params,
        Modes.STREAM_EXTERNAL_STATE_INFERENCE)

    interpreter = tf.lite.Interpreter(model_content=tflite_streaming_model)
    interpreter.allocate_tensors()

    # before processing new test sequence we reset model state
    inputs = []
    for detail in interpreter.get_input_details():
      inputs.append(np.zeros(detail['shape'], dtype=np.float32))

    stream_out = inference.run_stream_inference_classification_tflite(
        self.params, interpreter, self.input_data, inputs)

    self.assertAllClose(stream_out, self.non_stream_out, atol=1e-5)


if __name__ == '__main__':
  tf1.disable_eager_execution()
  tf.test.main()
