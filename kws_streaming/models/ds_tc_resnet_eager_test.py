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

"""Test for ds_tc_resnet model in eager mode."""
import numpy as np
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes
from kws_streaming.models import utils
import kws_streaming.models.ds_tc_resnet as ds_tc_resnet
from kws_streaming.train import inference


class DsTcResnetTest(tf.test.TestCase):
  """Test ds_tc_resnet model in non streaming and streaming modes."""

  def test_ds_tc_resnet_stream_internal_tflite(self):
    """Test tflite streaming with internal state."""
    test_utils.set_seed(123)
    tf.keras.backend.set_learning_phase(0)
    params = utils.ds_tc_resnet_model_params(True)

    model = ds_tc_resnet.model(params)
    model.summary()

    input_data = np.random.rand(params.batch_size, params.desired_samples)

    # run non streaming inference
    non_stream_out = model.predict(input_data)

    tflite_streaming_model = utils.model_to_tflite(
        None, model, params,
        Modes.STREAM_INTERNAL_STATE_INFERENCE)

    interpreter = tf.lite.Interpreter(model_content=tflite_streaming_model)
    interpreter.allocate_tensors()

    stream_out = inference.run_stream_inference_classification_tflite(
        params, interpreter, input_data, input_states=None)

    self.assertAllClose(stream_out, non_stream_out, atol=1e-5)

if __name__ == '__main__':
  tf.test.main()
