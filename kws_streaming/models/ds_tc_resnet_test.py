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

"""Tests for models."""
from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
from kws_streaming.models import model_flags
from kws_streaming.models import model_params
from kws_streaming.models import model_utils
from kws_streaming.models import utils
import kws_streaming.models.ds_tc_resnet as ds_tc_resnet
from kws_streaming.train import test


class DsTcResnetTest(tf.test.TestCase, parameterized.TestCase):
  """Test ds_tc_resnet model in non streaming and streaming modes."""

  def init_model(self, use_tf_fft=False):

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(self.sess)
    test_utils.set_seed(123)
    tf.keras.backend.set_learning_phase(0)

    # model parameters
    model_name = 'ds_tc_resnet'
    self.params = model_params.HOTWORD_MODEL_PARAMS[model_name]
    self.params.causal_data_frame_padding = 1  # causal padding on DataFrame
    self.params.clip_duration_ms = 160
    self.params.use_tf_fft = use_tf_fft
    self.params.mel_non_zero_only = not use_tf_fft
    self.params.feature_type = 'mfcc_tf'
    self.params.window_size_ms = 5.0
    self.params.window_stride_ms = 2.0
    self.params.wanted_words = 'a,b,c'
    self.params.ds_padding = "'causal','causal','causal','causal'"
    self.params.ds_filters = '4,4,4,2'
    self.params.ds_repeat = '1,1,1,1'
    self.params.ds_residual = '0,1,1,1'  # no residuals on strided layers
    self.params.ds_kernel_size = '3,3,3,1'
    self.params.ds_dilation = '1,1,1,1'
    self.params.ds_stride = '2,1,1,1'  # streaming conv with stride
    self.params.ds_pool = '1,2,1,1'  # streaming conv with pool
    self.params.ds_filter_separable = '1,1,1,1'

    # convert ms to samples and compute labels count
    self.params = model_flags.update_flags(self.params)

    # compute total stride
    pools = model_utils.parse(self.params.ds_pool)
    strides = model_utils.parse(self.params.ds_stride)
    time_stride = [1]
    for pool in pools:
      if pool > 1:
        time_stride.append(pool)
    for stride in strides:
      if stride > 1:
        time_stride.append(stride)
    total_stride = np.prod(time_stride)

    # overide input data shape for streaming model with stride/pool
    self.params.data_stride = total_stride
    self.params.data_shape = (total_stride * self.params.window_stride_samples,)

    # set desired number of frames in model
    frames_number = 16
    frames_per_call = total_stride
    frames_number = (frames_number // frames_per_call) * frames_per_call
    # number of input audio samples required to produce one output frame
    framing_stride = max(
        self.params.window_stride_samples,
        max(0, self.params.window_size_samples -
            self.params.window_stride_samples))
    signal_size = framing_stride * frames_number

    # desired number of samples in the input data to train non streaming model
    self.params.desired_samples = signal_size

    self.params.batch_size = 1
    self.model = ds_tc_resnet.model(self.params)
    self.model.summary()

    self.input_data = np.random.rand(self.params.batch_size,
                                     self.params.desired_samples)

    # run non streaming inference
    self.non_stream_out = self.model.predict(self.input_data)

  def test_ds_tc_resnet_stream(self):
    self.init_model()

    # prepare tf streaming model
    model_stream = utils.to_streaming_inference(
        self.model, self.params, Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run streaming inference
    stream_out = test.run_stream_inference_classification(
        self.params, model_stream, self.input_data)
    self.assertAllClose(stream_out, self.non_stream_out, atol=1e-5)

  @parameterized.parameters(False, True)
  def test_ds_tc_resnet_stream_tflite(self, use_tf_fft):
    self.init_model(use_tf_fft)
    tflite_streaming_model = utils.model_to_tflite(
        self.sess, self.model, self.params,
        Modes.STREAM_EXTERNAL_STATE_INFERENCE)

    interpreter = tf.lite.Interpreter(model_content=tflite_streaming_model)
    interpreter.allocate_tensors()

    # before processing new test sequence we reset model state
    inputs = []
    for detail in interpreter.get_input_details():
      inputs.append(np.zeros(detail['shape'], dtype=np.float32))

    stream_out = test.run_stream_inference_classification_tflite(
        self.params, interpreter, self.input_data, inputs)

    self.assertAllClose(stream_out, self.non_stream_out, atol=1e-5)


if __name__ == '__main__':
  tf1.disable_eager_execution()
  tf.test.main()
