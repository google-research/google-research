# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for kws_streaming.layers.stft."""
from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import modes
from kws_streaming.layers import stft
from kws_streaming.layers import temporal_padding
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import utils
from kws_streaming.train import inference


class STFTTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(STFTTest, self).setUp()
    test_utils.set_seed(123)

    self.frame_size = 40
    self.frame_step = 10
    # layer definition
    stft_layer = stft.STFT(
        self.frame_size,
        self.frame_step,
        mode=modes.Modes.TRAINING,
        inference_batch_size=1,
        padding='causal')

    if stft_layer.window_type == 'hann_tf':
      synthesis_window_fn = tf.signal.hann_window
    else:
      synthesis_window_fn = None

    # prepare input data
    self.input_signal = np.random.rand(1, 120)

    # prepare default tf stft
    frame_overlap = max(0, stft_layer.frame_size - stft_layer.frame_step)
    padding_layer = temporal_padding.TemporalPadding(
        padding_size=frame_overlap, padding=stft_layer.padding)
    # pylint: disable=g-long-lambda
    stft_default_layer = tf.keras.layers.Lambda(
        lambda x: tf.signal.stft(
            x,
            stft_layer.frame_size,
            stft_layer.frame_step,
            fft_length=stft_layer.fft_size,
            window_fn=synthesis_window_fn,
            pad_end=False))
    # pylint: enable=g-long-lambda
    input_tf = tf.keras.layers.Input(
        shape=(self.input_signal.shape[1],), batch_size=1)
    net = padding_layer(input_tf)
    net = stft_default_layer(net)

    model_stft = tf.keras.models.Model(input_tf, net)

    self.stft_out = model_stft.predict(self.input_signal)

  def testNonStreaming(self):
    # prepare non streaming model and compare it with default stft
    stft_layer = stft.STFT(
        self.frame_size,
        self.frame_step,
        mode=modes.Modes.TRAINING,
        inference_batch_size=1,
        padding='causal')
    input_tf = tf.keras.layers.Input(
        shape=(self.input_signal.shape[1],), batch_size=1)
    net = stft_layer(input_tf)
    model_non_stream = tf.keras.models.Model(input_tf, net)
    self.non_stream_out = model_non_stream.predict(self.input_signal)
    self.assertAllClose(self.non_stream_out, self.stft_out)

  @parameterized.named_parameters(
      {
          'testcase_name': 'streaming frame by frame',
          'input_samples': 1,
      }, {
          'testcase_name': 'streaming with 3 frames per call',
          'input_samples': 3,
      })
  def testStreaming(self, input_samples):

    window_type = 'hann_tf'
    # Note that with tf1.disable_eager_execution and window_type='hann_tf'
    # it will raise ValueError: Unable to save function
    # b'__inference_stft_1_layer_call_and_return_conditional_losses_1669'
    # because it captures graph tensor
    # Tensor("streaming/stft_1/windowing_1/hann_window/sub_2:0", shape=(40,),
    # dtype=float32) from a parent function which cannot be converted to a
    # constant with `tf.get_static_value`.
    # To address the above we should use window_type='hann'

    # Also with disabled eager mode and sess=None
    # model_to_tflite should use from_keras_model instead of from_saved_model
    # else ConverterError: Input 0 of node
    # StatefulPartitionedCall/stft_1/data_frame_1/AssignVariableOp
    # was passed float from Func/StatefulPartitionedCall/input/_18:0
    # incompatible with expected resource.

    # Prepare non streaming model
    stft_layer = stft.STFT(
        self.frame_size,
        self.frame_step,
        mode=modes.Modes.TRAINING,
        inference_batch_size=1,
        padding='causal',
        window_type=window_type,
    )

    input_tf = tf.keras.layers.Input(
        shape=(self.input_signal.shape[1],), batch_size=1
    )
    net = stft_layer(input_tf)
    model_non_stream = tf.keras.models.Model(input_tf, net)

    params = test_utils.Params([1])
    # Shape of input data in the inference streaming mode (excluding batch size)
    params.data_shape = (input_samples * stft_layer.frame_step,)
    params.step = input_samples

    # Convert TF non streaming model to streaming model
    model_stream = utils.to_streaming_inference(
        model_non_stream, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
    )
    model_stream.summary()

    # Run TF streaming inference and compare it with default stft
    stream_out = inference.run_stream_inference(
        params, model_stream, self.input_signal
    )
    stream_output_length = stream_out.shape[1]
    self.assertAllClose(stream_out, self.stft_out[:, 0:stream_output_length])

    # Convert TF non-streaming model to TFLite internal-state streaming model
    tflite_streaming_model = utils.model_to_tflite(
        None,
        model_non_stream,
        params,
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
    )
    self.assertTrue(tflite_streaming_model)

    # Run TFLite streaming inference.
    interpreter = tf.lite.Interpreter(model_content=tflite_streaming_model)
    interpreter.allocate_tensors()
    stream_out_tflite_external_st = inference.run_stream_inference_tflite(
        params, interpreter, self.input_signal, None, concat=True
    )

    # Compare streaming TFLite vs TF non-streaming
    self.assertAllClose(
        stream_out_tflite_external_st,
        self.stft_out[:, 0:stream_output_length],
        atol=1e-5,
    )


if __name__ == '__main__':
  tf1.enable_eager_execution()
  tf.test.main()
