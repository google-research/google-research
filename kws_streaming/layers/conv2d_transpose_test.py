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

"""Tests for kws_streaming.layers.conv2d_transpose."""

from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import conv2d_transpose
from kws_streaming.layers import modes
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import utils
from kws_streaming.train import inference


def conv2d_transpose_model(flags,
                           filters,
                           kernel_size,
                           strides,
                           features=1,
                           channels=1):
  """Toy model to up-scale input data with Conv2DTranspose.

  It can be used for speech enhancement.

  Args:
      flags: model and data settings
      filters: number of filters (output channels)
      kernel_size: 2d kernel_size of Conv2DTranspose layer
      strides: 2d strides of Conv2DTranspose layer
      features: feature size in the input data
      channels: number of channels in the input data
  Returns:
    Keras model
  """
  input_audio = tf.keras.layers.Input(
      shape=(flags.desired_samples, features, channels),
      batch_size=flags.batch_size)
  net = input_audio
  net = conv2d_transpose.Conv2DTranspose(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      use_bias=True,
      crop_output=True)(
          net)

  return tf.keras.Model(input_audio, net)


class Conv2DTransposeTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(Conv2DTransposeTest, self).setUp()
    test_utils.set_seed(123)
    self.input_channels = 2

  @parameterized.parameters(1, 2, 3)
  def test_streaming_on_1d_data_strides(self, stride):
    """Tests Conv2DTranspose on 1d in streaming mode with different strides.

    Args:
        stride: controls the upscaling factor
    """

    tf1.reset_default_graph()
    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(sess)

    # model and data parameters
    step = 1  # amount of data fed into streaming model on every iteration
    params = test_utils.Params([step], clip_duration_ms=0.25)

    # prepare input data: [batch, time, 1, channels]
    x = np.random.rand(1, params.desired_samples, 1, self.input_channels)
    inp_audio = x

    # prepare non-streaming model
    model = conv2d_transpose_model(
        params,
        filters=1,
        kernel_size=(3, 1),
        strides=(stride, 1),
        channels=self.input_channels)
    model.summary()

    # set weights with bias
    for layer in model.layers:
      if isinstance(layer, tf.keras.layers.Conv2DTranspose):
        layer.set_weights([
            np.ones(layer.weights[0].shape),
            np.zeros(layer.weights[1].shape) + 0.5
        ])

    params.data_shape = (1, 1, self.input_channels)

    # prepare streaming model
    model_stream = utils.to_streaming_inference(
        model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run inference
    non_stream_out = model.predict(inp_audio)
    stream_out = inference.run_stream_inference(params, model_stream, inp_audio)

    self.assertAllClose(stream_out, non_stream_out)

    # Convert TF non-streaming model to TFLite external-state streaming model.
    tflite_streaming_model = utils.model_to_tflite(
        sess, model, params, modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE)
    self.assertTrue(tflite_streaming_model)

    # Run TFLite external-state streaming inference.
    interpreter = tf.lite.Interpreter(model_content=tflite_streaming_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    input_states = []
    # before processing test sequence we create model state
    for s in range(len(input_details)):
      input_states.append(np.zeros(input_details[s]['shape'], dtype=np.float32))

    stream_out_tflite_external_st = inference.run_stream_inference_tflite(
        params, interpreter, inp_audio, input_states, concat=True)

    # compare streaming TFLite with external-state vs TF non-streaming
    self.assertAllClose(stream_out_tflite_external_st, non_stream_out)

  @parameterized.parameters(1, 2, 3)
  def test_streaming_on_2d_data_strides(self, stride):
    """Tests Conv2DTranspose on 2d in streaming mode with different strides.

    Args:
        stride: controls the upscaling factor
    """

    tf1.reset_default_graph()
    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(sess)

    # model and data parameters
    step = 1  # amount of data fed into streaming model on every iteration
    params = test_utils.Params([step], clip_duration_ms=0.25)

    input_features = 3
    # prepare input data: [batch, time, features, channels]
    x = np.random.rand(1, params.desired_samples, input_features,
                       self.input_channels)
    inp_audio = x

    # prepare non-streaming model
    model = conv2d_transpose_model(
        params,
        filters=1,
        kernel_size=(3, 3),
        strides=(stride, stride),
        features=input_features,
        channels=self.input_channels)
    model.summary()

    # set weights with bias
    for layer in model.layers:
      if isinstance(layer, tf.keras.layers.Conv2DTranspose):
        layer.set_weights([
            np.ones(layer.weights[0].shape),
            np.zeros(layer.weights[1].shape) + 0.5
        ])

    params.data_shape = (1, input_features, self.input_channels)

    # prepare streaming model
    model_stream = utils.to_streaming_inference(
        model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run inference
    non_stream_out = model.predict(inp_audio)
    stream_out = inference.run_stream_inference(params, model_stream, inp_audio)

    self.assertAllClose(stream_out, non_stream_out)

  def test_dynamic_shape(self):
    # model and data parameters
    params = test_utils.Params([1], clip_duration_ms=0.25)

    # prepare input data
    x = np.random.rand(1, params.desired_samples, 1, self.input_channels)
    inp_audio = x

    # prepare non stream model
    params.desired_samples = None
    model = conv2d_transpose_model(
        params,
        filters=1,
        kernel_size=(3, 1),
        strides=(1, 1),
        channels=self.input_channels)
    model.summary()

    # run inference on input with dynamic shape
    model.predict(inp_audio)

    with self.assertRaisesRegex(
        ValueError, 'in streaming mode time dimension of input packet '
        'should not be dynamic: TFLite limitation'):
      # streaming model expected to fail on input data with dynamic shape
      params.data_shape = (None, 1, self.input_channels)
      utils.to_streaming_inference(
          model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)

  @parameterized.parameters(
      ('same', (1, 2)),
      ('valid', (1, 2)),
      ('same', (2, 2)),
      ('valid', (2, 2)))
  def test_streaming_with_padding(self, padding, strides):
    # model and data parameters
    input_features = 4
    input_channels = 8
    desired_samples = 2
    batch_size = 1
    inputs = np.random.rand(batch_size, desired_samples, input_features,
                            input_channels)
    kernel_size = (1, 2)

    # prepare non stream model
    input_audio = tf.keras.layers.Input(
        shape=(desired_samples, input_features, input_channels),
        batch_size=batch_size)
    net = conv2d_transpose.Conv2DTranspose(
        filters=input_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding='valid',
        use_bias=False,
        crop_output=True,
        pad_time_dim='causal',
        pad_freq_dim=padding
        )(input_audio)
    non_stream_model = tf.keras.Model(input_audio, net)

    input_audio = tf.keras.layers.Input(
        shape=(desired_samples, input_features, input_channels),
        batch_size=batch_size)
    net = tf.keras.layers.Conv2DTranspose(
        filters=input_channels,
        kernel_size=kernel_size,
        strides=strides,
        use_bias=False,
        padding=padding)(input_audio)
    model = tf.keras.Model(input_audio, net)

    outputs = model.predict(inputs)
    weights = model.get_weights()
    non_stream_model.set_weights(weights)
    non_stream_outputs = non_stream_model.predict(inputs)

    # prepare streaming model
    params = test_utils.Params([1], 1)
    params.data_shape = (1, input_features, input_channels)
    model_stream = utils.to_streaming_inference(
        non_stream_model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run inference
    stream_outputs = inference.run_stream_inference(params, model_stream,
                                                    inputs)

    self.assertAllClose(outputs, non_stream_outputs)
    self.assertAllClose(stream_outputs, non_stream_outputs)


if __name__ == '__main__':
  tf1.disable_eager_execution()
  tf.test.main()
