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
"""Tests for kws_streaming.layers.conv1d_transpose."""

from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import conv1d_transpose
from kws_streaming.layers import modes
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import utils
from kws_streaming.train import test
tf1.disable_eager_execution()


def conv1d_transpose_model(flags, filters, kernel_size, stride):
  """Toy model to up-scale input data with Conv1DTranspose.

  It can be used for speech enhancement.

  Args:
      flags: model and data settings
      filters: numver of filters output channels
      kernel_size: kernel_size of Conv1DTranspose layer
      stride: stride of Conv1DTranspose layer
  Returns:
    Keras model
  """
  input_audio = tf.keras.layers.Input(
      shape=(flags.desired_samples,), batch_size=flags.batch_size)
  net = input_audio
  net = tf.keras.backend.expand_dims(net)
  net = conv1d_transpose.Conv1DTranspose(
      filters=filters,
      kernel_size=kernel_size,
      strides=stride,
      use_bias=True,
      crop_output=True)(
          net)

  return tf.keras.Model(input_audio, net)


class Conv1DTransposeTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(Conv1DTransposeTest, self).setUp()
    test_utils.set_seed(123)

  @parameterized.parameters(1, 2, 3, 4, 5, 6)
  def test_streaming_strides(self, stride):
    """Test Conv1DTranspose layer in streaming mode with different strides.

    Args:
        stride: controls the upscaling factor
    """

    # model and data parameters
    step = 1  # amount of data fed into streaming model on every iteration
    params = test_utils.Params([step], clip_duration_ms=0.25)

    # prepare input data
    x = np.arange(params.desired_samples)
    inp_audio = x
    inp_audio = np.expand_dims(inp_audio, 0)  # add batch dim

    # prepare non stream model
    model = conv1d_transpose_model(
        params, filters=1, kernel_size=3, stride=stride)
    model.summary()

    # set weights with bias
    for layer in model.layers:
      if isinstance(layer, tf.keras.layers.Conv1DTranspose):
        layer.set_weights([
            np.ones(layer.weights[0].shape),
            np.zeros(layer.weights[1].shape) + 0.5
        ])

    # prepare streaming model
    model_stream = utils.to_streaming_inference(
        model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run inference
    non_stream_out = model.predict(inp_audio)
    stream_out = test.run_stream_inference(params, model_stream, inp_audio)

    self.assertAllClose(stream_out, non_stream_out)

  def test_dynamic_shape(self):
    # model and data parameters
    params = test_utils.Params([1], clip_duration_ms=0.25)

    # prepare input data
    x = np.arange(10)
    inp_audio = x
    inp_audio = np.expand_dims(inp_audio, 0)  # add batch dim

    # prepare non stream model
    params.desired_samples = None
    model = conv1d_transpose_model(params, filters=1, kernel_size=3, stride=1)
    model.summary()

    # run inference on input with dynamic shape
    model.predict(inp_audio)

    with self.assertRaisesRegex(
        ValueError, 'in streaming mode time dimension of input packet '
        'should not be dynamic: TFLite limitation'):
      # streaming model expected to fail on input data with dynamic shape
      params.data_shape = (None,)
      utils.to_streaming_inference(
          model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)


if __name__ == '__main__':
  tf.test.main()
