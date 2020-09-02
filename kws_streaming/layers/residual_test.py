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
"""Tests for kws_streaming.layers.residual."""
import random as rn
from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import residual
from kws_streaming.layers import stream
from kws_streaming.layers import temporal_padding
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
from kws_streaming.models import utils
from kws_streaming.train import model_flags
from kws_streaming.train import test
tf1.disable_eager_execution()


class Params(object):
  """Parameters for data and other settings."""

  def __init__(self, cnn_strides):
    self.sample_rate = 16000
    self.clip_duration_ms = 16

    # it is a special case to customize input data shape
    self.preprocess = 'custom'

    # defines the step of feeding input data
    self.data_shape = (np.prod(cnn_strides),)

    self.batch_size = 1
    self.desired_samples = int(
        self.sample_rate * self.clip_duration_ms / model_flags.MS_PER_SECOND)

    # align data length with the step
    self.desired_samples = (self.desired_samples //
                            self.data_shape[0]) * self.data_shape[0]


def residual_model(flags, cnn_filters, cnn_kernel_size, cnn_act, cnn_use_bias,
                   cnn_padding):
  """Toy deep convolutional model with residual connections.

  It can be used for speech enhancement.

  Args:
      flags: model and data settings
      cnn_filters: list of filters in conv layer
      cnn_kernel_size: list of kernel_size in conv layer
      cnn_act: list of activation functions in conv layer
      cnn_use_bias: list of use_bias in conv layer
      cnn_padding: list of padding in conv layer

  Returns:
    Keras model

  Raises:
    ValueError: if any of input list has different length from any other
  """

  if not all(
      len(cnn_filters) == len(l) for l in [
          cnn_filters, cnn_kernel_size, cnn_act, cnn_use_bias, cnn_padding]):
    raise ValueError('all input lists have to be the same length')

  # it is an example of deep conv model for speech enhancement
  # which can be trained in non streaming mode and converted to streaming mode
  input_audio = tf.keras.layers.Input(
      shape=(flags.desired_samples,), batch_size=flags.batch_size)
  net = input_audio

  net = tf.keras.backend.expand_dims(net)
  for filters, kernel_size, activation, use_bias, padding in zip(
      cnn_filters, cnn_kernel_size,
      cnn_act, cnn_use_bias, cnn_padding):

    ring_buffer_size_in_time_dim = (kernel_size - 1)

    # it is a ring buffer in streaming mode and lambda x during training
    net = stream.Stream(
        cell=tf.identity,
        ring_buffer_size_in_time_dim=ring_buffer_size_in_time_dim,
        use_one_step=False,
        pad_time_dim=None)(net)

    # residual connection in streaming mode needs:
    # * kernel size in time dim of conv layer
    # * padding mode which was used to padd data in time dim
    net_residual = residual.Residual(
        padding=padding, kernel_size_time=ring_buffer_size_in_time_dim+1)(
            net)

    # it is easier to convert model to streaming mode when padding function
    # is decoupled from conv layer
    net = temporal_padding.TemporalPadding(
        padding=padding, padding_size=ring_buffer_size_in_time_dim)(
            net)

    net = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        use_bias=use_bias,
        padding='valid')(net)  # padding has to be valid!

    net = tf.keras.layers.Add()([net, net_residual])

  return tf.keras.Model(input_audio, net)


class ResidualStreamTest(tf.test.TestCase, parameterized.TestCase):
  """Test residual connection in streaming mode with conv layer."""

  def setUp(self):
    super(ResidualStreamTest, self).setUp()
    seed = 123
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)

  @parameterized.parameters(1, 4)
  def test_residual(self, step):

    # model and data parameters
    cnn_filters = [1, 1, 1, 1]
    cnn_kernel_size = [3, 3, 3, 3]
    cnn_act = ['linear', 'linear', 'elu', 'elu']
    cnn_use_bias = [False, False, False, False]
    cnn_padding = ['causal', 'causal', 'causal', 'causal']
    params = test_utils.Params([step], clip_duration_ms=2)

    # prepare input data
    x = np.arange(params.desired_samples)
    frequency = 2.0
    inp_audio = np.cos((2.0 * np.pi / params.desired_samples) * frequency *
                       x) + np.random.rand(1, params.desired_samples) * 0.5

    # prepare non stream model
    model = residual_model(params, cnn_filters, cnn_kernel_size, cnn_act,
                           cnn_use_bias, cnn_padding)
    model.summary()

    # prepare streaming model
    model_stream = utils.to_streaming_inference(
        model, params, Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run inference
    non_stream_out = model.predict(inp_audio)
    stream_out = test.run_stream_inference(params, model_stream, inp_audio)

    # normalize output data and compare them
    channel = 0
    non_stream_out = non_stream_out[0, :, channel]
    stream_out = stream_out[0, :, channel]

    min_len = min(stream_out.shape[0], non_stream_out.shape[0])
    stream_out = stream_out[0:min_len]
    non_stream_out = non_stream_out[0:min_len]
    self.assertAllEqual(non_stream_out.shape, (32,))
    self.assertAllClose(stream_out, non_stream_out)


if __name__ == '__main__':
  tf.test.main()
