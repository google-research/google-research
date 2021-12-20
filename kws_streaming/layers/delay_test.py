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

# Lint as: python3
"""Tests for kws_streaming.layers.residual."""

import itertools
from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import delay
from kws_streaming.layers import modes
from kws_streaming.layers import stream
from kws_streaming.layers import temporal_padding
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import utils
from kws_streaming.train import inference


def delay_model(flags, time_delay, also_in_non_streaming):
  """Model with delay for streaming mode.

  Args:
      flags: model and data settings
      time_delay: delay in time dim
      also_in_non_streaming: Apply delay also in non-streaming mode.

  Returns:
    Keras model
  """

  input_audio = tf.keras.layers.Input(
      shape=(flags.desired_samples,), batch_size=flags.batch_size)
  net = input_audio
  net = tf.keras.backend.expand_dims(net)
  net = delay.Delay(
      delay=time_delay, also_in_non_streaming=also_in_non_streaming)(
          net)
  return tf.keras.Model(input_audio, net)


def residual_model(flags,
                   cnn_filters,
                   cnn_kernel_size,
                   cnn_act,
                   cnn_use_bias,
                   cnn_padding,
                   delay_also_in_non_streaming,
                   dilation=1):
  """Toy deep convolutional model with residual connections.

  It can be used for speech enhancement.

  Args:
      flags: model and data settings
      cnn_filters: list of filters in conv layer
      cnn_kernel_size: list of kernel_size in conv layer
      cnn_act: list of activation functions in conv layer
      cnn_use_bias: list of use_bias in conv layer
      cnn_padding: list of padding in conv layer
      delay_also_in_non_streaming: Whether to apply delay also in non-streaming.
      dilation: dilation applied on all conv layers

  Returns:
    Keras model and sum delay

  Raises:
    ValueError: if any of input list has different length from any other
                or padding in not [same, causal]
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

  sum_delay = 0
  net = tf.keras.backend.expand_dims(net)
  for filters, kernel_size, activation, use_bias, padding in zip(
      cnn_filters, cnn_kernel_size,
      cnn_act, cnn_use_bias, cnn_padding):
    time_buffer_size = dilation * (kernel_size - 1)

    if padding == 'causal':
      # residual connection is simple with 'causal'  padding
      net_residual = net

    elif padding == 'same':
      # residual connection in streaming mode needs delay with 'same' padding
      delay_val = time_buffer_size // 2
      net_residual = delay.Delay(
          delay=delay_val, also_in_non_streaming=delay_also_in_non_streaming)(
              net)
      sum_delay += delay_val

    else:
      raise ValueError('wrong padding mode ', padding)

    # it is easier to convert model to streaming mode when padding function
    # is decoupled from conv layer
    net = temporal_padding.TemporalPadding(
        padding='causal' if delay_also_in_non_streaming else padding,
        padding_size=time_buffer_size)(
            net)

    # it is a ring buffer in streaming mode and lambda x during training
    net = stream.Stream(
        cell=tf.identity,
        ring_buffer_size_in_time_dim=time_buffer_size,
        use_one_step=False,
        pad_time_dim=None)(net)

    net = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        use_bias=use_bias,
        padding='valid')(net)  # padding has to be valid!

    net = tf.keras.layers.Add()([net, net_residual])

  return tf.keras.Model(input_audio, net), sum_delay


def conv_model(flags,
               cnn_filters,
               cnn_kernel_size,
               cnn_act,
               cnn_use_bias,
               cnn_padding,
               dilation=1):
  """Toy convolutional model with sequence of convs with different paddings.

  It can be used for speech enhancement.

  Args:
      flags: model and data settings
      cnn_filters: list of filters in conv layer
      cnn_kernel_size: list of kernel_size in conv layer
      cnn_act: list of activation functions in conv layer
      cnn_use_bias: list of use_bias in conv layer
      cnn_padding: list of padding in conv layer
      dilation: dilation applied on all conv layers

  Returns:
    Keras model and sum delay

  Raises:
    ValueError: if any of input list has different length from any other
                or padding in not [same, causal]
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

  sum_delay = 0
  sum_shift = 0
  net = tf.keras.backend.expand_dims(net)
  for filters, kernel_size, activation, use_bias, padding in zip(
      cnn_filters, cnn_kernel_size,
      cnn_act, cnn_use_bias, cnn_padding):
    time_buffer_size = dilation * (kernel_size - 1)

    if padding == 'same':
      # need a delay with 'same' padding in streaming mode
      delay_val = time_buffer_size // 2
      net = delay.Delay(delay=delay_val)(net)
      sum_delay += delay_val * 2
    elif padding == 'causal':
      sum_shift += kernel_size
    else:
      raise ValueError('wrong padding mode ', padding)

    # it is a ring buffer in streaming mode and lambda x during training
    net = stream.Stream(
        cell=tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding='valid'),
        use_one_step=False,
        pad_time_dim=padding)(net)

  return tf.keras.Model(input_audio, net), sum_delay, sum_shift


class DelayStreamTest(tf.test.TestCase, parameterized.TestCase):
  """Test delay layer."""

  def setUp(self):
    super(DelayStreamTest, self).setUp()
    test_utils.set_seed(123)

  @parameterized.parameters(
      itertools.product([1, 4], ['causal', 'same'], [False, True]))
  def test_residual(self, step, padding, delay_also_in_non_streaming):
    """Test residual connection in streaming mode with conv layer."""

    # model and data parameters
    cnn_filters = [1, 1]
    cnn_kernel_size = [5, 3]
    cnn_act = ['elu', 'elu']
    cnn_use_bias = [False, False]
    cnn_padding = [padding, padding]
    params = test_utils.Params([step], clip_duration_ms=2)

    # prepare input data
    x = np.arange(params.desired_samples)
    inp_audio = x
    inp_audio = np.expand_dims(inp_audio, 0)

    # prepare non stream model
    model, sum_delay = residual_model(params, cnn_filters, cnn_kernel_size,
                                      cnn_act, cnn_use_bias, cnn_padding,
                                      delay_also_in_non_streaming)
    model.summary()

    # prepare streaming model
    model_stream = utils.to_streaming_inference(
        model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run inference
    non_stream_out = model.predict(inp_audio)
    stream_out = inference.run_stream_inference(params, model_stream, inp_audio)

    # normalize output data and compare them
    channel = 0
    non_stream_out = non_stream_out[0, :, channel]
    stream_out = stream_out[0, :, channel]

    min_len = min(stream_out.shape[0], non_stream_out.shape[0])
    stream_out = stream_out[0:min_len]
    non_stream_out = non_stream_out[0:min_len]

    shift = 1
    if delay_also_in_non_streaming:
      # Delay was also applied in non-streaming, as well as streaming mode.
      non_stream_out = non_stream_out[shift + sum_delay:min_len]
    else:
      non_stream_out = non_stream_out[shift:min_len - sum_delay]
    stream_out = stream_out[sum_delay + shift:]

    self.assertAllEqual(non_stream_out.shape, (31-sum_delay,))
    self.assertAllClose(stream_out, non_stream_out)

  @parameterized.parameters(False, True)
  def test_delay_internal_state(self, delay_also_in_non_streaming):
    """Test delay layer with internal state."""

    # model and data parameters
    params = test_utils.Params([1], clip_duration_ms=1)

    # prepare non stream model
    time_delay = 3
    model = delay_model(params, time_delay, delay_also_in_non_streaming)
    model.summary()

    # prepare streaming model
    model_stream = utils.to_streaming_inference(
        model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model.summary()

    # fill the buffer
    for i in range(time_delay):
      output = model_stream.predict([i + 1])
      self.assertAllEqual(output[0, 0, 0], 0)

    # now get the data with delay
    for i in range(time_delay):
      output = model_stream.predict([0])
      self.assertAllEqual(output[0, 0, 0], i + 1)

  def test_conv(self):
    """Test conv model with 'same' padding."""

    # model and data parameters
    cnn_filters = [1, 1, 1]
    cnn_kernel_size = [5, 3, 5]
    cnn_act = ['elu', 'elu', 'elu']
    cnn_use_bias = [False, False, False]
    cnn_padding = ['same', 'causal', 'same']
    params = test_utils.Params([1], clip_duration_ms=2)

    # prepare input data
    x = np.arange(params.desired_samples)
    inp_audio = x
    inp_audio = np.expand_dims(inp_audio, 0)

    # prepare non stream model
    model, sum_delay, sum_shift = conv_model(params, cnn_filters,
                                             cnn_kernel_size, cnn_act,
                                             cnn_use_bias, cnn_padding)
    model.summary()
    non_stream_out = model.predict(inp_audio)

    # prepare streaming model
    model_stream = utils.to_streaming_inference(
        model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()
    stream_out = inference.run_stream_inference(params, model_stream, inp_audio)

    shift = sum_shift + 1
    # normalize output data and compare them
    non_stream_out = non_stream_out[0, shift:-(sum_delay),]
    stream_out = stream_out[0, sum_delay+shift:,]

    self.assertAllClose(stream_out, non_stream_out)


if __name__ == '__main__':
  tf1.disable_eager_execution()
  tf.test.main()
