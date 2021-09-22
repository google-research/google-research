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

"""Tests for kws_streaming.layers.stream."""

from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import modes
from kws_streaming.layers import stream
from kws_streaming.layers import temporal_padding
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import utils
from kws_streaming.train import inference


# Toy example which require signal processing in time
class Sum(tf.keras.layers.Layer):
  """Applies Sum on time_dim."""

  def __init__(self, time_dim=1, **kwargs):
    super(Sum, self).__init__(**kwargs)
    self.time_dim = time_dim

  def call(self, inputs):
    return tf.keras.backend.sum(inputs, axis=self.time_dim)

  def get_config(self):
    config = {'time_dim': self.time_dim}
    base_config = super(Sum, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def conv_model(flags, conv_cell, cnn_filters, cnn_kernel_size, cnn_act,
               cnn_dilation_rate, cnn_strides, cnn_use_bias, **kwargs):
  """Toy example of convolutional model with Stream wrapper.

  It can be used for speech enhancement.
  Args:
      flags: model and data settings
      conv_cell: cell for streaming, for example: tf.keras.layers.Conv1D
      cnn_filters: list of filters in conv layer
      cnn_kernel_size: list of kernel_size in conv layer
      cnn_act: list of activation functions in conv layer
      cnn_dilation_rate: list of dilation_rate in conv layer
      cnn_strides: list of strides in conv layer
      cnn_use_bias: list of use_bias in conv layer
      **kwargs: Additional kwargs passed on to conv_cell.
  Returns:
    Keras model

  Raises:
    ValueError: if any of input list has different length from any other
  """

  if not all(
      len(cnn_filters) == len(l) for l in [
          cnn_filters, cnn_kernel_size, cnn_act, cnn_dilation_rate, cnn_strides,
          cnn_use_bias
      ]):
    raise ValueError('all input lists have to be the same length')

  input_audio = tf.keras.layers.Input(
      shape=(flags.desired_samples,), batch_size=flags.batch_size)
  net = input_audio

  net = tf.keras.backend.expand_dims(net)

  for (filters, kernel_size, activation, dilation_rate, strides,
       use_bias) in zip(cnn_filters, cnn_kernel_size, cnn_act,
                        cnn_dilation_rate, cnn_strides, cnn_use_bias):

    if conv_cell == tf.keras.layers.DepthwiseConv1D:
      # DepthwiseConv has no filters arg
      net = stream.Stream(
          cell=conv_cell(
              kernel_size=kernel_size,
              activation=activation,
              dilation_rate=dilation_rate,
              strides=strides,
              use_bias=use_bias,
              padding='valid',
              **kwargs),
          use_one_step=False,
          pad_time_dim='causal')(net)
    else:
      net = stream.Stream(
          cell=conv_cell(
              filters=filters,
              kernel_size=kernel_size,
              activation=activation,
              dilation_rate=dilation_rate,
              strides=strides,
              use_bias=use_bias,
              padding='valid',
              **kwargs),
          use_one_step=False,
          pad_time_dim='causal')(net)

  return tf.keras.Model(input_audio, net)


def conv_model_no_stream_wrapper(flags, conv_cell, cnn_filters, cnn_kernel_size,
                                 cnn_act, cnn_dilation_rate, cnn_strides,
                                 cnn_use_bias, **kwargs):
  """Toy example of convolutional model.

  It has the same model topology as in conv_model() above, but without
  wrapping conv cell by Stream layer, so that all parameters set manually.
  Args:
      flags: model and data settings
      conv_cell: cell for streaming, for example: tf.keras.layers.Conv1D
      cnn_filters: list of filters in conv layer
      cnn_kernel_size: list of kernel_size in conv layer
      cnn_act: list of activation functions in conv layer
      cnn_dilation_rate: list of dilation_rate in conv layer
      cnn_strides: list of strides in conv layer
      cnn_use_bias: list of use_bias in conv layer
      **kwargs: Additional kwargs passed on to conv_cell.
  Returns:
    Keras model
  """

  if not all(
      len(cnn_filters) == len(l) for l in [
          cnn_filters, cnn_kernel_size, cnn_act, cnn_dilation_rate, cnn_strides,
          cnn_use_bias
      ]):
    raise ValueError('all input lists have to be the same length')

  input_audio = tf.keras.layers.Input(
      shape=(flags.desired_samples,), batch_size=flags.batch_size)
  net = input_audio

  net = tf.keras.backend.expand_dims(net)

  for filters, kernel_size, activation, dilation_rate, strides, use_bias in zip(
      cnn_filters, cnn_kernel_size,
      cnn_act, cnn_dilation_rate,
      cnn_strides, cnn_use_bias):

    ring_buffer_size_in_time_dim = max(
        dilation_rate * (kernel_size - 1) - (strides - 1), 0)
    net = stream.Stream(
        cell=tf.identity,
        ring_buffer_size_in_time_dim=ring_buffer_size_in_time_dim,
        use_one_step=False,
        pad_time_dim=None)(net)

    padding_size = ring_buffer_size_in_time_dim
    net = temporal_padding.TemporalPadding(
        padding='causal', padding_size=padding_size)(
            net)

    if conv_cell == tf.keras.layers.DepthwiseConv1D:
      # DepthwiseConv has no filters arg
      net = conv_cell(
          kernel_size=kernel_size,
          activation=activation,
          dilation_rate=dilation_rate,
          strides=strides,
          use_bias=use_bias,
          padding='valid',  # padding has to be valid!
          **kwargs)(net)
    else:
      net = conv_cell(
          filters=filters,
          kernel_size=kernel_size,
          activation=activation,
          dilation_rate=dilation_rate,
          strides=strides,
          use_bias=use_bias,
          padding='valid',  # padding has to be valid!
          **kwargs)(net)

  return tf.keras.Model(input_audio, net)


def conv_model_keras_native(flags, conv_cell, cnn_filters, cnn_kernel_size,
                            cnn_act, cnn_dilation_rate, cnn_strides,
                            cnn_use_bias, **kwargs):
  """Toy example of convolutional model without any streaming components.

  It has the same model topology as in conv_model() above, but using only native
  Keras layers and no streaming components.
  Args:
      flags: model and data settings
      conv_cell: cell for streaming, for example: tf.keras.layers.Conv1D
      cnn_filters: list of filters in conv layer
      cnn_kernel_size: list of kernel_size in conv layer
      cnn_act: list of activation functions in conv layer
      cnn_dilation_rate: list of dilation_rate in conv layer
      cnn_strides: list of strides in conv layer
      cnn_use_bias: list of use_bias in conv layer
      **kwargs: Additional kwargs passed on to conv_cell.
  Returns:
    Keras model
  """

  if not all(
      len(cnn_filters) == len(l) for l in [
          cnn_filters, cnn_kernel_size, cnn_act, cnn_dilation_rate, cnn_strides,
          cnn_use_bias
      ]):
    raise ValueError('all input lists have to be the same length')

  input_audio = tf.keras.layers.Input(
      shape=(flags.desired_samples,), batch_size=flags.batch_size)
  net = input_audio

  net = tf.keras.backend.expand_dims(net)

  for filters, kernel_size, activation, dilation_rate, strides, use_bias in zip(
      cnn_filters, cnn_kernel_size,
      cnn_act, cnn_dilation_rate,
      cnn_strides, cnn_use_bias):

    # Note: This explicit padding is different from calling
    # conv_cell(..., padding='causal) directly when strides > 1. The latter
    # is suboptimal because it pads `strides - 1` extra zeros and hence ignores
    # the rightmost `strides - 1` valid samples. See the comments in
    # test_strided_conv_alignment() for a more concrete example.
    net = tf.keras.layers.ZeroPadding1D(
        [max(dilation_rate * (kernel_size - 1) - (strides - 1), 0), 0])(
            net)
    if conv_cell == tf.keras.layers.DepthwiseConv1D:
      # DepthwiseConv has no filters arg
      net = conv_cell(
          kernel_size=kernel_size,
          activation=activation,
          dilation_rate=dilation_rate,
          strides=strides,
          use_bias=use_bias,
          padding='valid',
          **kwargs)(net)
    else:
      net = conv_cell(
          filters=filters,
          kernel_size=kernel_size,
          activation=activation,
          dilation_rate=dilation_rate,
          strides=strides,
          use_bias=use_bias,
          padding='valid',
          **kwargs)(net)

  return tf.keras.Model(input_audio, net)


class StreamTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(StreamTest, self).setUp()
    test_utils.set_seed(123)

  def test_streaming_with_effective_tdim(self):
    time_size = 10
    feature_size = 3
    batch_size = 1

    time_dim = 1  # index of time dimensions
    ring_buffer_size_in_time_dim = 3  # effective size of aperture in time dim

    inputs = tf.keras.layers.Input(
        shape=(time_size, feature_size),
        batch_size=batch_size,
        name='inp_sequence')

    mode = modes.Modes.TRAINING

    # in streaming mode it will create a
    # ring buffer with time dim size ring_buffer_size_in_time_dim
    outputs = stream.Stream(
        cell=Sum(time_dim=time_dim),
        mode=mode,
        ring_buffer_size_in_time_dim=ring_buffer_size_in_time_dim)(inputs)
    model_train = tf.keras.Model(inputs, outputs)
    model_train.summary()

    mode = modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
    input_tensors = [
        tf.keras.layers.Input(
            shape=(
                1,  # time dim is size 1 in streaming mode
                feature_size,
            ),
            batch_size=batch_size,
            name='inp_stream')
    ]
    # convert non streaming model to streaming one
    model_stream = utils.convert_to_inference_model(model_train,
                                                    input_tensors, mode)
    model_stream.summary()

    # second input tostream model is a state, so we can use its shape
    input_state_np = np.zeros(model_stream.inputs[1].shape, dtype=np.float32)

    # input test data
    non_stream_input = np.random.randint(
        1, 10, size=(batch_size, time_size, feature_size))

    # run streaming inference
    # iterate over time dim sample by sample
    for i in range(input_state_np.shape[1]):
      input_stream_np = np.expand_dims(non_stream_input[0][i], 0)
      input_stream_np = np.expand_dims(input_stream_np, 1)
      input_stream_np = input_stream_np.astype(np.float32)
      output_stream_np, output_state_np = model_stream.predict(
          [input_stream_np, input_state_np])
      input_state_np = output_state_np  # update input state

      # emulate sliding window summation
      target = np.sum(
          non_stream_input[:, max(0, i - ring_buffer_size_in_time_dim):i + 1],
          axis=time_dim)
      self.assertAllEqual(target, output_stream_np)

    # validate name tag of model's state
    expected_str = 'ExternalState'
    self.assertAllEqual(
        expected_str,
        model_stream.inputs[1].name.split('/')[-1][:len(expected_str)])

  @parameterized.parameters('causal', 'same')
  def test_padding(self, padding):
    batch_size = 1
    time_dim = 3
    feature_dim = 3
    kernel_size = 3
    inputs = tf.keras.layers.Input(
        shape=(time_dim, feature_dim), batch_size=batch_size)

    # set it in train mode (in stream mode padding is not applied)
    net = stream.Stream(
        mode=modes.Modes.TRAINING,
        cell=tf.keras.layers.Lambda(lambda x: x),
        ring_buffer_size_in_time_dim=kernel_size,
        pad_time_dim=padding)(inputs)
    model = tf.keras.Model(inputs, net)

    test_utils.set_seed(1)
    input_signal = np.random.rand(batch_size, time_dim, feature_dim)
    outputs = model.predict(input_signal)
    self.assertAllEqual(outputs.shape,
                        [batch_size, time_dim + kernel_size - 1, feature_dim])

  def test_strided_conv_alignment(self):
    kernel_size = 4
    strides = 2
    inputs = tf.keras.layers.Input(shape=(None, 1))
    net = inputs
    net = stream.Stream(
        cell=tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            kernel_initializer='ones'),
        use_one_step=False,
        pad_time_dim='causal')(
            net)
    model = tf.keras.Model(inputs=inputs, outputs=net)

    input_signal = np.arange(1, 5)  # [1, 2, 3, 4]
    # Sanity check for the test itself: We only care about the case when input
    # length is a multiple of strides. If not, streaming is not meaningful.
    assert len(input_signal) % strides == 0
    input_signal = input_signal[None, :, None]
    output_signal = model.predict(input_signal)
    outputs = output_signal[0, :, 0]

    # Make sure causal conv is right-aligned, so that the most recent samples
    # are never ignored. Thus we want:
    #           1  2  3  4
    # -> [0  0] 1  2  3  4  (padding)
    # ->           3    10  (conv with kernel of ones: 3=0+0+1+2, 10=1+2+3+4)
    # Note that this is different from tf.keras.layersConv1D(..., 'causal'),
    # which will pad 3 zeroes on the left and produce [1(=0+0+0+1), 6(=0+1+2+3)]
    # instead. The latter is less ideal, since it pads an extra zero and ignores
    # the last (and hence most recent) valid sample "4".
    self.assertAllEqual(outputs, [3, 10])

  @parameterized.named_parameters(
      {
          'testcase_name': 'model with stream wrapper on Conv1D',
          'get_model': conv_model,
          'conv_cell': tf.keras.layers.Conv1D
      }, {
          'testcase_name': 'model with stream wrapper on SeparableConv1D',
          'get_model': conv_model,
          'conv_cell': tf.keras.layers.SeparableConv1D
      }, {
          'testcase_name': 'model with stream wrapper on DepthwiseConv1D',
          'get_model': conv_model,
          'conv_cell': tf.keras.layers.DepthwiseConv1D
      }, {
          'testcase_name': 'model without stream wrapper on Conv1D',
          'get_model': conv_model_no_stream_wrapper,
          'conv_cell': tf.keras.layers.Conv1D
      })
  def test_stream_strided_convolution(self, get_model, conv_cell):
    # Test streaming convolutional layers with striding, dilation.
    cnn_filters = [1, 1, 1, 1]
    cnn_kernel_size = [3, 3, 3, 3]
    cnn_act = ['linear', 'linear', 'elu', 'elu']
    cnn_dilation_rate = [1, 1, 1, 2]
    cnn_strides = [2, 1, 3, 1]
    cnn_use_bias = [False, False, False, False]

    # prepare input data
    params = test_utils.Params(cnn_strides)
    x = np.arange(params.desired_samples)
    frequency = 2.0
    inp_audio = np.cos((2.0 * np.pi / params.desired_samples) * frequency *
                       x) + np.random.rand(1, params.desired_samples) * 0.5

    if conv_cell == tf.keras.layers.SeparableConv1D:
      kwargs = dict(
          depthwise_initializer=FixedRandomInitializer(seed=123),
          pointwise_initializer=FixedRandomInitializer(seed=456))
    elif conv_cell == tf.keras.layers.DepthwiseConv1D:
      kwargs = dict(
          depthwise_initializer=FixedRandomInitializer(seed=123))
    else:
      kwargs = dict(
          kernel_initializer=FixedRandomInitializer(seed=123))

    # Prepare Keras native model.
    model_native = conv_model_keras_native(params, conv_cell, cnn_filters,
                                           cnn_kernel_size, cnn_act,
                                           cnn_dilation_rate, cnn_strides,
                                           cnn_use_bias, **kwargs)
    model_native.summary()

    # prepare non stream model
    model = get_model(params, conv_cell, cnn_filters, cnn_kernel_size,
                      cnn_act, cnn_dilation_rate, cnn_strides, cnn_use_bias,
                      **kwargs)
    model.summary()

    # prepare streaming model
    model_stream = utils.to_streaming_inference(
        model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run inference
    non_stream_out = model.predict(inp_audio)
    native_out = model_native.predict(inp_audio)
    stream_out = inference.run_stream_inference(params, model_stream, inp_audio)

    # normalize output data and compare them
    channel = 0
    non_stream_out = non_stream_out[0, :, channel]
    native_out = native_out[0, :, channel]
    stream_out = stream_out[0, :, channel]

    min_len = min(stream_out.shape[0], non_stream_out.shape[0])
    stream_out = stream_out[0:min_len]
    native_out = native_out[0:min_len]
    non_stream_out = non_stream_out[0:min_len]
    self.assertAllEqual(non_stream_out.shape,
                        (params.desired_samples / np.prod(cnn_strides),))

    with self.subTest(name='stream_vs_non_stream'):
      self.assertAllClose(stream_out, non_stream_out)

    with self.subTest(name='non_stream_vs_native'):
      self.assertAllClose(non_stream_out, native_out)


@tf.keras.utils.register_keras_serializable()
class FixedRandomInitializer(tf.keras.initializers.Initializer):

  def __init__(self, seed, mean=0., stddev=1.):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed

  def __call__(self, shape, dtype=None):
    return tf.random.stateless_normal(
        shape, mean=self.mean, stddev=self.stddev, dtype=dtype,
        seed=[self.seed, 0])

  def get_config(self):
    return {'seed': self.seed,
            'mean': self.mean,
            'stddev': self.stddev}


if __name__ == '__main__':
  tf1.disable_eager_execution()
  tf.test.main()
