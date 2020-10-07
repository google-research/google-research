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

"""Model based on combination of (n by 1) convolutions with residual blocks."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
from kws_streaming.models.utils import parse


def model_parameters(parser_nn):
  """Temporal Convolution Resnet model parameters.

  Args:
    parser_nn: global command line args parser
  Returns: parser with updated arguments
  """

  parser_nn.add_argument(
      '--tc_filters',
      type=str,
      default='40, 36, 36, 48, 72, 72',
      help='Number of filters per convolutional in residual blocks',
  )
  parser_nn.add_argument(
      '--repeat_tc_convs',
      type=str,
      default='1, 1, 1, 1, 1, 1',
      help='How many times to repeat conv in time per resnet block',
  )
  parser_nn.add_argument(
      '--kernel_sizes',
      type=str,
      default='7, 5, 5, 5, 5, 5',
      help='Kernel sizes in time dim of conv layer in residual block',
  )
  parser_nn.add_argument(
      '--dilations',
      type=str,
      default='1, 1, 1, 1, 1, 1',
      help='Dilations in time dim of conv layers of residual blocks',
  )
  parser_nn.add_argument(
      '--residuals',
      type=str,
      default='0, 1, 1, 1, 0, 0',
      help='Apply residual in residual block',
  )
  parser_nn.add_argument(
      '--pool_sizes',
      type=str,
      default='1, 2, 2, 2, 1, 1',
      help='Pool size after every residual block',
  )
  parser_nn.add_argument(
      '--padding_in_time',
      type=str,
      default='same',
      help="Padding in time can 'same' or 'causal' "
      "with last one model is streamable",
  )
  parser_nn.add_argument(
      '--activation',
      type=str,
      default='relu',
      help='Activation function, more details at '
      'https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation',
  )
  parser_nn.add_argument(
      '--dropout',
      type=float,
      default=0.1,
      help='Percentage of data dropped',
  )


def resnet_block(inputs, repeat_tc_conv, kernel_size, filters, dilation,
                 residual, padding_in_time, dropout, activation):
  """TC(time conv) Residual block.

  Args:
    inputs: input tensor
    repeat_tc_conv: number of repeating Conv1D in time
    kernel_size: kernel size of Conv1D in time dim
    filters: number of filters in Conv1D in time and 1x1 conv
    dilation: dilation in time dim for Conv1D
    residual: if True residual connection is added
    padding_in_time: can be 'same' or 'causal'
    dropout: dropout value
    activation: type of activation function (string)

  Returns:
    output tensor

  Raises:
    ValueError: if padding has invalid value
  """
  if residual and (padding_in_time not in ('same', 'causal')):
    raise ValueError('padding should be same or causal')

  net = inputs
  if residual:
    # 1x1 conv
    layer_res = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        activation='linear')(
            net)
    layer_res = tf.keras.layers.BatchNormalization()(layer_res)

  for _ in range(repeat_tc_conv-1):
    # 1D conv in time
    net = stream.Stream(
        cell=tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, 1),
            dilation_rate=(dilation, 1),
            padding='valid',
            activation='linear'),
        pad_time_dim=padding_in_time)(
            net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(activation)(net)

  # 1D conv in time
  net = stream.Stream(
      cell=tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=(kernel_size, 1),
          dilation_rate=(dilation, 1),
          padding='valid',
          activation='linear'),
      pad_time_dim=padding_in_time)(
          net)
  net = tf.keras.layers.BatchNormalization()(net)

  # residual connection
  if residual:
    net = tf.keras.layers.Add()([net, layer_res])

  net = tf.keras.layers.Activation(activation)(net)
  net = tf.keras.layers.Dropout(rate=dropout)(net)
  return net


def model(flags):
  """Temporal Convolution ResNet model.

  It can be configured to reproduce model config as described in the paper below
  Temporal Convolution for Real-time Keyword Spotting on Mobile Devices
  https://arxiv.org/pdf/1904.03814.pdf
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """

  tc_filters = parse(flags.tc_filters)
  repeat_tc_convs = parse(flags.repeat_tc_convs)
  kernel_sizes = parse(flags.kernel_sizes)
  pool_sizes = parse(flags.pool_sizes)
  dilations = parse(flags.dilations)
  residuals = parse(flags.residuals)

  if len(
      set((len(repeat_tc_convs), len(kernel_sizes), len(pool_sizes),
           len(dilations), len(residuals), len(tc_filters)))) != 1:
    raise ValueError('all input lists have to be the same length')

  input_audio = tf.keras.layers.Input(
      shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
      batch_size=flags.batch_size)
  net = input_audio

  if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
    net = speech_features.SpeechFeatures(
        speech_features.SpeechFeatures.get_params(flags))(
            net)

  # make it [batch, time, 1, feature]
  net = tf.keras.backend.expand_dims(net, axis=2)

  for filters, repeat, kernel_size, pool_size, dilation, residual in zip(
      tc_filters, repeat_tc_convs, kernel_sizes, pool_sizes, dilations,
      residuals):
    net = resnet_block(net, repeat, kernel_size, filters, dilation,
                       residual, flags.padding_in_time, flags.dropout,
                       flags.activation)

    if pool_size > 1:
      net = tf.keras.layers.MaxPooling2D((pool_size, 1))(net)

  net = stream.Stream(cell=tf.keras.layers.GlobalAveragePooling2D())(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
