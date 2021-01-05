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

"""Mobilenet V2 - reduced version of keras/applications/mobilenet_v2.py."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
from kws_streaming.models import utils


def model_parameters(parser_nn):
  """Mobilenet V2 model parameters.

  Args:
    parser_nn: global command line args parser
  Returns: parser with updated arguments
  """
  parser_nn.add_argument(
      '--cnn1_filters',
      type=int,
      default=32,
      help='Number of filters in the first conv',
  )
  parser_nn.add_argument(
      '--cnn1_kernel_size',
      type=str,
      default='(3,1)',
      help='Kernel size of the first conv',
  )
  parser_nn.add_argument(
      '--cnn1_strides',
      type=str,
      default='(2,2)',
      help='Strides of the first conv',
  )
  parser_nn.add_argument(
      '--ds_kernel_size',
      type=str,
      default='(3,1),(3,1),(3,1),(3,1)',
      help='Kernel sizes of depthwise_conv_blocks',
  )
  parser_nn.add_argument(
      '--cnn_strides',
      type=str,
      default='(1,1),(2,2),(1,1),(1,1)',
      help='Strides of Inverted ResNet block '
      'if stride != 1 then there will be no residual connection',
  )
  parser_nn.add_argument(
      '--cnn_filters',
      type=str,
      default='32,32,64,64',
      help='Number of filters in Inverted ResNet block'
      'when filter number is changes there will be no residual connection',
  )
  parser_nn.add_argument(
      '--cnn_expansions',
      type=str,
      default='1.5,1.5,1.5,1.5',
      help='Filters expansion coefficient in Inverted ResNet block',
  )
  parser_nn.add_argument(
      '--dropout',
      type=float,
      default=0.2,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--bn_scale',
      type=int,
      default=0,
      help='If True, multiply by gamma. If False, gamma is not used. '
      'When the next layer is linear (also e.g. nn.relu), this can be disabled'
      'since the scaling will be done by the next layer.',
  )


def model(flags):
  """Mobilenet V2 model.

  It is based on paper:
  MobileNetV2: Inverted Residuals and Linear Bottlenecks
      https://arxiv.org/abs/1801.04381
  It is applied on sequence in time, so only 1D filters applied
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """
  input_audio = tf.keras.layers.Input(
      shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
      batch_size=flags.batch_size)
  net = input_audio

  if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
    net = speech_features.SpeechFeatures(
        speech_features.SpeechFeatures.get_params(flags))(
            net)

  # [batch, time, feature]
  net = tf.keras.backend.expand_dims(net, axis=2)
  # [batch, time, feature, 1]

  # it is conv_block
  net = tf.keras.layers.Conv2D(
      filters=flags.cnn1_filters,
      kernel_size=utils.parse(flags.cnn1_kernel_size),
      padding='valid',
      use_bias=False,
      strides=utils.parse(flags.cnn1_strides))(
          net)
  net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
  net = tf.keras.layers.ReLU(6.)(net)
  # [batch, time, feature, filters]

  for kernel_size, stride, filters, expansion in zip(
      utils.parse(flags.ds_kernel_size), utils.parse(flags.cnn_strides),
      utils.parse(flags.cnn_filters), utils.parse(flags.cnn_expansions)):
    # it is Inverted ResNet block
    net_input = net
    in_channels = tf.keras.backend.int_shape(net_input)[-1]

    net = tf.keras.layers.Conv2D(
        expansion * in_channels,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None)(net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.ReLU(6.)(net)
    # [batch, time, feature, filters]

    # depthwise
    net = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        activation=None,
        use_bias=False,
        padding='same')(net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.ReLU(6.)(net)

    # project
    net = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None)(net)
    net = tf.keras.layers.BatchNormalization()(net)

    if in_channels == filters and stride == (1, 1):
      net = tf.keras.layers.Add()([net_input, net])

  # [batch, time, feature, filters]
  net = tf.keras.layers.GlobalAveragePooling2D()(net)
  # [batch, filters]
  net = tf.keras.layers.Dropout(flags.dropout)(net)
  net = tf.keras.layers.Dense(flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
