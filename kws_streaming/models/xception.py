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

"""Xception - reduced version of keras/applications/xception.py."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
from kws_streaming.models.utils import parse


def model_parameters(parser_nn):
  """Xception model parameters.

  Args:
    parser_nn: global command line args parser
  Returns: parser with updated arguments
  """

  parser_nn.add_argument(
      '--cnn1_kernel_size',
      type=str,
      default='(3,3),(3,1)',
      help='kernel_size of the first conv block',
  )
  parser_nn.add_argument(
      '--cnn1_filters',
      type=str,
      default='32,32',
      help='Number of filters in the first conv block',
  )
  parser_nn.add_argument(
      '--cnn1_strides',
      type=str,
      default='(2,2),(1,1)',
      help='strides of the first conv block',
  )
  parser_nn.add_argument(
      '--cnn2_filters',
      type=str,
      default='64,64',
      help='filters in the first residual block, which has striding',
  )
  parser_nn.add_argument(
      '--cnn3_blocks',
      type=int,
      default=2,
      help='number of residual blocks in second pass'
      ', which does not have striding',
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
      default=1,
      help='If True, multiply by gamma. If False, gamma is not used. '
      'When the next layer is linear (also e.g. nn.relu), this can be disabled'
      'since the scaling will be done by the next layer.',
  )


def model(flags):
  """Xception model.

  It is based on paper:
  Xception: Deep Learning with Depthwise Separable Convolutions
      https://arxiv.org/abs/1610.02357
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
  net = tf.keras.backend.expand_dims(net, axis=-1)
  # [batch, time, feature, 1]

  # conv block
  for kernel_size, stride, filters in zip(
      parse(flags.cnn1_kernel_size), parse(flags.cnn1_strides),
      parse(flags.cnn1_filters)):
    net = tf.keras.layers.Conv2D(
        filters, kernel_size,
        strides=stride,
        use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.Activation('relu')(net)
    # [batch, time, feature, filters]

  # first residual block
  for filters in parse(flags.cnn2_filters):
    residual = tf.keras.layers.Conv2D(
        filters, (1, 1), strides=(2, 2), padding='same', use_bias=False)(
            net)
    residual = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(
        residual)
    net = tf.keras.layers.SeparableConv2D(
        filters, (3, 3), padding='same', use_bias=False)(
            net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(
        net)
    net = tf.keras.layers.add([net, residual])
    # [batch, time, feature, filters]

  # second residual block
  filters = parse(flags.cnn2_filters)[-1]
  for _ in range(flags.cnn3_blocks):
    residual = net
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.SeparableConv2D(
        filters, (3, 3),
        padding='same',
        use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.SeparableConv2D(
        filters, (3, 3),
        padding='same',
        use_bias=False,)(net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.SeparableConv2D(
        filters, (3, 3),
        padding='same',
        use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.add([net, residual])
    # [batch, time, feature, filters]

  net = tf.keras.layers.GlobalAveragePooling2D()(net)
  # [batch, filters]
  net = tf.keras.layers.Dropout(flags.dropout)(net)
  net = tf.keras.layers.Dense(flags.label_count)(net)
  # [batch, label_count]
  return tf.keras.Model(input_audio, net)
