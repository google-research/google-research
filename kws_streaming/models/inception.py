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

"""Inception - reduced version of keras/applications/inception_v3.py ."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
from kws_streaming.models import utils


def model_parameters(parser_nn):
  """Inception model parameters.

  Args:
    parser_nn: global command line args parser
  Returns: parser with updated arguments
  """
  parser_nn.add_argument(
      '--cnn_filters0',
      type=str,
      default='24',
      help='Number of filters in conv blocks',
  )
  parser_nn.add_argument(
      '--cnn_strides',
      type=str,
      default='2,2,1',
      help='strides applied in inception block',
  )
  parser_nn.add_argument(
      '--cnn_filters1',
      type=str,
      default='16,24,24',
      help='internal number of filters inside of inception block',
  )
  parser_nn.add_argument(
      '--cnn_filters2',
      type=str,
      default='10,10,16',
      help='number of filters inside of inception block '
      'will be multipled by 4 because of concatenation of 4 branches',
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
  """Inception model.

  It is based on paper:
  Rethinking the Inception Architecture for Computer Vision
      http://arxiv.org/abs/1512.00567
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

  for filters in utils.parse(flags.cnn_filters0):
    net = tf.keras.layers.SeparableConv2D(
        filters, (3, 3), padding='valid', use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(net)
    # [batch, time, feature, filters]

  filters = utils.parse(flags.cnn_filters0)[-1]
  net = utils.conv2d_bn(
      net, filters, (3, 1), padding='valid', scale=flags.bn_scale)
  net = utils.conv2d_bn(
      net, filters, (1, 3), padding='valid', scale=flags.bn_scale)

  for stride, filters1, filters2 in zip(
      utils.parse(flags.cnn_strides), utils.parse(flags.cnn_filters1),
      utils.parse(flags.cnn_filters2)):

    if stride > 1:
      net = tf.keras.layers.MaxPooling2D((3, 3), strides=stride)(net)

    branch1 = utils.conv2d_bn(net, filters2, (1, 1), scale=flags.bn_scale)

    branch2 = utils.conv2d_bn(net, filters1, (1, 1), scale=flags.bn_scale)
    branch2 = utils.conv2d_bn(branch2, filters1, (3, 1), scale=flags.bn_scale)
    branch2 = utils.conv2d_bn(branch2, filters2, (1, 3), scale=flags.bn_scale)

    branch3 = utils.conv2d_bn(net, filters1, (1, 1), scale=flags.bn_scale)
    branch3 = utils.conv2d_bn(branch3, filters1, (3, 1), scale=flags.bn_scale)
    branch3 = utils.conv2d_bn(branch3, filters1, (1, 3), scale=flags.bn_scale)
    branch3 = utils.conv2d_bn(branch3, filters1, (3, 1), scale=flags.bn_scale)
    branch3 = utils.conv2d_bn(branch3, filters2, (1, 3), scale=flags.bn_scale)

    branch4 = tf.keras.layers.AveragePooling2D((3, 3),
                                               strides=(1, 1),
                                               padding='same')(
                                                   net)
    branch4 = utils.conv2d_bn(branch4, filters2, (1, 1), scale=flags.bn_scale)
    net = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4])
    # [batch, time, feature, filters*4]

  net = tf.keras.layers.GlobalAveragePooling2D()(net)
  # [batch, filters*4]
  net = tf.keras.layers.Dropout(flags.dropout)(net)
  net = tf.keras.layers.Dense(flags.label_count)(net)
  return tf.keras.Model(input_audio, net)
