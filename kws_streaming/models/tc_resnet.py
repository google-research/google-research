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

"""Model based on combination of n by 1 convolutions with residual blocks."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
import kws_streaming.models.model_utils as utils


def model_parameters(parser_nn):
  """Temporal Convolution Resnet model parameters.

  In more details parameters are described at:
  https://arxiv.org/pdf/1904.03814.pdf
  We converted model to Keras and made it compatible with TF V2
  https://github.com/hyperconnect/TC-ResNet


  Args:
    parser_nn: global command line args parser
  Returns: parser with updated arguments
  """

  parser_nn.add_argument(
      '--channels',
      type=str,
      default='24, 36, 36, 48, 48, 72, 72',
      help='Number of channels per convolutional block (including first conv)',
  )
  parser_nn.add_argument(
      '--debug_2d',
      type=int,
      default=0,
      help='If 0 conv_kernel will be [3, 3], else conv_kernel [3, 1]',
  )
  parser_nn.add_argument(
      '--pool_size',
      type=str,
      default='',
      help="Pool size for example '4,4'",
  )
  parser_nn.add_argument(
      '--kernel_size',
      type=str,
      default='(9,1)',
      help='Kernel size of conv layer',
  )
  parser_nn.add_argument(
      '--pool_stride',
      type=int,
      default=0,
      help='Pool stride, for example 4',
  )
  parser_nn.add_argument(
      '--bn_momentum',
      type=float,
      default=0.997,
      help='Momentum for the moving average',
  )
  parser_nn.add_argument(
      '--bn_center',
      type=int,
      default=1,
      help='If True, add offset of beta to normalized tensor.'
      'If False, beta is ignored',
  )
  parser_nn.add_argument(
      '--bn_scale',
      type=int,
      default=1,
      help='If True, multiply by gamma. If False, gamma is not used. '
      'When the next layer is linear (also e.g. nn.relu), this can be disabled'
      'since the scaling will be done by the next layer.',
  )
  parser_nn.add_argument(
      '--bn_renorm',
      type=int,
      default=0,
      help='Whether to use Batch Renormalization',
  )
  parser_nn.add_argument(
      '--dropout',
      type=float,
      default=0.2,
      help='Percentage of data dropped',
  )


def model(flags):
  """Temporal Convolution ResNet model.

  It is based on paper:
  Temporal Convolution for Real-time Keyword Spotting on Mobile Devices
  https://arxiv.org/pdf/1904.03814.pdf
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

  time_size, feature_size = net.shape[1:3]

  channels = utils.parse(flags.channels)

  net = tf.keras.backend.expand_dims(net)

  if flags.debug_2d:
    conv_kernel = first_conv_kernel = (3, 3)
  else:
    net = tf.reshape(
        net, [-1, time_size, 1, feature_size])  # [batch, time, 1, feature]
    first_conv_kernel = (3, 1)
    conv_kernel = utils.parse(flags.kernel_size)

  net = tf.keras.layers.Conv2D(
      filters=channels[0],
      kernel_size=first_conv_kernel,
      strides=1,
      padding='same',
      activation='linear')(
          net)
  net = tf.keras.layers.BatchNormalization(
      momentum=flags.bn_momentum,
      center=flags.bn_center,
      scale=flags.bn_scale,
      renorm=flags.bn_renorm)(
          net)
  net = tf.keras.layers.Activation('relu')(net)

  if utils.parse(flags.pool_size):
    net = tf.keras.layers.AveragePooling2D(
        pool_size=utils.parse(flags.pool_size), strides=flags.pool_stride)(
            net)

  channels = channels[1:]

  # residual blocks
  for n in channels:
    if n != net.shape[-1]:
      stride = 2
      layer_in = tf.keras.layers.Conv2D(
          filters=n,
          kernel_size=1,
          strides=stride,
          padding='same',
          activation='linear')(
              net)
      layer_in = tf.keras.layers.BatchNormalization(
          momentum=flags.bn_momentum,
          center=flags.bn_center,
          scale=flags.bn_scale,
          renorm=flags.bn_renorm)(
              layer_in)
      layer_in = tf.keras.layers.Activation('relu')(layer_in)
    else:
      layer_in = net
      stride = 1

    net = tf.keras.layers.Conv2D(
        filters=n,
        kernel_size=conv_kernel,
        strides=stride,
        padding='same',
        activation='linear')(
            net)
    net = tf.keras.layers.BatchNormalization(
        momentum=flags.bn_momentum,
        center=flags.bn_center,
        scale=flags.bn_scale,
        renorm=flags.bn_renorm)(
            net)
    net = tf.keras.layers.Activation('relu')(net)

    net = tf.keras.layers.Conv2D(
        filters=n,
        kernel_size=conv_kernel,
        strides=1,
        padding='same',
        activation='linear')(
            net)
    net = tf.keras.layers.BatchNormalization(
        momentum=flags.bn_momentum,
        center=flags.bn_center,
        scale=flags.bn_scale,
        renorm=flags.bn_renorm)(
            net)

    # residual connection
    net = tf.keras.layers.Add()([net, layer_in])
    net = tf.keras.layers.Activation('relu')(net)

  net = tf.keras.layers.AveragePooling2D(
      pool_size=net.shape[1:3], strides=1)(
          net)

  net = tf.keras.layers.Dropout(rate=flags.dropout)(net)

  # fully connected layer
  net = tf.keras.layers.Conv2D(
      filters=flags.label_count,
      kernel_size=1,
      strides=1,
      padding='same',
      activation='linear')(
          net)

  net = tf.reshape(net, shape=(-1, net.shape[3]))
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
