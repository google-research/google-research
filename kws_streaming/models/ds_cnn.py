# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Model based on combination of 2D depthwise and 1x1 convolutions."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
import kws_streaming.models.model_utils as utils


def model_parameters(parser_nn):
  """Depthwise Convolutional(DS CNN) model parameters.

  In more details parameters are described at:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

  Args:
    parser_nn: global command line args parser
  Returns: parser with updated arguments
  """

  parser_nn.add_argument(
      '--cnn1_kernel_size',
      type=str,
      default='3,3',
      help='Heights and widths of the first 2D convolution',
  )
  parser_nn.add_argument(
      '--cnn1_dilation_rate',
      type=str,
      default='2,1',
      help='Dilation rate of the first 2D convolution',
  )
  parser_nn.add_argument(
      '--cnn1_strides',
      type=str,
      default='1,1',
      help='Strides of the first 2D convolution along the height and width',
  )
  parser_nn.add_argument(
      '--cnn1_padding',
      type=str,
      default='valid',
      help="One of 'valid' or 'same'",
  )
  parser_nn.add_argument(
      '--cnn1_filters',
      type=int,
      default=300,
      help='Number of output filters in the first 2D convolution layers',
  )
  parser_nn.add_argument(
      '--cnn1_act',
      type=str,
      default='relu',
      help='Activation function in the first 2D convolution layers',
  )
  parser_nn.add_argument(
      '--bn_momentum',
      type=float,
      default=0.98,
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
      default=0,
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
      '--dw2_kernel_size',
      type=str,
      default='(3,3),(3,3),(10,3),(5,3),(10,3)',
      help='Height and width of the 2D Depthwise convolutions',
  )
  parser_nn.add_argument(
      '--dw2_dilation_rate',
      type=str,
      default='(1,1),(2,2),(1,1),(2,2),(1,1)',
      help='Dilation rate of the 2D Depthwise convolutions',
  )
  parser_nn.add_argument(
      '--dw2_strides',
      type=str,
      default='(1,1),(1,1),(1,1),(1,1),(1,1)',
      help='Strides of the 2D Depthwise convolutions',
  )
  parser_nn.add_argument(
      '--dw2_padding',
      type=str,
      default='valid',
      help="One of 'valid' or 'same'",
  )
  parser_nn.add_argument(
      '--dw2_act',
      type=str,
      default="'relu','relu','relu','relu','relu'",
      help='Activation functions in the Depthwise convolution layers',
  )
  parser_nn.add_argument(
      '--cnn2_filters',
      type=str,
      default='300,300,300,300,300',
      help='Number of output filters in 1x1 convolution layers',
  )
  parser_nn.add_argument(
      '--cnn2_act',
      type=str,
      default="'relu','relu','relu','relu','relu'",
      help='Activation functions in 1x1 convolution layers',
  )
  parser_nn.add_argument(
      '--dropout1',
      type=float,
      default=0.2,
      help='Percentage of data dropped',
  )


def model(flags):
  """Depthwise convolutional model.

  It is based on paper:
  MobileNets: Efficient Convolutional Neural Networks for
  Mobile Vision Applications https://arxiv.org/abs/1704.04861
  Model topology is similar with "Hello Edge: Keyword Spotting on
  Microcontrollers" https://arxiv.org/pdf/1711.07128.pdf
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

  net = tf.keras.backend.expand_dims(net)

  net = stream.Stream(
      cell=tf.keras.layers.Conv2D(
          kernel_size=utils.parse(flags.cnn1_kernel_size),
          dilation_rate=utils.parse(flags.cnn1_dilation_rate),
          filters=flags.cnn1_filters,
          padding=flags.cnn1_padding,
          strides=utils.parse(flags.cnn1_strides)))(
              net)
  net = tf.keras.layers.BatchNormalization(
      momentum=flags.bn_momentum,
      center=flags.bn_center,
      scale=flags.bn_scale,
      renorm=flags.bn_renorm)(
          net)
  net = tf.keras.layers.Activation('relu')(net)

  for kernel_size, dw2_act, dilation_rate, strides, filters, cnn2_act in zip(
      utils.parse(flags.dw2_kernel_size), utils.parse(flags.dw2_act),
      utils.parse(flags.dw2_dilation_rate), utils.parse(flags.dw2_strides),
      utils.parse(flags.cnn2_filters), utils.parse(flags.cnn2_act)):
    net = stream.Stream(
        cell=tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=flags.dw2_padding,
            strides=strides))(
                net)
    net = tf.keras.layers.BatchNormalization(
        momentum=flags.bn_momentum,
        center=flags.bn_center,
        scale=flags.bn_scale,
        renorm=flags.bn_renorm)(
            net)
    net = tf.keras.layers.Activation(dw2_act)(net)
    net = tf.keras.layers.Conv2D(kernel_size=(1, 1), filters=filters)(net)
    net = tf.keras.layers.BatchNormalization(
        momentum=flags.bn_momentum,
        center=flags.bn_center,
        scale=flags.bn_scale,
        renorm=flags.bn_renorm)(
            net)
    net = tf.keras.layers.Activation(cnn2_act)(net)

  net = stream.Stream(
      cell=tf.keras.layers.AveragePooling2D(
          pool_size=(int(net.shape[1]), int(net.shape[2]))))(
              net)

  net = stream.Stream(cell=tf.keras.layers.Flatten())(net)
  net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
