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

"""Inception resnet - reduced version of keras/applications/inception_resnet_v2.py."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
from kws_streaming.models import utils


def model_parameters(parser_nn):
  """Inception resnet model parameters.

  Args:
    parser_nn: global command line args parser
  Returns: parser with updated arguments
  """
  parser_nn.add_argument(
      '--cnn1_filters',
      type=str,
      default='32',
      help='Number of filters in conv blocks',
  )
  parser_nn.add_argument(
      '--cnn1_strides',
      type=str,
      default='1',
      help='Strides applied in pooling of blocks',
  )
  parser_nn.add_argument(
      '--cnn1_kernel_sizes',
      type=str,
      default='5',
      help='Kernel sizes in time dim of conv block',
  )
  parser_nn.add_argument(
      '--cnn2_scales',
      type=str,
      default='0.2,0.5,1.0',
      help='Scaling factor to scale the residuals inside of inception block',
  )
  parser_nn.add_argument(
      '--cnn2_filters_branch0',
      type=str,
      default='32,32,32',
      help='Number of filters inside of inception block branch0'
  )
  parser_nn.add_argument(
      '--cnn2_filters_branch1',
      type=str,
      default='32,32,32',
      help='Number of filters inside of inception block branch1'
  )
  parser_nn.add_argument(
      '--cnn2_filters_branch2',
      type=str,
      default='32,32,64',
      help='Number of filters outsize of inception block'
  )
  parser_nn.add_argument(
      '--cnn2_strides',
      type=str,
      default='2,2,1',
      help='Stride parameter of poolling after inception block'
  )
  parser_nn.add_argument(
      '--cnn2_kernel_sizes',
      type=str,
      default='3,5,5',
      help='Kernel sizes of conv in inception block branch1, applied twice'
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


def inception_resnet_block(x,
                           scale,
                           filters_branch0,
                           filters_branch1,
                           kernel_size,
                           activation='relu',
                           bn_scale=False):
  """Adds a Inception-ResNet block.

  This function builds only one types of Inception-ResNet block
  reduced version of keras/applications/inception_resnet_v2.py
  Arguments:
    x: input tensor.
    scale: scaling factor to scale the residuals (i.e., the output of
      passing `x` through an inception module) before adding them
      to the shortcut branch.
      Let `r` be the output from the residual branch,
      the output of this block will be `x + scale * r`.
    filters_branch0: number of filters in branch0
    filters_branch1: number of filters in branch1
    kernel_size: kernel size of conv in branch1
    activation: activation function to use at the end of the block
    bn_scale: use scale in batch normalization layer

  Returns:
      Output tensor for the block.
  """

  # only one type of branching is supported
  branch_0 = utils.conv2d_bn(x, filters_branch0, 1, scale=bn_scale)
  branch_1 = utils.conv2d_bn(x, filters_branch0, 1, scale=bn_scale)
  branch_1 = utils.conv2d_bn(
      branch_1, filters_branch1, [kernel_size, 1], scale=bn_scale)
  branch_1 = utils.conv2d_bn(
      branch_1, filters_branch1, [kernel_size, 1], scale=bn_scale)
  branches = [branch_0, branch_1]

  mixed = tf.keras.layers.Concatenate()(branches)
  up = utils.conv2d_bn(
      mixed,
      tf.keras.backend.int_shape(x)[-1],
      1,
      activation=None,
      scale=bn_scale,
      use_bias=True)

  x = tf.keras.layers.Lambda(
      lambda inputs, scale: inputs[0] + inputs[1] * scale,
      output_shape=tf.keras.backend.int_shape(x)[1:],
      arguments={'scale': scale})([x, up])
  if activation is not None:
    x = tf.keras.layers.Activation(activation)(x)
  return x


def model(flags):
  """Inception resnet model.

  It is based on paper:
  Inception-v4, Inception-ResNet and the Impact of
     Residual Connections on Learning https://arxiv.org/abs/1602.07261
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
  # [batch, time, 1, feature]

  for filters, kernel_size, stride in zip(
      utils.parse(flags.cnn1_filters), utils.parse(flags.cnn1_kernel_sizes),
      utils.parse(flags.cnn1_strides)):
    net = utils.conv2d_bn(
        net, filters, (kernel_size, 1), scale=flags.bn_scale, padding='valid')
    if stride > 1:
      net = tf.keras.layers.MaxPooling2D((3, 1), strides=(stride, 1))(net)
    # [batch, time, 1, filters]

  for stride, scale, filters_branch0, filters_branch1, filters_branch2, kernel_size in zip(
      utils.parse(flags.cnn2_strides), utils.parse(flags.cnn2_scales),
      utils.parse(flags.cnn2_filters_branch0),
      utils.parse(flags.cnn2_filters_branch1),
      utils.parse(flags.cnn2_filters_branch2),
      utils.parse(flags.cnn2_kernel_sizes)):
    net = inception_resnet_block(
        net,
        scale,
        filters_branch0,
        filters_branch1,
        kernel_size,
        bn_scale=flags.bn_scale)
    net = utils.conv2d_bn(
        net, filters_branch2, (1, 1), scale=flags.bn_scale, padding='valid')
    if stride > 1:
      net = tf.keras.layers.MaxPooling2D((3, 1),
                                         strides=(stride, 1),
                                         padding='valid')(
                                             net)
    # [batch, time, 1, filters]

  net = tf.keras.layers.GlobalAveragePooling2D()(net)
  # [batch, filters]
  net = tf.keras.layers.Dropout(flags.dropout)(net)
  net = tf.keras.layers.Dense(flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
