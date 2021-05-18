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

"""SVDF model with Mel spectrum and fully connected layers + residual."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import svdf
from kws_streaming.layers.compat import tf
import kws_streaming.models.model_utils as utils


def model_parameters(parser_nn):
  """SVDF+residual model parameters."""

  parser_nn.add_argument(
      '--block1_memory_size',
      type=str,
      default='4,10',
      help='Number of time steps to keep in memory (time dim) in svdf layers'
      'in the first residual block'
  )
  parser_nn.add_argument(
      '--block2_memory_size',
      type=str,
      default='10,10',
      help='Number of time steps to keep in memory (time dim) in svdf layers'
      'in the second residual block'
  )
  parser_nn.add_argument(
      '--block3_memory_size',
      type=str,
      default='10,10',
      help='Number of time steps to keep in memory (time dim) in svdf layers'
      'in the third residual block'
  )
  parser_nn.add_argument(
      '--block1_units1',
      type=str,
      default='256,256',
      help='Number of units (feature dim) in the first part of svdf layers'
      'in the first residual block'
  )
  parser_nn.add_argument(
      '--block2_units1',
      type=str,
      default='256,256',
      help='Number of units (feature dim) in the first part of svdf layers'
      'in the second residual block'
  )
  parser_nn.add_argument(
      '--block3_units1',
      type=str,
      default='256,256',
      help='Number of units (feature dim) in the first part of svdf layers'
      'in the third residual block'
  )
  parser_nn.add_argument(
      '--blocks_pool',
      type=str,
      default='1,2,2',
      help='pooling size after each block'
  )
  parser_nn.add_argument(
      '--use_batch_norm',
      type=int,
      default=1,
      help='Use batch normalization in svdf module'
  )
  parser_nn.add_argument(
      '--bn_scale',
      type=int,
      default=0,
      help='Use scaling in batch normalization '
      'if the last one is enabled by use_batch_norm'
  )
  parser_nn.add_argument(
      '--activation',
      type=str,
      default='relu',
      help='Activation type for all layers',
  )
  parser_nn.add_argument(
      '--svdf_dropout',
      type=float,
      default=0.0,
      help='Percentage of data dropped in all svdf layers',
  )
  parser_nn.add_argument(
      '--svdf_use_bias',
      type=int,
      default=1,
      help='Use bias in depthwise 1d conv',
  )
  parser_nn.add_argument(
      '--svdf_pad',
      type=int,
      default=1,
      help='If 1, causal pad svdf input data with zeros, else valid pad',
  )
  parser_nn.add_argument(
      '--dropout1',
      type=float,
      default=0.0,
      help='Percentage of data dropped after svdf layers',
  )
  parser_nn.add_argument(
      '--units2',
      type=str,
      default='',
      help='Number of units in the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--flatten',
      type=int,
      default=0,
      help='Apply flatten 1 or average pooling 0 in the last layer',
  )


def model(flags):
  """SVDF model with residual connections.

  This model is based on decomposition of a densely connected ops
  into low rank filters.
  It is based on paper
  END-TO-END STREAMING KEYWORD SPOTTING https://arxiv.org/pdf/1812.02802.pdf
  In addition we added residual connection
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

  blocks_pool = utils.parse(flags.blocks_pool)
  if len(blocks_pool) != 3:
    raise ValueError('number of pooling blocks has to be 3, but get: ',
                     len(blocks_pool))

  # for streaming mode it is better to use causal padding
  padding = 'causal' if flags.svdf_pad else 'valid'

  # first residual block
  number_of_blocks = len(utils.parse(flags.block1_units1))
  activations = [flags.activation] * number_of_blocks
  activations[-1] = 'linear'  # last layer is linear
  residual = net
  for i, (units1, memory_size, activation) in enumerate(
      zip(
          utils.parse(flags.block1_units1),
          utils.parse(flags.block1_memory_size), activations)):
    # [batch, time, feature]
    net = svdf.Svdf(
        units1=units1,
        memory_size=memory_size,
        units2=-1,
        dropout=flags.svdf_dropout,
        activation=activation,
        pad=padding,
        use_bias=flags.svdf_use_bias,
        use_batch_norm=flags.use_batch_norm,
        bn_scale=flags.bn_scale,
        name='svdf_1_%d' % i)(
            net)

  # number of channels in the last layer
  units1_last = utils.parse(flags.block1_units1)[-1]

  # equivalent to 1x1 convolution
  residual = tf.keras.layers.Dense(units1_last, use_bias=False)(residual)
  residual = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(residual)

  # residual connection
  net = tf.keras.layers.Add()([net, residual])
  # [batch, time, feature]
  net = tf.keras.layers.Activation(flags.activation)(net)
  net = tf.keras.layers.MaxPool1D(
      3, strides=blocks_pool[0], padding='valid')(
          net)

  # second residual block
  number_of_blocks = len(utils.parse(flags.block2_units1))
  activations = [flags.activation] * number_of_blocks
  activations[-1] = 'linear'  # last layer is linear
  residual = net
  for i, (units1, memory_size, activation) in enumerate(
      zip(
          utils.parse(flags.block2_units1),
          utils.parse(flags.block2_memory_size), activations)):
    # [batch, time, feature]
    net = svdf.Svdf(
        units1=units1,
        memory_size=memory_size,
        units2=-1,
        dropout=flags.svdf_dropout,
        activation=activation,
        pad=padding,
        use_bias=flags.svdf_use_bias,
        use_batch_norm=flags.use_batch_norm,
        bn_scale=flags.bn_scale,
        name='svdf_2_%d' % i)(
            net)

  # number of channels in the last layer
  units1_last = utils.parse(flags.block2_units1)[-1]

  # equivalent to 1x1 convolution
  residual = tf.keras.layers.Dense(units1_last, use_bias=False)(residual)
  residual = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(residual)

  # residual connection
  net = tf.keras.layers.Add()([net, residual])
  net = tf.keras.layers.Activation(flags.activation)(net)
  # [batch, time, feature]
  net = tf.keras.layers.MaxPool1D(
      3, strides=blocks_pool[1], padding='valid')(
          net)

  # third residual block
  number_of_blocks = len(utils.parse(flags.block3_units1))
  activations = [flags.activation] * number_of_blocks
  activations[-1] = 'linear'  # last layer is linear
  residual = net
  for i, (units1, memory_size, activation) in enumerate(
      zip(
          utils.parse(flags.block3_units1),
          utils.parse(flags.block3_memory_size), activations)):
    net = svdf.Svdf(
        units1=units1,
        memory_size=memory_size,
        units2=-1,
        dropout=flags.svdf_dropout,
        activation=activation,
        pad=padding,
        use_bias=flags.svdf_use_bias,
        use_batch_norm=flags.use_batch_norm,
        bn_scale=flags.bn_scale,
        name='svdf_3_%d' % i)(
            net)

  # number of channels in the last layer
  units1_last = utils.parse(flags.block3_units1)[-1]

  # equivalent to 1x1 convolution
  residual = tf.keras.layers.Dense(units1_last, use_bias=False)(residual)
  residual = tf.keras.layers.BatchNormalization(scale=flags.bn_scale)(residual)

  # residual connection
  net = tf.keras.layers.Add()([net, residual])
  net = tf.keras.layers.Activation(flags.activation)(net)
  net = tf.keras.layers.MaxPool1D(
      3, strides=blocks_pool[2], padding='valid')(
          net)
  # [batch, time, feature]

  # convert all feature to one vector
  if flags.flatten:
    net = tf.keras.layers.Flatten()(net)
  else:
    net = tf.keras.layers.GlobalAveragePooling1D()(net)

  # [batch, feature]
  net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

  for units in utils.parse(flags.units2):
    net = tf.keras.layers.Dense(units=units, activation=flags.activation)(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
