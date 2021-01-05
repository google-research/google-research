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

"""BiRNN model with attention."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
from kws_streaming.models import utils


def model_parameters(parser_nn):
  """BiRNN attention model parameters."""

  parser_nn.add_argument(
      '--cnn_filters',
      type=str,
      default='10,1',
      help='Number of output filters in the convolution layers',
  )
  parser_nn.add_argument(
      '--cnn_kernel_size',
      type=str,
      default='(5,1),(5,1)',
      help='Heights and widths of the 2D convolution window',
  )
  parser_nn.add_argument(
      '--cnn_act',
      type=str,
      default="'relu','relu'",
      help='Activation function in the convolution layers',
  )
  parser_nn.add_argument(
      '--cnn_dilation_rate',
      type=str,
      default='(1,1),(1,1)',
      help='Dilation rate to use for dilated convolutions',
  )
  parser_nn.add_argument(
      '--cnn_strides',
      type=str,
      default='(1,1),(1,1)',
      help='Strides of the convolution layers along the height and width',
  )
  parser_nn.add_argument(
      '--rnn_layers',
      type=int,
      default=2,
      help='Number of RNN layers (each RNN is wrapped by Bidirectional)',
  )
  parser_nn.add_argument(
      '--rnn_type',
      type=str,
      default='gru',
      help='RNN type: it can be gru or lstm',
  )
  parser_nn.add_argument(
      '--rnn_units',
      type=int,
      default=128,
      help='Units number in RNN cell',
  )
  parser_nn.add_argument(
      '--dropout1',
      type=float,
      default=0.1,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--units2',
      type=str,
      default='64,32',
      help='Number of units in the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--act2',
      type=str,
      default="'relu','linear'",
      help='Activation function of the last set of hidden layers',
  )


def model(flags):
  """BiRNN attention model.

  It is based on paper:
  A neural attention model for speech command recognition
  https://arxiv.org/pdf/1808.08929.pdf

  Depending on parameter rnn_type, model can be biLSTM or biGRU

  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """

  rnn_types = {'lstm': tf.keras.layers.LSTM, 'gru': tf.keras.layers.GRU}

  if flags.rnn_type not in rnn_types:
    ValueError('not supported RNN type ', flags.rnn_type)
  rnn = rnn_types[flags.rnn_type]

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
  for filters, kernel_size, activation, dilation_rate, strides in zip(
      utils.parse(flags.cnn_filters), utils.parse(flags.cnn_kernel_size),
      utils.parse(flags.cnn_act), utils.parse(flags.cnn_dilation_rate),
      utils.parse(flags.cnn_strides)):
    net = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        dilation_rate=dilation_rate,
        strides=strides,
        padding='same')(
            net)
    net = tf.keras.layers.BatchNormalization()(net)

  shape = net.shape
  # input net dimension: [batch, time, feature, channels]
  # reshape dimension: [batch, time, feature * channels]
  # so that GRU/RNN can process it
  net = tf.keras.layers.Reshape((-1, shape[2] * shape[3]))(net)

  # dims: [batch, time, feature]
  for _ in range(flags.rnn_layers):
    net = tf.keras.layers.Bidirectional(
        rnn(flags.rnn_units, return_sequences=True, unroll=True))(
            net)
  feature_dim = net.shape[-1]
  middle = net.shape[1] // 2  # index of middle point of sequence

  # feature vector at middle point [batch, feature]
  mid_feature = net[:, middle, :]
  # apply one projection layer with the same dim as input feature
  query = tf.keras.layers.Dense(feature_dim)(mid_feature)

  # attention weights [batch, time]
  att_weights = tf.keras.layers.Dot(axes=[1, 2])([query, net])
  att_weights = tf.keras.layers.Softmax(name='attSoftmax')(att_weights)

  # apply attention weights [batch, feature]
  net = tf.keras.layers.Dot(axes=[1, 1])([att_weights, net])

  net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

  for units, activation in zip(
      utils.parse(flags.units2), utils.parse(flags.act2)):
    net = tf.keras.layers.Dense(units=units, activation=activation)(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
