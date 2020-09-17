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

"""LSTM with Mel spectrum and fully connected layers."""
from kws_streaming.layers import lstm
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
from kws_streaming.models.utils import parse


def model_parameters(parser_nn):
  """LSTM model parameters."""
  parser_nn.add_argument(
      '--lstm_units',
      type=str,
      default='500',
      help='Output space dimensionality of lstm layer ',
  )
  parser_nn.add_argument(
      '--return_sequences',
      type=str,
      default='0',
      help='Whether to return the last output in the output sequence,'
      'or the full sequence',
  )
  parser_nn.add_argument(
      '--stateful',
      type=int,
      default='1',
      help='If True, the last state for each sample at index i'
      'in a batch will be used as initial state for the sample '
      'of index i in the following batch',
  )
  parser_nn.add_argument(
      '--num_proj',
      type=str,
      default='200',
      help='The output dimensionality for the projection matrices.',
  )
  parser_nn.add_argument(
      '--use_peepholes',
      type=int,
      default='1',
      help='True to enable diagonal/peephole connections',
  )
  parser_nn.add_argument(
      '--dropout1',
      type=float,
      default=0.3,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--units1',
      type=str,
      default='',
      help='Number of units in the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--act1',
      type=str,
      default='',
      help='Activation function of the last set of hidden layers',
  )


def model(flags):
  """LSTM model.

  Similar model in papers:
  Convolutional Recurrent Neural Networks for Small-Footprint Keyword Spotting
  https://arxiv.org/pdf/1703.05390.pdf (with no conv layer)
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

  for units, return_sequences, num_proj in zip(
      parse(flags.lstm_units), parse(flags.return_sequences),
      parse(flags.num_proj)):
    net = lstm.LSTM(
        units=units,
        return_sequences=return_sequences,
        stateful=flags.stateful,
        use_peepholes=flags.use_peepholes,
        num_proj=num_proj)(
            net)

  net = stream.Stream(cell=tf.keras.layers.Flatten())(net)
  net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

  for units, activation in zip(parse(flags.units1), parse(flags.act1)):
    net = tf.keras.layers.Dense(units=units, activation=activation)(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
