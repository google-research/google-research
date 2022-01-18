# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""DNN model with Mel spectrum and fully connected layers."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
import kws_streaming.models.model_utils as utils


def model_parameters(parser_nn):
  """DNN model parameters."""
  parser_nn.add_argument(
      '--units1', type=str, default='64,128',
      help='List of units in the first set of hidden layers',)
  parser_nn.add_argument(
      '--act1', type=str, default="'relu','relu'",
      help='List of activation functions of the first set hidden layers',)
  parser_nn.add_argument(
      '--pool_size', type=int, default=2,
      help='Pooling size after first set of hidden layer',)
  parser_nn.add_argument(
      '--strides', type=int, default=2,
      help='Stride after first set of hidden layer',)
  parser_nn.add_argument(
      '--dropout1', type=float, default=0.1,
      help='Percentage of data dropped',)
  parser_nn.add_argument(
      '--units2', type=str, default='128,256',
      help='List of units in the second set of hidden layers',)
  parser_nn.add_argument(
      '--act2', type=str, default="'linear','relu'",
      help='List of activation functions of the second set of hidden layers',)


def model(flags):
  """Fully connected layer based model.

  It is based on paper (with added pooling):
  SMALL-FOOTPRINT KEYWORD SPOTTING USING DEEP NEURAL NETWORKS
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42537.pdf
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

  for units, activation in zip(
      utils.parse(flags.units1), utils.parse(flags.act1)):
    net = tf.keras.layers.Dense(units=units, activation=activation)(net)

  net = stream.Stream(cell=tf.keras.layers.Flatten())(net)

  # after flattening data in time, we can apply any layer: pooling, bi-lstm etc
  if flags.pool_size > 1:
    # add fake dim for compatibility with pooling
    net = tf.keras.backend.expand_dims(net, axis=-1)
    net = tf.keras.layers.MaxPool1D(
        pool_size=flags.pool_size,
        strides=flags.strides,
        data_format='channels_last')(net)
    # remove fake dim
    net = tf.keras.backend.squeeze(net, axis=-1)

  net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

  for units, activation in zip(
      utils.parse(flags.units2), utils.parse(flags.act2)):
    net = tf.keras.layers.Dense(units=units, activation=activation)(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
