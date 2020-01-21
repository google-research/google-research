# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""SVDF model with Mel spectrum and fully connected layers."""

from kws_streaming.layers import speech_features
from kws_streaming.layers import svdf
from kws_streaming.layers.compat import tf
from kws_streaming.layers.stream import Stream
from kws_streaming.models.utils import parse


def model_parameters(parser_nn):
  """SVDF model parameters."""

  parser_nn.add_argument(
      '--svdf_memory_size',
      type=str,
      default='4,10,10,10,10,10',
      help='Number of time steps to keep in memory (time dim) in svdf layers',
  )
  parser_nn.add_argument(
      '--svdf_units1',
      type=str,
      default='256,256,256,256,256,256',
      help='Number of units (feature dim) in the first part of svdf layers',
  )
  parser_nn.add_argument(
      '--svdf_act',
      type=str,
      default='relu,selu,selu,selu,selu,selu',
      help='Number of units in the first set of hidden layers',
  )
  parser_nn.add_argument(
      '--svdf_units2',
      type=str,
      default='128,128,128,128,128,-1',
      help='Number of units (feature dim) in projection op of svdf layers',
  )
  parser_nn.add_argument(
      '--svdf_dropout',
      type=str,
      default='0.5,0.5,0.5,0.5,0.5,0.5',
      help='Percentage of data dropped in svdf layers',
  )
  parser_nn.add_argument(
      '--svdf_pad',
      type=int,
      default=0,
      help='If 1, pad svdf input data with zeros',
  )
  parser_nn.add_argument(
      '--dropout1',
      type=float,
      default=0.5,
      help='Percentage of data dropped after svdf layers',
  )
  parser_nn.add_argument(
      '--units2',
      type=str,
      default='',
      help='Number of units in the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--act2',
      type=str,
      default='',
      help='Activation function of the last set of hidden layers',
  )


def model(flags):
  """SVDF model.

  This model is based on decomposition of a densely connected ops
  into low rank filters.
  It is based on paper: https://arxiv.org/pdf/1812.02802.pdf
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """

  input_audio = tf.keras.layers.Input(
      shape=(flags.desired_samples,), batch_size=flags.batch_size)

  net = speech_features.SpeechFeatures(
      frame_size_ms=flags.window_size_ms,
      frame_step_ms=flags.window_stride_ms,
      sample_rate=flags.sample_rate,
      use_tf_fft=flags.use_tf_fft,
      preemph=flags.preemph,
      window_type=flags.window_type,
      mel_num_bins=flags.mel_num_bins,
      mel_lower_edge_hertz=flags.mel_lower_edge_hertz,
      mel_upper_edge_hertz=flags.mel_upper_edge_hertz,
      mel_non_zero_only=flags.mel_non_zero_only,
      fft_magnitude_squared=flags.fft_magnitude_squared,
      dct_num_features=flags.dct_num_features)(
          input_audio)

  for i, (units1, memory_size, units2, dropout, activation) in enumerate(
      zip(
          parse(flags.svdf_units1), parse(flags.svdf_memory_size),
          parse(flags.svdf_units2), parse(flags.svdf_dropout),
          parse(flags.svdf_act))):
    net = svdf.Svdf(
        units1=units1,
        memory_size=memory_size,
        units2=units2,
        dropout=dropout,
        activation=activation,
        pad=flags.svdf_pad,
        name='svdf_%d' % i)(
            net)

  net = Stream(cell=tf.keras.layers.Flatten())(net)
  net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

  for units, activation in zip(parse(flags.units2), parse(flags.act2)):
    net = tf.keras.layers.Dense(units=units, activation=activation)(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)
  return tf.keras.Model(input_audio, net)
