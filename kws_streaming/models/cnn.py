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

"""CNN model with Mel spectrum."""
from kws_streaming.layers import modes
from kws_streaming.layers import quantize
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
from kws_streaming.models import utils
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import AllValuesQuantizer


def model_parameters(parser_nn):
  """Covolutional Neural Network(CNN) model parameters."""

  parser_nn.add_argument(
      '--cnn_filters',
      type=str,
      default='64,64,64,64,128,64,128',
      help='Number of output filters in the convolution layers',
  )
  parser_nn.add_argument(
      '--cnn_kernel_size',
      type=str,
      default='(3,3),(5,3),(5,3),(5,3),(5,2),(5,1),(10,1)',
      help='Heights and widths of the 2D convolution window',
  )
  parser_nn.add_argument(
      '--cnn_act',
      type=str,
      default="'relu','relu','relu','relu','relu','relu','relu'",
      help='Activation function in the convolution layers',
  )
  parser_nn.add_argument(
      '--cnn_dilation_rate',
      type=str,
      default='(1,1),(1,1),(2,1),(1,1),(2,1),(1,1),(2,1)',
      help='Dilation rate to use for dilated convolutions',
  )
  parser_nn.add_argument(
      '--cnn_strides',
      type=str,
      default='(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)',
      help='Strides of the convolution layers along the height and width',
  )
  parser_nn.add_argument(
      '--dropout1',
      type=float,
      default=0.5,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--units2',
      type=str,
      default='128,256',
      help='Number of units in the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--act2',
      type=str,
      default="'linear','relu'",
      help='Activation function of the last set of hidden layers',
  )


def model(flags):
  """CNN model.

  It is based on paper:
  Convolutional Neural Networks for Small-footprint Keyword Spotting
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
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

  if flags.quantize:
    net = quantize_layer.QuantizeLayer(
        AllValuesQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False))(
                net)

  net = tf.keras.backend.expand_dims(net)
  for filters, kernel_size, activation, dilation_rate, strides in zip(
      utils.parse(flags.cnn_filters), utils.parse(flags.cnn_kernel_size),
      utils.parse(flags.cnn_act), utils.parse(flags.cnn_dilation_rate),
      utils.parse(flags.cnn_strides)):
    net = stream.Stream(
        cell=quantize.quantize_layer(
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                activation='linear',
                strides=strides), flags.quantize,
            quantize.NoOpActivationConfig(['kernel'], ['activation'], False)),
        pad_time_dim='causal',
        use_one_step=False)(
            net)
    net = quantize.quantize_layer(
        tf.keras.layers.BatchNormalization(),
        default_8bit_quantize_configs.NoOpQuantizeConfig())(
            net)
    net = quantize.quantize_layer(tf.keras.layers.Activation(activation))(net)

  net = stream.Stream(
      cell=quantize.quantize_layer(
          tf.keras.layers.Flatten(), apply_quantization=flags.quantize))(
              net)

  net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

  for units, activation in zip(
      utils.parse(flags.units2), utils.parse(flags.act2)):
    net = quantize.quantize_layer(
        tf.keras.layers.Dense(units=units, activation=activation),
        apply_quantization=flags.quantize)(
            net)

  net = quantize.quantize_layer(
      tf.keras.layers.Dense(units=flags.label_count),
      apply_quantization=flags.quantize)(
          net)
  if flags.return_softmax:
    net = quantize.quantize_layer(
        tf.keras.layers.Activation('softmax'),
        apply_quantization=flags.quantize)(
            net)
  return tf.keras.Model(input_audio, net)
