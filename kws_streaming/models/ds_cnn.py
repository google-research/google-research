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

"""Model based on combination of depthwise and 1x1 convolutions."""
from kws_streaming.layers import speech_features
from kws_streaming.layers.compat import tf
from kws_streaming.layers.stream import Stream
from kws_streaming.models.utils import parse


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
      default='10,1',
      help='Heights and widths of the first 2D convolution',
  )
  parser_nn.add_argument(
      '--cnn1_dilation_rate',
      type=str,
      default='1,1',
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
      help="one of 'valid' or 'same'",
  )
  parser_nn.add_argument(
      '--cnn1_filters',
      type=int,
      default=276,
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
      default=0.96,
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
      default='(10,2),(5,2),(5,3),(5,3),(6,3)',
      help='Height and width of the 2D Depthwise convolutions',
  )
  parser_nn.add_argument(
      '--dw2_dilation_rate',
      type=str,
      default='(1,1),(1,1),(1,1),(1,1),(1,1)',
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
      help="one of 'valid' or 'same'",
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
      default='276,276,276,276,276',
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
      default=0.0,
      help='Percentage of data dropped',
  )


def model(flags):
  """Depthwise convolutional model.

  It is based on paper:
  MobileNets: Efficient Convolutional Neural Networks for
  Mobile Vision Applications https://arxiv.org/abs/1704.04861
  Hello Edge: Keyword Spotting on Microcontrollers
  https://arxiv.org/pdf/1711.07128.pdf
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

  net = tf.keras.backend.expand_dims(net)

  net = Stream(
      cell=tf.keras.layers.Conv2D(
          kernel_size=parse(flags.cnn1_kernel_size),
          dilation_rate=parse(flags.cnn1_dilation_rate),
          filters=flags.cnn1_filters,
          padding=flags.cnn1_padding,
          strides=parse(flags.cnn1_strides)))(
              net)
  net = tf.keras.layers.BatchNormalization(
      momentum=flags.bn_momentum,
      center=flags.bn_center,
      scale=flags.bn_scale,
      renorm=flags.bn_renorm)(
          net)
  net = tf.keras.layers.Activation('relu')(net)

  for kernel_size, dw2_act, dilation_rate, strides, filters, cnn2_act in zip(
      parse(flags.dw2_kernel_size), parse(flags.dw2_act),
      parse(flags.dw2_dilation_rate), parse(flags.dw2_strides),
      parse(flags.cnn2_filters), parse(flags.cnn2_act)):
    net = Stream(
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

  net = Stream(
      cell=tf.keras.layers.AveragePooling2D(
          pool_size=(int(net.shape[1]), int(net.shape[2]))))(
              net)

  net = Stream(cell=tf.keras.layers.Flatten())(net)
  net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)
  return tf.keras.Model(input_audio, net)
