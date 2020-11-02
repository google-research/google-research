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

"""Model based on 1D depthwise and 1x1 convolutions in time + residual."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
from kws_streaming.models import utils


def model_parameters(parser_nn):
  """MatchboxNet model parameters."""

  parser_nn.add_argument(
      '--activation',
      type=str,
      default='relu',
      help='activation function'
  )
  parser_nn.add_argument(
      '--dropout',
      type=float,
      default=0.0,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--ds_filters',
      type=str,
      default='128, 64, 64, 64, 128, 128',
      help='Number of filters in every residual block'
  )
  parser_nn.add_argument(
      '--ds_repeat',
      type=str,
      default='1, 1, 1, 1, 1, 1',
      help='Number of repeating conv blocks inside of residual block'
  )
  parser_nn.add_argument(
      '--ds_filter_separable',
      type=str,
      default='1, 1, 1, 1, 1, 1',
      help='If 1 - use separable filter: depthwise conv in time and 1x1 conv '
      'If 0 - use conv filter in time'
  )
  parser_nn.add_argument(
      '--ds_residual',
      type=str,
      default='0, 1, 1, 1, 0, 0',
      help='Apply/not apply residual connection in residual block'
  )
  parser_nn.add_argument(
      '--ds_padding',
      type=str,
      default="'same', 'same', 'same', 'same', 'same', 'same'",
      help='padding can be same or causal, causal should be used for streaming'
  )
  parser_nn.add_argument(
      '--ds_kernel_size',
      type=str,
      default='11, 13, 15, 17, 29, 1',
      help='Kernel size of DepthwiseConv1D in time dim for every residual block'
  )
  parser_nn.add_argument(
      '--ds_stride',
      type=str,
      default='1, 1, 1, 1, 1, 1',
      help='stride value in time dim of DepthwiseConv1D for residual block'
  )
  parser_nn.add_argument(
      '--ds_dilation',
      type=str,
      default='1, 1, 1, 1, 2, 1',
      help='dilation value of DepthwiseConv1D for every residual block'
  )
  parser_nn.add_argument(
      '--ds_pool',
      type=str,
      default='1, 1, 1, 1, 1, 1',
      help='Apply pooling after every residual block: pooling size'
  )
  parser_nn.add_argument(
      '--ds_max_pool',
      type=int,
      default=0,
      help='Pooling type: 0 - average pooling; 1 - max pooling'
  )
  parser_nn.add_argument(
      '--ds_scale',
      type=int,
      default=1,
      help='apply scaling in batch normalization layer'
  )


def resnet_block(inputs,
                 repeat,
                 kernel_size,
                 filters,
                 dilation,
                 stride,
                 filter_separable,
                 residual=False,
                 padding='same',
                 dropout=0.0,
                 activation='relu',
                 scale=True):
  """Residual block.

  It is based on paper
  Jasper: An End-to-End Convolutional Neural Acoustic Model
  https://arxiv.org/pdf/1904.03288.pdf

  Args:
    inputs: input tensor
    repeat: number of repeating DepthwiseConv1D and Conv1D block
    kernel_size: kernel size of DepthwiseConv1D in time dim
    filters: number of filters in DepthwiseConv1D and Conv1D
    dilation: dilation in time dim for DepthwiseConv1D
    stride: stride in time dim for DepthwiseConv1D
    filter_separable: use separable conv or standard conv
    residual: if True residual connection is added
    padding: can be 'same' or 'causal'
    dropout: dropout value
    activation: type of activation function (string)
    scale: apply scaling in batchnormalization layer

  Returns:
    output tensor

  Raises:
    ValueError: if padding has invalid value
  """
  if residual and (padding not in ('same', 'causal')):
    raise ValueError('padding should be same or causal')

  net = inputs
  for _ in range(repeat-1):
    if filter_separable:  # apply separable conv
      if kernel_size > 0:
        # DepthwiseConv1D
        net = stream.Stream(
            cell=tf.keras.layers.DepthwiseConv2D(
                kernel_size=(kernel_size, 1),
                strides=(stride, 1),
                padding='valid',
                dilation_rate=(dilation, 1),
                use_bias=False),
            pad_time_dim=padding)(
                net)

      # Conv1D 1x1 - streamable by default
      net = tf.keras.layers.Conv2D(
          filters=filters, kernel_size=1, use_bias=False, padding='valid')(
              net)
    else:  # apply 1D conv in time
      net = stream.Stream(
          cell=tf.keras.layers.Conv2D(
              filters=filters,
              kernel_size=(kernel_size, 1),
              dilation_rate=(dilation, 1),
              padding='valid',
              activation='linear',
              use_bias=False),
          pad_time_dim=padding)(
              net)

    net = tf.keras.layers.BatchNormalization(scale=scale)(net)
    net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(rate=dropout)(net)

  if filter_separable:  # apply separable conv
    if kernel_size > 0:
      # DepthwiseConv1D
      net = stream.Stream(
          cell=tf.keras.layers.DepthwiseConv2D(
              kernel_size=(kernel_size, 1),
              strides=(stride, 1),
              padding='valid',
              dilation_rate=(dilation, 1),
              use_bias=False),
          pad_time_dim=padding)(
              net)

    # Conv1D 1x1 - streamable by default
    net = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=1, use_bias=False, padding='valid')(
            net)
  else:  # apply 1D conv in time
    net = stream.Stream(
        cell=tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, 1),
            dilation_rate=(dilation, 1),
            padding='valid',
            activation='linear',
            use_bias=False),
        pad_time_dim=padding)(
            net)

  net = tf.keras.layers.BatchNormalization(scale=scale)(net)

  if residual:
    # Conv1D 1x1 - streamable by default
    net_res = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=1, use_bias=False, padding='valid')(
            inputs)
    net_res = tf.keras.layers.BatchNormalization(scale=scale)(net_res)

    net = tf.keras.layers.Add()([net, net_res])

  net = tf.keras.layers.Activation(activation)(net)
  net = tf.keras.layers.Dropout(rate=dropout)(net)
  return net


def model(flags):
  """MatchboxNet model.

  It is based on paper
  MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network
  Architecture for Speech Commands Recognition
  https://arxiv.org/pdf/2004.08531.pdf

  Args:
    flags: data/model parameters

  Returns:
    Keras model for training

  Raises:
    ValueError: if any of input list has different length from any other;
    or if padding is not supported
  """

  ds_filters = utils.parse(flags.ds_filters)
  ds_repeat = utils.parse(flags.ds_repeat)
  ds_kernel_size = utils.parse(flags.ds_kernel_size)
  ds_stride = utils.parse(flags.ds_stride)
  ds_dilation = utils.parse(flags.ds_dilation)
  ds_residual = utils.parse(flags.ds_residual)
  ds_pool = utils.parse(flags.ds_pool)
  ds_padding = utils.parse(flags.ds_padding)
  ds_filter_separable = utils.parse(flags.ds_filter_separable)

  for l in (ds_repeat, ds_kernel_size, ds_stride, ds_dilation, ds_residual,
            ds_pool, ds_padding, ds_filter_separable):
    if len(ds_filters) != len(l):
      raise ValueError('all input lists have to be the same length')

  input_audio = tf.keras.layers.Input(
      shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
      batch_size=flags.batch_size)
  net = input_audio

  if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
    net = speech_features.SpeechFeatures(
        speech_features.SpeechFeatures.get_params(flags))(
            net)

  # make it [batch, time, 1, feature]
  net = tf.keras.backend.expand_dims(net, axis=2)

  # encoder
  for filters, repeat, ksize, stride, sep, dilation, res, pool, pad in zip(
      ds_filters, ds_repeat, ds_kernel_size, ds_stride, ds_filter_separable,
      ds_dilation, ds_residual, ds_pool, ds_padding):
    net = resnet_block(net, repeat, ksize, filters, dilation, stride,
                       sep, res, pad, flags.dropout,
                       flags.activation, flags.ds_scale)
    if pool > 1:
      if flags.ds_max_pool:
        net = tf.keras.layers.MaxPooling2D(
            pool_size=(pool, 1),
            strides=(pool, 1)
            )(net)
      else:
        net = tf.keras.layers.AveragePooling2D(
            pool_size=(pool, 1),
            strides=(pool, 1)
            )(net)

  # decoder
  net = stream.Stream(cell=tf.keras.layers.GlobalAveragePooling2D())(net)

  net = tf.keras.layers.Flatten()(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)

  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
