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

"""Model based on Broadcasted Residual Learning for Efficient Keyword Spotting.

It is not official model version based on paper:
https://arxiv.org/pdf/2106.04140.pdf
"""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers import sub_spectral_normalization
from kws_streaming.layers.compat import tf
import kws_streaming.models.model_utils as utils


def model_parameters(parser_nn):
  """Model parameters."""

  parser_nn.add_argument(
      '--dropouts',
      type=str,
      default='0.5, 0.5, 0.5, 0.5',
      help='List of dropouts for BC-ResBlock.',
  )
  parser_nn.add_argument(
      '--filters',
      type=str,
      default='8, 12, 16, 20',
      help='Number of filters in every BC-ResBlock.'
  )
  parser_nn.add_argument(
      '--blocks_n',
      type=str,
      default='2, 2, 4, 4',
      help='Number of BC-ResBlocks.'
  )
  parser_nn.add_argument(
      '--strides',
      type=str,
      default='(1,1),(1,2),(1,2),(1,1)',
      help='Strides applied in every TransitionBlock.',
  )
  parser_nn.add_argument(
      '--dilations',
      type=str,
      default='(1,1), (2,1), (3,1), (3,1)',
      help='Dilations applied in every BC-ResBlocks.',
  )
  parser_nn.add_argument(
      '--paddings',
      type=str,
      default='same',
      help='Paddings in time applied in every BC-ResBlocks.',
  )
  parser_nn.add_argument(
      '--first_filters',
      type=int,
      default=16,
      help='Number of filters in the first conv layer.',
  )
  parser_nn.add_argument(
      '--last_filters',
      type=int,
      default=32,
      help='Number of filters in the last conv layer.',
  )
  parser_nn.add_argument(
      '--sub_groups',
      type=int,
      default=5,
      help='Number of groups for SubSpectralNormalization.',
  )


class TransitionBlock(tf.keras.layers.Layer):
  """TransitionBlock.

  It is based on paper:
    Broadcasted Residual Learning for Efficient Keyword Spotting
    https://arxiv.org/pdf/2106.04140.pdf

  Attributes:
    filters: number of filters/channels in conv layer
    dilation: dilation of conv layer
    stride: stride of conv layer
    padding: padding of conv layer (can be same or causal only)
    dropout: dropout rate
    use_one_step: this parameter will be used for streaming only
    sub_groups: number of groups for SubSpectralNormalization
    **kwargs: additional layer arguments
  """

  def __init__(self,
               filters=8,
               dilation=1,
               stride=1,
               padding='same',
               dropout=0.5,
               use_one_step=True,
               sub_groups=5,
               **kwargs):
    super(TransitionBlock, self).__init__(**kwargs)
    self.filters = filters
    self.dilation = dilation
    self.stride = stride
    self.padding = padding
    self.dropout = dropout
    self.use_one_step = use_one_step
    self.sub_groups = sub_groups

    self.frequency_dw_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(1, 3),
        strides=self.stride,
        dilation_rate=self.dilation,
        padding='same',
        use_bias=False)
    if self.padding == 'same':
      self.temporal_dw_conv = tf.keras.layers.DepthwiseConv2D(
          kernel_size=(3, 1),
          strides=self.stride,
          dilation_rate=self.dilation,
          padding='same',
          use_bias=False)
    else:
      self.temporal_dw_conv = stream.Stream(
          cell=tf.keras.layers.DepthwiseConv2D(
              kernel_size=(3, 1),
              strides=self.stride,
              dilation_rate=self.dilation,
              padding='valid',
              use_bias=False),
          use_one_step=use_one_step,
          pad_time_dim=self.padding,
          pad_freq_dim='same')
    self.batch_norm1 = tf.keras.layers.BatchNormalization()
    self.batch_norm2 = tf.keras.layers.BatchNormalization()
    self.conv1x1_1 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=1,
        padding='valid',
        use_bias=False)
    self.conv1x1_2 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=1,
        padding='valid',
        use_bias=False)

  def call(self, inputs):

    # expected input: [N, Time, Frequency, Channels]
    if inputs.shape.rank != 4:
      raise ValueError('input_shape.rank:%d must be 4' % inputs.shape.rank)

    net = inputs
    net = self.conv1x1_1(net)
    net = self.batch_norm1(net)
    net = tf.keras.activations.relu(net)
    net = self.frequency_dw_conv(net)
    net = sub_spectral_normalization.SubSpectralNormalization(self.sub_groups)(
        net)

    residual = net
    net = tf.keras.backend.mean(net, axis=2, keepdims=True)
    net = self.temporal_dw_conv(net)
    net = self.batch_norm2(net)
    net = tf.keras.activations.swish(net)
    net = self.conv1x1_2(net)
    net = tf.keras.layers.SpatialDropout2D(rate=self.dropout)(net)

    net = net + residual
    net = tf.keras.activations.relu(net)
    return net

  def get_config(self):
    config = {
        'filters': self.filters,
        'dilation': self.dilation,
        'stride': self.stride,
        'padding': self.padding,
        'dropout': self.dropout,
        'use_one_step': self.use_one_step,
        'sub_groups': self.sub_groups,
        }
    base_config = super(TransitionBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return self.temporal_dw_conv.get_input_state()

  def get_output_state(self):
    return self.temporal_dw_conv.get_output_state()


class NormalBlock(tf.keras.layers.Layer):
  """NormalBlock.

  It is based on paper:
    Broadcasted Residual Learning for Efficient Keyword Spotting
    https://arxiv.org/pdf/2106.04140.pdf

  Attributes:
    filters: number of filters/channels in conv layer
    dilation: dilation of conv layer
    stride: stride of conv layer
    padding: padding of conv layer (can be same or causal only)
    dropout: dropout rate
    use_one_step: this parameter will be used for streaming only
    sub_groups: number of groups for SubSpectralNormalization
    **kwargs: additional layer arguments
  """

  def __init__(
      self,
      filters,
      dilation=1,
      stride=1,
      padding='same',
      dropout=0.5,
      use_one_step=True,
      sub_groups=5,
      **kwargs):
    super(NormalBlock, self).__init__(**kwargs)
    self.filters = filters
    self.dilation = dilation
    self.stride = stride
    self.padding = padding
    self.dropout = dropout
    self.use_one_step = use_one_step
    self.sub_groups = sub_groups

    self.frequency_dw_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(1, 3),
        strides=self.stride,
        dilation_rate=self.dilation,
        padding=self.padding,
        use_bias=False)
    if self.padding == 'same':
      self.temporal_dw_conv = tf.keras.layers.DepthwiseConv2D(
          kernel_size=(3, 1),
          strides=self.stride,
          dilation_rate=self.dilation,
          padding='same',
          use_bias=False)
    else:
      self.temporal_dw_conv = stream.Stream(
          cell=tf.keras.layers.DepthwiseConv2D(
              kernel_size=(3, 1),
              strides=self.stride,
              dilation_rate=self.dilation,
              padding='valid',
              use_bias=False),
          use_one_step=use_one_step,
          pad_time_dim=self.padding,
          pad_freq_dim='same')
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.conv1x1 = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=1,
        padding=self.padding,
        use_bias=False)

  def call(self, inputs):

    # expected input: [N, Time, Frequency, Channels]
    if inputs.shape.rank != 4:
      raise ValueError('input_shape.rank:%d must be 4' % inputs.shape.rank)

    identity = inputs
    net = inputs
    net = self.frequency_dw_conv(net)
    net = sub_spectral_normalization.SubSpectralNormalization(self.sub_groups)(
        net)

    residual = net
    net = tf.keras.backend.mean(net, axis=2, keepdims=True)
    net = self.temporal_dw_conv(net)
    net = self.batch_norm(net)
    net = tf.keras.activations.swish(net)
    net = self.conv1x1(net)
    net = tf.keras.layers.SpatialDropout2D(rate=self.dropout)(net)

    net = net + identity + residual
    net = tf.keras.activations.relu(net)
    return net

  def get_config(self):
    config = {
        'filters': self.filters,
        'dilation': self.dilation,
        'stride': self.stride,
        'padding': self.padding,
        'dropout': self.dropout,
        'use_one_step': self.use_one_step,
        'sub_groups': self.sub_groups,
        }
    base_config = super(NormalBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return self.temporal_dw_conv.get_input_state()

  def get_output_state(self):
    return self.temporal_dw_conv.get_output_state()


def model(flags):
  """BC-ResNet model.

  It is based on paper
  Broadcasted Residual Learning for Efficient Keyword Spotting
  https://arxiv.org/pdf/2106.04140.pdf

  Args:
    flags: data/model parameters

  Returns:
    Keras model for training

  Raises:
    ValueError: if any of input list has different length from any other;
    or if padding is not supported
  """

  dropouts = utils.parse(flags.dropouts)
  filters = utils.parse(flags.filters)
  blocks_n = utils.parse(flags.blocks_n)
  strides = utils.parse(flags.strides)
  dilations = utils.parse(flags.dilations)

  for l in (dropouts, filters, strides, dilations):
    if len(blocks_n) != len(l):
      raise ValueError('all input lists have to be the same length '
                       'but get %s and %s ' % (blocks_n, l))

  input_audio = tf.keras.layers.Input(
      shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
      batch_size=flags.batch_size)
  net = input_audio

  if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
    net = speech_features.SpeechFeatures(
        speech_features.SpeechFeatures.get_params(flags))(
            net)

  # make it [batch, time, feature, 1]
  net = tf.keras.backend.expand_dims(net, axis=3)

  if flags.paddings == 'same':
    net = tf.keras.layers.Conv2D(
        filters=flags.first_filters,
        kernel_size=5,
        strides=(1, 2),
        padding='same')(net)
  else:
    net = stream.Stream(
        cell=tf.keras.layers.Conv2D(
            filters=flags.first_filters,
            kernel_size=5,
            strides=(1, 2),
            padding='valid'),
        use_one_step=True,
        pad_time_dim=flags.paddings,
        pad_freq_dim='same')(
            net)

  for n, n_filters, dilation, stride, dropout in zip(blocks_n, filters,
                                                     dilations, strides,
                                                     dropouts):
    net = TransitionBlock(
        n_filters,
        dilation,
        stride,
        flags.paddings,
        dropout,
        sub_groups=flags.sub_groups)(
            net)
    for _ in range(n):
      net = NormalBlock(
          n_filters,
          dilation,
          1,
          flags.paddings,
          dropout,
          sub_groups=flags.sub_groups)(
              net)

  if flags.paddings == 'same':
    net = tf.keras.layers.DepthwiseConv2D(
        kernel_size=5,
        padding='same')(net)
  else:
    net = stream.Stream(
        cell=tf.keras.layers.DepthwiseConv2D(
            kernel_size=5,
            padding='valid'),
        use_one_step=True,
        pad_time_dim=flags.paddings,
        pad_freq_dim='same')(
            net)

  # average out frequency dim
  net = tf.keras.backend.mean(net, axis=2, keepdims=True)

  net = tf.keras.layers.Conv2D(
      filters=flags.last_filters, kernel_size=1, use_bias=False)(
          net)

  # average out time dim
  if flags.paddings == 'same':
    net = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(net)
  else:
    net = stream.Stream(
        cell=tf.keras.layers.GlobalAveragePooling2D(keepdims=True))(
            net)

  net = tf.keras.layers.Conv2D(
      filters=flags.label_count, kernel_size=1, use_bias=False)(
          net)
  # 1 and 2 dims are equal to 1
  net = tf.squeeze(net, [1, 2])

  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
