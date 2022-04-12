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

"""Image observation encoder.
"""

import math

from typing import Tuple

import tensorflow as tf


class ConvStack(tf.keras.Model):
  """Conv2D stack for ImageEncoder.

  In DRQ, the Conv2D weights are shared between the Actor and Critic
  ImageEncoder instances.
  """

  def __init__(self, obs_shape):
    """Creates an actor.

    Args:
      obs_shape: Image observation shape, typically (height, width, chans).
    """
    super().__init__()

    # Network ported from here:
    # https://github.com/denisyarats/drq/blob/master/drq.py#L12

    assert len(obs_shape) == 3
    self.obs_shape = obs_shape
    self.num_filters = 32
    self.num_layers = 4

    sqrt2 = math.sqrt(2.0)
    conv_stack = tf.keras.Sequential()
    conv_stack.add(tf.keras.layers.Conv2D(
        filters=self.num_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.Orthogonal(gain=sqrt2)))
    for _ in range(self.num_layers - 1):
      conv_stack.add(tf.keras.layers.Conv2D(
          filters=self.num_filters,
          kernel_size=(3, 3),
          strides=(1, 1),
          activation=tf.keras.activations.relu,
          kernel_initializer=tf.keras.initializers.Orthogonal(gain=sqrt2)))
    conv_stack.add(tf.keras.layers.Flatten())

    inputs = tf.keras.Input(shape=obs_shape)
    outputs = conv_stack(inputs)
    self.output_size = outputs.shape[-1]
    self.conv_stack = tf.keras.Model(inputs=inputs, outputs=outputs)

  @tf.function
  def call(self, obs):
    obs = tf.cast(obs, tf.float32) / 255.0
    return self.conv_stack(obs)


class ImageEncoder(tf.keras.Model):
  """Image observation encoder."""

  def __init__(self,
               conv_stack,
               feature_dim,
               bprop_conv_stack):
    """Creates an actor.

    Args:
      conv_stack: Conv2D stack to use on input.
      feature_dim: Desired output state size.
      bprop_conv_stack: If False, adds tf.stop_gradient to output of conv_stack.
    """
    super().__init__()

    self.conv_stack = conv_stack
    self.feature_dim = feature_dim
    self.bprop_conv_stack = bprop_conv_stack

    # Network ported from here:
    # https://github.com/denisyarats/drq/blob/master/drq.py#L12

    fc_stack = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=self.feature_dim,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0)),
        tf.keras.layers.LayerNormalization(epsilon=1e-5),
        tf.keras.layers.Activation(tf.keras.activations.tanh),
    ])

    inputs = tf.keras.Input(shape=(conv_stack.output_size,))
    outputs = fc_stack(inputs)

    self.fc_stack = tf.keras.Model(inputs=inputs, outputs=outputs)

  @tf.function
  def call(self, obs):
    hidden = self.conv_stack(obs)
    if not self.bprop_conv_stack:
      hidden = tf.stop_gradient(hidden)
    return self.fc_stack(hidden)


class SimpleImageEncoder(tf.keras.Model):
  """Image observation encoder without FC layer."""

  def __init__(self,
               conv_stack,
               # feature_dim: int,
               bprop_conv_stack):
    """Creates an actor.

    Args:
      conv_stack: Conv2D stack to use on input.
      bprop_conv_stack: If False, adds tf.stop_gradient to output of conv_stack.
    """
    super().__init__()

    self.conv_stack = conv_stack
    self.bprop_conv_stack = bprop_conv_stack

  @tf.function
  def call(self, obs):
    hidden = self.conv_stack(obs)
    if not self.bprop_conv_stack:
      hidden = tf.stop_gradient(hidden)
    return hidden


class ImpalaConvLayer(tf.keras.Model):
  """Impala convolutional layer.
  """

  def __init__(self,
               depth,
               dropout_rate=0.0,
               use_batch_norm=False,
               name=None,
               **kwargs):
    super(ImpalaConvLayer, self).__init__(name=name)
    self.conv = tf.keras.layers.Conv2D(depth, 3, padding='SAME')
    self.bn = tf.keras.layers.BatchNormalization()
    self.dropout_rate = dropout_rate
    self.use_batch_norm = use_batch_norm

  def __call__(self, inputs, is_training=True, **kwargs):
    del kwargs
    out = self.conv(inputs)
    if is_training:
      out = tf.nn.dropout(out, rate=self.dropout_rate)
    if self.use_batch_norm:
      out = self.bn(out, training=is_training)
    return out


class ImpalaResidualBlock(tf.keras.Model):
  """Impala resblock.
  """

  def __init__(self, depth, conv_layer=ImpalaConvLayer, name=None, **kwargs):
    super(ImpalaResidualBlock, self).__init__(name=name)
    self.conv1 = conv_layer(depth=depth, name='c1', **kwargs)
    self.conv2 = conv_layer(depth=depth, name='c2', **kwargs)

  def __call__(self, inputs, is_training=True, **kwargs):
    out = tf.nn.relu(inputs)
    out = self.conv1(out, is_training=is_training, **kwargs)
    out = tf.nn.relu(out)
    out = self.conv2(out, is_training=is_training, **kwargs)
    return out + inputs


class ImpalaConvSequence(tf.keras.Model):
  """Impala sequence of layers.
  """

  def __init__(self,
               depth,
               conv_layer=ImpalaConvLayer,
               residual_block=ImpalaResidualBlock,
               name=None,
               **kwargs):
    super(ImpalaConvSequence, self).__init__(name=name)
    self.conv = conv_layer(depth=depth, name='c', **kwargs)
    self.residual1 = residual_block(
        depth=depth, conv_layer=conv_layer, name='r1', **kwargs)
    self.residual2 = residual_block(
        depth=depth, conv_layer=conv_layer, name='r2', **kwargs)

  def __call__(self, inputs, is_training=True, **kwargs):
    out = self.conv(inputs, is_training=is_training, **kwargs)
    out = tf.nn.max_pool2d(out, ksize=3, strides=2, padding='SAME')
    out = self.residual1(out, is_training=is_training, **kwargs)
    out = self.residual2(out, is_training=is_training, **kwargs)
    return out


class ImpalaCNN(tf.keras.Model):
  """Impala encoder.
  """

  def __init__(self,  # pylint: disable=dangerous-default-value
               impala_sequence=ImpalaConvSequence,
               depths=[16, 32, 32],
               name=None,
               **kwargs):
    super(ImpalaCNN, self).__init__(name=name)

    temp_list = []
    for i, d in enumerate(depths):
      temp_list.append(
          impala_sequence(
              depth=d,
              name='impala_conv_seq_' + str(i) + '_' + str(d),
              **kwargs))

    self.conv_section = temp_list  # tf.keras.Sequential(temp_list)
    self.linear1 = tf.keras.layers.Dense(256)
    self.linear2 = tf.keras.layers.Dense(256)

  def __call__(self, inputs, is_training=True, **kwargs):
    out = self.conv_section[0](inputs, is_training=is_training, **kwargs)
    out = self.conv_section[1](out, is_training=is_training, **kwargs)
    out = self.conv_section[2](out, is_training=is_training, **kwargs)
    out = tf.keras.layers.Flatten()(out)
    out = tf.nn.relu(out)
    out = self.linear1(out)
    out = tf.nn.relu(out)
    out = self.linear2(out)
    return out


def make_impala_cnn_network(  # pylint: disable=dangerous-default-value
    conv_layer=ImpalaConvLayer,
    depths=[16, 32, 32],
    use_batch_norm=False,
    dropout_rate=0.0):
  return ImpalaCNN(
      depths=depths,
      name='impala',
      use_batch_norm=use_batch_norm,
      dropout_rate=dropout_rate,
      conv_layer=conv_layer)
