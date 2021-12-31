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

"""Implementation of RAFT."""

# pylint:skip-file

import tensorflow as tf
import tensorflow_addons as tfa


def create_extractor_Conv2d(c_in, c_out, k_size, stride=1):
  kernel_scale = 2.0
  if isinstance(k_size, list) or isinstance(k_size, tuple):
    bias_scale = c_out / (3.0 * c_in * k_size[0] * k_size[1])
  else:
    bias_scale = c_out / (3.0 * c_in * k_size * k_size)
  return tf.keras.layers.Conv2D(
      filters=c_out,
      kernel_size=k_size,
      strides=stride,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          distribution='normal', scale=kernel_scale, mode='fan_out'),
      bias_initializer=tf.keras.initializers.VarianceScaling(
          distribution='uniform', scale=bias_scale, mode='fan_in'))


class ResidualBlock(tf.keras.layers.Layer):

  def __init__(self, in_planes, planes, norm_fn='batch', stride=1, **kwargs):
    super(ResidualBlock, self).__init__(**kwargs)
    self.conv1 = create_extractor_Conv2d(
        c_in=in_planes, c_out=planes, k_size=3, stride=stride)
    self.conv2 = create_extractor_Conv2d(c_in=planes, c_out=planes, k_size=3)
    self.relu = tf.keras.layers.ReLU()

    num_groups = planes // 8

    beta_initializer = 'zeros'
    gamma_initializer = 'ones'

    if norm_fn == 'group':
      self.norm1 = tfa.layers.GroupNormalization(
          groups=num_groups,
          axis=-1,
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm2 = tfa.layers.GroupNormalization(
          groups=num_groups,
          axis=-1,
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      if stride != 1:
        self.norm3 = tfa.layers.GroupNormalization(
            groups=num_groups,
            axis=-1,
            epsilon=1e-5,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer)
    elif norm_fn == 'batch':
      self.norm1 = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm2 = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      if stride != 1:
        self.norm3 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer)
    elif norm_fn == 'instance':
      self.norm1 = tfa.layers.InstanceNormalization(
          axis=3,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm2 = tfa.layers.InstanceNormalization(
          axis=3,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      if stride != 1:
        self.norm3 = tfa.layers.InstanceNormalization(
            axis=3,
            epsilon=1e-5,
            center=False,
            scale=False,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer)
    elif norm_fn == 'none':
      self.norm1 = tf.keras.Sequential()
      self.norm2 = tf.keras.Sequential()
      if stride != 1:
        self.norm3 = tf.keras.Sequential()
    else:
      raise Exception('norm_fn %s not implemented' % norm_fn)

    if stride == 1:
      self.downsample = tf.keras.Sequential()
    else:
      conv = create_extractor_Conv2d(
          c_in=in_planes, c_out=planes, k_size=1, stride=stride)
      self.downsample = tf.keras.Sequential(layers=[conv, self.norm3])

  def call(self, x, training=True):
    y = x
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
    y = tf.pad(y, paddings)
    y = self.relu(self.norm1(self.conv1(y), training=training))
    y = tf.pad(y, paddings)
    y = self.relu(self.norm2(self.conv2(y), training=training))

    x = self.downsample(x, training=training)
    return self.relu(x + y)


class BottleneckBlock(tf.keras.layers.Layer):

  def __init__(self, in_planes, planes, norm_fn='group', stride=1, **kwargs):
    super(BottleneckBlock, self).__init__(**kwargs)

    hidden_planes = planes // 4

    self.conv1 = create_extractor_Conv2d(
        c_in=in_planes, c_out=hidden_planes, k_size=1)
    self.conv2 = create_extractor_Conv2d(
        c_in=hidden_planes, c_out=hidden_planes, k_size=3, stride=stride)
    self.conv3 = create_extractor_Conv2d(
        c_in=hidden_planes, c_out=planes, k_size=1)
    self.relu = tf.keras.layers.ReLU()

    num_groups = planes // 8

    beta_initializer = 'zeros'
    gamma_initializer = 'ones'

    if norm_fn == 'group':
      self.norm1 = tfa.layers.GroupNormalization(
          groups=num_groups,
          axis=-1,
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm2 = tfa.layers.GroupNormalization(
          groups=num_groups,
          axis=-1,
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm3 = tfa.layers.GroupNormalization(
          groups=num_groups,
          axis=-1,
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      if stride != 1:
        self.norm4 = tfa.layers.GroupNormalization(
            groups=num_groups,
            axis=-1,
            epsilon=1e-5,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer)
    elif norm_fn == 'batch':
      self.norm1 = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm2 = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm3 = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      if stride != 1:
        self.norm4 = tf.keras.layers.BatchNormalization(
            epsilon=1e-5,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer)
    elif norm_fn == 'instance':
      self.norm1 = tfa.layers.InstanceNormalization(
          axis=3,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm2 = tfa.layers.InstanceNormalization(
          axis=3,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm3 = tfa.layers.InstanceNormalization(
          axis=3,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      if stride != 1:
        self.norm4 = tfa.layers.InstanceNormalization(
            axis=3,
            epsilon=1e-5,
            center=False,
            scale=False,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer)
    elif norm_fn == 'none':
      self.norm1 = tf.keras.Sequential()
      self.norm2 = tf.keras.Sequential()
      self.norm3 = tf.keras.Sequential()
      if stride != 1:
        self.norm4 = tf.keras.Sequential()
    else:
      raise Exception('norm_fn %s not implemented' % norm_fn)

    if stride == 1:
      self.downsample = tf.keras.Sequential()
    else:
      conv = create_extractor_Conv2d(
          c_in=in_planes, c_out=planes, k_size=1, stride=stride)
      self.downsample = tf.keras.Sequential(layers=[conv, self.norm4])

  def call(self, x, training=True):
    y = x
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
    y = self.relu(self.norm1(self.conv1(y), training=training))
    y = tf.pad(y, paddings)
    y = self.relu(self.norm2(self.conv2(y), training=training))
    y = self.relu(self.norm3(self.conv3(y), training=training))

    x = self.downsample(x, training=training)
    return self.relu(x + y)


class BasicEncoder(tf.keras.layers.Layer):

  def __init__(self, output_dim=128, norm_fn='none', dropout=0.0, **kwargs):
    super(BasicEncoder, self).__init__(**kwargs)

    self.norm_fn = norm_fn

    beta_initializer = 'zeros'
    gamma_initializer = 'ones'

    if norm_fn == 'group':
      self.norm1 = tfa.layers.GroupNormalization(
          groups=8,
          axis=-1,
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    elif norm_fn == 'batch':
      self.norm1 = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    elif norm_fn == 'instance':
      self.norm1 = tfa.layers.InstanceNormalization(
          axis=-1,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    elif norm_fn == 'none':
      self.norm1 = tf.keras.Sequential()
    else:
      raise Exception('norm_fn %s not implemented' % norm_fn)

    self.conv1 = create_extractor_Conv2d(c_in=3, c_out=64, k_size=7, stride=2)
    self.relu1 = tf.keras.layers.ReLU()

    self.in_planes = 64
    self.layer1 = self._make_layer(64, stride=1)
    self.layer2 = self._make_layer(96, stride=2)
    self.layer3 = self._make_layer(128, stride=2)

    self.conv2 = create_extractor_Conv2d(c_in=128, c_out=output_dim, k_size=1)

    if dropout > 0:
      self.dropout = tf.keras.layers.Dropout(rate=dropout)
    else:
      self.dropout = tf.keras.Sequential()

    # initialize according to RAFT

  def _make_layer(self, dim, stride=1):
    layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
    layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
    layers = [layer1, layer2]

    self.in_planes = dim
    return tf.keras.Sequential(layers=layers)

  def call(self, x, training=True):

    paddings = [[0, 0], [3, 3], [3, 3], [0, 0]]
    x = tf.pad(x, paddings=paddings)
    x = self.conv1(x)
    x = self.norm1(x, training=training)
    x = self.relu1(x)

    x = self.layer1(x, training=training)
    x = self.layer2(x, training=training)
    x = self.layer3(x, training=training)

    x = self.conv2(x)

    x = self.dropout(x, training=training)

    return x


class SmallEncoder(tf.keras.layers.Layer):

  def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, **kwargs):
    super(SmallEncoder, self).__init__(**kwargs)

    self.norm_fn = norm_fn

    beta_initializer = 'zeros'
    gamma_initializer = 'ones'

    if norm_fn == 'group':
      self.norm1 = tfa.layers.GroupNormalization(
          groups=8,
          axis=-1,
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    elif norm_fn == 'batch':
      self.norm1 = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    elif norm_fn == 'instance':
      self.norm1 = tfa.layers.InstanceNormalization(
          axis=-1,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    elif norm_fn == 'none':
      self.norm1 = tf.keras.Sequential()
    else:
      raise Exception('norm_fn %s not implemented' % norm_fn)

    self.conv1 = create_extractor_Conv2d(c_in=3, c_out=32, k_size=7, stride=2)
    self.relu1 = tf.keras.layers.ReLU()

    self.in_planes = 32
    self.layer1 = self._make_layer(32, stride=1)
    self.layer2 = self._make_layer(64, stride=2)
    self.layer3 = self._make_layer(96, stride=2)

    self.conv2 = create_extractor_Conv2d(c_in=96, c_out=output_dim, k_size=1)

  def _make_layer(self, dim, stride=1):
    layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
    layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
    layers = [layer1, layer2]

    self.in_planes = dim
    return tf.keras.Sequential(layers=layers)

  def call(self, x, training=True):

    paddings = [[0, 0], [3, 3], [3, 3], [0, 0]]
    x = tf.pad(x, paddings=paddings)
    x = self.conv1(x)
    x = self.norm1(x, training=training)
    x = self.relu1(x)

    x = self.layer1(x, training=training)
    x = self.layer2(x, training=training)
    x = self.layer3(x, training=training)

    x = self.conv2(x)

    return x
