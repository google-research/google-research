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

# Lint as python3
"""Modular normalization type."""

from typing import Optional, Sequence, Text, Union

import tensorflow.compat.v2 as tf


class BasicBlock(tf.keras.Model):
  """Basic ResNet block."""

  EXPANSION = 1

  def __init__(self,
               channels,
               stride,
               use_projection,
               norm,
               name = None):

    super(BasicBlock, self).__init__(name=name)

    self._channels = channels
    self._stride = stride
    self._use_projection = use_projection
    if self._use_projection:
      self._proj_conv = tf.keras.layers.Conv2D(
          filters=channels,
          kernel_size=1,
          strides=stride,
          padding='same',
          use_bias=False,
          name='shortcut_conv')
      self._proj_norm = norm(name='shortcut_' + 'bn')

    self._layers = []
    conv_0 = tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=3,
        strides=stride,
        padding='same',
        use_bias=False,
        name='conv_0')
    self._layers.append([
        conv_0,
        norm(name='bn' + '_0'),
    ])

    conv_1 = tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=3,
        padding='same',
        use_bias=False,
        name='conv_1')

    self._layers.append([
        conv_1,
        norm(name='bn' + '_1', gamma_initializer='zeros'),
    ])

  def __call__(self, x, **norm_kwargs):
    if self._use_projection:
      shortcut = self._proj_conv(x)
      shortcut = self._proj_norm(shortcut, **norm_kwargs)
    else:
      shortcut = x

    for i, [conv_layer, norm_layer] in enumerate(self._layers):
      x = conv_layer(x)
      x = norm_layer(x, **norm_kwargs)
      if i == 0:
        x = tf.nn.relu(x)  # Don't apply relu on last layer

    return tf.nn.relu(x + shortcut)


class BottleNeckBlockV1(tf.keras.Model):
  """Bottleneck Block for a ResNet implementation."""

  EXPANSION = 4

  def __init__(self,
               channels,
               stride,
               use_projection,
               norm,
               name = None):
    super(BottleNeckBlockV1, self).__init__(name=name)
    self._channels = channels
    self._stride = stride
    self._use_projection = use_projection

    if self._use_projection:
      self._proj_conv = tf.keras.layers.Conv2D(
          filters=channels,
          kernel_size=1,
          strides=stride,
          padding='same',
          use_bias=False,
          name='shortcut_conv')
      self._proj_norm = norm(name='shortcut_' + 'bn')

    self._layers = []
    conv_0 = tf.keras.layers.Conv2D(
        filters=channels // 4,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        name='conv_0')
    self._layers.append([
        conv_0,
        norm(name='bn' + '_0'),
    ])

    conv_1 = tf.keras.layers.Conv2D(
        filters=channels // 4,
        kernel_size=3,
        strides=stride,
        padding='same',
        use_bias=False,
        name='conv_1')
    self._layers.append([
        conv_1,
        norm(name='bn' + '_1'),
    ])

    conv_2 = tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        name='conv_2')
    self._layers.append([
        conv_2,
        norm(name='bn' + '_2', gamma_initializer='zeros'),
    ])

  def __call__(self, x, **norm_kwargs):
    if self._use_projection:
      shortcut = self._proj_conv(x)
      shortcut = self._proj_norm(shortcut, **norm_kwargs)
    else:
      shortcut = x

    for i, [conv_layer, norm_layer] in enumerate(self._layers):
      x = conv_layer(x)
      x = norm_layer(x, **norm_kwargs)
      x = tf.nn.relu(x) if i < 2 else x  # Don't apply relu on last layer
      # x = tf.nn.relu(x)

    return tf.nn.relu(x + shortcut)
