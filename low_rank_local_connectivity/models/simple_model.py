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

"""Simple model for image classification.

The model is multiple
conv/locally_connected/wide_conv/low_rank_locally_connected layers followed
by a fully connected layer. Changes to the model architecture can be made by
modifying simple_model_config.py file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import tensorflow.compat.v1 as tf

from low_rank_local_connectivity import layers
from low_rank_local_connectivity import utils

MOMENTUM = 0.9
EPS = 1e-5


class SimpleNetwork(tf.keras.Model):
  """Locally Connected Network."""

  def __init__(self, config, variable_scope='simple_network'):
    super(SimpleNetwork, self).__init__()
    self.variable_scope = variable_scope
    self.config = copy.deepcopy(config)
    filters_list = self.config.num_filters_list
    depth = len(filters_list)
    self.pass_is_training_list = []
    self.layers_list = []

    if self.config.num_channels < 1:
      raise ValueError('num_channels should be > 0')

    input_channels = self.config.num_channels
    if self.config.coord_conv:
      # Add two coordinate conv channels.
      input_channels = input_channels + 2

    if len(self.config.layer_types) < depth:
      self.config.layer_types.extend(
          ['conv2d'] * (depth - len(self.config.layer_types)))
    chin = input_channels
    for i, (kernel_size, num_filters, strides, layer_type) in enumerate(zip(
        self.config.kernel_size_list,
        filters_list,
        self.config.strides_list,
        self.config.layer_types)):
      padding = 'valid'
      if layer_type == 'conv2d':
        chout = num_filters
        layer = tf.keras.layers.Conv2D(
            filters=chout,
            kernel_size=kernel_size,
            strides=(strides, strides),
            padding=padding,
            activation=None,
            use_bias=not self.config.batch_norm,
            kernel_initializer=self.config.kernel_initializer,
            name=os.path.join(self.variable_scope, 'layer%d' %i, layer_type))
      elif layer_type == 'wide_conv2d':
        # Conv. layer with equivalent params to low rank locally connected.
        if self.config.rank < 1:
          raise ValueError('rank should be > 0 for %s layer.' % layer_type)
        chout = int((self.config.rank * chin + num_filters) / float(
            chin + num_filters) * num_filters)
        layer = tf.keras.layers.Conv2D(
            filters=chout if i < (depth-1)
            else int(num_filters * self.config.rank),
            kernel_size=kernel_size, strides=(strides, strides),
            padding=padding,
            activation=None,
            use_bias=not self.config.batch_norm,
            kernel_initializer=self.config.kernel_initializer,
            name=os.path.join(self.variable_scope, 'layer%d' %i, layer_type))

      elif layer_type == 'locally_connected2d':
        # Full locally connected layer.
        chout = num_filters
        layer = tf.keras.layers.LocallyConnected2D(
            filters=chout,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides),
            padding=padding,
            activation=None,
            use_bias=True,  # not self.config.batch_norm,
            name=os.path.join(self.variable_scope, 'layer%d' %i, layer_type),
            kernel_initializer=self.config.kernel_initializer)
      elif layer_type == 'low_rank_locally_connected2d':
        if self.config.rank < 1:
          raise ValueError('rank should be > 0 for %s layer.' % layer_type)
        chout = num_filters
        layer = layers.LowRankLocallyConnected2D(
            filters=chout,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides),
            padding=padding,
            activation=None,
            use_bias=not self.config.batch_norm,
            name=os.path.join(self.variable_scope, 'layer%d' %i, layer_type),
            kernel_initializer=self.config.kernel_initializer,
            combining_weights_initializer=(
                self.config.combining_weights_initializer),
            spatial_rank=self.config.rank,
            normalize_weights=self.config.normalize_weights,
            input_dependent=config.input_dependent,
            share_row_combining_weights=self.config.share_row_combining_weights,
            share_col_combining_weights=self.config.share_col_combining_weights)
      else:
        raise ValueError('Can not recognize layer %s type.' % layer_type)

      chin = chout

      self.layers_list.append(layer)
      self.pass_is_training_list.append(False)
      if self.config.batch_norm:
        layer = tf.keras.layers.BatchNormalization(
            trainable=True, momentum=MOMENTUM, epsilon=EPS)

        self.layers_list.append(layer)
        self.pass_is_training_list.append(True)

      layer = tf.keras.layers.ReLU()
      self.layers_list.append(layer)
      self.pass_is_training_list.append(False)

    if self.config.global_avg_pooling:
      self.layers_list.append(tf.keras.layers.GlobalAveragePooling2D())
    else:
      self.layers_list.append(tf.keras.layers.Flatten())

    self.pass_is_training_list.append(False)

    self.layers_list.append(tf.keras.layers.Dense(
        units=self.config.num_classes, activation=None, use_bias=True,
        name='logits'))
    self.pass_is_training_list.append(False)

  def __call__(self, images, is_training):
    endpoints = {}
    if self.config.coord_conv:
      # Append position channels.
      net = tf.concat([images, utils.position_channels(images)], axis=3)
    else:
      net = images

    for i, (pass_is_training, layer) in enumerate(
        zip(self.pass_is_training_list, self.layers_list)):
      net = layer(net, training=is_training) if pass_is_training else layer(net)
      endpoints['layer%d' % i] = net
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, layer.updates)
      self.add_update(layer.updates)
    logits = net
    return logits, endpoints
