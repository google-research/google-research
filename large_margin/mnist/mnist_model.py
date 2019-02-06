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

"""Model for MNIST classification.

The model is a two layer convolutional network followed by a fully connected
layer. Changes to the model architecture can be made by modifying
mnist_config.py file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

MOMENTUM = 0.9
EPS = 1e-5


def pool2d_layer(inputs, pool_type, pool_size=2, pool_stride=2):
  """Pooling layer.

  Args:
    inputs: Tensor of size [batch, H, W, channels].
    pool_type: String ("max", or "average"), specifying pooling type.
    pool_size: Integer > 1 pooling size.
    pool_stride: Integer > 1 pooling stride.

  Returns:
    Pooling result.
  """
  if pool_type == "max":
    # Max pooling layer
    return tf.layers.max_pooling2d(
        inputs, pool_size=[pool_size] * 2, strides=pool_stride)

  elif pool_type == "average":
    # Average pooling layer
    return tf.layers.average_pooling2d(
        inputs, pool_size=[pool_size] * 2, strides=pool_stride)


class MNISTNetwork(object):
  """MNIST model. """

  def __init__(self, config):
    self.num_classes = config.num_classes
    self.var_list = []
    self.init_ops = None
    self.regularizer = config.regularizer
    self.activation = config.activation
    self.filter_sizes_conv_layers = config.filter_sizes_conv_layers
    self.num_units_fc_layers = config.num_units_fc_layers
    self.pool_params = config.pool_params
    self.dropout_rate = config.dropout_rate
    self.batch_norm = config.batch_norm

  def __call__(self, images, is_training=False):
    """Builds model."""
    endpoints = {}
    net = images
    reuse = tf.AUTO_REUSE
    for i, filter_size in enumerate(self.filter_sizes_conv_layers):
      layer_suffix = "layer%d" % i
      with tf.variable_scope(
          os.path.join("mnist_network", "conv_" + layer_suffix), reuse=reuse):
        net = tf.layers.conv2d(
            net,
            kernel_size=filter_size[0],
            filters=filter_size[1],
            strides=(1, 1),
            padding="same",
            activation=self.activation,
            kernel_regularizer=self.regularizer,
            use_bias=not self.batch_norm)

        net = self.activation(net)

        if self.pool_params:
          net = pool2d_layer(
              net,
              pool_type=self.pool_params["type"],
              pool_size=self.pool_params["size"],
              pool_stride=self.pool_params["stride"])

        if self.dropout_rate > 0:
          net = tf.layers.dropout(
              net,
              rate=self.dropout_rate,
              training=is_training,
          )
        if self.batch_norm:
          net = tf.layers.batch_normalization(
              net, training=is_training, momentum=MOMENTUM, epsilon=EPS)

      endpoints["conv_" + layer_suffix] = net

    net = tf.layers.flatten(net)

    for i, num_units in enumerate(self.num_units_fc_layers):
      layer_suffix = "layer%d" % i
      with tf.variable_scope(
          os.path.join("mnist_network", "fc_" + layer_suffix), reuse=reuse):
        net = tf.layers.dense(
            net,
            num_units,
            activation=self.activation,
            kernel_regularizer=self.regularizer,
            use_bias=True)

      endpoints["fc_" + layer_suffix] = net

    with tf.variable_scope(
        os.path.join("mnist_network", "output_layer"), reuse=reuse):
      logits = tf.layers.dense(
          net,
          self.num_classes,
          activation=None,
          kernel_regularizer=self.regularizer)
    endpoints["logits"] = net

    return logits, endpoints
