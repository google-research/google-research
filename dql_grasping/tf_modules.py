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

"""Reused modules for building actors/critics for grasping task.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin.tf

slim = tf.contrib.slim

# Register normalization functions as configurables.
gin.config.external_configurable(tf.contrib.layers.layer_norm,
                                 module='tf.contrib.layers')
gin.config.external_configurable(tf.contrib.layers.batch_norm,
                                 module='tf.contrib.layers')


@gin.configurable
def argscope(is_training=None, normalizer_fn=slim.layer_norm):
  """Default TF argscope used for convnet-based grasping models.

  Args:
    is_training: Whether this argscope is for training or inference.
    normalizer_fn: Which conv/fc normalizer to use.
  Returns:
    Dictionary of argument overrides.
  """
  with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn):
      with slim.arg_scope(
          [slim.conv2d, slim.max_pool2d], stride=2, padding='VALID') as scope:
        return scope


def tile_to_match_context(net, context):
  """Tiles net along a new axis=1 to match context.

  Repeats minibatch elements of `net` tensor to match multiple corresponding
  minibatch elements from `context`.
  Args:
    net: Tensor of shape [num_batch_net, ....].
    context: Tensor of shape [num_batch_net, num_examples, context_size].
  Returns:
    Tensor of shape [num_batch_net, num_examples, ...], where each minibatch
    element of net has been tiled M times where M = num_batch_context /
    num_batch_net.
  """
  with tf.name_scope('tile_to_context'):
    num_samples = tf.shape(context)[1]
    net_examples = tf.expand_dims(net, 1)  # [batch_size, 1, ...]

    net_ndim = len(net_examples.get_shape().as_list())
    # Tile net by num_samples in axis=1.
    multiples = [1]*net_ndim
    multiples[1] = num_samples
    net_examples = tf.tile(net_examples, multiples)
  return net_examples


def add_context(net, context):
  """Merges visual perception with context using elementwise addition.

  Actions are reshaped to match net dimension depth-wise, and are added to
  the conv layers by broadcasting element-wise across H, W extent.

  Args:
    net: Tensor of shape [batch_size, H, W, C].
    context: Tensor of shape [batch_size * num_examples, C].
  Returns:
    Tensor with shape [batch_size * num_examples, H, W, C]
  """
  num_batch_net = tf.shape(net)[0]
  _, h, w, d1 = net.get_shape().as_list()
  _, d2 = context.get_shape().as_list()
  assert d1 == d2
  context = tf.reshape(context, [num_batch_net, -1, d2])
  net_examples = tile_to_match_context(net, context)
  # Flatten first two dimensions.
  net = tf.reshape(net_examples, [-1, h, w, d1])
  context = tf.reshape(context, [-1, 1, 1, d2])
  context = tf.tile(context, [1, h, w, 1])
  net = tf.add_n([net, context])
  return net
