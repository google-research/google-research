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

"""Convolutional subnetwork."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from capsule_em import utils
from tensorflow.contrib import layers as contrib_layers
FLAGS = tf.app.flags.FLAGS


def conv_pos(grid, kernel_size, stride, padding):
  """Keep track of the receptive field offsets."""
  x_kernel = tf.stack([
      tf.zeros((kernel_size, kernel_size)),
      tf.ones((kernel_size, kernel_size))
  ],
                      axis=2)
  y_kernel = tf.stack([
      tf.ones((kernel_size, kernel_size)),
      tf.zeros((kernel_size, kernel_size))
  ],
                      axis=2)
  pos_kernel = tf.stack([x_kernel, y_kernel], axis=3)
  conv_position = tf.nn.conv2d(
      grid,
      pos_kernel, [1, 1, stride, stride],
      padding=padding,
      data_format='NCHW')
  return conv_position / (kernel_size * kernel_size)


def add_convs(features):
  """Stack Convolution layers."""
  image_dim = features['height']
  image_depth = features['depth']
  image = features['images']
  position_grid = tf.reshape(
      tf.constant(
          np.mgrid[(-image_dim // 2):((image_dim + 1) // 2), (-image_dim // 2):(
              (image_dim + 1) // 2)],
          dtype=tf.float32) / 100.0, (1, 2, image_dim, image_dim))
  if FLAGS.verbose_image:
    with tf.name_scope('input_reshape'):
      image_shaped_input = tf.reshape(image,
                                      [-1, image_dim, image_dim, image_depth])
      tf.summary.image('input', image_shaped_input, 10)

  with tf.variable_scope('conv1') as scope:
    kernel = utils.weight_variable(
        shape=[
            FLAGS.kernel_size, FLAGS.kernel_size, image_depth,
            FLAGS.num_start_conv
        ],
        stddev=5e-2)

    image_reshape = tf.reshape(image, [-1, image_depth, image_dim, image_dim])
    conv = tf.nn.conv2d(
        image_reshape,
        kernel, [1, 1, FLAGS.stride_1, FLAGS.stride_1],
        padding=FLAGS.padding,
        data_format='NCHW')
    position_grid = conv_pos(position_grid, FLAGS.kernel_size, FLAGS.stride_1,
                             FLAGS.padding)
    biases = utils.bias_variable([FLAGS.num_start_conv])
    pre_activation = tf.nn.bias_add(conv, biases, data_format='NCHW')
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    if FLAGS.verbose:
      tf.summary.histogram('activation', conv1)
    if FLAGS.pooling:
      pool1 = contrib_layers.max_pool2d(
          conv1, kernel_size=2, stride=2, data_format='NCHW', padding='SAME')
      convs = [pool1]
    else:
      convs = [conv1]

  conv_outputs = [FLAGS.num_start_conv]

  for i in range(int(FLAGS.extra_conv)):
    conv_outputs += [int(FLAGS.conv_dims.split(',')[i])]
    with tf.variable_scope('conv{}'.format(i + 2)) as scope:
      kernel = utils.weight_variable(
          shape=[
              int(FLAGS.conv_kernels.split(',')[i]),
              int(FLAGS.conv_kernels.split(',')[i]), conv_outputs[i],
              conv_outputs[i + 1]
          ],
          stddev=5e-2)
      conv = tf.nn.conv2d(
          convs[i],
          kernel, [
              1, 1,
              int(FLAGS.conv_strides.split(',')[i]),
              int(FLAGS.conv_strides.split(',')[i])
          ],
          padding=FLAGS.padding,
          data_format='NCHW')
      position_grid = conv_pos(position_grid,
                               int(FLAGS.conv_kernels.split(',')[i]),
                               int(FLAGS.conv_strides.split(',')[i]),
                               FLAGS.padding)
      biases = utils.bias_variable([conv_outputs[i + 1]])
      pre_activation = tf.nn.bias_add(conv, biases, data_format='NCHW')
      cur_conv = tf.nn.relu(pre_activation, name=scope.name)
      if FLAGS.pooling:
        convs += [
            contrib_layers.max_pool2d(
                cur_conv,
                kernel_size=2,
                stride=2,
                data_format='NCHW',
                padding='SAME')
        ]
      else:
        convs += [cur_conv]
      if FLAGS.verbose:
        tf.summary.histogram('activation', convs[-1])
  return convs[-1], conv_outputs[-1], position_grid


def conv_inference(features):
  """Inference for a CNN. Conv + FC."""
  conv, _, _ = add_convs(features)
  hidden1 = contrib_layers.flatten(conv)
  if FLAGS.extra_fc > 0:
    hidden = contrib_layers.fully_connected(
        hidden1,
        FLAGS.extra_fc,
        activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=0.1, dtype=tf.float32),
        biases_initializer=tf.constant_initializer(0.1))
    if FLAGS.dropout and FLAGS.train:
      hidden = tf.nn.dropout(hidden, 0.5)
  else:
    hidden = hidden1
  logits = contrib_layers.fully_connected(
      hidden,
      features['num_classes'],
      activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=0.1, dtype=tf.float32),
      biases_initializer=tf.constant_initializer(0.1))
  return logits, None, None
