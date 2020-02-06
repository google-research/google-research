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
import tensorflow.compat.v1 as tf
from capsule_em import utils
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
  if FLAGS.cpu_way:
    grid = tf.transpose(grid, [0, 2, 3, 1])
    data_format = 'NHWC'
    strides = [1, stride, stride, 1]
  else:
    data_format = 'NCHW'
    strides = [1, 1, stride, stride]

  conv_position = tf.nn.conv2d(
      grid, pos_kernel, strides, padding=padding, data_format=data_format)
  if FLAGS.cpu_way:
    conv_position = tf.transpose(conv_position, [0, 3, 1, 2])
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
    if FLAGS.cpu_way:
      image_reshape = tf.transpose(image_reshape, [0, 2, 3, 1])
      data_format = 'NHWC'
      strides = [1, FLAGS.stride_1, FLAGS.stride_1, 1]
    else:
      data_format = 'NCHW'
      strides = [1, 1, FLAGS.stride_1, FLAGS.stride_1]
    conv = tf.nn.conv2d(
        image_reshape,
        kernel,
        strides,
        padding=FLAGS.padding,
        data_format=data_format)
    biases = utils.bias_variable([FLAGS.num_start_conv])
    pre_activation = tf.nn.bias_add(conv, biases, data_format=data_format)
    if FLAGS.cpu_way:
      pre_activation = tf.transpose(pre_activation, [0, 3, 1, 2])
    position_grid = conv_pos(position_grid, FLAGS.kernel_size, FLAGS.stride_1,
                             FLAGS.padding)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    if FLAGS.verbose:
      tf.summary.histogram('activation', conv1)
    if FLAGS.pooling:
      pool1 = tf.nn.max_pool2d(
          conv1, ksize=2, strides=2, data_format='NCHW', padding='SAME')
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
            tf.nn.max_pool2d(
                cur_conv,
                ksize=2,
                strides=2,
                data_format='NCHW',
                padding='SAME')
        ]
      else:
        convs += [cur_conv]
      if FLAGS.verbose:
        tf.summary.histogram('activation', convs[-1])
  return convs[-1], conv_outputs[-1], position_grid
