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

"""Network definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import tensorflow.compat.v1 as tf


def ed_3d_net(inputs, ch_final, scope='ed_3d_net'):
  """3D encoder-decoder conv net to predict initial MPI."""

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    ksize = 3
    def conv(net, width, ksize=ksize, s=1, d=1):
      """Conv helper function."""

      return tf.layers.conv3d(
          net,
          width,
          ksize,
          strides=s,
          padding='SAME',
          dilation_rate=(d, d, d),
          activation=tf.nn.relu)

    def down_block(net, width, ksize=ksize, do_down=False):
      """Strided convs followed by convs."""

      down = conv(net, width, ksize, 2) if do_down else net
      out = conv(conv(down, width), width)
      return out, out

    def tf_repeat(tensor, repeats):
      """NN upsampling from https://github.com/tensorflow/tensorflow/issues/8246."""

      with tf.variable_scope('repeat'):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
      return repeated_tensor

    def up_block(net, skip, width, ksize=ksize):
      """Nearest neighbor upsampling followed by convs."""

      ch = net.get_shape().as_list()[-1]
      net_repeat = tf_repeat(net, [1, 2, 2, 2, 1])
      net_repeat.set_shape([None, None, None, None, ch])
      up = net_repeat
      up = tf.cond(
          tf.equal(tf.shape(up)[1],
                   tf.shape(skip)[1]), lambda: up, lambda: up[:, :-1, Ellipsis])
      up = tf.cond(
          tf.equal(tf.shape(up)[2],
                   tf.shape(skip)[2]), lambda: up, lambda: up[:, :, :-1, Ellipsis])
      out = tf.concat([up, skip], -1)
      out = conv(conv(out, width, ksize), width, ksize)
      return out

    skips = []
    net = inputs
    sh = inputs.get_shape().as_list()
    w_list = [np.maximum(8, sh[-1]), 16, 32, 64, 128]
    for i in range(len(w_list)):
      net, skip = down_block(net, w_list[i], do_down=(i > 0))
      skips.append(skip)
    net = skips.pop()

    # dilated conv layers
    d_list = [2, 4, 8, 1]
    for i in range(len(d_list)):
      net = conv(net, w_list[-1], d=d_list[i])

    w_list = [
        64,
        np.maximum(32, ch_final),
        np.maximum(16, ch_final),
        np.maximum(8, ch_final)
    ]
    for i in range(len(w_list)):
      with tf.variable_scope('up_block{}'.format(i)):
        skip = skips.pop()
        net = up_block(net, skip, w_list[i])

    penultimate = net
    # Final 3d conv
    # RGBA output
    net = tf.layers.conv3d(
        net, ch_final, 3, padding='SAME', activation=tf.nn.tanh)

    return net, penultimate


def refine_net(inputs, scope='refine_net'):
  """3D encoder-decoder conv net to predict refined MPI."""

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    ksize = 3
    def conv(net, width, ksize=ksize, s=1, d=1):
      """Conv helper function."""

      return tf.layers.conv3d(
          net,
          width,
          ksize,
          strides=s,
          padding='SAME',
          dilation_rate=(d, d, d),
          activation=tf.nn.relu)

    def down_block(net, width, ksize=ksize, do_down=False):
      """Strided convs followed by convs."""

      down = conv(net, width, ksize, 2) if do_down else net
      out = conv(conv(down, width), width)
      return out, out

    def tf_repeat(tensor, repeats):
      """NN upsampling from https://github.com/tensorflow/tensorflow/issues/8246."""

      with tf.variable_scope('repeat'):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
      return repeated_tensor

    def up_block(net, skip, width, ksize=ksize):
      """Nearest neighbor upsampling followed by convs."""

      ch = net.get_shape().as_list()[-1]
      net_repeat = tf_repeat(net, [1, 2, 2, 2, 1])
      net_repeat.set_shape([None, None, None, None, ch])
      up = net_repeat
      up = tf.cond(
          tf.equal(tf.shape(up)[1],
                   tf.shape(skip)[1]), lambda: up, lambda: up[:, :-1, Ellipsis])
      up = tf.cond(
          tf.equal(tf.shape(up)[2],
                   tf.shape(skip)[2]), lambda: up, lambda: up[:, :, :-1, Ellipsis])
      out = tf.concat([up, skip], -1)
      out = conv(conv(out, width, ksize), width, ksize)
      return out

    skips = []
    net = inputs
    sh = inputs.get_shape().as_list()
    w_list = [np.maximum(8, sh[-1]), 16, 32, 64, 128]
    for i in range(len(w_list)):
      net, skip = down_block(net, w_list[i], do_down=(i > 0))
      skips.append(skip)
    net = skips.pop()

    w_list = [64, 32, 16, 8]
    for i in range(len(w_list)):
      with tf.variable_scope('up_block{}'.format(i)):
        skip = skips.pop()
        net = up_block(net, skip, w_list[i])

    # Final 3d conv
    chout = 3  # 2D flow and alpha
    net = tf.layers.conv3d(net, chout, ksize, padding='SAME', activation=None)
    alpha_act = tf.nn.tanh(net[Ellipsis, -1:])
    flow_act = net[Ellipsis, :2]
    net = tf.concat([flow_act, alpha_act], axis=-1)

    return net


# ******************************************************************************
# The VGG code below is copied from Qifeng Chen
# https://github.com/CQFIO/PhotographicImageSynthesis/blob/master/demo_1024p.py
def build_net(ntype, nin, nwb=None, name=None):
  """VGG layer model."""

  if ntype == 'conv':
    return tf.nn.relu(
        tf.nn.conv2d(
            nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) +
        nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(
        nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i):
  """Convert vgg weights to constant tensors."""

  weights = vgg_layers[i][0][0][2][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[i][0][0][2][0][1]
  bias = tf.constant(np.reshape(bias, (bias.size)))
  return weights, bias


def build_vgg19(input_tensor, model_filepath, reuse=False):
  """Set up VGG network."""

  with tf.variable_scope('vgg', reuse=reuse):
    net = {}
    input_tensor = tf.cast(input_tensor, tf.float32)
    with tf.gfile.Open(model_filepath, 'r') as f:
      vgg_rawnet = sio.loadmat(f)
    vgg_layers = vgg_rawnet['layers'][0]
    imagenet_mean = tf.constant([123.6800, 116.7790, 103.9390],
                                shape=[1, 1, 1, 3])
    net['input'] = input_tensor - imagenet_mean
    net['conv1_1'] = build_net(
        'conv',
        net['input'],
        get_weight_bias(vgg_layers, 0),
        name='vgg_conv1_1')
    net['conv1_2'] = build_net(
        'conv',
        net['conv1_1'],
        get_weight_bias(vgg_layers, 2),
        name='vgg_conv1_2')
    net['pool1'] = build_net('pool', net['conv1_2'])
    net['conv2_1'] = build_net(
        'conv',
        net['pool1'],
        get_weight_bias(vgg_layers, 5),
        name='vgg_conv2_1')
    net['conv2_2'] = build_net(
        'conv',
        net['conv2_1'],
        get_weight_bias(vgg_layers, 7),
        name='vgg_conv2_2')
    net['pool2'] = build_net('pool', net['conv2_2'])
    net['conv3_1'] = build_net(
        'conv',
        net['pool2'],
        get_weight_bias(vgg_layers, 10),
        name='vgg_conv3_1')
    net['conv3_2'] = build_net(
        'conv',
        net['conv3_1'],
        get_weight_bias(vgg_layers, 12),
        name='vgg_conv3_2')
    net['conv3_3'] = build_net(
        'conv',
        net['conv3_2'],
        get_weight_bias(vgg_layers, 14),
        name='vgg_conv3_3')
    net['conv3_4'] = build_net(
        'conv',
        net['conv3_3'],
        get_weight_bias(vgg_layers, 16),
        name='vgg_conv3_4')
    net['pool3'] = build_net('pool', net['conv3_4'])
    net['conv4_1'] = build_net(
        'conv',
        net['pool3'],
        get_weight_bias(vgg_layers, 19),
        name='vgg_conv4_1')
    net['conv4_2'] = build_net(
        'conv',
        net['conv4_1'],
        get_weight_bias(vgg_layers, 21),
        name='vgg_conv4_2')
    net['conv4_3'] = build_net(
        'conv',
        net['conv4_2'],
        get_weight_bias(vgg_layers, 23),
        name='vgg_conv4_3')
    net['conv4_4'] = build_net(
        'conv',
        net['conv4_3'],
        get_weight_bias(vgg_layers, 25),
        name='vgg_conv4_4')
    net['pool4'] = build_net('pool', net['conv4_4'])
    net['conv5_1'] = build_net(
        'conv',
        net['pool4'],
        get_weight_bias(vgg_layers, 28),
        name='vgg_conv5_1')
    net['conv5_2'] = build_net(
        'conv',
        net['conv5_1'],
        get_weight_bias(vgg_layers, 30),
        name='vgg_conv5_2')
  return net
# ******************************************************************************
