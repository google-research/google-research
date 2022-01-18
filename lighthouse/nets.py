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

import tensorflow.compat.v1 as tf


def mpi_net(inputs, scope='mpi_net'):
  """3D encoder-decoder conv net for predicting MPI."""

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    ksize = 3
    norm = tf.contrib.layers.layer_norm
    act = tf.nn.relu
    initializer = tf.contrib.layers.variance_scaling_initializer()

    def conv(net, width, ksize=ksize, strides=1, d=1):
      """3D conv helper function."""

      return tf.layers.conv3d(
          act(norm(net)),
          width,
          ksize,
          strides=strides,
          padding='SAME',
          dilation_rate=(d, d, d),
          activation=None,
          kernel_initializer=initializer)

    def down_block(net, width, ksize=ksize, do_down=False, do_double=True):
      """strided conv + convs."""

      out = conv(net, width, ksize, 2) if do_down else conv(net, width, ksize)
      out = out + conv(conv(out, width, ksize), width,
                       ksize) if do_double else out
      return out, out

    def tf_repeat(tensor, repeats):
      """Nearest neighbor upsampling."""

      # from https://github.com/tensorflow/tensorflow/issues/8246

      with tf.variable_scope('repeat'):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
      return repeated_tensor

    def up_block(net, skip, width, ksize=ksize, do_double=True):
      """upsample + convs."""

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
      out = conv(out, width, ksize)
      out = out + conv(conv(out, width, ksize), width,
                       ksize) if do_double else out
      return out

    skips = []
    net = inputs
    width_list = [8, 16, 32, 64, 128]
    net = tf.layers.conv3d(
        net,
        width_list[0],
        3,
        padding='SAME',
        activation=None,
        kernel_initializer=initializer)
    for i in range(len(width_list)):
      net, skip = down_block(
          net, width_list[i], do_down=(i > 0), do_double=(i > 0))
      skips.append(skip)
    net = skips.pop()

    width_list = [64, 32, 16, 8]
    for i in range(len(width_list)):
      with tf.variable_scope('up_block{}'.format(i)):
        skip = skips.pop()
        net = up_block(
            net, skip, width_list[i], do_double=(i < len(width_list) - 1))

    # final 3d conv
    chout = 5  # bg RGB + alpha + weights
    net = tf.layers.conv3d(
        act(norm(net)),
        chout,
        3,
        padding='SAME',
        activation=None,
        kernel_initializer=initializer)

    rgb_bg = tf.reduce_mean(tf.nn.sigmoid(net[Ellipsis, :3]), axis=3, keepdims=True)

    weights = tf.nn.sigmoid(net[Ellipsis, 3:4])
    alpha = tf.nn.sigmoid(net[Ellipsis, -1:])

    ref = inputs[Ellipsis, 0:3]
    rgb = weights * ref + (1.0 - weights) * rgb_bg

    mpi = tf.concat([rgb, alpha], axis=4)

    return mpi


def cube_net_multires(inputs,
                      cube_rel_shapes,
                      cube_nest_inds,
                      scope='cube_net_multires'):
  """Multiresolution 3D encoder-decoder conv net for predicting lighting cubes."""

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    ksize = 3
    norm = tf.contrib.layers.layer_norm
    act = tf.nn.relu
    initializer = tf.contrib.layers.variance_scaling_initializer()

    def conv(net, width, ksize=ksize, strides=1, d=1):
      """3D conv helper function."""
      return tf.layers.conv3d(
          act(norm(net)),
          width,
          ksize,
          strides=strides,
          padding='SAME',
          dilation_rate=(d, d, d),
          activation=None,
          kernel_initializer=initializer)

    def down_block(net, width, ksize=ksize, do_down=False, do_double=True):
      """strided conv + convs."""

      out = conv(net, width, ksize, 2) if do_down else conv(net, width, ksize)
      out = out + conv(conv(out, width, ksize), width,
                       ksize) if do_double else out
      return out, out

    def tf_repeat(tensor, repeats):
      """Nearest neighbor upsampling."""

      # from https://github.com/tensorflow/tensorflow/issues/8246
      with tf.variable_scope('repeat'):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
      return repeated_tensor

    def up_block(net, skip, width, ksize=ksize, do_double=True):
      """upsample + convs."""

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
      out = conv(out, width)
      out = out + conv(conv(out, width, ksize), width,
                       ksize) if do_double else out
      return out

    def unet(net, width_list_down, width_list_up, chout):
      """3D encoder-decoder with skip connections."""

      skips = []
      net = tf.layers.conv3d(
          net,
          width_list_down[0],
          3,
          padding='SAME',
          activation=None,
          kernel_initializer=initializer)
      for i in range(len(width_list_down)):
        net, skip = down_block(
            net, width_list_down[i], do_down=(i > 0), do_double=(i > 0))
        skips.append(skip)
      net = skips.pop()

      for i in range(len(width_list_up)):
        with tf.variable_scope('up_block{}'.format(i)):
          skip = skips.pop()
          net = up_block(
              net,
              skip,
              width_list_up[i],
              do_double=(i < len(width_list_up) - 1))

      net = tf.layers.conv3d(
          act(norm(net)),
          chout,
          3,
          padding='SAME',
          activation=None,
          kernel_initializer=initializer)
      outvol = net
      return outvol

    width_list_down = [8, 16, 32, 64, 128]
    width_list_up = [64, 32, 16, 8]
    chout = 4

    outvols = []
    i_outvol_next = None
    for i in range(len(inputs)):
      with tf.variable_scope('multires_level{}'.format(i)):
        if i == 0:
          i_input = tf.stop_gradient(inputs[0])
        else:
          i_input = tf.concat([tf.stop_gradient(inputs[i]), i_outvol_next],
                              axis=-1)

        # outvol is convex combo of prenet vol and predicted vol
        i_net_out = unet(i_input, width_list_down, width_list_up, chout + 1)
        i_outvol_weights = tf.nn.sigmoid(i_net_out[Ellipsis, -1:])
        i_outvol = tf.nn.sigmoid(i_net_out[
            Ellipsis, :-1]) * i_outvol_weights + inputs[i] * (1.0 - i_outvol_weights)

        outvols.append(i_outvol)

        if i < len(inputs) - 1:
          # slice and upsample region of volume
          # corresponding to next finer resolution level
          i_outvol_next = i_outvol[:,
                                   cube_nest_inds[i][0]:cube_nest_inds[i][0] +
                                   cube_rel_shapes[i],
                                   cube_nest_inds[i][1]:cube_nest_inds[i][1] +
                                   cube_rel_shapes[i],
                                   cube_nest_inds[i][2]:cube_nest_inds[i][2] +
                                   cube_rel_shapes[i], :]
          i_outvol_next = tf_repeat(i_outvol_next, [
              1,
              tf.shape(i_input)[1] // tf.shape(i_outvol_next)[1],
              tf.shape(i_input)[2] // tf.shape(i_outvol_next)[2],
              tf.shape(i_input)[3] // tf.shape(i_outvol_next)[3], 1
          ])
          i_outvol_next.set_shape([None, None, None, None, chout])

    return outvols


def discriminator(x_init, do_inorm=False, scope='discriminator'):
  """Image discriminator from SPADE paper."""

  # code from https://github.com/taki0112/SPADE-Tensorflow/

  def lrelu(x, alpha=0.01):
    return tf.nn.leaky_relu(x, alpha)

  def instance_norm(x, scope='instance_norm'):
    if do_inorm:
      return tf.contrib.layers.instance_norm(
          x, epsilon=1e-05, center=True, scale=True, scope=scope)
    else:
      return x

  def spectral_norm(w, iteration=1):
    """Spectral normalization of a weight matrix."""

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(
        'u', [1, w_shape[-1]],
        initializer=tf.random_normal_initializer(),
        trainable=False)

    u_hat = u
    v_hat = None
    for _ in range(iteration):
      # power iteration, usually iteration = 1 will be enough
      v_ = tf.matmul(u_hat, tf.transpose(w))
      v_hat = tf.nn.l2_normalize(v_)

      u_ = tf.matmul(v_hat, w)
      u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
      w_norm = w / sigma
      w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

  weight_init = tf.contrib.layers.variance_scaling_initializer()

  def conv(x,
           channels,
           kernel=4,
           stride=2,
           pad=0,
           pad_type='zero',
           use_bias=True,
           sn=False,
           scope='conv_0'):
    """2D conv helper function."""

    with tf.variable_scope(scope):
      if pad > 0:
        h = x.get_shape().as_list()[1]
        if h % stride == 0:
          pad = pad * 2
        else:
          pad = max(kernel - (h % stride), 0)

        pad_top = pad // 2
        pad_bottom = pad - pad_top
        pad_left = pad // 2
        pad_right = pad - pad_left

        if pad_type == 'zero':
          x = tf.pad(
              x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect':
          x = tf.pad(
              x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
              mode='REFLECT')

      if sn:
        w = tf.get_variable(
            'kernel',
            shape=[kernel, kernel, x.get_shape()[-1], channels],
            initializer=weight_init,
            regularizer=None)
        x = tf.nn.conv2d(
            input=x,
            filter=spectral_norm(w),
            strides=[1, stride, stride, 1],
            padding='VALID')
        if use_bias:
          bias = tf.get_variable(
              'bias', [channels], initializer=tf.constant_initializer(0.0))
          x = tf.nn.bias_add(x, bias)

      else:
        x = tf.layers.conv2d(
            inputs=x,
            filters=channels,
            kernel_size=kernel,
            kernel_initializer=weight_init,
            kernel_regularizer=None,
            strides=stride,
            use_bias=use_bias)

      return x

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    d_logit = []
    n_scale = 1
    for scale in range(n_scale):
      feature_loss = []
      channel = 64
      x = x_init

      x = conv(
          x,
          channel,
          kernel=4,
          stride=2,
          pad=1,
          use_bias=True,
          sn=False,
          scope='ms_' + str(scale) + 'conv_0')
      x = lrelu(x, 0.2)

      feature_loss.append(x)

      n_dis = 4
      for i in range(1, n_dis):
        stride = 1 if i == n_dis - 1 else 2

        x = conv(
            x,
            channel * 2,
            kernel=4,
            stride=stride,
            pad=1,
            use_bias=True,
            sn=True,
            scope='ms_' + str(scale) + 'conv_' + str(i))
        x = instance_norm(x, scope='ms_' + str(scale) + 'ins_norm_' + str(i))
        x = lrelu(x, 0.2)

        feature_loss.append(x)

        channel = min(channel * 2, 512)

      x = conv(
          x,
          channels=1,
          kernel=4,
          stride=1,
          pad=1,
          use_bias=True,
          sn=True,
          scope='ms_' + str(scale) + 'd_logit')

      feature_loss.append(x)
      d_logit.append(feature_loss)

  return d_logit


# To complete this codebase, you must copy lines 10 through 48 from
# https://github.com/CQFIO/PhotographicImageSynthesis/blob/master/demo_1024p.py
# Into this code here.
