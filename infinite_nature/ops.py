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

"""Helper functions for defining network components."""
import math

import config
import numpy as np
import tensorflow as tf


def instance_norm(inputs, scope="instance_norm"):
  with tf.compat.v1.variable_scope(scope):
    beta = None
    gamma = None
    epsilon = 1e-05
    # All axes except first (batch) and last (channels).
    axes = list(range(1, inputs.shape.ndims - 1))
    mean, variance = tf.nn.moments(inputs, axes, keepdims=True)
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)


def fully_connected(x, units, use_bias=True, scope="linear"):
  """Creates a fully connected layer.

  Args:
    x: [B, ...] batch of vectors
    units: (int) Number of output features
    use_bias: (bool) If true defines a bias term
    scope: (str) variable scope

  Returns:
    [B, units] output of the fully connected layer on x.
  """
  with tf.compat.v1.variable_scope(scope):
    x = tf.compat.v1.layers.flatten(x)
    x = tf.compat.v1.layers.dense(
        x,
        units=units,
        use_bias=use_bias)

    return x


def double_size(image):
  """Double the size of an image or batch of images.

  This just duplicates each pixel into a 2x2 block – i.e. nearest-neighbor
  upsampling. The result is identical to using tf.image.resize_area to double
  the size, with the addition that we can take the gradient.

  Args:
    image: [..., H, W, C] image to double in size

  Returns:
    [..., H*2, W*2, C] scaled up.
  """
  image = tf.convert_to_tensor(image)
  shape = image.shape.as_list()
  multiples = [1] * (len(shape) - 2) + [2, 2]
  tiled = tf.tile(image, multiples)
  newshape = shape[:-3] + [shape[-3] * 2, shape[-2] * 2, shape[-1]]
  return tf.reshape(tiled, newshape)


def leaky_relu(x, alpha=0.01):
  return tf.nn.leaky_relu(x, alpha)


def spectral_norm(w, iteration=1, update_variable=False):
  """Applies spectral normalization to a weight tensor.

  When update_variable is True, updates the u vector of spectral normalization
  with its power-iteration method. If spectral norm is called multiple
  times within the same scope (like in Infinite Nature), the normalization
  variable u will be shared between them, and any prior assign operations on u
  will be executed before the current assign. Because power iteration is
  convergent, it does not matter if multiple updates take place in a single
  forward pass.

  Args:
    w: (tensor) A weight tensor to apply spectral normalization to
    iteration: (int) The number of times to run power iteration when called
    update_variable: (bool) If true, update the u variable.

  Returns:
    A tensor of the same shape as w.
  """
  w_shape = w.shape.as_list()
  w = tf.reshape(w, [-1, w_shape[-1]])

  u = tf.compat.v1.get_variable(
      "u", [1, w_shape[-1]],
      initializer=tf.random_normal_initializer(),
      trainable=False)

  u_hat = u
  v_hat = None
  for _ in range(iteration):
    # Power iteration. Usually iteration = 1 will be enough.
    v_ = tf.matmul(u_hat, tf.transpose(w))
    v_hat = tf.nn.l2_normalize(v_)

    u_ = tf.matmul(v_hat, w)
    u_hat = tf.nn.l2_normalize(u_)

  u_hat = tf.stop_gradient(u_hat)
  v_hat = tf.stop_gradient(v_hat)

  sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

  if update_variable:
    # Force any previous assign_ops that are upstream of w to assign
    # to prevent race conditions.
    update_op = u.assign(u_hat)
  else:
    update_op = tf.no_op()
  with tf.control_dependencies([update_op]):
    w_norm = w / sigma
    w_norm = tf.reshape(w_norm, w_shape)

  return w_norm


def sn_conv(tensor, channels, kernel_size=3, stride=1,
            use_bias=True, use_spectral_norm=True, scope="conv",
            pad_type="REFLECT"):
  """A convolutional layer with support for padding and optional spectral norm.

  Args:
    tensor: [B, H, W, C] A tensor to perform a convolution on
    channels: (int) The number of output channels
    kernel_size: (int) The size of a square convolutional filter
    stride: (int) The stride to apply the convolution
    use_bias: (bool) If true, adds a learned bias term
    use_spectral_norm: (bool) If true, applies spectral normalization to the
      weights
    scope: (str) The scope of the variables
    pad_type: (str) The padding to use

  Returns:
    The result of the convolution layer on a tensor.
  """
  tensor_shape = tensor.shape
  with tf.compat.v1.variable_scope(scope):
    h, w = tensor_shape[1], tensor_shape[2]
    output_h, output_w = int(math.ceil(h / stride)), int(
        math.ceil(w / stride))

    p_h = (output_h) * stride + kernel_size - h - 1
    p_w = (output_w) * stride + kernel_size - w - 1

    pad_top = p_h // 2
    pad_bottom = p_h - pad_top
    pad_left = p_w // 2
    pad_right = p_w - pad_left
    tensor = tf.pad(
        tensor,
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        mode=pad_type)
    if use_spectral_norm:
      w = tf.compat.v1.get_variable(
          "kernel",
          shape=[kernel_size, kernel_size, tensor_shape[-1], channels])
      x = tf.nn.conv2d(
          tensor,
          spectral_norm(w, update_variable=config.is_training()),
          [1, stride, stride, 1],
          "VALID")
      if use_bias:
        bias = tf.compat.v1.get_variable(
            "bias", [channels], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)

    else:
      x = tf.compat.v1.layers.conv2d(
          tensor,
          channels,
          kernel_size,
          strides=stride,
          use_bias=use_bias)

    return x
