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

"""Contains definitions for 'Representation Flow' layer [1].

Representation flow layer is a generalization of optical flow extraction; the
layer could be inserted anywhere within a CNN to capture feature movements. This
is the version taking 4D tensor with the shape [batch*time, height, width,
channels], to make this run on TPU.

[1] AJ Piergiovanni and Michael S. Ryoo,
    Representation Flow for Action Recognition. CVPR 2019.
    https://arxiv.org/abs/1810.01455
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import numpy as np

import tensorflow.compat.v1 as tf

flags.DEFINE_string(
    'combine', 'add',
    'Method to combine rep. flow with input (none, concat, add, multiply)')
flags.DEFINE_bool('train_feature_grad', False, 'Train _ param')
flags.DEFINE_bool('train_divergence', False, 'Train _ param')
flags.DEFINE_bool('train_flow_grad', False, 'Train _ param')
flags.DEFINE_bool('train_hyper', False, 'Train _ param')
flags.DEFINE_float('bn_decay', 0.99, 'batch norm decay')
flags.DEFINE_float('bn_epsilon', 1e-5, 'batch norm epsilon')
FLAGS = flags.FLAGS

BATCH_NORM_DECAY = 0.99
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs,
                    is_training,
                    relu=True,
                    swish=False,
                    init_zero=False,
                    bn_decay=BATCH_NORM_DECAY,
                    bn_epsilon=BATCH_NORM_EPSILON,
                    data_format='channels_first'):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    swish: `bool`. True to use swish activation function, False for ReLU.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    bn_decay: `float` batch norm decay parameter to use.
    bn_epsilon: `float` batch norm epsilon parameter to use.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """

  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  assert data_format == 'channels_last'

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = -1

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=bn_decay,
      epsilon=bn_epsilon,
      center=True,
      scale=True,
      training=is_training,
      fused=True,
      gamma_initializer=gamma_initializer)

  if swish:
    inputs = tf.keras.activations.swish(inputs)
  elif relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def rep_flow(inputs,
             batch,
             time,
             num_iter=20,
             is_training=False,
             bottleneck=32,
             data_format='channels_last',
             scope='RepFlow'):
  """Computes the representation flow motivated by TV-L1 optical flow.

  Args:
    inputs: list of `Tensors` of shape `[batch*time, height, width, channels]`.
    batch: 'int' number of examples in the batch.
    time: 'int' number of frames in the input tensor.
    num_iter: 'int' number of iterations to use for the flow computation.
    is_training: `bool` for whether the model is training.
    bottleneck: 'int' number of filters to be used for the flow computation.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`. Only
      works for channels_last currently.
    scope: 'str' to give names to the TF variables used in this flow layer.

  Returns:
    A `Tensor` with the same `data_format` and shape as input.
  """
  dtype = inputs.dtype

  def divergence(p1, p2, f_grad_x, f_grad_y, name,
                 data_format='channels_last'):
    """Computes the divergence value used with TV-L1 optical flow algorithm.

    Args:
      p1: 'Tensor' input.
      p2: 'Tensor' input in the next frame.
      f_grad_x: 'Tensor' x gradient of F value used in TV-L1.
      f_grad_y: 'Tensor' y gradient of F value used in TV-L1.
      name: 'str' name for the variable scope.
      data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

    Returns:
      A `Tensor` with the same `data_format` and shape as input.
    """
    df = 'NHWC' if data_format == 'channels_last' else 'NCHW'

    with tf.variable_scope('divergence_' + name):
      if data_format == 'channels_last':
        p1 = tf.pad(p1[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]])
        p2 = tf.pad(p2[:, :-1, :, :], [[0, 0], [1, 0], [0, 0], [0, 0]])
      else:
        p1 = tf.pad(p1[:, :, :, :-1], [[0, 0], [0, 0], [0, 0], [1, 0]])
        p2 = tf.pad(p2[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]])

      grad_x = tf.nn.conv2d(p1, f_grad_x, [1, 1, 1, 1], 'SAME', data_format=df)
      grad_y = tf.nn.conv2d(p2, f_grad_y, [1, 1, 1, 1], 'SAME', data_format=df)
      return grad_x + grad_y

  def forward_grad(x, f_grad_x, f_grad_y, name, data_format='channels_last'):
    with tf.variable_scope('forward_grad_' + name):
      df = 'NHWC' if data_format == 'channels_last' else 'NCHW'
      grad_x = tf.nn.conv2d(x, f_grad_x, [1, 1, 1, 1], 'SAME', data_format=df)
      grad_y = tf.nn.conv2d(x, f_grad_y, [1, 1, 1, 1], 'SAME', data_format=df)
      return grad_x, grad_y

  def norm_img(x):
    mx = tf.reduce_max(x)
    mn = tf.reduce_min(x)
    if mx == mn:
      return x
    else:
      return 255 * (x - mn) / (mx - mn)

  assert data_format == 'channels_last'

  residual = inputs

  with tf.variable_scope(scope, scope, [inputs]):

    df = 'NHWC' if data_format == 'channels_last' else 'NCHW'
    axis = 3 if data_format == 'channels_last' else 1

    img_grad = np.asarray([-0.5, 0, 0.5]).astype('float32')
    img_grad_x = np.repeat(
        np.reshape(img_grad, (1, 3, 1, 1)), bottleneck, axis=2) * np.eye(
            bottleneck, dtype='float32')
    img_grad_x = tf.get_variable(
        'img_grad_x', initializer=img_grad_x,
        trainable=FLAGS.train_feature_grad)
    img_grad_y = np.repeat(
        np.reshape(img_grad, (3, 1, 1, 1)), bottleneck, axis=2) * np.eye(
            bottleneck, dtype='float32')
    img_grad_y = tf.get_variable(
        'img_grad_y', initializer=img_grad_y,
        trainable=FLAGS.train_feature_grad)

    img_grad_x = tf.cast(img_grad_x, dtype)
    img_grad_y = tf.cast(img_grad_y, dtype)

    f_grad = np.asarray([-1, 1]).astype('float32')
    f_grad_x = np.repeat(
        np.reshape(f_grad, (1, 2, 1, 1)), bottleneck, axis=2) * np.eye(
            bottleneck, dtype='float32')
    f_grad_x = tf.get_variable(
        'f_grad_x', initializer=f_grad_x, trainable=FLAGS.train_divergence)
    f_grad_y = np.repeat(
        np.reshape(f_grad, (2, 1, 1, 1)), bottleneck, axis=2) * np.eye(
            bottleneck, dtype='float32')
    f_grad_y = tf.get_variable(
        'f_grad_y', initializer=f_grad_y, trainable=FLAGS.train_divergence)

    f_grad_x = tf.cast(f_grad_x, dtype)
    f_grad_y = tf.cast(f_grad_y, dtype)

    f_grad_x2 = np.repeat(
        np.reshape(f_grad, (1, 2, 1, 1)), bottleneck, axis=2) * np.eye(
            bottleneck, dtype='float32')
    f_grad_x2 = tf.get_variable(
        'f_grad_x2', initializer=f_grad_x2, trainable=FLAGS.train_flow_grad)
    f_grad_y2 = np.repeat(
        np.reshape(f_grad, (2, 1, 1, 1)), bottleneck, axis=2) * np.eye(
            bottleneck, dtype='float32')
    f_grad_y2 = tf.get_variable(
        'f_grad_y2', initializer=f_grad_y2, trainable=FLAGS.train_flow_grad)

    f_grad_x2 = tf.cast(f_grad_x2, dtype)
    f_grad_y2 = tf.cast(f_grad_y2, dtype)

    t = tf.get_variable(
        'theta', initializer=0.3, trainable=FLAGS.train_hyper)
    l = tf.get_variable(
        'lambda', initializer=0.15, trainable=FLAGS.train_hyper)
    a = tf.get_variable(
        'tau', initializer=0.25, trainable=FLAGS.train_hyper)

    t = tf.cast(t, dtype)
    l = tf.cast(l, dtype)
    a = tf.cast(a, dtype)

    with tf.variable_scope('rf'):

      depth = inputs.shape[axis]

      if bottleneck == 1:
        inputs = tf.reduce_mean(inputs, axis=axis)
        inputs = tf.expand_dims(inputs, -1)
      elif depth != bottleneck:
        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=bottleneck,
            kernel_size=1,
            strides=1,  # filters=inputs.shape[3]
            padding='SAME',
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

      inp = norm_img(inputs)
      inp = tf.reshape(
          inp,
          (batch, time, tf.shape(inp)[1], tf.shape(inp)[2], tf.shape(inp)[3]))
      img1 = tf.reshape(
          inp[:, :-1],
          (-1, tf.shape(inp)[2], tf.shape(inp)[3], tf.shape(inp)[4]))
      img2 = tf.reshape(
          inp[:, 1:],
          (-1, tf.shape(inp)[2], tf.shape(inp)[3], tf.shape(inp)[4]))

      u1 = tf.zeros_like(img1, dtype=dtype)
      u2 = tf.zeros_like(img2, dtype=dtype)

      t = tf.abs(t) + 1e-12
      l_t = l * t
      taut = a / t

      grad2_x = tf.nn.conv2d(
          img2, img_grad_x, [1, 1, 1, 1], 'SAME', data_format=df)
      grad2_y = tf.nn.conv2d(
          img2, img_grad_y, [1, 1, 1, 1], 'SAME', data_format=df)

      p11 = tf.zeros_like(img1, dtype=dtype)
      p12 = tf.zeros_like(img1, dtype=dtype)
      p21 = tf.zeros_like(img1, dtype=dtype)
      p22 = tf.zeros_like(img1, dtype=dtype)

      gsqx = grad2_x**2
      gsqy = grad2_y**2

      grad = gsqx + gsqy + 1e-12

      rho_c = img2 - grad2_x * u1 - grad2_y * u2 - img1

      for _ in range(num_iter):
        rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12

        v1 = tf.zeros_like(img1, dtype=dtype)
        v2 = tf.zeros_like(img2, dtype=dtype)

        mask1 = rho < -l_t * grad
        tmp11 = tf.where(mask1, l_t * grad2_x,
                         tf.zeros_like(grad2_x, dtype=dtype))
        tmp12 = tf.where(mask1, l_t * grad2_y,
                         tf.zeros_like(grad2_y, dtype=dtype))

        mask2 = rho > l_t * grad
        tmp21 = tf.where(mask2, -l_t * grad2_x,
                         tf.zeros_like(grad2_x, dtype=dtype))
        tmp22 = tf.where(mask2, -l_t * grad2_y,
                         tf.zeros_like(grad2_y, dtype=dtype))

        mask3 = (~mask1) & (~mask2) & (grad > 1e-12)
        tmp31 = tf.where(mask3, (-rho / grad) * grad2_x,
                         tf.zeros_like(grad2_x, dtype=dtype))
        tmp32 = tf.where(mask3, (-rho / grad) * grad2_y,
                         tf.zeros_like(grad2_y, dtype=dtype))

        v1 = tmp11 + tmp21 + tmp31 + u1
        v2 = tmp12 + tmp22 + tmp32 + u2

        u1 = v1 + t * divergence(
            p11, p12, f_grad_x, f_grad_y, 'div_p1', data_format=data_format)
        u2 = v2 + t * divergence(
            p21, p22, f_grad_x, f_grad_y, 'div_p2', data_format=data_format)

        u1x, u1y = forward_grad(
            u1, f_grad_x2, f_grad_y2, 'u1', data_format=data_format)
        u2x, u2y = forward_grad(
            u2, f_grad_x2, f_grad_y2, 'u2', data_format=data_format)

        p11 = (p11 + taut * u1x) / (1. +
                                    taut * tf.sqrt(u1x**2 + u1y**2 + 1e-12))
        p12 = (p12 + taut * u1y) / (1. +
                                    taut * tf.sqrt(u1x**2 + u1y**2 + 1e-12))
        p21 = (p21 + taut * u2x) / (1. +
                                    taut * tf.sqrt(u2x**2 + u2y**2 + 1e-12))
        p22 = (p22 + taut * u2y) / (1. +
                                    taut * tf.sqrt(u2x**2 + u2y**2 + 1e-12))

      u1 = tf.reshape(
          u1,
          (batch, time - 1, tf.shape(u1)[1], tf.shape(u1)[2], tf.shape(u1)[3]))
      u2 = tf.reshape(
          u2,
          (batch, time - 1, tf.shape(u2)[1], tf.shape(u2)[2], tf.shape(u2)[3]))
      flow = tf.concat([u1, u2], axis=axis + 1)
      flow = tf.concat([
          flow,
          tf.reshape(flow[:, -1, :, :, :],
                     (batch, 1, tf.shape(u1)[2], tf.shape(u1)[3], -1))
      ],
                       axis=1)
      flow = tf.reshape(flow,
                        (batch * (time), tf.shape(u1)[2], tf.shape(u2)[3], -1))

      if bottleneck == 1:
        return flow
      else:
        flow = tf.layers.conv2d(
            inputs=flow,
            filters=depth,
            kernel_size=1,
            strides=1,  # filters=inputs.shape[3]
            padding='SAME',
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

        flow = batch_norm_relu(
            flow, is_training, relu=False, init_zero=True,
            bn_decay=FLAGS.bn_decay, bn_epsilon=FLAGS.bn_epsilon,
            data_format=data_format)
        return tf.nn.relu(flow + residual)
