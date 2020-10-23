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

"""Spectral normalization library."""

import tensorflow.compat.v1 as tf


def _l2normalize(v, eps=1e-12):
  """L2 normalizes the input vector v."""
  return v / (tf.reduce_sum(v**2)**0.5 + eps)


def spectral_normed_weight(weights, num_iters=1, update_collection=None):
  """Performs Spectral Normalization on a weight tensor.

  Specifically, it divides the weight tensor by its largest singular value.

  Args:
    weights: The weight tensor which requires spectral normalization
    num_iters: Number of SN iterations.
    update_collection: The update collection for assigning persisted variable u.
      If None, the function will update u during the forward pass. Else if the
      update_collection equals 'NO_OPS', the function will not update the u
      during the forward pass. This is useful for the discriminator, since it
      does not update u in the second pass. Else, it will put the assignment in
      a collection defined by the user. Then the user needs to run the
      assignment explicitly.

  Returns:
    w_bar: The normalized weight tensor
    sigma: The estimated singular value for the weight tensor.
  """
  w_shape = weights.shape.as_list()
  w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
  u = tf.get_variable(
      'u', [1, w_shape[-1]],
      initializer=tf.truncated_normal_initializer(),
      trainable=False)
  u_ = u
  for _ in range(num_iters):
    v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
    u_ = _l2normalize(tf.matmul(v_, w_mat))

  sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
  w_mat = w_mat / sigma
  if update_collection is None:
    with tf.control_dependencies([u.assign(u_)]):
      w_bar = tf.reshape(w_mat, w_shape)
  else:
    w_bar = tf.reshape(w_mat, w_shape)
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_))
  return w_bar


def snconv2d(input_,
             output_dim,
             ksz=3,
             stride=2,
             sn_iters=1,
             is_training=True,
             name='snconv2d',
             padding='VALID',
             use_bias=True):
  """Creates a 2d conv-layer with Spectral Norm applied to the weights.

  Args:
    input_: 4D input tensor (batch size, height, width, channel).
    output_dim: Number of features in the output layer.
    ksz: The height of the convolutional kernel.
    stride: The width stride of the convolutional kernel.
    sn_iters: The number of SN iterations.
    is_training: If true, updates the spectral norm variables.
    name: The name of the variable scope.
    padding: Padding argument for convolutions
    use_bias: If true, learns biases

  Returns:
    conv: The normalized tensor.
  """
  if is_training:
    update_collection = None
  else:
    update_collection = 'NO_OPS'
  with tf.variable_scope(name):
    w = tf.get_variable('w', [ksz, ksz, input_.get_shape()[-1], output_dim])
    w_bar = spectral_normed_weight(
        w, num_iters=sn_iters, update_collection=update_collection)
    conv = tf.nn.conv2d(
        input_, w_bar, strides=[1, stride, stride, 1], padding=padding)
    if use_bias:
      biases = tf.get_variable(
          'biases', [output_dim], initializer=tf.zeros_initializer())
      conv = tf.nn.bias_add(conv, biases)
    return conv


def snlinear(x,
             output_size,
             bias_start=0.0,
             sn_iters=1,
             is_training=True,
             name='snlinear'):
  """Creates a linear layer with Spectral Normalization applied.

  Args:
    x: 2D input tensor (batch size, features).
    output_size: Integer number of features in output of layer.
    bias_start: Float to which bias parameters are initialized.
    sn_iters: Integer number of SN iterations.
    is_training: If true, updates the spectral norm variables.
    name: Optional, variable scope to put the layer's parameters into.

  Returns:
    The normalized output tensor of the linear layer.
  """
  if is_training:
    update_collection = None
  else:
    update_collection = 'NO_OPS'
  shape = x.get_shape().as_list()

  with tf.variable_scope(name):
    matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32)
    matrix_bar = spectral_normed_weight(
        matrix, num_iters=sn_iters, update_collection=update_collection)
    bias = tf.get_variable(
        'bias', [output_size], initializer=tf.constant_initializer(bias_start))
    out = tf.matmul(x, matrix_bar) + bias
    return out
