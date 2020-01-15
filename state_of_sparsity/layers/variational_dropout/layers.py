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

"""Defines an implementation of tensorflow core layers with vd pruning.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from state_of_sparsity.layers.utils import layer_utils
from state_of_sparsity.layers.variational_dropout import common
from state_of_sparsity.layers.variational_dropout import nn
from tensorflow.python.layers import base  # pylint: disable=g-direct-tensorflow-import

THETA_LOGSIGMA2_COLLECTION = "theta_logsigma2"


class Conv2D(base.Layer):
  r"""Base implementation of a conv2d layer with variational dropout.

   Instead of deterministic parameters, parameters are drawn from a
   distribution with mean \theta and variance \sigma^2.  A log-uniform prior
   for the distribution is used to encourage sparsity.

    Args:
      x: Input, float32 tensor.
      num_outputs: Int representing size of output tensor.
      kernel_size: The size of the convolutional window, int of list of ints.
      strides: stride length of convolution, a single int is expected.
      padding: May be populated as `"VALID"` or `"SAME"`.
      activation: If None, a linear activation is used.
      kernel_initializer: Initializer for the convolution weights.
      bias_initializer: Initalizer of the bias vector.
      kernel_regularizer: Regularization method for the convolution weights.
      bias_regularizer: Optional regularizer for the bias vector.
      log_sigma2_initializer: Specified initializer of the log_sigma2 term.
      data_format: Either'"channels_last"','"NHWC"','"NCHW"','"channels_first".
      is_training: Boolean specifying whether it is training or eval.
      use_bias: Boolean specifying whether bias vector should be used.
      eps: Small epsilon value to prevent math op saturation.
      threshold: Threshold for masking log alpha at test time. The relationship
        between \sigma^2, \theta, and \alpha as defined in the
        paper https://arxiv.org/abs/1701.05369 is \sigma^2 = \alpha * \theta^2
      clip_alpha: Int that specifies range for clipping log alpha values during
        training.
      name: String speciying name scope of layer in network.

    Returns:
      Output Tensor of the conv2d operation.
  """

  def __init__(self,
               num_outputs,
               kernel_size,
               strides,
               padding,
               activation,
               kernel_initializer,
               bias_initializer,
               kernel_regularizer,
               bias_regularizer,
               log_sigma2_initializer,
               data_format,
               activity_regularizer=None,
               is_training=True,
               trainable=True,
               use_bias=False,
               eps=common.EPSILON,
               threshold=3.,
               clip_alpha=8.,
               name="",
               **kwargs):
    super(Conv2D, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.num_outputs = num_outputs
    self.kernel_size = kernel_size
    self.strides = [1, strides[0], strides[1], 1]
    self.padding = padding.upper()
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.log_sigma2_initializer = log_sigma2_initializer
    self.data_format = layer_utils.standardize_data_format(data_format)
    self.is_training = is_training
    self.use_bias = use_bias
    self.eps = eps
    self.threshold = threshold
    self.clip_alpha = clip_alpha

  def build(self, input_shape):
    input_shape = input_shape.as_list()
    dims = input_shape[3]
    kernel_shape = [
        self.kernel_size[0], self.kernel_size[1], dims, self.num_outputs
    ]

    self.kernel = tf.get_variable(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        dtype=tf.float32,
        trainable=True)

    if not self.log_sigma2_initializer:
      self.log_sigma2_initializer = tf.constant_initializer(
          value=-10, dtype=tf.float32)

    self.log_sigma2 = tf.get_variable(
        "log_sigma2",
        shape=kernel_shape,
        initializer=self.log_sigma2_initializer,
        dtype=tf.float32,
        trainable=True)

    layer_utils.add_variable_to_collection(
        (self.kernel, self.log_sigma2),
        [THETA_LOGSIGMA2_COLLECTION],
        None)

    if self.use_bias:
      self.bias = self.add_variable(
          name="bias",
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):

    if self.is_training:
      output = nn.conv2d_train(
          x=inputs,
          variational_params=(self.kernel, self.log_sigma2),
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          clip_alpha=self.clip_alpha,
          eps=self.eps)
    else:
      output = nn.conv2d_eval(
          x=inputs,
          variational_params=(self.kernel, self.log_sigma2),
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          threshold=self.threshold,
          eps=self.eps)

    if self.use_bias:
      output = tf.nn.bias_add(output, self.bias)
    if self.activation is not None:
      return self.activation(output)
    else:
      return output


class FullyConnected(base.Layer):
  r"""Base implementation of a fully connected layer with variational dropout.

   Instead of deterministic parameters, parameters are drawn from a
   distribution with mean \theta and variance \sigma^2.  A log-uniform prior
   for the distribution is used to encourage sparsity.

    Args:
      x: Input, float32 tensor.
      num_outputs: Int representing size of output tensor.
      activation: If None, a linear activation is used.
      kernel_initializer: Initializer for the convolution weights.
      bias_initializer: Initalizer of the bias vector.
      kernel_regularizer: Regularization method for the convolution weights.
      bias_regularizer: Optional regularizer for the bias vector.
      log_sigma2_initializer: Specified initializer of the log_sigma2 term.
      is_training: Boolean specifying whether it is training or eval.
      use_bias: Boolean specifying whether bias vector should be used.
      eps: Small epsilon value to prevent math op saturation.
      threshold: Threshold for masking log alpha at test time. The relationship
        between \sigma^2, \theta, and \alpha as defined in the
        paper https://arxiv.org/abs/1701.05369 is \sigma^2 = \alpha * \theta^2
      clip_alpha: Int that specifies range for clipping log alpha values during
        training.
      name: String speciying name scope of layer in network.

    Returns:
      Output Tensor of the fully connected operation.
  """

  def __init__(self,
               num_outputs,
               activation,
               kernel_initializer,
               bias_initializer,
               kernel_regularizer,
               bias_regularizer,
               log_sigma2_initializer,
               activity_regularizer=None,
               is_training=True,
               trainable=True,
               use_bias=True,
               eps=common.EPSILON,
               threshold=3.,
               clip_alpha=8.,
               name="FullyConnected",
               **kwargs):
    super(FullyConnected, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.log_sigma2_initializer = log_sigma2_initializer
    self.is_training = is_training
    self.use_bias = use_bias
    self.eps = eps
    self.threshold = threshold
    self.clip_alpha = clip_alpha

  def build(self, input_shape):
    input_shape = input_shape.as_list()
    input_hidden_size = input_shape[1]
    kernel_shape = [input_hidden_size, self.num_outputs]

    self.kernel = tf.get_variable(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        dtype=tf.float32,
        trainable=True)

    if not self.log_sigma2_initializer:
      self.log_sigma2_initializer = tf.constant_initializer(
          value=-10, dtype=tf.float32)

    self.log_sigma2 = tf.get_variable(
        "log_sigma2",
        shape=kernel_shape,
        initializer=self.log_sigma2_initializer,
        dtype=tf.float32,
        trainable=True)

    layer_utils.add_variable_to_collection(
        (self.kernel, self.log_sigma2),
        [THETA_LOGSIGMA2_COLLECTION],
        None)

    if self.use_bias:
      self.bias = self.add_variable(
          name="bias",
          shape=(self.num_outputs,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    if self.is_training:
      x = nn.matmul_train(
          inputs, (self.kernel, self.log_sigma2), clip_alpha=self.clip_alpha)
    else:
      x = nn.matmul_eval(
          inputs, (self.kernel, self.log_sigma2), threshold=self.threshold)

    if self.use_bias:
      x = tf.nn.bias_add(x, self.bias)
    if self.activation is not None:
      return self.activation(x)
    return x
