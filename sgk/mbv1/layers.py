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

"""Custom layers for sparse/dense inference."""
import tensorflow.compat.v1 as tf

from sgk.sparse import connectors
from sgk.sparse import ops
from sgk.sparse import sparse_matrix


class SparseConv2D(tf.keras.layers.Layer):
  """Sparse 1x1 convolution.

  NOTE: Only supports 1x1 convolutions, batch_size == 1, CHW format, unit
  stride, and no padding.
  """

  def __init__(self,
               filters,
               nonzeros,
               use_bias=False,
               activation=None,
               name=None):
    super(SparseConv2D, self).__init__(name=name)
    self.filters = filters
    self.nonzeros = nonzeros
    self.use_bias = use_bias
    self.activation = activation

  def build(self, input_shape):
    input_shape = input_shape.as_list()

    input_channels = input_shape[1]
    with tf.variable_scope(self.name, default_name="sparse_conv2d"):
      # TODO(tgale): This is a hack to make sure the sparsities
      # match exactly, not a general solution.
      sparsity = 1.0 - self.nonzeros / (self.filters * input_channels)
      self.kernel = sparse_matrix.SparseMatrix(
          "kernel", [self.filters, input_channels],
          connector=connectors.Uniform(sparsity))

      if self.use_bias:
        self.bias = tf.get_variable("bias", [self.filters])

  def call(self, inputs, training=None):
    # TODO(tgale): The following code assumes that the input channels,
    # height, and width are all defined and that the batch dimesnion
    # is undefined. Fix this to handle arbitrary input shapes correctly.
    input_shape = inputs.shape.as_list()
    flat_inputs = tf.reshape(inputs, [-1, input_shape[2] * input_shape[3]])

    output_shape = [-1, self.filters, input_shape[2], input_shape[3]]

    # Use the fused kernel if possible.
    if self.use_bias and self.activation == tf.nn.relu:
      flat_output = ops.fused_spmm(self.kernel, flat_inputs, self.bias)
      return tf.reshape(flat_output, output_shape)
    flat_output = ops.spmm(self.kernel, flat_inputs)
    out = tf.reshape(flat_output, output_shape)

    if self.use_bias:
      out = tf.nn.bias_add(out, self.bias, data_format="NCHW")
    if self.activation:
      out = self.activation(out)
    return out


class Conv2D(tf.keras.layers.Layer):
  """Fast convolution layers that uses GeMM for 1x1.

  NOTE: This layer only supports batch_size == 1, and CHW format. It does
  not support padding, and only supports strides for non-1x1 kernels.
  """

  def __init__(self,
               filters,
               use_bias=False,
               activation=None,
               kernel_size=1,
               stride=1,
               padding="SAME",
               name=None):
    super(Conv2D, self).__init__(name=name)
    self.filters = filters
    self.use_bias = use_bias
    self.activation = activation

    if kernel_size == 1:
      assert stride == 1
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

  def build(self, input_shape):
    input_shape = input_shape.as_list()
    # TODO(tgale): This doesn't work with undefined batch dimensions.
    # Figure out a better way to guard against this.
    # assert input_shape[0] == 1
    input_channels = input_shape[1]
    with tf.variable_scope(self.name, default_name="conv2d"):
      # TODO(tgale): We make the kernel the correct shape to match
      # the checkpoint and then call matmul with transpose_a=True.
      # Figure out if this or the alternative is faster.
      kernel_shape = [
          self.kernel_size, self.kernel_size, input_channels, self.filters
      ]
      self.kernel = tf.get_variable(
          "kernel", shape=kernel_shape, dtype=tf.float32)

      self.bias = None
      if self.use_bias:
        self.bias = tf.get_variable(
            "bias", shape=[self.filters], dtype=tf.float32)

  def call(self, inputs, training=None):
    if self.kernel_size == 1:
      # TODO(tgale): The following code assumes that the input channels,
      # height, and width are all defined and that the batch dimesnion
      # is undefined. Fix this to handle arbitrary input shapes correctly.
      input_shape = inputs.shape.as_list()
      flat_inputs = tf.reshape(inputs, [-1, input_shape[2] * input_shape[3]])
      flat_output = tf.matmul(
          tf.squeeze(self.kernel), flat_inputs, transpose_a=True)
      output_shape = [-1, self.filters, input_shape[2], input_shape[3]]
      output = tf.reshape(flat_output, output_shape)
    else:
      output = tf.nn.conv2d(
          inputs,
          self.kernel,
          strides=[self.stride, self.stride],
          padding=self.padding,
          data_format="NCHW")

    if self.use_bias and self.activation == tf.nn.relu:
      # Use our fused bias relu kernel if possible.
      output = ops.bias_relu(output, self.bias)
    else:
      if self.use_bias:
        output = tf.nn.bias_add(output, self.bias, data_format="NCHW")
      if self.activation:
        output = self.activation(output)
    return output


class DepthwiseConv2D(tf.keras.layers.Layer):
  """DepthwiseConv2D that supports fusion and explicit padding."""

  def __init__(self,
               kernel_size,
               strides,
               padding,
               use_bias=False,
               activation=None,
               name=None):
    super(DepthwiseConv2D, self).__init__(name=name)
    if kernel_size != 3:
      raise ValueError("DepthwiseConv2D only supports 3x3 kernels.")

    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.use_bias = use_bias
    self.activation = activation

  def build(self, input_shape):
    input_shape = input_shape.as_list()
    input_channels = input_shape[1]
    with tf.variable_scope(self.name, default_name="depthwise_conv2d"):
      self.filters = tf.get_variable(
          "depthwise_weights",
          [input_channels, self.kernel_size, self.kernel_size, 1])
      if self.use_bias:
        self.bias = tf.get_variable("bias", [input_channels])

  def call(self, inputs, training=None):
    if self.use_bias and self.activation == tf.nn.relu:
      return ops.fused_depthwise_conv2d(
          inputs,
          self.filters,
          self.bias,
          strides=self.strides,
          padding=self.padding)
    out = ops.depthwise_conv2d(
        inputs, self.filters, strides=self.strides, padding=self.padding)
    if self.use_bias:
      out = tf.nn.bias_add(out, self.bias, data_format="NCHW")
    if self.activation:
      out = self.activation(out)
    return out
