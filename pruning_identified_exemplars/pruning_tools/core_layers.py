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

# Lint as: python3

"""Contains the core layer classes for model pruning and its functional aliases.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
# pylint: enable=g-direct-tensorflow-import

MASK_COLLECTION = 'masks'
THRESHOLD_COLLECTION = 'thresholds'
MASKED_WEIGHT_COLLECTION = 'masked_weights'
WEIGHT_COLLECTION = 'kernel'
# The 'weights' part of the name is needed for the quantization library
# to recognize that the kernel should be quantized.
MASKED_WEIGHT_NAME = 'weights/masked_weight'


class _MaskedConv(base.Layer):
  """Abstract nD convolution layer."""

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(_MaskedConv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
    self.strides = utils.normalize_tuple(strides, rank, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.dilation_rate = utils.normalize_tuple(dilation_rate, rank,
                                               'dilation_rate')
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.input_spec = input_spec.InputSpec(ndim=self.rank + 2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = 1 if self.data_format == 'channels_first' else -1
    if tensor_shape.dimension_value(input_shape[channel_axis]) is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = tensor_shape.dimension_value(input_shape[channel_axis])
    kernel_shape = self.kernel_size + (input_dim, self.filters)
    self.mask = self.add_variable(
        name='mask',
        shape=kernel_shape,
        initializer=init_ops.ones_initializer(),
        trainable=False,
        dtype=self.dtype)

    self.kernel = self.add_variable(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        trainable=True,
        dtype=self.dtype)

    self.threshold = self.add_variable(
        name='threshold',
        shape=[],
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=self.dtype)

    # Add masked_weights in the weights namescope so as to make it easier
    # for the quantization library to add quant ops.
    self.masked_kernel = math_ops.multiply(self.mask, self.kernel,
                                           MASKED_WEIGHT_NAME)

    ops.add_to_collection(MASK_COLLECTION, self.mask)
    ops.add_to_collection(MASKED_WEIGHT_COLLECTION, self.masked_kernel)
    ops.add_to_collection(THRESHOLD_COLLECTION, self.threshold)
    ops.add_to_collection(WEIGHT_COLLECTION, self.kernel)

    if self.use_bias:
      self.bias = self.add_variable(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.input_spec = input_spec.InputSpec(
        ndim=self.rank + 2, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    outputs = nn.convolution(
        input=inputs,
        filter=self.masked_kernel,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format, self.rank + 2))

    if self.bias is not None:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        if self.rank == 2:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
        if self.rank == 3:
          # As of Mar 2017, direct addition is significantly slower than
          # bias_add when computing gradients. To use bias_add, we collapse Z
          # and Y into a single dimension to obtain a 4D input tensor.
          outputs_shape = outputs.shape.as_list()
          outputs_4d = array_ops.reshape(outputs, [
              outputs_shape[0], outputs_shape[1],
              outputs_shape[2] * outputs_shape[3], outputs_shape[4]
          ])
          outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
          outputs = array_ops.reshape(outputs_4d, outputs_shape)
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)


class MaskedConv2D(_MaskedConv):
  """2D convolution layer (e.g. spatial convolution over images)."""

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(MaskedConv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        **kwargs)
