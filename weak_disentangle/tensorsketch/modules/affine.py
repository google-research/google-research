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

"""Affine modules.
"""

# pylint: disable=g-bad-import-order, g-importing-member
import numpy as np
import tensorflow.compat.v1 as tf
from collections import OrderedDict

from weak_disentangle.tensorsketch import utils as tsu
from weak_disentangle.tensorsketch.modules.base import build_with_name_scope
from weak_disentangle.tensorsketch.modules.base import Module


class Affine(Module):
  """Abstract class for modules that apply an affine transformation to input.

  Affine includes several special functionalities to ensure that classes that
  extend it are amenable to the injection of kernel normalizers (based on the
  respects_kernel_norm flag). All classes that extend Affine should adhere to
  the following contract: Never access self.orig_kernel directly in forward
  call, and parameter initialization/building.
  """

  def __init__(self, bias=True, name=None, initializer=None):
    super().__init__(name=name)
    self.use_bias = bias
    self.kernel = None
    self.bias = None
    self.initializer = initializer
    self.kernel_normalizers = OrderedDict()

  @property
  def normalized_kernel(self):
    kernel = self.kernel
    for km in self.kernel_normalizers.values():
      kernel = km(kernel)
    return kernel

  @build_with_name_scope
  def build_parameters(self, x):
    raise NotImplementedError("Implement parameter building for Affine class")

  def reset_parameters(self):
    if self.initializer is not None:
      self.initializer(self.kernel, self.bias)
      return

    # By default, all affine layers are initialized via
    # Unif(-a, a), where a = sqrt(1 / fan_in)
    fan_in, _ = tsu.compute_fan(self.kernel)
    limit = np.sqrt(1 / fan_in)
    self.kernel.assign(tf.random.uniform(self.kernel.shape, -limit, limit))

    if self.use_bias:
      self.bias.assign(tf.random.uniform(self.bias.shape, -limit, limit))


class Dense(Affine):
  """Applies a dense affine transformation to input.
  """

  def __init__(self, out_dims, bias=True, initializer=None, name=None):
    super().__init__(bias=bias, initializer=initializer, name=name)
    self.out_dims = out_dims

  @build_with_name_scope
  def build_parameters(self, x):
    self.in_dims = int(x.shape[-1])
    self.kernel = tf.Variable(tf.random.normal((self.in_dims, self.out_dims)),
                              trainable=True)

    if self.use_bias:
      self.bias = tf.Variable(tf.random.normal([self.out_dims]), trainable=True)

    self.reset_parameters()

  def forward(self, x):
    x = tf.matmul(x, self.normalized_kernel)

    if self.bias is not None:
      x = tf.nn.bias_add(x, self.bias)
    return x

  def extra_repr(self):
    return "({}, bias={})".format(self.out_dims, self.use_bias)


class Conv2d(Affine):
  """Applies 2d convolutional transformation (and bias) to input.
  """

  def __init__(self,
               out_channels,
               kernel_size,
               strides,
               padding="same",
               dilation=1,
               bias=True,
               initializer=None,
               name=None):
    super().__init__(bias=bias, initializer=initializer, name=name)
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.dilation = dilation

  @build_with_name_scope
  def build_parameters(self, x):
    self.in_channels = int(x.shape[-1])
    self.kernel = tf.Variable(tf.random.normal((self.kernel_size,
                                                self.kernel_size,
                                                self.in_channels,
                                                self.out_channels)),
                              trainable=True)

    if self.use_bias:
      self.bias = tf.Variable(tf.random.normal([self.out_channels]),
                              trainable=True)
    self.reset_parameters()

  def forward(self, x):
    x = tf.nn.conv2d(
        x, filter=self.normalized_kernel,
        strides=self.strides,
        padding=self.padding.upper(),
        dilations=self.dilation)

    if self.use_bias:
      x = tf.nn.bias_add(x, self.bias)
    return x

  def extra_repr(self):
    return "({}, {}, {}, {}, bias={})".format(self.out_channels,
                                              self.kernel_size,
                                              self.strides,
                                              self.padding,
                                              self.use_bias)


class ConvTranspose2d(Affine):
  """Applies 2d transposed convolutional transformation (and bias) to input.
  """

  def __init__(self,
               out_channels,
               kernel_size,
               strides,
               padding="same",
               output_padding=None,
               dilation=1,
               bias=True,
               initializer=None,
               name=None):
    super().__init__(bias=bias, initializer=initializer, name=name)
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.output_padding = output_padding
    self.dilation = dilation

  @build_with_name_scope
  def build_parameters(self, x):
    self.in_channels = int(x.shape[-1])
    self.kernel = tf.Variable(tf.random.normal((self.kernel_size,
                                                self.kernel_size,
                                                self.out_channels,
                                                self.in_channels)),
                              trainable=True)

    if self.use_bias:
      self.bias = tf.Variable(tf.random.normal([self.out_channels]),
                              trainable=True)
    self.reset_parameters()

  def forward(self, x):
    n, h, w, _ = x.shape
    h = tsu.compute_out_dims(h, self.kernel_size,
                             self.strides,
                             self.padding,
                             self.output_padding,
                             self.dilation)

    w = tsu.compute_out_dims(w, self.kernel_size,
                             self.strides,
                             self.padding,
                             self.output_padding,
                             self.dilation)
    output_shape = (n, h, w, self.out_channels)

    x = tf.nn.conv2d_transpose(
        x, filter=self.normalized_kernel,
        strides=self.strides,
        padding=self.padding.upper(),
        output_shape=output_shape,
        dilations=self.dilation)

    if self.use_bias:
      x = tf.nn.bias_add(x, self.bias)
    return x

  def extra_repr(self):
    return "({}, {}, {}, {}, bias={})".format(self.out_channels,
                                              self.kernel_size,
                                              self.strides,
                                              self.padding,
                                              self.use_bias)
