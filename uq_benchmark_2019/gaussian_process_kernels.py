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

"""Definitions of kernels for Gaussian Process models for UQ experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


class RBFKernelFn(tf.keras.layers.Layer):
  """ExponentiatedQuadratic kernel provider."""

  def __init__(self,
               num_classes,
               per_class_kernel,
               feature_size,
               initial_amplitude,
               initial_length_scale,
               initial_linear_bias,
               initial_linear_slope,
               add_linear=False,
               name='vgp_kernel',
               **kwargs):
    super(RBFKernelFn, self).__init__(**kwargs)
    self._per_class_kernel = per_class_kernel
    self._initial_linear_bias = initial_linear_bias
    self._initial_linear_slope = initial_linear_slope
    self._add_linear = add_linear

    with tf.compat.v1.variable_scope(name):
      if self._per_class_kernel and num_classes > 1:
        amplitude_shape = (num_classes,)
        length_scale_shape = (num_classes, feature_size)
      else:
        amplitude_shape = ()
        length_scale_shape = (feature_size,)

      self._amplitude = self.add_variable(
          initializer=tf.constant_initializer(initial_amplitude),
          shape=amplitude_shape,
          name='amplitude')

      self._length_scale = self.add_variable(
          initializer=tf.constant_initializer(initial_length_scale),
          shape=length_scale_shape,
          name='length_scale')

      if self._add_linear:
        self._linear_bias = self.add_variable(
            initializer=tf.constant_initializer(self._initial_linear_bias),
            shape=amplitude_shape,
            name='linear_bias')
        self._linear_slope = self.add_variable(
            initializer=tf.constant_initializer(self._initial_linear_slope),
            shape=amplitude_shape,
            name='linear_slope')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
    k = tfp.math.psd_kernels.FeatureScaled(
        tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(self._amplitude)),
        scale_diag=tf.math.sqrt(tf.nn.softplus(self._length_scale)))
    if self._add_linear:
      k += tfp.math.psd_kernels.Linear(
          bias_variance=self._linear_bias,
          slope_variance=self._linear_slope)
    return k


class MaternKernelFn(tf.keras.layers.Layer):
  """Matern kernel provider."""

  def __init__(self,
               num_classes,
               degree,
               per_class_kernel,
               feature_size,
               initial_amplitude,
               initial_length_scale,
               initial_linear_bias,
               initial_linear_slope,
               add_linear=False,
               name='vgp_kernel',
               **kwargs):
    super(MaternKernelFn, self).__init__(**kwargs)
    self._per_class_kernel = per_class_kernel
    self._initial_linear_bias = initial_linear_bias
    self._initial_linear_slope = initial_linear_slope
    self._add_linear = add_linear

    if degree not in [1, 3, 5]:
      raise ValueError(
          'Matern degree must be one of [1, 3, 5]: {}'.format(degree))

    self._degree = degree

    with tf.compat.v1.variable_scope(name):
      if self._per_class_kernel and num_classes > 1:
        amplitude_shape = (num_classes,)
        length_scale_shape = (num_classes, feature_size)
      else:
        amplitude_shape = ()
        length_scale_shape = (feature_size,)

      self._amplitude = self.add_variable(
          initializer=tf.constant_initializer(initial_amplitude),
          shape=amplitude_shape,
          name='amplitude')

      self._length_scale = self.add_variable(
          initializer=tf.constant_initializer(initial_length_scale),
          shape=length_scale_shape,
          name='length_scale')

      if self._add_linear:
        self._linear_bias = self.add_variable(
            initializer=tf.constant_initializer(self._initial_linear_bias),
            shape=amplitude_shape,
            name='linear_bias')
        self._linear_slope = self.add_variable(
            initializer=tf.constant_initializer(self._initial_linear_slope),
            shape=amplitude_shape,
            name='linear_slope')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
    if self._degree == 1:
      kernel_class = tfp.math.psd_kernels.MaternOneHalf
    if self._degree == 3:
      kernel_class = tfp.math.psd_kernels.MaternThreeHalves
    if self._degree == 5:
      kernel_class = tfp.math.psd_kernels.MaternFiveHalves

    k = tfp.math.psd_kernels.FeatureScaled(
        kernel_class(amplitude=tf.nn.softplus(self._amplitude)),
        scale_diag=tf.math.sqrt(tf.nn.softplus(self._length_scale)))
    if self._add_linear:
      k += tfp.math.psd_kernels.Linear(
          bias_variance=self._linear_bias,
          slope_variance=self._linear_slope)
    return k


class LinearKernelFn(tf.keras.layers.Layer):
  """Matern kernel provider."""

  def __init__(self,
               num_classes,
               per_class_kernel,
               initial_linear_bias,
               initial_linear_slope,
               name='vgp_kernel',
               **kwargs):
    super(LinearKernelFn, self).__init__(**kwargs)
    self._per_class_kernel = per_class_kernel
    self._initial_linear_bias = initial_linear_bias
    self._initial_linear_slope = initial_linear_slope

    with tf.compat.v1.variable_scope(name):
      if self._per_class_kernel and num_classes > 1:
        shape = (num_classes,)
      else:
        shape = ()

      self._linear_bias = self.add_variable(
          initializer=tf.constant_initializer(self._initial_linear_bias),
          shape=shape,
          name='linear_bias')
      self._linear_slope = self.add_variable(
          initializer=tf.constant_initializer(self._initial_linear_slope),
          shape=shape,
          name='linear_slope')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
    return tfp.math.psd_kernels.Linear(
        bias_variance=self._linear_bias,
        slope_variance=self._linear_slope)
