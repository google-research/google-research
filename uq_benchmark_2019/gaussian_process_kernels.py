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

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


def _pad_shape_with_ones(x, ndims):
  """Maybe add `ndims` ones to `x.shape`.

  If `ndims` is zero, this is a no-op; otherwise, we will create and return a
  new `Tensor` whose shape is that of `x` with `ndims` ones concatenated on the
  right side. If the shape of `x` is known statically, the shape of the return
  value will be as well.

  Args:
    x: The `Tensor` we'll return a reshaping of.
    ndims: Python `integer` number of ones to pad onto `x.shape`.
  Returns:
    If `ndims` is zero, `x`; otherwise, a `Tensor` whose shape is that of `x`
    with `ndims` ones concatenated on the right side. If possible, returns a
    `Tensor` whose shape is known statically.
  """
  if ndims == 0:
    return x
  x = tf.convert_to_tensor(value=x)
  original_shape = x.shape
  first_shape = x.get_shape().as_list()
  new_shape = first_shape + [1] * ndims
  x = tf.reshape(x, new_shape)
  x.set_shape(original_shape.concatenate([1] * ndims))
  return x


def _sum_rightmost_ndims_preserving_shape(x, ndims):
  """Return `Tensor` with right-most ndims summed.

  Args:
    x: the `Tensor` whose right-most `ndims` dimensions to sum
    ndims: number of right-most dimensions to sum.

  Returns:
    A `Tensor` resulting from calling `reduce_sum` on the `ndims` right-most
    dimensions. If the shape of `x` is statically known, the result will also
    have statically known shape. Otherwise, the resulting shape will only be
    known at runtime.
  """
  x = tf.convert_to_tensor(value=x)
  if x.shape.ndims is not None:
    axes = tf.range(x.shape.ndims - ndims, x.shape.ndims)
  else:
    axes = tf.range(tf.rank(x) - ndims, tf.rank(x))
  return tf.reduce_sum(x, axis=axes)


class _ExponentiatedQuadratic(
    tfp.positive_semidefinite_kernels.ExponentiatedQuadratic):
  """ExponentiatedQuadratic kernel function with per-dimension parameters."""

  def _apply(self, x1, x2, example_ndims=0):
    exponent = -0.5 * _sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2) / self.length_scale,
        self.feature_ndims)

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = _pad_shape_with_ones(amplitude, example_ndims)
      exponent += 2. * tf.math.log(amplitude)

    return tf.exp(exponent)


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
               **kwargs):
    super(RBFKernelFn, self).__init__(**kwargs)
    self._per_class_kernel = per_class_kernel
    self._initial_linear_bias = initial_linear_bias
    self._initial_linear_slope = initial_linear_slope
    self._add_linear = add_linear

    with tf.compat.v1.variable_scope('kernel'):
      if self._per_class_kernel and num_classes > 1:
        amplitude_shape = (num_classes,)
        length_scale_shape = (num_classes, 1, 1, feature_size)
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
    k = _ExponentiatedQuadratic(
        amplitude=tf.nn.softplus(self._amplitude),
        length_scale=tf.nn.softplus(self._length_scale))
    if self._add_linear:
      k += tfp.positive_semidefinite_kernels.Linear(
          bias_variance=self._linear_bias,
          slope_variance=self._linear_slope)
    return k


class _MaternOneHalf(tfp.positive_semidefinite_kernels.MaternOneHalf):
  """Matern 1/2 kernel function with per-dimension parameters."""

  def _apply(self, x1, x2, example_ndims=0):
    norm = tf.sqrt(
        _sum_rightmost_ndims_preserving_shape(
            tf.math.squared_difference(x1, x2) / self.length_scale,
            self.feature_ndims))
    log_result = -norm

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = _pad_shape_with_ones(amplitude, example_ndims)
      log_result += 2. * tf.math.log(amplitude)
    return tf.exp(log_result)


class _MaternThreeHalves(tfp.positive_semidefinite_kernels.MaternThreeHalves):
  """Matern 3/2 kernel function with per-dimension parameters."""

  def _apply(self, x1, x2, example_ndims=0):
    norm = tf.sqrt(
        _sum_rightmost_ndims_preserving_shape(
            tf.math.squared_difference(x1, x2) / self.length_scale,
            self.feature_ndims))
    series_term = np.sqrt(3) * norm
    log_result = tf.math.log1p(series_term) - series_term

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = _pad_shape_with_ones(amplitude, example_ndims)
      log_result += 2. * tf.math.log(amplitude)
    return tf.exp(log_result)


class _MaternFiveHalves(tfp.positive_semidefinite_kernels.MaternFiveHalves):
  """Matern 5/2 kernel function with per-dimension parameters."""

  def _apply(self, x1, x2, example_ndims=0):
    norm = tf.sqrt(
        _sum_rightmost_ndims_preserving_shape(
            tf.math.squared_difference(x1, x2) / self.length_scale,
            self.feature_ndims))
    series_term = np.sqrt(5) * norm
    log_result = (
        tf.math.log1p(series_term + series_term ** 2 / 3.) - series_term)

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = _pad_shape_with_ones(amplitude, example_ndims)
      log_result += 2. * tf.math.log(amplitude)
    return tf.exp(log_result)


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

    with tf.compat.v1.variable_scope('kernel'):
      if self._per_class_kernel and num_classes > 1:
        amplitude_shape = (num_classes,)
        length_scale_shape = (num_classes, 1, 1, feature_size)
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
      kernel_class = _MaternOneHalf
    if self._degree == 3:
      kernel_class = _MaternThreeHalves
    if self._degree == 5:
      kernel_class = _MaternFiveHalves

    k = kernel_class(
        amplitude=tf.nn.softplus(self._amplitude),
        length_scale=tf.nn.softplus(self._length_scale))
    if self._add_linear:
      k += tfp.positive_semidefinite_kernels.Linear(
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
               **kwargs):
    super(LinearKernelFn, self).__init__(**kwargs)
    self._per_class_kernel = per_class_kernel
    self._initial_linear_bias = initial_linear_bias
    self._initial_linear_slope = initial_linear_slope

    with tf.compat.v1.variable_scope('kernel'):
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
    return tfp.positive_semidefinite_kernels.Linear(
        bias_variance=self._linear_bias,
        slope_variance=self._linear_slope)
