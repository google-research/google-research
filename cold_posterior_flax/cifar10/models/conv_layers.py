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

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Variants of Conv Modules."""
from flax.deprecated import nn
from flax.deprecated.nn import initializers
from jax import lax
import jax.numpy as jnp

import numpy as onp


class ConvFixedScale(nn.Module):
  """Convolutional layer that uses Weight Norm [1] with the scale fixed at 1.

  [1] Salimans, T., & Kingma, D. P. (2016, February 25). Weight Normalization:
  A Simple Reparameterization to Accelerate Training of Deep Neural Networks.
  arXiv [cs.LG]. http://arxiv.org/abs/1602.07868.
  """

  def apply(self,
            inputs,
            features,
            kernel_size,
            strides=None,
            padding='SAME',
            lhs_dilation=None,
            rhs_dilation=None,
            feature_group_count=1,
            bias=True,
            dtype=jnp.float32,
            precision=None,
            kernel_init=nn.linear.default_kernel_init,
            bias_init=initializers.zeros,
            compensate_padding=True):
    """Applies a convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: a sequence of `n` integers, representing the inter-window
        strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      lhs_dilation: `None`, or a sequence of `n` integers, giving the dilation
        factor to apply in each spatial dimension of `lhs`. LHS dilation is also
        known as transposed convolution.
      rhs_dilation: `None`, or a sequence of `n` integers, giving the dilation
        factor to apply in each spatial dimension of `rhs`. RHS dilation is also
        known as atrous convolution.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
      compensate_padding: Renormalize output based on introduced zero padding.

    Returns:
      The convolved data.
    """

    inputs = jnp.asarray(inputs, dtype)

    if strides is None:
      strides = (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    assert in_features % feature_group_count == 0
    kernel_shape = kernel_size + (in_features // feature_group_count, features)
    kernel_unnorm = self.param('kernel', kernel_shape, kernel_init)
    kernel_unnorm = jnp.asarray(kernel_unnorm, dtype)
    kernel_unnorm = jnp.reshape(
        kernel_unnorm,
        (-1, features),
    )
    kernel = kernel_unnorm / (
        jnp.linalg.norm(kernel_unnorm, axis=0, keepdims=True) + 1e-5)
    kernel = jnp.reshape(kernel, kernel_shape)

    # pylint: disable=protected-access
    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)
    # pylint: enable=protected-access
    y = lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision)

    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
      y = y + bias

    if compensate_padding:
      y = padding_compensate(inputs, kernel_size, lhs_dilation, padding,
                             precision, rhs_dilation, strides, y)
    return y


class ConvLearnedScale(nn.Module):
  """Convolutional layer that uses Weight Norm [1].

  [1] Salimans, T., & Kingma, D. P. (2016, February 25). Weight Normalization:
  A Simple Reparameterization to Accelerate Training of Deep Neural Networks.
  arXiv [cs.LG]. http://arxiv.org/abs/1602.07868.

  Convolution Module wrapping lax.conv_general_dilated.
  """

  def apply(self,
            inputs,
            features,
            kernel_size,
            strides=None,
            padding='SAME',
            lhs_dilation=None,
            rhs_dilation=None,
            feature_group_count=1,
            bias=True,
            dtype=jnp.float32,
            precision=None,
            kernel_init=nn.linear.default_kernel_init,
            bias_init=initializers.zeros,
            scale_init=initializers.ones,
            compensate_padding=True):
    """Applies a convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: a sequence of `n` integers, representing the inter-window
        strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      lhs_dilation: `None`, or a sequence of `n` integers, giving the dilation
        factor to apply in each spatial dimension of `lhs`. LHS dilation is also
        known as transposed convolution.
      rhs_dilation: `None`, or a sequence of `n` integers, giving the dilation
        factor to apply in each spatial dimension of `rhs`. RHS dilation is also
        known as atrous convolution.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
      scale_init: initializer for the scale.
      compensate_padding: Renormalize output based on introduced zero padding.

    Returns:
      The convolved data.
    """

    inputs = jnp.asarray(inputs, dtype)

    if strides is None:
      strides = (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    assert in_features % feature_group_count == 0
    kernel_shape = kernel_size + (in_features // feature_group_count, features)
    kernel_unnorm = self.param('kernel', kernel_shape, kernel_init)
    kernel_unnorm = jnp.asarray(kernel_unnorm, dtype)
    kernel_unnorm = jnp.reshape(
        kernel_unnorm,
        (-1, features),
    )
    kernel = kernel_unnorm / (
        jnp.linalg.norm(kernel_unnorm, axis=0, keepdims=True) + 1e-5)

    scale = self.param('scale', (features,), scale_init)
    kernel *= scale.reshape((-1, features))
    kernel = jnp.reshape(kernel, kernel_shape)

    # pylint: disable=protected-access
    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)
    # pylint: enable=protected-access
    y = lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision)

    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
      y = y + bias

    if compensate_padding:
      y = padding_compensate(inputs, kernel_size, lhs_dilation, padding,
                             precision, rhs_dilation, strides, y)
    return y


class Conv(nn.Module):
  """Plain Convolution Module that supports padding compensation."""

  def apply(self,
            inputs,
            features,
            kernel_size,
            strides=None,
            padding='SAME',
            input_dilation=None,
            kernel_dilation=None,
            feature_group_count=1,
            bias=True,
            dtype=jnp.float32,
            precision=None,
            kernel_init=nn.linear.default_kernel_init,
            bias_init=initializers.zeros,
            compensate_padding=True):
    """Applies a convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: a sequence of `n` integers, representing the inter-window
        strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      input_dilation: `None`, or a sequence of `n` integers, giving the dilation
        factor to apply in each spatial dimension of `inputs`. Convolution with
        input dilation `d` is equivalent to transposed convolution with stride
        `d`.
      kernel_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel. Convolution with kernel dilation is also known as 'atrous
        convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
      compensate_padding: Renormalize output based on introduced zero padding.

    Returns:
      The convolved data.
    """

    inputs = jnp.asarray(inputs, dtype)

    if strides is None:
      strides = (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    assert in_features % feature_group_count == 0
    kernel_shape = kernel_size + (in_features // feature_group_count, features)
    kernel = self.param('kernel', kernel_shape, kernel_init)
    kernel = jnp.asarray(kernel, dtype)

    # pylint: disable=protected-access
    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)
    # pylint: enable=protected-access
    y = lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision)

    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
      y = y + bias

    if compensate_padding:
      y = padding_compensate(inputs, kernel_size, input_dilation, padding,
                             precision, kernel_dilation, strides, y)
    return y


def padding_compensate(inputs, kernel_size, lhs_dilation, padding, precision,
                       rhs_dilation, strides, y):
  """Divide inputs by the expected reduction in std-dev induced by zero padding."""
  if padding != 'VALID':
    # Figure out input count by conv:
    ones = jnp.ones((1, inputs.shape[1], inputs.shape[2], 1))
    ones_kernel = jnp.ones(kernel_size + (1, 1))
    count = lax.conv_general_dilated(
        ones,
        ones_kernel,
        strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        # pylint: disable=protected-access
        dimension_numbers=nn.linear._conv_dimension_numbers(ones.shape),
        # pylint: enable=protected-access
        feature_group_count=1,
        precision=precision)
    var = count / (onp.prod(kernel_size))
    var_avg = jnp.mean(var)
    std_var = jnp.sqrt(var_avg)
    y /= std_var
  return y


class ConvWS(nn.Module):
  """Convolution Module using weight standardization [1].

  - [1] Qiao, S., Wang, H., Liu, C., Shen, W., & Yuille, A. (2019, March 25).
  Micro-Batch Training with Batch-Channel Normalization and Weight
  Standardization. arXiv [cs.CV]. http://arxiv.org/abs/1903.10520.
  """

  def apply(self,
            inputs,
            features,
            kernel_size,
            strides=None,
            padding='SAME',
            lhs_dilation=None,
            rhs_dilation=None,
            feature_group_count=1,
            bias=True,
            dtype=jnp.float32,
            precision=None,
            kernel_init=nn.linear.default_kernel_init,
            bias_init=initializers.zeros,
            kaiming_scaling=True,
            compensate_padding=True):
    """Applies a convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: a sequence of `n` integers, representing the inter-window
        strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      lhs_dilation: `None`, or a sequence of `n` integers, giving the dilation
        factor to apply in each spatial dimension of `lhs`. LHS dilation is also
        known as transposed convolution.
      rhs_dilation: `None`, or a sequence of `n` integers, giving the dilation
        factor to apply in each spatial dimension of `rhs`. RHS dilation is also
        known as atrous convolution.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
      kaiming_scaling: Scale kernel according to Kaiming initialization scaling.
      compensate_padding: Renormalize output based on introduced zero padding.

    Returns:
      The convolved data.
    """

    inputs = jnp.asarray(inputs, dtype)

    if strides is None:
      strides = (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    assert in_features % feature_group_count == 0
    input_channels = in_features // feature_group_count
    kernel_shape = kernel_size + (input_channels, features)
    kernel_unnorm = self.param('kernel', kernel_shape, kernel_init)
    kernel_unnorm = jnp.asarray(kernel_unnorm, dtype)
    # Normalize mean.
    kernel = kernel_unnorm - jnp.mean(
        kernel_unnorm, keepdims=True, axis=[0, 1, 2])
    # Normalize stdev.
    std_estimate = (
        jnp.sqrt(jnp.mean(kernel**2, keepdims=True, axis=[0, 1, 2]) + 1e-5))
    # Sample estimate compensation:
    kernel = kernel / std_estimate
    # Normalize by number of inputs:
    if kaiming_scaling:
      kernel = kernel / jnp.sqrt(int(input_channels * onp.prod(kernel_size)))

    # pylint: disable=protected-access
    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)
    # pylint: enable=protected-access
    y = lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision)

    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
      y = y + bias

    if compensate_padding:
      y = padding_compensate(inputs, kernel_size, lhs_dilation, padding,
                             precision, rhs_dilation, strides, y)
    return y
