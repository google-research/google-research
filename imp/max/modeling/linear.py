# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Linear layers with SPMD-friendly named axes params."""

import string
from typing import Callable, Sequence

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from imp.max.utils import sharding
from imp.max.utils import typing


def _normalize_axes(axes, ndim):
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Sequence):
    return tuple(x)
  else:
    return (x,)


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def canonicalize_padding(
    padding,
    rank,
):
  """Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding, lambda x, y, z: x
  if isinstance(padding, int):
    return [(padding, padding)] * rank, lambda x, y, z: x
  if isinstance(padding, Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad, lambda x, y, z: x
  if isinstance(padding, Callable):
    return 'VALID', padding
  raise ValueError(
      f'Invalid padding format: {padding}, should be str, int,'
      f' or a sequence of len {rank} where each element is an'
      ' int or pair of ints.'
  )


def _apply_lora_kernels(
    module,
    kernel,
    kernel_shardings,
    features,
    rank,
    scale,
    dtype,
    param_dtype = jnp.float32,
    dot_general = lax.dot_general,
):
  """Adds the LoRA kernels to the main kernel.

  Args:
    module: The parent module in which this method is being called.
    kernel: The dense layer's kernel to be transformed.
    kernel_shardings: sharding axes for the kernel.
    features: the number of output features.
    rank: the maximum rank of LoRA approximation.
    scale: it controls the scale of LoRA wights to be added.
      * lora_scale = 1.0 means fully add LoRA weights.
      * lora_scale = 0.0 means add no LoRA weights.
    dtype: the dtype of the computation.
    param_dtype: the dtype of the kernels.
    dot_general: the function that performs dot product between the weights
      and the inputs.

  Returns:
    The transformed kernel as: original_kernel + low_rank_kernel.
  """
  if rank <= 0:
    raise ValueError('The LoRA kernel rank should be a positive integer. '
                     f'Instead, received {rank=}.')
  features = _canonicalize_tuple(features)
  einsum_str = string.ascii_lowercase
  if len(features) > len(einsum_str):
    raise NotImplementedError(
        f'Total number of features={len(features)} cannot exceed '
        f'{len(einsum_str)}.')
  feat_einsum = einsum_str[:len(features)]
  kernel_left_shape = tuple(list(kernel.shape[:-len(features)]) + [rank])
  kernel_right_shape = tuple([rank] + list(features))
  if kernel_shardings:
    kernel_left_shardings = tuple(
        kernel_shardings[:len(kernel_left_shape)-1]) + (None,)
    kernel_right_shardings = (None,) + tuple(
        kernel_shardings[-(len(kernel_right_shape)-1):])
  else:
    kernel_left_shardings = ()
    kernel_right_shardings = ()

  kernel_left_init = sharding.modulate_param_init(
      nn.initializers.normal(), kernel_left_shardings)
  kernel_right_init = sharding.modulate_param_init(
      nn.initializers.zeros_init(), kernel_right_shardings)
  kernel_left = module.param(
      name='kernel_left',
      init_fn=kernel_left_init,
      shape=kernel_left_shape,
      dtype=param_dtype,
      unbox=True,
  )
  kernel_right = module.param(
      name='kernel_right',
      init_fn=kernel_right_init,
      shape=kernel_right_shape,
      dtype=param_dtype,
      unbox=True,
  )
  kernel_left = jnp.asarray(kernel_left, dtype)
  kernel_right = jnp.asarray(kernel_right, dtype)
  kernel_delta = jnp.einsum(
      f'...z, z{feat_einsum}->...{feat_einsum}',
      kernel_left, kernel_right,
      _dot_general=dot_general)
  kernel = jnp.asarray(kernel, dtype)
  kernel = kernel + scale * kernel_delta
  return kernel


class Dense(nn.Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    kernel_shardings: sharding annotations to use for kernel sharding.
    dot_general: the function that performs dot product between the weights
      and the inputs.
    lora_rank: the maximum rank of LoRA approximation.
    lora_scale: it controls the scale of LoRA wights to be added.
      * lora_scale = 1.0 means fully add LoRA weights.
      * lora_scale = 0.0 means add no LoRA weights.
  """
  features: int
  use_bias: bool = True
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32
  precision: typing.Precision = None
  kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
  bias_init: nn.initializers.Initializer = nn.initializers.zeros
  kernel_shardings: typing.ShardingAxes = ()
  dot_general: typing.DotGeneral = lax.dot_general
  lora_rank: int = 4
  lora_scale: float = 0.

  @nn.compact
  def __call__(self, inputs):
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel_init = sharding.modulate_param_init(
        self.kernel_init, self.kernel_shardings)
    kernel = self.param(
        name='kernel',
        init_fn=kernel_init,
        shape=(inputs.shape[-1], self.features),
        dtype=self.param_dtype,
        unbox=True,
    )
    kernel = jnp.asarray(kernel, self.dtype)

    if self.lora_scale > 0.:
      kernel = _apply_lora_kernels(
          module=self,
          kernel=kernel,
          kernel_shardings=self.kernel_shardings,
          features=self.features,
          rank=self.lora_rank,
          scale=self.lora_scale,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          dot_general=self.dot_general,
      )

    y = self.dot_general(inputs, kernel,
                         (((inputs.ndim - 1,), (0,)), ((), ())),
                         precision=self.precision)
    if self.use_bias:
      bias_shardings = (
          self.kernel_shardings[-1:] if self.kernel_shardings else ()
      )
      bias_init = sharding.modulate_param_init(self.bias_init, bias_shardings)
      bias = self.param(
          name='bias',
          init_fn=bias_init,
          shape=(self.features,),
          dtype=self.param_dtype,
          unbox=True,
      )
      bias = jnp.asarray(bias, self.dtype)
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    use_bias: whether to add a bias to the output (default: False).
    axis: tuple with axes to apply the transformation on.
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_shardings: sharding annotations to use for kernel sharding.
    dot_general: the function that performs dot product between the weights
      and the inputs.
    lora_rank: the maximum rank of LoRA approximation.
    lora_scale: it controls the scale of LoRA wights to be added.
      * lora_scale = 1.0 means fully add LoRA weights.
      * lora_scale = 0.0 means add no LoRA weights.
  """
  features: typing.Axes
  axis: typing.Axes = -1
  use_bias: bool = True
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32
  kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
  bias_init: nn.initializers.Initializer = nn.initializers.zeros
  precision: typing.Precision = None
  kernel_shardings: typing.ShardingAxes = ()
  dot_general: typing.DotGeneral = lax.dot_general
  lora_rank: int = 4
  lora_scale: float = 0.

  @nn.compact
  def __call__(self, inputs):
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features

    if self.kernel_shardings:
      if len(self.kernel_shardings) != len(kernel_shape):
        raise ValueError(
            f"Kernel sharding axes {self.kernel_shardings} does not match "
            f'kernel shape {kernel_shape}.'
        )

    kernel_init = sharding.modulate_param_init(
        self.kernel_init, self.kernel_shardings)
    kernel = self.param(
        name='kernel',
        init_fn=kernel_init,
        shape=kernel_shape,
        dtype=self.param_dtype,
        unbox=True,
    )
    kernel = jnp.asarray(kernel, self.dtype)

    if self.lora_scale > 0.:
      kernel = _apply_lora_kernels(
          module=self,
          kernel=kernel,
          kernel_shardings=self.kernel_shardings,
          features=self.features,
          rank=self.lora_rank,
          scale=self.lora_scale,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          dot_general=self.dot_general,
      )

    contract_ind = tuple(range(0, len(axis)))
    out = self.dot_general(
        inputs,
        kernel, ((axis, contract_ind), ((), ())),
        precision=self.precision
    )

    if self.use_bias:
      bias_shape = features
      bias_shardings = self.kernel_shardings[-len(features):]
      if bias_shardings:
        if len(bias_shardings) != len(bias_shape):
          raise ValueError(
              f"Bias sharding axes {bias_shardings} does not match "
              f'bias shape {bias_shape}.'
          )

      bias_init = sharding.modulate_param_init(self.bias_init, bias_shardings)
      bias = self.param(
          name='bias',
          init_fn=bias_init,
          shape=bias_shape,
          dtype=self.param_dtype,
          unbox=True,
      )
      bias = jnp.asarray(bias, self.dtype)

      # Expand bias for broadcast.
      expand_dims = tuple(range(out.ndim - bias.ndim))
      bias = jnp.expand_dims(bias, expand_dims)

      out = out + bias

    return out


class _Conv(nn.Module):
  """Convolution Module wrapping `lax.conv_general_dilated[_local]`.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it must
      be a sequence of integers.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpreted as applying the same
      padding in all dims and assign a single int in a sequence causes the same
      padding to be used on both sides. `'CAUSAL'` padding for a 1D convolution
      will left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `inputs`
      (default: 1). Convolution with input dilation `d` is equivalent to
      transposed convolution with stride `d`.
    kernel_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    kernel_shardings: sharding annotations to use for kernel sharding.
    bias_shardings: sharding annotations to use for bias sharding.
    conv_general_dilated: the function that performs a generic dilated
      convolution.
    conv_general_dilated_local: the function that performs the local dilated
      convolution.
  """

  features: int
  kernel_size: Sequence[int]
  strides: typing.Axes | None = 1
  padding: typing.PaddingLike = 'SAME'
  input_dilation: typing.Axes | None = 1
  kernel_dilation: typing.Axes | None = 1
  feature_group_count: int = 1
  use_bias: bool = True
  mask: jax.Array | None = None
  dtype: jax.typing.DTypeLike | None = None
  param_dtype: jax.typing.DTypeLike = jnp.float32
  precision: typing.Precision = None
  kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
  bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
  kernel_shardings: typing.ShardingAxes = ()
  bias_shardings: typing.ShardingAxes = ()
  conv_general_dilated: typing.ConvGeneralDilated = lax.conv_general_dilated
  conv_general_dilated_local: typing.ConvGeneralDilated = (
      lax.conv_general_dilated_local
  )

  @property
  def shared_weights(self):  # type: ignore
    """Defines whether weights are shared or not between different pixels.

    Returns:
      `True` to use shared weights in convolution (regular convolution).
      `False` to use different weights at different pixels, a.k.a.
      'locally connected layer', 'unshared convolution', or 'local convolution'.

    """
    Ellipsis  # pytype: disable=bad-return-type

  @nn.compact
  def __call__(self, inputs):
    """Applies a (potentially unshared) convolution to the inputs.

    Args:
      inputs: input data with dimensions (*batch_dims, spatial_dims...,
        features). This is the channels-last convention, i.e. NHWC for a 2d
        convolution and NDHWC for a 3D convolution. Note that this is different
        from the input convention used by `lax.conv_general_dilated`, which puts
        the spatial dimensions last.
        If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """

    if isinstance(self.kernel_size, int):
      raise TypeError(
          'Expected Conv kernel_size to be a'
          ' tuple/list of integers (eg.: [3, 3]) but got'
          f' {self.kernel_size}.'
      )
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x):
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
          num_batch_dimensions:
      ]
      inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    # Perform pre-conv padding (if function provided)
    padding_lax, pre_conv_padding = canonicalize_padding(
        self.padding, len(kernel_size))
    inputs = pre_conv_padding(inputs, kernel_size, kernel_dilation)

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    in_features = jnp.shape(inputs)[-1]

    if self.shared_weights:
      # One shared convolutional kernel for all pixels in the output.
      assert in_features % self.feature_group_count == 0
      kernel_shape = kernel_size + (
          in_features // self.feature_group_count,
          self.features,
      )

    else:
      if self.feature_group_count != 1:
        raise NotImplementedError(
            '`lax.conv_general_dilated_local` does not support '
            f'`feature_group_count != 1`, got `{self.feature_group_count}`.'
        )

      # Need to know the spatial output shape of a standard convolution to
      # create the unshared convolution kernel.
      conv_output_shape = jax.eval_shape(
          lambda lhs, rhs: self.conv_general_dilated(  # pylint: disable=g-long-lambda
              lhs=lhs,
              rhs=rhs,
              window_strides=strides,
              padding=padding_lax,
              dimension_numbers=dimension_numbers,
              lhs_dilation=input_dilation,
              rhs_dilation=kernel_dilation,
          ),
          inputs,
          jax.core.ShapedArray(kernel_size + (in_features, self.features),
                               inputs.dtype),
      ).shape

      # One (unshared) convolutional kernel per each pixel in the output.
      kernel_shape = conv_output_shape[1:-1] + (
          np.prod(kernel_size) * in_features,
          self.features,
      )

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError(
          'Mask needs to have the same shape as weights. '
          f'Shapes are: {self.mask.shape}, {kernel_shape}'
      )

    if self.kernel_shardings:
      if len(self.kernel_shardings) != len(kernel_shape):
        raise ValueError(
            f"Kernel axis names {self.kernel_shardings} does not match "
            f'kernel shape {kernel_shape}.'
        )

    kernel_init = sharding.modulate_param_init(
        self.kernel_init, self.kernel_shardings)
    kernel = self.param(
        name='kernel',
        init_fn=kernel_init,
        shape=kernel_shape,
        dtype=self.param_dtype,
        unbox=True,
    )
    kernel = jnp.asarray(kernel, dtype=self.dtype)

    if self.mask is not None:
      kernel *= self.mask

    if self.use_bias:
      if self.shared_weights:
        # One bias weight per output channel, shared between pixels.
        bias_shape = (self.features,)
      else:
        # One bias weight per output entry, unshared between pixels.
        bias_shape = conv_output_shape[1:]  # pylint: disable=undefined-variable  # pytype: disable=name-error  # jax-api-types

      if self.bias_shardings:
        if len(self.bias_shardings) != len(bias_shape):
          raise ValueError(
              f"Bias axis names {self.bias_shardings} does not match "
              f'bias shape {bias_shape}.'
          )

      bias_init = sharding.modulate_param_init(
          self.bias_init, self.bias_shardings)
      bias = self.param(
          name='bias',
          init_fn=bias_init,
          shape=bias_shape,
          dtype=self.param_dtype,
          unbox=True,
      )
      bias = jnp.asarray(bias, dtype=self.dtype)
    else:
      bias = None

    inputs, kernel, bias = nn.dtypes.promote_dtype(inputs, kernel, bias,
                                                   dtype=self.dtype)
    if self.shared_weights:
      y = self.conv_general_dilated(
          inputs,
          kernel,
          strides,
          padding_lax,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          feature_group_count=self.feature_group_count,
          precision=self.precision,
      )
    else:
      y = self.conv_general_dilated_local(
          lhs=inputs,
          rhs=kernel,
          window_strides=strides,
          padding=padding_lax,
          filter_shape=kernel_size,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          precision=self.precision,
      )

    if self.use_bias:
      bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
      y += bias

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]  # pylint: disable=undefined-variable
      y = jnp.reshape(y, output_shape)
    return y


class Conv(_Conv):
  """Convolution Module wrapping `lax.conv_general_dilated`.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it must
      be a sequence of integers.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpreted as applying the same
      padding in all dims and assign a single int in a sequence causes the same
      padding to be used on both sides. `'CAUSAL'` padding for a 1D convolution
      will left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `inputs`
      (default: 1). Convolution with input dilation `d` is equivalent to
      transposed convolution with stride `d`.
    kernel_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    kernel_shardings: sharding annotations to use for kernel sharding.
    bias_shardings: sharding annotations to use for bias sharding.
    conv_general_dilated: the function that performs a generic dilated
      convolution.
    conv_general_dilated_local: the function that performs the local dilated
      convolution.
  """

  @property
  def shared_weights(self):
    return True


class ConvLocal(_Conv):
  """Local convolution Module wrapping `lax.conv_general_dilated_local`.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it must
      be a sequence of integers.
    strides: an integer or a sequence of `n` integers, representing the
      inter-window strides (default: 1).
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpreted as applying the same
      padding in all dims and assign a single int in a sequence causes the same
      padding to be used on both sides. `'CAUSAL'` padding for a 1D convolution
      will left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `inputs`
      (default: 1). Convolution with input dilation `d` is equivalent to
      transposed convolution with stride `d`.
    kernel_dilation: an integer or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    kernel_shardings: sharding annotations to use for kernel sharding.
    bias_shardings: sharding annotations to use for bias sharding.
    conv_general_dilated: the function that performs a generic dilated
      convolution.
    conv_general_dilated_local: the function that performs the local dilated
      convolution.
  """

  @property
  def shared_weights(self):
    return False


class ConvTranspose(nn.Module):
  """Convolution Module wrapping lax.conv_transpose.

  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it must
      be a sequence of integers.
    strides: a sequence of `n` integers, representing the inter-window strides.
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpreted as applying the same
      padding in all dims and assign a single int in a sequence causes the same
      padding to be used on both sides.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel. Convolution with kernel dilation is also known as 'atrous
      convolution'.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    kernel_shardings: sharding annotations to use for kernel sharding.
    bias_shardings: sharding annotations to use for bias sharding.
    transpose_kernel: if True flips spatial axes and swaps the input/output
      channel axes of the kernel.
  """

  features: int
  kernel_size: typing.Axes
  strides: Sequence[int] | None = None
  padding: typing.PaddingLike = 'SAME'
  kernel_dilation: Sequence[int] | None = None
  use_bias: bool = True
  mask: jax.Array | None = None
  dtype: jax.typing.DTypeLike | None = None
  param_dtype: jax.typing.DTypeLike = jnp.float32
  precision: typing.Precision = None
  kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
  bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
  kernel_shardings: typing.ShardingAxes = ()
  bias_shardings: tuple[str, Ellipsis] = ()
  transpose_kernel: bool = False

  @nn.compact
  def __call__(self, inputs):
    """Applies a transposed convolution to the inputs.

    Behaviour mirrors of `jax.lax.conv_transpose`.

    Args:
      inputs: input data with dimensions (*batch_dims, spatial_dims...,
        features). This is the channels-last convention, i.e. NHWC for a 2d
        convolution and NDHWC for a 3D convolution. Note that this is different
        from the input convention used by `lax.conv_general_dilated`, which puts
        the spatial dimensions last.
        If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """
    kernel_size: tuple[int, Ellipsis]
    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = tuple(self.kernel_size)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
          num_batch_dimensions:
      ]
      inputs = jnp.reshape(inputs, flat_input_shape)

    strides: tuple[int, Ellipsis]
    if self.strides is None:
      strides = (1,) * (inputs.ndim - 2)
    else:
      strides = tuple(self.strides)

    in_features = jnp.shape(inputs)[-1]
    if self.transpose_kernel:
      kernel_shape = kernel_size + (self.features, in_features)
    else:
      kernel_shape = kernel_size + (in_features, self.features)

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError(
          'Mask needs to have the same shape as weights. '
          f'Shapes are: {self.mask.shape}, {kernel_shape}'
      )

    if self.kernel_shardings:
      if len(self.kernel_shardings) != len(kernel_shape):
        raise ValueError(
            f"Kernel axis names {self.kernel_shardings} does not match "
            f'kernel shape {kernel_shape}.'
        )

    kernel_init = sharding.modulate_param_init(
        self.kernel_init, self.kernel_shardings)
    kernel = self.param(
        name='kernel',
        init_fn=kernel_init,
        shape=kernel_shape,
        dtype=self.param_dtype,
        unbox=True,
    )
    kernel = jnp.asarray(kernel, dtype=self.dtype)

    if self.mask is not None:
      kernel *= self.mask

    if self.use_bias:
      bias_shape = (self.features,)

      if self.bias_shardings:
        if len(self.bias_shardings) != len(bias_shape):
          raise ValueError(
              f"Bias axis names {self.bias_shardings} does not match "
              f'bias shape {bias_shape}.'
          )

      bias_init = sharding.modulate_param_init(
          self.bias_init, self.bias_shardings)
      bias = self.param(
          name='bias',
          init_fn=bias_init,
          shape=bias_shape,
          dtype=self.param_dtype,
          unbox=True,
      )
      bias = jnp.asarray(bias, dtype=self.dtype)
    else:
      bias = None

    inputs, kernel, bias = nn.dtypes.promote_dtype(
        inputs, kernel, bias, dtype=self.dtype)

    # Get post-conv padding function and canonolize lax padding method
    padding_lax, post_conv_padding = canonicalize_padding(
        self.padding, len(kernel_size))

    y = lax.conv_transpose(
        inputs,
        kernel,
        strides,
        padding_lax,
        rhs_dilation=self.kernel_dilation,
        transpose_kernel=self.transpose_kernel,
        precision=self.precision,
    )
    y = post_conv_padding(y, inputs.shape, strides)

    if self.use_bias:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]  # pylint: disable=undefined-variable
      y = jnp.reshape(y, output_shape)

    return y
