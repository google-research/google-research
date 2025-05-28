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

"""Normalization layers with SPMD-friendly named axes params."""

from typing import Optional, Sequence

from flax import linen as nn
import jax
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

from imp.max.utils import sharding
from imp.max.utils import typing


def _canonicalize_axes(rank, axes):
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, Sequence):
    axes = (axes,)
  return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


def _compute_stats(
    inputs,
    axes,
    mask = None,
    use_mean = True,
    use_fast_variance = True,
):
  """Computes mean and variance statistics.

  This implementation takes care of a few important details:
  - Computes in float32 precision for stability in half precision training.
  - If `use_fast_variance` is `True`, mean and variance are computed using
    Var = E[|x|^2] - |E[x]|^2, instead of Var = E[|x - E[x]|^2]), in a single
    XLA fusion.
  - Clips negative variances to zero which can happen due to roundoff errors.
    This avoids downstream NaNs.

  Arguments:
    inputs: Input array.
    axes: The axes in ``x`` to compute mean and variance statistics for.
    mask: Binary array of shape broadcastable to `inputs` tensor, indicating
      the positions for which the mean and variance should be computed.
    use_mean: If true, calculate the mean from the input and use it when
      computing the variance. If false, set the mean to zero and compute the
      variance without subtracting the mean.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.

  Returns:
    A pair ``(mean, var)``.
  """
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  dtype = jnp.promote_types(jnp.float32, jnp.result_type(inputs))
  inputs = jnp.asarray(inputs, dtype)

  if use_mean:
    if use_fast_variance:
      mean = jnp.mean(inputs, axes, where=mask)
      mean2 = jnp.mean(_abs_sq(inputs), axes, where=mask)
      # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
      # to floating point round-off errors.
      var = nn.relu(mean2 - _abs_sq(mean))
    else:
      mean = jnp.mean(inputs, axes, where=mask)
      var = jnp.mean(_abs_sq(inputs - jnp.expand_dims(mean, axes)),
                     axes, where=mask)
  else:
    var = jnp.mean(_abs_sq(inputs), axes, where=mask)
    mean = jnp.zeros_like(var)
  return mean, var


def _normalize(module,
               inputs,
               mean,
               var,
               reduction_axes,
               feature_axes,
               dtype,
               param_dtype,
               epsilon,
               use_bias,
               use_scale,
               bias_init,
               scale_init,
               shardings):
  """'Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

  Arguments:
    module: The parent Flax module from which this function is called.
    inputs: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: Dtype of the returned result.
    param_dtype: Dtype of the parameters.
    epsilon: Normalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.
    shardings: Sharding annotations for the scale and bias parameters

  Returns:
    The normalized input.
  """
  reduction_axes = _canonicalize_axes(inputs.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(inputs.ndim, feature_axes)

  feature_shape = [1] * inputs.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = inputs.shape[ax]
    reduced_feature_shape.append(inputs.shape[ax])

  if (use_scale or use_bias) and shardings:
    if len(shardings) != len(reduced_feature_shape):
      raise ValueError(
          f'Shardings `{shardings}` do not match '
          f'reduced feature shape {reduced_feature_shape}.'
      )

  mean = jnp.expand_dims(mean, reduction_axes)
  var = jnp.expand_dims(var, reduction_axes)
  y = inputs - mean
  multiplier = lax.rsqrt(var + epsilon)

  if use_scale:
    scale_init = sharding.modulate_param_init(scale_init, shardings)
    scale = module.param(
        name='scale',
        init_fn=scale_init,
        shape=reduced_feature_shape,
        dtype=param_dtype,
        unbox=True,
    )
    scale = jnp.asarray(scale, param_dtype).reshape(feature_shape)
    multiplier *= scale

  y *= multiplier

  if use_bias:
    bias_init = sharding.modulate_param_init(bias_init, shardings)
    bias = module.param(
        name='bias',
        init_fn=bias_init,
        shape=reduced_feature_shape,
        dtype=param_dtype,
        unbox=True,
    )
    bias = jnp.asarray(bias, param_dtype).reshape(feature_shape)
    y += bias

  return jnp.asarray(y, dtype)


class LayerNorm(nn.Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  Operates on the last axis of the input data.

  It normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    shardings: Sharding annotations for the scale and bias parameters
  """
  epsilon: float = 1e-6
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  scale_init: nn.initializers.Initializer = initializers.ones
  bias_init: nn.initializers.Initializer = initializers.zeros
  reduction_axes: typing.Axes = -1
  feature_axes: typing.Axes = -1
  use_fast_variance: bool = True
  shardings: typing.ShardingAxes = ()

  @nn.compact
  def __call__(self,
               inputs,
               *,
               mask = None):
    """Applies layer normalization on the input.

    Args:
      inputs: The inputs array.
      mask: Binary array of shape broadcastable to ``inputs`` array, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

    mean, var = _compute_stats(
        inputs=inputs,
        axes=self.reduction_axes,
        mask=mask,
        use_mean=True,
        use_fast_variance=self.use_fast_variance,
    )

    return _normalize(
        module=self,
        inputs=inputs,
        mean=mean,
        var=var,
        reduction_axes=self.reduction_axes,
        feature_axes=self.feature_axes,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        epsilon=self.epsilon,
        use_bias=self.use_bias,
        use_scale=self.use_scale,
        bias_init=self.bias_init,
        scale_init=self.scale_init,
        shardings=self.shardings,
    )


class RMSNorm(nn.Module):
  """RMS Layer normalization (https://arxiv.org/abs/1910.07467).

  RMSNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  Unlike LayerNorm which re-centers the mean to be 0 and normalizes by the
  standard deviation of the activations, RMSNorm does not re-center at all
  and instead normalizes by the root mean square of the activations.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: Axes for computing normalization statistics.
    feature_axes: Feature axes for learned bias and scaling.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See ``jax.pmap`` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap or shard
      map. For SPMD jit, you do not need to manually synchronize. Just make sure
      that the axes are correctly annotated and XLA:SPMD will insert the
      necessary collectives.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
      examples on the first two and last two devices. See ``jax.lax.psum`` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  epsilon: float = 1e-6
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32
  use_scale: bool = True
  scale_init: nn.initializers.Initializer = initializers.ones
  reduction_axes: typing.Axes = -1
  feature_axes: typing.Axes = -1
  use_fast_variance: bool = True
  shardings: typing.ShardingAxes = ()

  @nn.compact
  def __call__(self,
               inputs,
               *,
               mask = None):
    """Applies RMS layer normalization on the input.

    Args:
      inputs: the inputs
      mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    mean, var = _compute_stats(
        inputs=inputs,
        axes=self.reduction_axes,
        mask=mask,
        use_mean=False,
        use_fast_variance=self.use_fast_variance,
    )

    return _normalize(
        module=self,
        inputs=inputs,
        mean=mean,
        var=var,
        reduction_axes=self.reduction_axes,
        feature_axes=self.feature_axes,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        epsilon=self.epsilon,
        use_bias=False,
        use_scale=self.use_scale,
        bias_init=initializers.zeros,
        scale_init=self.scale_init,
        shardings=self.shardings,
    )
