# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""EvoNorm layers for Flax.

Liu, H., Brock, A., Simonyan, K., & Le, Q. V. (2020, April 6). Evolving
Normalization-Activation Layers. arXiv [cs.LG]. http://arxiv.org/abs/2004.02967
"""
from flax import nn
from flax.nn import initializers
from jax import lax
import jax.numpy as jnp
import jax.scipy.stats
EPSILON = 1e-5
MOVING_AVERAGE_DECAY = 0.9

LAYER_EVONORM_S0 = 'evonorm_s0'

LAYER_EVONORM_B0 = 'evonorm_b0'
LAYER_EVONORMS = [LAYER_EVONORM_B0, LAYER_EVONORM_S0]


def _instance_std(inputs, reduction_axis, epsilon=EPSILON):
  """Numerically stable computation of st.dev. of inputs over reduction_axis."""
  return jnp.sqrt(inputs.var(axis=reduction_axis, keepdims=True) + epsilon)


class EvoNorm(nn.Module):
  """EvoNorm Module."""

  def apply(self,
            x,
            layer=LAYER_EVONORM_B0,
            nonlinearity=True,
            num_groups=32,
            group_size=None,
            batch_stats=None,
            use_running_average=False,
            axis=-1,
            momentum=0.99,
            epsilon=1e-5,
            dtype=jnp.float32,
            bias=True,
            scale=True,
            bias_init=initializers.zeros,
            scale_init=initializers.ones,
            axis_name=None,
            axis_index_groups=None):
    """Normalizes the input using batch statistics.

    Args:
      x: the input to be normalized.
      layer: LAYER_EVONORM_B0 or LAYER_EVONORM_S0.
      nonlinearity: use the EvoNorm nonlinearity.
      num_groups: number of groups to use for group statistics.
      group_size: size of groups, see nn.GroupNorm.
      batch_stats: a `flax.nn.Collection` used to store an exponential moving
        average of the batch statistics (default: None).
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of the batch
        statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      bias:  if True, bias (beta) is added.
      scale: if True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
          example, `[[0, 1], [2, 3]]` would independently batch-normalize over
          the examples on the first two and last two devices. See `jax.lax.psum`
          for more details.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)

    axis = axis if isinstance(axis, tuple) else (axis,)
    # pylint: disable=protected-access
    axis = nn.normalization._absolute_dims(x.ndim, axis)
    # pylint: enable=protected-access
    feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
    reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
    instance_reduction_axis = tuple(
        i for i in range(x.ndim) if i not in axis and i > 0)
    batch_reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

    if nonlinearity:
      v = self.param('v', reduced_feature_shape,
                     jax.nn.initializers.ones).reshape(feature_shape)
      if layer == LAYER_EVONORM_S0:
        den, group_shape, input_shape = _GroupStd(
            x,
            num_groups=num_groups,
            group_size=group_size,
            epsilon=epsilon,
            dtype=dtype,
        )
        x = x * nn.sigmoid(v * x)
        x = x.reshape(group_shape)
        x /= den
        x = x.reshape(input_shape)
      elif layer == LAYER_EVONORM_B0:
        if self.is_stateful() or batch_stats:
          ra_var = self.state(
              'var',
              reduced_feature_shape,
              initializers.ones,
              collection=batch_stats)
        else:
          ra_var = None

        if use_running_average:
          if ra_var is None:
            raise ValueError(
                'when use_running_averages is True '
                'either use a stateful context or provide batch_stats')
          var = ra_var.value
        else:
          mean = jnp.mean(x, axis=batch_reduction_axis, keepdims=False)
          mean2 = jnp.mean(
              lax.square(x), axis=batch_reduction_axis, keepdims=False)
          if axis_name is not None and not self.is_initializing():
            concatenated_mean = jnp.concatenate([mean, mean2])
            mean, mean2 = jnp.split(
                lax.pmean(
                    concatenated_mean,
                    axis_name=axis_name,
                    axis_index_groups=axis_index_groups), 2)
          var = mean2 - lax.square(mean)

          if ra_var and not self.is_initializing():
            ra_var.value = momentum * ra_var.value + (1 - momentum) * var

        left = lax.sqrt(var + epsilon)

        instance_std = jnp.sqrt(
            x.var(axis=instance_reduction_axis, keepdims=True) + epsilon)
        right = v * x + instance_std
        x = x / jnp.maximum(left, right)
      else:
        raise ValueError('Unknown EvoNorm layer: {}'.format(layer))

    if scale:
      x *= self.param('scale', reduced_feature_shape,
                      scale_init).reshape(feature_shape)
    if bias:
      x = x + self.param('bias', reduced_feature_shape,
                         bias_init).reshape(feature_shape)
    return jnp.asarray(x, dtype)


class _BatchStd(nn.Module):
  """BatchNorm Module."""

  def apply(self,
            x,
            batch_stats=None,
            use_running_average=False,
            axis=-1,
            momentum=0.99,
            epsilon=1e-5,
            dtype=jnp.float32,
            axis_name=None,
            axis_index_groups=None):
    """Normalizes the input using batch statistics.

    Args:
      x: the input to be normalized.
      batch_stats: a `flax.nn.Collection` used to store an exponential moving
        average of the batch statistics (default: None).
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of the batch
        statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
          example, `[[0, 1], [2, 3]]` would independently batch-normalize over
          the examples on the first two and last two devices. See `jax.lax.psum`
          for more details.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    axis = axis if isinstance(axis, tuple) else (axis,)
    # pylint: disable=protected-access
    axis = nn.normalization._absolute_dims(x.ndim, axis)
    # pylint: enable=protected-access
    reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
    reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)
    if self.is_stateful() or batch_stats:
      ra_var = self.state(
          'var',
          reduced_feature_shape,
          initializers.ones,
          collection=batch_stats)
    else:
      ra_var = None

    if use_running_average:
      if ra_var is None:
        raise ValueError('when use_running_averages is True '
                         'either use a stateful context or provide batch_stats')
      var = ra_var.value
    else:
      mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
      mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
      if axis_name is not None and not self.is_initializing():
        concatenated_mean = jnp.concatenate([mean, mean2])
        mean, mean2 = jnp.split(
            lax.pmean(
                concatenated_mean,
                axis_name=axis_name,
                axis_index_groups=axis_index_groups), 2)
      var = mean2 - lax.square(mean)

      if ra_var and not self.is_initializing():
        ra_var.value = momentum * ra_var.value + (1 - momentum) * var

    mul = lax.sqrt(var + epsilon)

    return jnp.asarray(mul, dtype)


class _GroupStd(nn.Module):
  """Group normalization (arxiv.org/abs/1803.08494)."""

  def apply(
      self,
      x,
      num_groups=32,
      group_size=None,
      epsilon=1e-6,
      dtype=jnp.float32,
  ):
    """Applies group normalization to the input (arxiv.org/abs/1803.08494).

    This op is similar to batch normalization, but statistics are shared across
    equally-sized groups of channels and not shared across batch dimension.
    Thus, group normalization does not depend on the batch composition and does
    not require maintaining internal state for storing statistics.

    The user should either specify the total number of channel groups or the
    number of channels per group.

    Args:
      x: the input of shape N...C, where N is a batch dimension and C is a
        channels dimensions. `...` represents an arbitrary number of extra
        dimensions that are used to accumulate statistics over.
      num_groups: the total number of channel groups. The default value of 32 is
        proposed by the original group normalization paper.
      group_size: the number of channels in a group.
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).

    Returns:
      Normalized inputs (the same shape as inputs).

    """
    x = jnp.asarray(x, jnp.float32)
    if ((num_groups is None and group_size is None) or
        (num_groups is not None and group_size is not None)):
      raise ValueError('Either `num_groups` or `group_size` should be '
                       'specified, but not both of them.')

    channels = x.shape[-1]
    if group_size is not None:
      if channels % group_size != 0:
        raise ValueError('Number of channels ({}) is not multiple of the '
                         'group size ({}).'.format(channels, group_size))
      num_groups = channels // group_size
    while num_groups > 1:
      if channels % num_groups == 0:
        break
      num_groups -= 1

    group_shape = x.shape[:-1] + (num_groups, x.shape[-1] // num_groups)

    input_shape = x.shape
    x = x.reshape(group_shape)

    reduction_axis = list(range(1, x.ndim - 2)) + [x.ndim - 1]

    mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
    mean_of_squares = jnp.mean(
        jnp.square(x), axis=reduction_axis, keepdims=True)
    var = mean_of_squares - jnp.square(mean)

    std = lax.sqrt(var + epsilon)

    return std.astype(dtype), group_shape, input_shape
