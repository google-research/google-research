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

"""Helper class and methods for computing and tracking statistics."""

import functools
from typing import Any, Callable, Optional, Iterable

import flax
import jax
from jax import lax
import jax.numpy as jnp

from aqt.jax import shape_utils
from aqt.jax.flax import struct as flax_struct


def is_in_pmapped_context(paxis_name):
  """Returns whether context is pmapped."""
  return paxis_name is not None and len(jax.devices()) > 1


def masked_reduction(
    x,
    *,
    mask,
    axis,
    paxis_name,
    local_reduction_fn,
    pmap_reduction_fn,
    identity,
    keepdims,
):
  """Reduce an array locally and across devices, excluding masked elements.

  Args:
    x: Tensor to take the mean of.
    mask: Boolean array of same shape as 'x'. True elements are included in the
      mean, false elements are excluded. None means to not mask out any
      elements
    axis: Axis of 'x' to compute the mean over.
    paxis_name: Optional. If not None, will take a distributed mean of 'x'
      across devices using the specified parallel axis.
    local_reduction_fn: Reduction function to apply locally to 'x'.
    pmap_reduction_fn: A distributed reduction function to apply after
      'x' has been reduced locally. A member of the lax.p* family of functions.
    identity: Identity element for the reduction performs by 'reduction_fn'.
      eg, 0 for sum, 1 for multipy, etc.
    keepdims: Same meaning as the corresponding parameter in `numpy.mean`.
      Whether to keep the reduction axes or squeeze them out.

  Returns:
    A Jax array resulting from applying the reduction to 'x'.
  """
  if mask is None:
    x_masked = x
  else:
    if x.shape != mask.shape:
      raise ValueError(
          'Masked reduction received input and mask with different shapes')
    if mask.dtype != jnp.bool_:
      raise ValueError(
          f'Mask should be boolean-typed, but has dtype {mask.dtype}')

    x_masked = jnp.where(mask, x, identity)
  x_reduced = local_reduction_fn(x_masked, axis=axis, keepdims=keepdims)
  if is_in_pmapped_context(paxis_name):
    x_reduced = pmap_reduction_fn(
        x_reduced, axis_name=paxis_name)
  return x_reduced


masked_sum = functools.partial(
    masked_reduction,
    local_reduction_fn=jnp.sum,
    pmap_reduction_fn=lax.psum,
    identity=0.0)

# TODO(shivaniagrawal): The pmap_function_fn used here should be 'lax.pmax' for
# masked_max and 'lax.pmin' for masked_min, but Jax currently doesn't support
# using those in multihost training. Jax team says it will be trivial to add and
# they welcome PRs, so we just approximate distributed max using lax.pmean until
# we get that PR into Jax.
masked_mean_of_max = functools.partial(
    masked_reduction,
    local_reduction_fn=jnp.max,
    pmap_reduction_fn=lax.pmean,
    identity=-jnp.inf)
masked_mean_of_min = functools.partial(
    masked_reduction,
    local_reduction_fn=jnp.min,
    pmap_reduction_fn=lax.pmean,
    identity=jnp.inf)


def masked_mean(x, *, mask, axis,
                paxis_name, keepdims):
  """Calculates the mean of a tensor, excluding masked-out entries.

  Args:
    x: Tensor to take the mean of.
    mask: Boolean array of same shape as 'x'. True elements are included in the
      mean, false elements are excluded.
    axis: Axis of 'x' to compute the mean over.
    paxis_name: Optional. If not None, will take a distributed mean of 'x'
      across devices using the specified parallel axis.
    keepdims: Same meaning as the corresponding parameter in `numpy.mean`.
      Whether to keep the reduction axes or squeeze them out.

  Returns:
    Tensor resulting from reducing 'x' over axes in 'axis'.
  """
  assert x.shape == mask.shape
  x_masked_sum = masked_sum(
      x, mask=mask, axis=axis, paxis_name=paxis_name, keepdims=keepdims)
  mask_count = masked_sum(
      x=mask, mask=None, axis=axis, paxis_name=paxis_name, keepdims=keepdims)
  x_masked_mean = x_masked_sum / mask_count
  return x_masked_mean


@flax_struct.dataclass
class Stats:
  """Dataclass to keep track of statistics."""

  n: int  # samples count
  mean: jnp.ndarray  # mean of values over axis
  mean_abs: jnp.ndarray  # mean of absolute values over axis
  mean_sq: jnp.ndarray  # mean of square values over axis
  # minimum value over axis in each device's batch, averaged over devices.
  mean_batch_minimum: jnp.ndarray
  # maximum value over axis in each device's batch, averaged over devices.
  mean_batch_maximum: jnp.ndarray

  @classmethod
  def stats_initializer(cls, shape, *, dtype = jnp.float32):
    """Constructor to init a empty Stats instance.

    Args:
      shape: shape of the statistics (mean, mean_sq and mean_abs)
      dtype: the dtype of the stats (default: float32).

    Returns:
      A new instance of Stats, with statistics intialized to 0.
    """
    key = jax.random.PRNGKey(
        1)  # no effect, but needed for flax.linen.initializers.zeros.
    shape = tuple(shape)
    return cls(
        n=0,
        mean=flax.linen.initializers.zeros(key, shape, dtype),
        mean_abs=flax.linen.initializers.zeros(key, shape, dtype),
        mean_sq=flax.linen.initializers.zeros(key, shape, dtype),
        mean_batch_maximum=flax.linen.initializers.zeros(key, shape, dtype),
        mean_batch_minimum=flax.linen.initializers.zeros(key, shape, dtype))

  @classmethod
  def create_updated_stats(cls,
                           stats,
                           samples,
                           *,
                           axis = None,
                           paxis_name = None,
                           alpha = None,
                           mask = None,
                           exclude_zeros = False):
    """Create a new Stats instance that represents the updated statistics.

    Since flax.struct.dataclass objects are frozen, this method creates a new
    instance of Stats with updated stats and returns it.

    Args:
      stats: A Stats dataclass object to be updated.
      samples: An array to update the current statistics with.
      axis: axis to average input samples over, e.g. to calculate stats per
        channel.
      paxis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names.
      alpha: Smoothing parameter to use for moving average. If None, will use
        1/n, where n is the stat count.
      mask: Optional boolean tensor of the same shape as 'samples' specifying
        which values of 'samples' to use as part of the bounds calculation.
        'True' indicates the corresponding value from 'samples' should be used.
        If None, all values are used.
      exclude_zeros: Whether to exclude zeros in samples when computing
        statistics, e.g. when calculating mean absolute values.

    Returns:
      A new Stats instance with updated stats and count.
    """

    if mask is None:
      mask = jnp.full(samples.shape, True)
    shape_utils.assert_shapes_compatible(samples.shape, mask.shape)
    mask = jnp.broadcast_to(mask, samples.shape)
    if exclude_zeros:
      # Where samples are zero, set mask to False. This way they won't be
      # included in statistics.
      mask = mask & (samples != 0)

    def _moving_avg(old_avg, new_val,
                    masked_reduction_fn):
      masked_new_val_reduced = masked_reduction_fn(
          new_val, mask=mask, axis=axis, paxis_name=paxis_name, keepdims=True)
      valid_mask = jnp.isfinite(masked_new_val_reduced)
      # Only update average where means are valid, so set deltas corresponding
      # to invalid entries to 0.
      delta = jnp.where(valid_mask, masked_new_val_reduced - old_avg, 0)
      # TODO(lew): This is slightly incorrect, alpha should be proportional to
      # the mask size.
      new_avg = old_avg + alpha * delta
      return new_avg

    new_n = stats.n + 1
    if alpha is None:
      alpha = 1. / new_n

    new_mean = _moving_avg(stats.mean, samples, masked_reduction_fn=masked_mean)
    new_mean_abs = _moving_avg(
        stats.mean_abs, jnp.abs(samples), masked_reduction_fn=masked_mean)
    new_mean_sq = _moving_avg(
        stats.mean_sq, jnp.square(samples), masked_reduction_fn=masked_mean)
    new_mean_batch_minimum = _moving_avg(
        stats.mean_batch_minimum,
        samples,
        masked_reduction_fn=masked_mean_of_min)
    new_mean_batch_maximum = _moving_avg(
        stats.mean_batch_maximum,
        samples,
        masked_reduction_fn=masked_mean_of_max)
    return cls(
        n=new_n,
        mean=new_mean,
        mean_abs=new_mean_abs,
        mean_sq=new_mean_sq,
        mean_batch_minimum=new_mean_batch_minimum,
        mean_batch_maximum=new_mean_batch_maximum)
