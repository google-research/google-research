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

"""Implements a flax module for tracking distributions and statistics.
"""
import math
from typing import Optional, Iterable, Union

from flax import linen as nn
import jax.numpy as jnp

from aqt.jax import shape_utils
from aqt.jax import stats
from aqt.jax.utils import normalize_axes


def _take_subset_of_axes(x,
                         axis,
                         num_indices_per_ax = 10):
  """Take the first num_indices_per_ax of specified axes."""
  if not isinstance(axis, Iterable):
    axis = (axis,)
  for ax in axis:  # exclude batch dimension 0
    x = x.take(indices=range(min(num_indices_per_ax, x.shape[ax])), axis=ax)
  return x


class StatsTag(nn.Module):
  """This module is intended for tracking distributions and simple statistics.

  For example, if you want track the distribution of activations and its
  statistics over the
  course of training, you can tag them with this module. You can retrieve the
  values from the state dict, and add them to tensorboard summaries for
  histogram / distribution visualization.

  Example usage:
    x = 2 * x
    stats_tag.StatsTag(update_stats=True, name='activations')(x)
    x = log(x)

  Attributes:
    update_stats: If True, will update states in this module.
    channel_axis: Axis to reduce over for computing statistics.
      E.g. for a tensor with shape (a, b, c) where c is is the channel
      dimension, then channel_axis should be 2 or -1.
    num_indices_per_ax: Number of indices to take from each channel axis.
  """
  update_stats: bool
  channel_axis: Optional[Union[int, Iterable[int]]]
  num_indices_per_ax: int = 5

  @nn.compact
  def __call__(
      self,
      x,
      *,
      mask,
  ):
    """Applies a tag to track distributions.

    Args:
      x: the array to compute statistics distributions over.
      mask: boolean array indicating which elements of 'x' should be
        included in the stats calculation ('True' means to include).

    Returns:
      x unchanged. The return value can also be ignored.
    """
    if mask is None:
      mask = jnp.full(x.shape, True)
    shape_utils.assert_shapes_compatible(x.shape, mask.shape)
    mask = jnp.broadcast_to(mask, x.shape)
    channel_axis = self.channel_axis
    if channel_axis is not None:
      if not isinstance(channel_axis, Iterable):
        channel_axis = (channel_axis,)
      channel_axis = normalize_axes(channel_axis, x.ndim)
      x = _take_subset_of_axes(
          x, axis=channel_axis, num_indices_per_ax=self.num_indices_per_ax)
      mask = _take_subset_of_axes(
          mask, axis=channel_axis, num_indices_per_ax=self.num_indices_per_ax)
      reduction_axis = tuple(
          [ax for ax in range(x.ndim) if ax not in channel_axis])
    else:
      reduction_axis = None

    distr_shape = ()
    if channel_axis:
      distr_shape = tuple(d for i, d in enumerate(x.shape) if i in channel_axis)

    # TODO(wanglisa): Consider adding configurability to specify which
    # statistics are collected.
    init_with_zeros = lambda shape: jnp.zeros(shape, dtype=jnp.float32)
    is_initializing = not self.has_variable('stats_tag', 'min_per_ch')
    min_per_ch = self.variable(
        'stats_tag',
        'min_per_ch',
        init_with_zeros,
        distr_shape,
    )
    max_per_ch = self.variable('stats_tag', 'max_per_ch', init_with_zeros,
                               distr_shape)
    mean_per_ch = self.variable(
        'stats_tag',
        'mean_per_ch',
        init_with_zeros,
        distr_shape,
    )
    stddev_per_ch = self.variable(
        'stats_tag',
        'stddev_per_ch',
        init_with_zeros,
        distr_shape,
    )
    absdev_per_ch = self.variable(
        'stats_tag',
        'absdev_per_ch',
        init_with_zeros,
        distr_shape,
    )
    stddev_per_ch_uncentered = self.variable(
        'stats_tag',
        'stddev_per_ch_uncentered',
        init_with_zeros,
        distr_shape,
    )
    absdev_per_ch_uncentered = self.variable(
        'stats_tag',
        'absdev_per_ch_uncentered',
        init_with_zeros,
        distr_shape,
    )
    if self.update_stats and not is_initializing:
      min_per_ch.value = jnp.min(
          jnp.where(mask, x, math.inf), axis=reduction_axis)
      max_per_ch.value = jnp.max(
          jnp.where(mask, x, -math.inf), axis=reduction_axis)
      mean_per_ch_keepdims = stats.masked_mean(
          x, mask=mask, axis=reduction_axis, paxis_name=None, keepdims=True)
      mean_per_ch.value = mean_per_ch_keepdims.squeeze(axis=reduction_axis)
      stddev_per_ch.value = jnp.sqrt(
          stats.masked_mean(
              (x - mean_per_ch_keepdims)**2,
              mask=mask,
              axis=reduction_axis,
              paxis_name=None,
              keepdims=False))
      absdev_per_ch.value = stats.masked_mean(
          jnp.abs(x - mean_per_ch_keepdims),
          mask=mask,
          axis=reduction_axis,
          paxis_name=None,
          keepdims=False)
      stddev_per_ch_uncentered.value = jnp.sqrt(
          stats.masked_mean(
              jnp.square(x),
              mask=mask,
              axis=reduction_axis,
              paxis_name=None,
              keepdims=False))
      absdev_per_ch_uncentered.value = stats.masked_mean(
          jnp.abs(x),
          mask=mask,
          axis=reduction_axis,
          paxis_name=None,
          keepdims=False)
