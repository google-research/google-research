# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Inputs and outputs normalization and clipping."""

from collections.abc import Callable
from typing import Any, Optional
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


class Normalizer:
  """A class for normalizing and clipping tf/np/jnp tensors."""

  def __init__(self, center, scale, lower_bound=None, upper_bound=None,
               bc_lambda=None):
    center = np.asarray(center)
    scale = np.asarray(scale)
    if lower_bound is not None:
      lower_bound = np.asarray(lower_bound)
    if upper_bound is not None:
      upper_bound = np.asarray(upper_bound)
    if bc_lambda is not None:
      bc_lambda = np.asarray(bc_lambda)
    assert center.shape == scale.shape
    assert center.ndim == 1
    self._center = np.nan_to_num(
        center, nan=0., posinf=np.inf, neginf=-np.inf)
    self._center.setflags(write=False)
    self._scale = np.nan_to_num(
        scale, nan=1., posinf=np.inf, neginf=-np.inf)
    self._scale.setflags(write=False)

    if lower_bound is not None:
      assert center.shape == lower_bound.shape
      self._lower_bound = np.nan_to_num(lower_bound, nan=-np.inf,
                                        posinf=np.inf, neginf=-np.inf)
      self._lower_bound.setflags(write=False)
    else:
      self._lower_bound = None
    if upper_bound is not None:
      assert center.shape == upper_bound.shape
      self._upper_bound = np.nan_to_num(upper_bound, nan=np.inf,
                                        posinf=np.inf, neginf=-np.inf)
      self._upper_bound.setflags(write=False)
    else:
      self._upper_bound = None
    if bc_lambda is not None:
      assert center.shape == bc_lambda.shape
      # replace NaN with 1 and subtract 1 to the center to remove BC effect.
      self._center = self._where(np.isnan(bc_lambda),
                                 self._center - 1,
                                 self._center)
      self._bc_lambda = np.nan_to_num(bc_lambda, nan=1., posinf=np.inf,
                                      neginf=-np.inf)
      self._bc_lambda.setflags(write=False)
    else:
      self._bc_lambda = None

    self._is_identity = (
        np.all(self._center == 0)
        and np.all(self._scale == 1)
        and self._lower_bound is None
        and self._upper_bound is None
        and self._bc_lambda is None
    )

  @property
  def center(self):
    return self._center

  @property
  def scale(self):
    return self._scale

  @property
  def lower_bound(self):
    return self._lower_bound

  @property
  def upper_bound(self):
    return self._upper_bound

  @property
  def bc_lambda(self):
    """Box-Cox lambda for each channel."""
    return self._bc_lambda

  @property
  def num_channels(self):
    return self._center.size

  def get_lower_bound(self):
    if self.lower_bound is None:
      return np.full(self.center.shape, -np.inf)
    else:
      return self.lower_bound

  def get_upper_bound(self):
    if self.upper_bound is None:
      return np.full(self.center.shape, np.inf)
    else:
      return self.upper_bound

  def _cast(self, source, target):
    if isinstance(target, tf.Tensor):
      return tf.convert_to_tensor(source, target.dtype)
    elif isinstance(target, np.ndarray):
      return np.array(source, target.dtype)
    elif isinstance(target, jnp.ndarray):
      return jnp.array(source, target.dtype)
    else:
      raise ValueError(f'The target must be a tf/np/jnp tensor. Got: {target}.')

  def _clip(self, t, t_min, t_max):
    if isinstance(t, tf.Tensor):
      f = tf.clip_by_value
    elif isinstance(t, np.ndarray):
      f = np.clip
    elif isinstance(t, jnp.ndarray):
      f = jnp.clip
    else:
      raise ValueError(f't must be a tf/np/jnp tensor. Got: {t}.')
    return f(t, t_min, t_max)

  def _log(self, t):
    if isinstance(t, tf.Tensor):
      f = tf.math.log
    elif isinstance(t, np.ndarray):
      f = np.log
    elif isinstance(t, jnp.ndarray):
      f = jnp.log
    else:
      raise ValueError(f't must be a tf/np/jnp tensor. Got: {t}.')
    return f(t)

  def _exp(self, t):
    if isinstance(t, tf.Tensor):
      f = tf.exp
    elif isinstance(t, np.ndarray):
      f = np.exp
    elif isinstance(t, jnp.ndarray):
      f = jnp.exp
    else:
      raise ValueError(f't must be a tf/np/jnp tensor. Got: {t}.')
    return f(t)

  def _where(self, cond, true, false):
    if isinstance(cond, tf.Tensor):
      f = tf.where
    elif isinstance(cond, np.ndarray):
      f = np.where
    elif isinstance(cond, jnp.ndarray):
      f = jnp.where
    else:
      raise ValueError(f't must be a tf/np/jnp tensor. Got: {cond}.')
    return f(cond, true, false)

  def _is_full_precision(self, dtype):
    return 'float32' in str(dtype) or 'float64' in str(dtype)

  # pylint: disable=missing-function-docstring
  def _get_reshaped_stats(self, t, channel_idx=None):
    def prepare_tensor(stat):
      if stat is None:
        return None
      if channel_idx is not None:
        assert 0 <= channel_idx and channel_idx < self.num_channels
        stat = stat[channel_idx]
      else:
        ndim = len(t.shape)
        stat = np.reshape(stat, [1] * (ndim - 1) + [-1])
      return self._cast(stat, t)
    return map(prepare_tensor,
               [self.center, self.scale, self.lower_bound, self.upper_bound,
                self.bc_lambda])

  def normalize(self, t, channel_idx=None, clip=True):
    if self.is_identity:
      return t

    assert self._is_full_precision(
        t.dtype), ('Unnormalized values require at least float32 precision.'
                   'Cast the input to the desired output dtype.')
    center, scale, lower, upper, bc_lambda = self._get_reshaped_stats(
        t, channel_idx)
    if clip and (lower is not None or upper is not None):
      if lower is None:
        lower = -np.inf
      if upper is None:
        upper = np.inf
      t = self._clip(t, lower, upper)
    if bc_lambda is not None:
      t = self._where(bc_lambda == 0,
                      self._log(t + 1e-9),
                      (t**bc_lambda - 1) / bc_lambda)
    return (t - center) / scale

  @property
  def is_bijective(self):
    return self.lower_bound is None and self.upper_bound is None

  @property
  def is_identity(self):
    return self._is_identity

  def denormalize(self, t, channel_idx=None, accept_clipped=False):
    if self.is_identity:
      return t

    assert self._is_full_precision(t.dtype), (
        'Unnormalized values require at least float32 precision.')
    assert self.is_bijective or accept_clipped, (
        'Can\'t fully denormalize because the values are already clipped. '
        'Pass accept_clipped=True if you are fine with getting clipped '
        'denormalized values.')
    center, scale, _, _, bc_lambda = self._get_reshaped_stats(t, channel_idx)
    t = t * scale + center
    if bc_lambda is not None:
      t = self._where(bc_lambda == 0,
                      self._exp(t) - 1e-9,
                      (t * bc_lambda + 1) ** (1 / bc_lambda))
    return t

  def serialize(self):
    out = {'center': self.center,
           'scale': self.scale,
           'lower_bound': self.lower_bound,
           'upper_bound': self.upper_bound,
           'bc_lambda': self.bc_lambda}
    return {k: list(v) for k, v in out.items() if v is not None}


def identity(num_channels):
  return Normalizer(center=np.zeros(num_channels),
                    scale=np.ones(num_channels))


def deserialize(kwargs):
  return Normalizer(**kwargs)


def make_normalizer_fn_from_stats(
    stats, min_scale = 0
):
  """Crete a normalizer function which expect a list of channels.

  Args:
    stats: A dict containing (center, scale) values.
    min_scale: Minimal allowed value of scale.

  Returns:
    A list of channels -> normalize function.
  """
  stats = stats.copy()
  stats['timedelta'] = (0, 1)
  def normalizer_fn(current_channels):
    return Normalizer(
        center=[stats[ch][0] for ch in current_channels],
        scale=np.clip([stats[ch][1] for ch in current_channels],
                      min_scale, np.inf))
  return normalizer_fn
