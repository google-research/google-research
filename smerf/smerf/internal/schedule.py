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

"""Hyperparameter schedules."""

import gin
import jax
import jax.numpy as jnp


def log_lerp(t, v0, v1):
  """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
  if v0 <= 0 or v1 <= 0:
    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
  lv0 = jnp.log(v0)
  lv1 = jnp.log(v1)
  return jnp.exp(jnp.clip(t, 0, 1) * (lv1 - lv0) + lv0)


class Schedule:
  pass


@gin.configurable
class ConstSchedule(Schedule):
  """Fixes the hyperparameter to a constant value: no schedule is used."""

  def __init__(self, val):
    self.val = val

  def __call__(self, step):
    return self.val

  def __repr__(self):
    return f'ConstSchedule: {self.val}'


@gin.configurable
class DictSchedule(Schedule):
  """Dictionary maps iterations to hyperparameter values."""

  def __init__(self, schedule):
    self.schedule = schedule

  def __call__(self, step):
    return [
        self.schedule[t] for t in sorted(self.schedule.keys()) if step >= t
    ][-1]

  def __repr__(self):
    return f'DictSchedule: {self.schedule}'


@gin.configurable
class LogLerpSchedule(Schedule):
  """Log-linearly interpolates a hyperparameter."""

  def __init__(self, start, end, v0, v1, zero_before_start=False):
    assert start <= end, f"{start=} must be before {end=}"
    self.start = start
    self.end = end
    self.v0 = v0
    self.v1 = v1
    self.zero_before_start = zero_before_start

  def __call__(self, step):
    def h(step):
      t = (step - self.start) / (self.end - self.start)
      return log_lerp(t, self.v0, self.v1)

    if self.zero_before_start:
      return jax.lax.cond(step < self.start, lambda x: 0.0, h, step)
    else:
      return h(step)

  def __repr__(self):
    return (
        f'LogLerpSchedule: start: {self.start}, end: {self.end}, v0: {self.v0},'
        f' v1: {self.v1}'
    )
