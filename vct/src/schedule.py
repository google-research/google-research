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

"""Defines functions for 'schedules', e.g. for a learning rate."""

import functools
from typing import Sequence

import tensorflow as tf

TensorLike = tf.types.experimental.TensorLike


def piecewise_constant_schedule(step, boundaries, values):
  """Piecewise constant between boundaries and interval values."""
  # If no boundaries, function is constant.
  if len(values) != len(boundaries) + 1:
    raise ValueError("The number of values must be one more than the number "
                     f"of boundaries: {len(values)} != {len(boundaries)+1}")
  step = tf.convert_to_tensor(step)
  # Cast `boundaries` to have the same type as `step`.
  boundaries = tf.convert_to_tensor(boundaries, dtype=step.dtype)
  values = tf.convert_to_tensor(values)
  index = tf.math.reduce_sum(
      tf.cast(boundaries <= tf.expand_dims(step, axis=-1), tf.int32), axis=-1)
  return tf.gather(values, index)


def schedule_at_step(step,
                     vals,
                     boundaries = (),
                     warmup_steps = 0,
                     ):
  """Computes the schedule value at a given step `step`.

  Args:
    step: The step to compute the schedule value at.
    vals: Sequence of values.
    boundaries: Locations where the schedule changes between values in `vals`.
      If empty, `vals` should be a sequence with exactly one element.
    warmup_steps:  Number of steps at the beginning of training to use as
      warmup. Set to non-positive to disable.

  Returns:
    The computed schedule value at step `step`.
  """
  step = tf.convert_to_tensor(step)
  value = piecewise_constant_schedule(step, boundaries, vals)
  if warmup_steps > 0:
    # Applies linear warmup, over the first `warmup_steps` steps.
    value *= tf.minimum(1., (tf.cast(step, tf.float32) + 1) / warmup_steps)
  return value


class KerasSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Returns `schedule_at_step` above in the form of a KerasSchedule.

  Here the schedule is multiplicative over a provided base value.
  Example usage:

  learning_rate_schedule = schedule.KerasSchedule(
      base_value=0.1,
      vals=[8, 4, 2],
      boundaries=[10, 15],
      interpolation="linear",
  )
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
  """

  def __init__(self, base_value = 1.0, **kwargs):
    """Initializes the schedule.

    Args:
      base_value: A base value that is multiplied with the scheduled value.
      **kwargs: Schedule configuration compatible with
        `schedules.schedule_at_step`.
    """
    self._base_value = tf.convert_to_tensor(base_value, tf.float32)
    self._schedule_at_step = functools.partial(schedule_at_step, **kwargs)

  def __call__(self, step):
    return self._base_value * self._schedule_at_step(step)
