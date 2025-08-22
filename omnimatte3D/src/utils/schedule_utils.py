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

"""Functions to schedule the loss alpha."""

import optax


def cons_then_decay(cons_value, switch_iter, total_iter):
  """Keep the schedule constant util switch iter and then cosine decay.

  Args:
    cons_value: intital constant value.
    switch_iter: iteration at which to start decay.
    total_iter: total number of iterations.

  Returns:
    schedule_fn: scheduling function.
  """
  constant_schedule = optax.constant_schedule(cons_value)
  decay_fn = optax.cosine_decay_schedule(
      init_value=cons_value, decay_steps=total_iter - switch_iter
  )

  schedule_fn = optax.join_schedules(
      [constant_schedule, decay_fn], [switch_iter]
  )
  return schedule_fn


def cons_then_step_decay(cons_value, switch_iter, decay_rate):
  """Keep the schedule constant util switch iter and then cosine decay.

  Args:
    cons_value: intital constant value.
    switch_iter: iteration at which to start decay.
    decay_rate: decay rate for schedule.

  Returns:
    schedule_fn: scheduling function.
  """
  constant_schedule1 = optax.constant_schedule(cons_value)
  constant_schedule2 = optax.constant_schedule(cons_value * decay_rate)
  zero_schedule = optax.constant_schedule(0.0)

  schedule_fn = optax.join_schedules(
      [constant_schedule1, constant_schedule2, zero_schedule],
      [switch_iter, switch_iter * 2],
  )
  return schedule_fn


def cons_step(cons_value1, cons_value2, switch_iter):
  """Keep the schedule constant util switch iter and then cosine decay.

  Args:
    cons_value1: intital constant value.
    cons_value2: second constant value.
    switch_iter: iteration at which to start decay.

  Returns:
    schedule_fn: scheduling function.
  """
  constant_schedule1 = optax.constant_schedule(cons_value1)
  constant_schedule2 = optax.constant_schedule(cons_value2)

  schedule_fn = optax.join_schedules(
      [constant_schedule1, constant_schedule2], [switch_iter]
  )
  return schedule_fn


def cons(cons_value):
  """Keep the schedule constant.

  Args:
    cons_value: intital constant value.

  Returns:
    schedule_fn: scheduling function.
  """
  constant_schedule = optax.constant_schedule(cons_value)
  return constant_schedule


def warmup_cons_then_decay(cons_value, warmup_steps, switch_iter, total_iter):
  """Keep the schedule constant util switch iter and then cosine decay.

  Args:
    cons_value: intital constant value.
    warmup_steps: number of warmup steps.
    switch_iter: iteration at which to start decay.
    total_iter: total number of iterations.

  Returns:
    schedule_fn: scheduling function.
  """
  warmup_schedule = optax.linear_schedule(
      init_value=0.0, end_value=cons_value, transition_steps=warmup_steps
  )
  constant_schedule = optax.constant_schedule(cons_value)
  decay_fn = optax.cosine_decay_schedule(
      init_value=cons_value, decay_steps=total_iter - switch_iter
  )

  schedule_fn = optax.join_schedules(
      [warmup_schedule, constant_schedule, decay_fn],
      [warmup_steps, switch_iter],
  )
  return schedule_fn


def warmup_then_cons(cons_value, switch_iter):
  """Keep the schedule zero util switch iter and then at constant value.

  Args:
    cons_value: intital constant value.
    switch_iter: iteration at which to start decay.

  Returns:
    schedule_fn: scheduling function.
  """
  warmup_schedule = optax.linear_schedule(
      init_value=0.0, end_value=cons_value, transition_steps=switch_iter
  )
  constant_schedule = optax.constant_schedule(cons_value)
  schedule_fn = optax.join_schedules(
      [warmup_schedule, constant_schedule], [switch_iter]
  )
  return schedule_fn


def zero_then_cons(cons_value, switch_iter):
  """Keep the schedule zero util switch iter and then at constant value.

  Args:
    cons_value: intital constant value.
    switch_iter: iteration at which to start decay.

  Returns:
    schedule_fn: scheduling function.
  """
  zero_schedule = optax.constant_schedule(0.0)
  constant_schedule = optax.constant_schedule(cons_value)
  schedule_fn = optax.join_schedules(
      [zero_schedule, constant_schedule], [switch_iter]
  )
  return schedule_fn


def cons_then_zero(cons_value, switch_iter):
  """Keep the schedule zero util switch iter and then at constant value.

  Args:
    cons_value: intital constant value.
    switch_iter: iteration at which to start decay.

  Returns:
    schedule_fn: scheduling function.
  """
  zero_schedule = optax.constant_schedule(0.0)
  constant_schedule = optax.constant_schedule(cons_value)
  schedule_fn = optax.join_schedules(
      [constant_schedule, zero_schedule], [switch_iter]
  )
  return schedule_fn
