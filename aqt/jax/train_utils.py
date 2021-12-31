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

"""Util functions for training, which can be shared across models.
"""

from aqt.jax import quant_config


def should_quantize_weights(weight_quant_start_step, step):
  return step >= weight_quant_start_step


def should_update_bounds(activation_bound_update_freq,
                         activation_bound_start_step, step):
  """Returns whether activation bounds should be updated.

  Args:
    activation_bound_update_freq: How frequently to update bounds after the
      initial bounds update. A value of '-1' indicates to not update the bounds
      after the first update.
    activation_bound_start_step: The first step to update bounds on. '-1'
      indicates to never update bounds.
    step: The current training step.

  Returns:
    Boolean indicating whether to update the bounds on the current step.
  """
  if activation_bound_start_step < -1:
    raise ValueError("Start step must be >= -1.")
  if activation_bound_update_freq < -1 or activation_bound_update_freq == 0:
    raise ValueError("Update frequency must be a positive integer or -1.")
  steps_since_start = step - activation_bound_start_step
  if activation_bound_start_step == -1 or steps_since_start < 0:
    return False
  if activation_bound_update_freq == -1:
    return steps_since_start == 0
  else:
    return steps_since_start % activation_bound_update_freq == 0


def get_quant_context_for_step(
    *,
    activation_bound_update_freq,
    activation_bound_start_step,
    step,
    collect_acts_stats,
    prefer_int8_to_int32_dot):
  """Returns correct quantization context for a given step.

  Args:
    activation_bound_update_freq: How frequently to update bounds after the
      initial bounds update. A value of '-1' indicates to not update the bounds
      after the first update.
    activation_bound_start_step: The first step to update bounds on. '-1'
      indicates to never update bounds.
    step: The current training step.
    collect_acts_stats: Whether to collect activation statistics.
    prefer_int8_to_int32_dot: Whether to feed lax.dot inputs with an int8 dtype
      and accumulate to int32.

  Returns:
    A quant_config.QuantContext instance.
  """
  update_bounds = should_update_bounds(
      activation_bound_start_step=activation_bound_start_step,
      activation_bound_update_freq=activation_bound_update_freq,
      step=step)
  quantize_acts = step >= activation_bound_start_step
  return quant_config.QuantContext(
      update_bounds=update_bounds,
      quantize_acts=quantize_acts,
      collect_acts_stats=collect_acts_stats,
      prefer_int8_to_int32_dot=prefer_int8_to_int32_dot,
  )
