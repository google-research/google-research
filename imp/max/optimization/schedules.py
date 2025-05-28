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

"""Collection of the supported schedules for the optimizers."""

import optax

from imp.max.core import constants
from imp.max.optimization import config as opt_config
from imp.max.utils import typing

Schedule = constants.Schedule


def pre_warmup_cosine_decay_schedule(
    init_value,
    peak_value,
    warmup_steps,
    decay_steps,
    end_value,
    pre_warmup_steps,
    pre_warmup_init_value):
  """Applies a linear warmup before the cosine schedule."""
  schedule = optax.warmup_cosine_decay_schedule(
      init_value=init_value,
      peak_value=peak_value,
      warmup_steps=warmup_steps,
      decay_steps=decay_steps,
      end_value=end_value,
  )
  if pre_warmup_steps > 0:
    schedule = optax.join_schedules([
        optax.linear_schedule(
            init_value=pre_warmup_init_value,
            end_value=init_value,
            transition_steps=pre_warmup_steps),
        schedule,
    ], boundaries=[pre_warmup_steps])
  return schedule


_SCHEDULES = {
    Schedule.CONSTANT_LR: optax.constant_schedule,
    Schedule.COSINE_DECAY_LR: optax.cosine_decay_schedule,
    Schedule.WARMUP_COSINE_DECAY_LR: optax.warmup_cosine_decay_schedule,
    Schedule.PRE_WARMUP_COSINE_DECAY_LR: pre_warmup_cosine_decay_schedule,
}


def get_schedule(config):
  """Get the desired Learning Rate Schedule for the optimizers."""

  config = config.as_dict()
  name = config.pop("name")
  schedule = _SCHEDULES[name]

  return schedule(**config)
