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

"""Training utils."""

import math
from typing import Any, Callable, Optional

from big_vision import utils as bv_utils
from big_vision.utils import create_learning_rate_schedule as bv_create_learning_rate_schedule
import flax
from flax import struct
import jax
import jax.numpy as jnp


# pytype:disable=attribute-error
@struct.dataclass
class ExponentialMovingAverage:
  """Exponential Moving Average as implemented in Tensorflow."""

  # Moving average of the parameters.
  state: Any
  # Decay to use for the update (typical values are 0.999, 0.9999, etc...).
  decay: float
  # For how many steps we should just keep the new parameters instead of an
  # average (useful if we don't want the initial weights to be included in the
  # average).
  warmup_steps: int

  def update_moving_average(self, new_target,
                            step):
    """Updates the moving average of the target.

    Args:
      new_target: New values of the target (example: weights of a network
        after gradient step).
      step: Current step (used only for warmup).

    Returns:
      The updated ExponentialMovingAverage.
    """
    factor = jnp.float32(step >= self.warmup_steps)
    delta = step - self.warmup_steps
    decay = jnp.minimum(self.decay, (1. + delta) / (10. + delta))
    decay *= factor
    new_target = flax.core.FrozenDict(new_target)
    state = flax.core.FrozenDict(self.state)
    weight_ema = jax.tree.map(lambda a, b: (1 - decay) * a + decay * b,
                              new_target, state)
    return self.replace(state=weight_ema)
# pytype:enable=attribute-error


def create_exponential_rate_schedule(global_batch_size,
                                     total_steps,
                                     steps_per_epoch = None,
                                     base = 0.0,
                                     scale_with_batchsize = False,
                                     warmup_steps = 0,
                                     cooldown_steps = 0,
                                     warmup_epochs = 0,
                                     cooldown_epochs = 0,
                                     **kw):
  """Creates exponential learning rate schedule.

  Args:
    global_batch_size: The global batch-size optionally used for scaling.
    total_steps: The total number of steps to run.
    steps_per_epoch: How many steps form an epoch. Needed only if anything is
      passed in terms of epochs.
    base: The starting learning-rate (without warmup).
    scale_with_batchsize: Whether or not to scale lr automatically.
    warmup_steps: how many steps to warm up for.
    cooldown_steps: how many steps to cool down for.
    warmup_epochs: how many epochs to warm up for.
    cooldown_epochs: how many epochs to cool down for.
    **kw: extra arguments specific to individual decay_types.

  Returns:
    A function learning_rate(step): float -> {"learning_rate": float}.
  """

  # For convenience, convert {warmup,cooldown}_epochs to _steps.
  assert bool(warmup_epochs) + bool(warmup_steps) < 2, "Only one!"
  assert bool(cooldown_epochs) + bool(cooldown_steps) < 2, "Only one!"
  if warmup_epochs:
    warmup_steps = warmup_epochs * steps_per_epoch
  assert warmup_steps < total_steps, "warmup_steps is >= total_steps"
  if cooldown_epochs:
    cooldown_steps = cooldown_epochs * steps_per_epoch

  def step_fn(step):
    """Step to learning rate function."""
    lr = base

    # This implements the linear scaling rule following
    # Goyal et al. at arxiv.org/abs/1706.02677.
    # The reference batch size in literature is 256, so we scale the lr to
    # adjust to the literature lr when bach_size changes.
    if scale_with_batchsize:
      lr = lr * global_batch_size / 256.0

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = jnp.clip(progress, 0.0, 1.0)

    # At the end of the training, lr should be 1.2% of original value.
    # This mimic the behavior from the efficientnet paper.
    end_lr_ratio = kw.get("end_lr_ratio", 0.012)
    lr = lr * jnp.exp(progress * math.log(end_lr_ratio))

    if warmup_steps:
      lr = lr * jnp.minimum(1., step / warmup_steps)
    if cooldown_steps:
      lr = lr * jnp.minimum(1., (total_steps - step) / cooldown_steps)

    return jnp.asarray(lr, dtype=jnp.float32)

  return step_fn


def create_learning_rate_schedule(*args,
                                  decay_type = "stair",
                                  **kwargs):
  if decay_type != "exponential":
    return bv_create_learning_rate_schedule(*args, decay_type=decay_type,
                                            **kwargs)
  else:
    return create_exponential_rate_schedule(*args, **kwargs)


bv_utils.create_learning_rate_schedule = create_learning_rate_schedule
