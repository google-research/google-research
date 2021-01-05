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

# Lint as: python3
"""Learning rate schedules."""

import abc
import functools
from typing import Optional

import dataclasses
import gin
import numpy as np

from gfsa import jax_util


class LearningRateSchedule(abc.ABC):
  """Defines a learning rate schedule.

  All implementations of this class should be Flax-serializable objects, so they
  can work seamlessly with the rest of checkpointing.
  """

  @abc.abstractmethod
  def learning_rate_for_step(self, step):
    """Returns the learning rate to use for this training step.

    Args:
      step: The training step.

    Returns:
      Learning rate.
    """
    raise NotImplementedError()

  def update_with_validation(
      self,
      validation_loss,
  ):
    """Update the learning rate schedule with a validation loss.

    Args:
      validation_loss: Current validation loss.
    """
    # By default, we ignore validation loss.
    del validation_loss
    pass


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class ConstantLearningRateSchedule(LearningRateSchedule):
  """Learning rate schedule that just uses a constant value.

  Attributes:
    learning_rate: The constant learning rate.
  """
  learning_rate: float

  def learning_rate_for_step(self, step):
    return self.learning_rate


# Gin tries to wrap classes in subclasses, which flax can't serialize. So we
# wrap the constructor instead.
gin.external_configurable(
    functools.partial(ConstantLearningRateSchedule),
    "ConstantLearningRateSchedule",
    module="learning_rate_schedules")


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class InverseTimeLearningRateSchedule(LearningRateSchedule):
  """Learning rate schedule that decays as 1/(1 + t/c).

  Attributes:
    initial_rate: Initial learning rate.
    time_scale: Scale of the learning rate.
  """
  initial_rate: float
  time_scale: float

  def learning_rate_for_step(self, step):
    return self.initial_rate / (1 + step / self.time_scale)


gin.external_configurable(
    functools.partial(InverseTimeLearningRateSchedule),
    "InverseTimeLearningRateSchedule",
    module="learning_rate_schedules")


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class LinearThenInverseSquaredLearningRateSchedule(LearningRateSchedule):
  """Learning rate schedule based on "Attention is all you need".

  Attributes:
    warmup_steps: Number of steps to warm up the optimizer.
    max_learning_rate: Maximum learning rate. "Attention is all you need"
      suggest using 1 / sqrt(warmup_steps * model_feature_dim_size).
  """
  max_learning_rate: float
  warmup_steps: int

  def learning_rate_for_step(self, step):
    linear_ramp = step * np.power(self.warmup_steps, -1.5)
    inverse_falloff = 1 / np.sqrt(step)
    coefficient = self.max_learning_rate * np.sqrt(self.warmup_steps)
    return coefficient * np.minimum(linear_ramp, inverse_falloff)


gin.external_configurable(
    functools.partial(LinearThenInverseSquaredLearningRateSchedule),
    "LinearThenInverseSquaredLearningRateSchedule")


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class ValidationBasedLearningRateSchedule(LearningRateSchedule):
  """Learning rate schedule that drops the learning rate based on validation.

  Attributes:
    rate: Learning rate to use.
    stagnant_valid_steps_per_decay: How many validation steps must not show
      improvement before we drop the learning rate.
    decay_factor: Multiplicative factor to apply when the validation loss
      doesn't improve for `stagnant_valid_steps_per_decay` steps.
  """

  rate: float
  stagnant_valid_steps_per_decay: int
  decay_factor: float

  _steps_since_improvement: int = 0
  _best_valid: Optional[float] = None

  def learning_rate_for_step(self, step):
    return self.rate

  def update_with_validation(self, validation_loss):
    if self._best_valid is None or validation_loss < self._best_valid:
      self._best_valid = validation_loss
      self._steps_since_improvement = 0
    elif self._steps_since_improvement + 1 >= self.stagnant_valid_steps_per_decay:
      self.rate *= self.decay_factor
      # Reset our best validation to our current performance
      self._best_valid = validation_loss
      self._steps_since_improvement = 0
    else:
      self._steps_since_improvement += 1


gin.external_configurable(
    functools.partial(ValidationBasedLearningRateSchedule),
    "ValidationBasedLearningRateSchedule",
    module="learning_rate_schedules")
