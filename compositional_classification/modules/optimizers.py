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

# coding=utf-8
"""Defines the optimizers used in the classification experiments."""

import math

import tensorflow as tf
from tensorflow.keras.optimizers import schedules


class InverseSqrtDecay(schedules.LearningRateSchedule):
  """Inverse square root learning rate decay with linear warm-up."""

  def __init__(self, hparams):
    self.warmup_steps = hparams.learning_rate_warmup_steps
    self.base_learning_rate = hparams.learning_rate

  def __call__(self, step):
    if step < self.warmup_steps:
      factor = step / self.warmup_steps
    else:
      factor = math.sqrt(self.warmup_steps / step)
    learning_rate = self.base_learning_rate * factor
    return learning_rate

  def get_config(self):
    return {
        'warmup_steps': self.warmup_steps,
        'base_learning_rate': self.base_learning_rate
    }


def get_lr_schedule(
    hparams
):
  """Creates a learning rate schedule from hparams."""
  if hparams.learning_rate_schedule == 'constant':
    lr = hparams.learning_rate
  elif hparams.learning_rate_schedule == 'decay':
    lr = schedules.ExponentialDecay(hparams.learning_rate,
                                    hparams.learning_rate_decay_steps,
                                    hparams.learning_rate_decay_rate,
                                    hparams.learning_rate_decay_staircase)
  elif hparams.learning_rate_schedule == 'rsqrt':
    lr = InverseSqrtDecay(hparams)
  else:
    raise ValueError(f'Wrong lr schedule: {hparams.learning_rate_schedule}')
  return lr


def get_optimizer(hparams):
  """Takes lr schedule and creates an optimizer."""
  lr_schedule = get_lr_schedule(hparams)
  if hparams.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=hparams.optimizer_adam_beta1,
        beta_2=hparams.optimizer_adam_beta2,
        epsilon=hparams.optimizer_adam_epsilon)
  else:
    raise ValueError(f'Wrong optimizer: {hparams.optimizer}')
  return optimizer


def get_lr(optimizer):
  """Retrieves the learning rate of current step."""
  learning_rate = optimizer.learning_rate
  if isinstance(learning_rate, schedules.LearningRateSchedule):
    learning_rate = learning_rate(optimizer.iterations)
  return float(learning_rate)
