# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Define learning rate schedule classes."""

import numpy as np
import tensorflow as tf

from vatt.configs import experiment


class CosineLearningRateWithLinearWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Class to generate learning rate tensor."""

  def __init__(self, warmup_learning_rate, warmup_steps, init_learning_rate,
               total_steps):
    """Creates the consine learning rate tensor with linear warmup."""
    super(CosineLearningRateWithLinearWarmup, self).__init__()
    self.warmup_learning_rate = warmup_learning_rate
    self.warmup_steps = warmup_steps
    self.init_learning_rate = init_learning_rate
    self.total_steps = total_steps

  def __call__(self, global_step):
    global_step = tf.cast(global_step, dtype=tf.float32)
    warmup_lr = self.warmup_learning_rate
    warmup_steps = self.warmup_steps
    init_lr = self.init_learning_rate
    total_steps = self.total_steps
    linear_warmup = (
        warmup_lr + global_step / warmup_steps * (init_lr - warmup_lr))
    cosine_learning_rate = (
        init_lr * (tf.cos(np.pi * (global_step - warmup_steps) /
                          (total_steps - warmup_steps)) + 1.0) / 2.0)
    learning_rate = tf.where(global_step < warmup_steps, linear_warmup,
                             cosine_learning_rate)
    return learning_rate

  def get_config(self):
    return dict(
        warmup_learning_rate=self.warmup_learning_rate,
        warmup_steps=self.warmup_steps,
        init_learning_rate=self.init_learning_rate,
        total_steps=self.total_steps)


class StepCosineLearningRateWithLinearWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Class to generate learning rate tensor."""

  def __init__(self, warmup_learning_rate, warmup_steps, learning_rate_levels,
               learning_rate_steps, total_steps):
    """Creates the consine learning rate tensor with linear warmup."""
    super(StepCosineLearningRateWithLinearWarmup, self).__init__()
    self.warmup_learning_rate = warmup_learning_rate
    self.warmup_steps = warmup_steps
    self.learning_rate_levels = learning_rate_levels
    self.learning_rate_steps = learning_rate_steps
    self.total_steps = (
        [learning_rate_steps[n + 1] - learning_rate_steps[n]
         for n in range(len(learning_rate_steps) - 1)]
        + [total_steps - learning_rate_steps[-1]]
        )
    err_msg = ("First level's steps should be equal to warmup steps"
               ", but received {} warmup steps and {} first level steps".format(
                   self.warmup_steps, self.learning_rate_steps[0])
               )
    assert self.warmup_steps == self.learning_rate_steps[0], err_msg

  def __call__(self, global_step):
    global_step = tf.cast(global_step, dtype=tf.float32)
    warmup_lr = self.warmup_learning_rate
    warmup_steps = self.warmup_steps
    lr_levels = self.learning_rate_levels
    lr_steps = self.learning_rate_steps
    total_steps = self.total_steps
    num_levels = len(lr_levels)

    init_lr = lr_levels[0]
    next_init_lr = lr_levels[1] if num_levels > 1 else 0.

    init_total_steps = total_steps[0]
    linear_warmup = (
        warmup_lr + global_step / warmup_steps * (init_lr - warmup_lr))

    cosine_learning_rate = (
        (init_lr - next_init_lr)
        * (tf.cos(np.pi * (global_step - warmup_steps) / (init_total_steps))
           + 1.0) / 2.0
        + next_init_lr
        )
    learning_rate = tf.where(global_step < warmup_steps, linear_warmup,
                             cosine_learning_rate)

    for n in range(1, num_levels):
      next_init_lr = lr_levels[n]
      next_start_step = lr_steps[n]
      next_total_steps = total_steps[n]
      next_next_init_lr = lr_levels[n+1] if num_levels > n+1 else 0.

      next_cosine_learning_rate = (
          (next_init_lr - next_next_init_lr)
          * (tf.cos(np.pi * (global_step - next_start_step)
                    / (next_total_steps))
             + 1.0) / 2.0
          + next_next_init_lr
          )
      learning_rate = tf.where(global_step >= next_start_step,
                               next_cosine_learning_rate, learning_rate)

    return learning_rate

  def get_config(self):
    return dict(
        warmup_learning_rate=self.warmup_learning_rate,
        warmup_steps=self.warmup_steps,
        learning_rate_levels=self.learning_rate_levels,
        learning_rate_steps=self.learning_rate_steps,
        total_steps=self.total_steps)


def get_learning_rate(lr_config):
  """Factory function for learning rate object."""
  if isinstance(lr_config, experiment.CosineDecayLearningRate):
    learning_rate = CosineLearningRateWithLinearWarmup(
        warmup_learning_rate=lr_config.warmup_learning_rate,
        warmup_steps=lr_config.warmup_steps,
        init_learning_rate=lr_config.learning_rate_base,
        total_steps=lr_config.total_steps)
  elif isinstance(lr_config, experiment.StepwiseCosineDecayLearningRate):
    learning_rate = StepCosineLearningRateWithLinearWarmup(
        warmup_learning_rate=lr_config.warmup_learning_rate,
        warmup_steps=lr_config.warmup_steps,
        learning_rate_levels=lr_config.learning_rate_levels,
        learning_rate_steps=lr_config.learning_rate_steps,
        total_steps=lr_config.total_steps)
  elif isinstance(lr_config, experiment.LearningRate):
    learning_rate = lr_config.learning_rate_base
  else:
    raise ValueError('Unknown type of learning rate: {!r}'.format(lr_config))
  return learning_rate
