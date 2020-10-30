# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Learning rate scheduler."""

import functools
import numpy as np
import tensorflow as tf


# pylint: disable=unused-argument
def constant_lr(step, lr=None):
  return tf.constant(1.0) if lr is None else tf.constant(lr)


def linear_lr(step, max_step, min_rate=0.0, lr=None):
  step = tf.cast(step, dtype=tf.float32)
  max_step = tf.cast(max_step, dtype=tf.float32)
  rate = min_rate + (1 - min_rate) * tf.clip_by_value(1.0 - step / max_step,
                                                      0.0, 1.0)
  return rate if lr is None else lr * rate


def step_lr(step, step_size=1, gamma=0.995, lr=None):
  """Implements step learning rate scheduler."""
  # Reference
  # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR
  step = tf.cast(step, dtype=tf.float32)
  step_size = tf.cast(step_size, dtype=tf.float32)
  rate = tf.pow(gamma, tf.math.floordiv(step, step_size))
  rate = tf.clip_by_value(rate, 0.0, 1.0)
  return rate if lr is None else lr * rate


def cosine_annealing_lr(step, max_step, min_rate=0.0, lr=None):
  """Implements cosine learning rate scheduler."""
  # Reference
  # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR

  step = tf.cast(step, dtype=tf.float32)
  max_step = tf.cast(max_step, dtype=tf.float32)
  rate = min_rate + 0.5 * (1 - min_rate) * (1 + tf.cos(np.pi *
                                                       (step / max_step)))
  rate = tf.clip_by_value(rate, 0.0, 1.0)
  return rate if lr is None else lr * rate


def half_cosine_annealing_lr(step, max_step, level=7, lr=None):
  """Implements cosine learning rate decay schedule proposed by FixMatch."""
  # Reference:
  #  https://arxiv.org/abs/2001.07685
  #  https://github.com/google-research/fixmatch/blob/08d9b83d7cc87e853e6afc5a86b12aacff56cdea/fixmatch.py#L93
  step = tf.cast(step, dtype=tf.float32)
  max_step = tf.cast(max_step, dtype=tf.float32)
  assert level <= 8, 'level should be an integer less than or equal to 8'
  ratio = tf.divide(tf.cast(level, dtype=tf.float32), 16.0)
  rate = tf.cos(np.pi * (step / max_step) * ratio)
  rate = tf.clip_by_value(rate, 0.0, 1.0)
  return rate if lr is None else lr * rate


class CustomLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Custom Learning Rate Schedule."""

  def __init__(self,
               step_per_epoch,
               max_step,
               base_lr,
               mode='constant',
               **kwargs):
    super(CustomLearningRateSchedule, self).__init__()
    self.step_per_epoch = step_per_epoch
    self.max_step = max_step
    self.base_lr = base_lr
    assert mode in [
        'const', 'constant', 'linear', 'step', 'cosine', 'cos', 'halfcosine',
        'halfcos'
    ], 'Scheduler {} is not implemented'.format(mode)
    self.mode = mode
    for key in kwargs:
      if not hasattr(self, key):
        setattr(self, key, kwargs[key])
    self._check_mode()

  def _check_mode(self):
    """Check whether all variables are correctly set for each mode."""
    if self.mode in ['const', 'constant']:
      self.scheduler = functools.partial(constant_lr, lr=self.base_lr)
      self.name = 'const'

    if self.mode in ['linear']:
      if not hasattr(self, 'min_rate'):
        setattr(self, 'min_rate', 0.0)
      self.scheduler = functools.partial(
          linear_lr,
          max_step=self.max_step,
          min_rate=self.min_rate,
          lr=self.base_lr)
      self.name = 'linear_step{}_rate{}'.format(self.step_per_epoch,
                                                self.min_rate)

    elif self.mode in ['step']:
      if not hasattr(self, 'step_size'):
        setattr(self, 'step_size', 1)
      if not hasattr(self, 'gamma'):
        setattr(self, 'gamma', 0.995)
      self.scheduler = functools.partial(
          step_lr,
          step_size=self.step_size * self.step_per_epoch,
          gamma=self.gamma,
          lr=self.base_lr)
      self.name = 'step_step{}_size{}_gamma{}'.format(self.step_per_epoch,
                                                      self.step_size,
                                                      self.gamma)

    elif self.mode in ['cosine', 'cos']:
      if not hasattr(self, 'min_rate'):
        setattr(self, 'min_rate', 0.0)
      self.scheduler = functools.partial(
          cosine_annealing_lr,
          max_step=self.max_step,
          min_rate=self.min_rate,
          lr=self.base_lr)
      self.name = 'cos_step{}_rate{}'.format(self.step_per_epoch, self.min_rate)

    elif self.mode in ['halfcosine', 'halfcos']:
      if not hasattr(self, 'level'):
        setattr(self, 'level', 7)
      self.scheduler = functools.partial(
          half_cosine_annealing_lr,
          max_step=self.max_step,
          level=self.level,
          lr=self.base_lr)
      self.name = 'hcos_step{}_level{}'.format(self.step_per_epoch, self.level)

  def __call__(self, step):
    lr_step = self.step_per_epoch * tf.cast(
        tf.math.floordiv(step, self.step_per_epoch), dtype=tf.float32)
    return self.scheduler(step=lr_step)
