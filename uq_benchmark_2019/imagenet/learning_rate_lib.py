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

# Lint as: python2, python3
"""Library for managing the learning-rate schedule for ResNet50 on Imagenet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
keras = tf.keras
kb = tf.keras.backend

BASE_LEARNING_RATE = 0.4
LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def learning_rate_schedule_wrapper(training_steps_per_epoch):
  """Wrapper around the learning rate schedule."""

  def learning_rate_schedule(current_epoch, current_batch):
    """Handles linear scaling rule, gradual warmup, and LR decay.

    The learning rate starts at 0, then it increases linearly per step.
    After 5 epochs we reach the base learning rate (scaled to account
      for batch size).
    After 30, 60 and 80 epochs the learning rate is divided by 10.
    After 90 epochs training stops and the LR is set to 0. This ensures
      that we train for exactly 90 epochs for reproducibility.

    Args:
      current_epoch: integer, current epoch indexed from 0.
      current_batch: integer, current batch in current epoch, indexed from 0.

    Returns:
      Adjusted learning rate.
    """
    epoch = current_epoch + float(current_batch) / training_steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
    if epoch < warmup_end_epoch:
      # Learning rate increases linearly per step.
      return (BASE_LEARNING_RATE * warmup_lr_multiplier * epoch /
              warmup_end_epoch)
    for mult, start_epoch in LR_SCHEDULE:
      if epoch >= start_epoch:
        learning_rate = BASE_LEARNING_RATE * mult
      else:
        break
    return learning_rate

  return learning_rate_schedule


class LearningRateBatchScheduler(keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.
  """

  def __init__(self, schedule):
    """Initialize LearningRateBatchScheduler object.

    Args:
      schedule: a function that takes an epoch index and a batch index as input
        (both integer, indexed from 0) and returns a new learning rate as output
        (float).
    """
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.epochs = -1
    self.prev_lr = -1

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    lr = self.schedule(self.epochs, batch)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      kb.set_value(self.model.optimizer.lr, lr)
      self.prev_lr = lr
      logging.debug(
          'Epoch %05d Batch %05d: LearningRateBatchScheduler change '
          'learning rate to %s.', self.epochs, batch, lr)
