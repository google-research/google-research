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

"""Custom optimizers.

Currently only contains AdamWithSeparateLRSchedule, which is used to train
product key memory layers that have a fixed learning rate schedule while the
rest of the model uses a variable one.
Name contains redundant mention of AT5 to facilitate gin invocation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re

import gin
import mesh_tensorflow as mtf
from mesh_tensorflow import optimize
from mesh_tensorflow.transformer import learning_rate_schedules
from mesh_tensorflow.transformer import utils

import tensorflow.compat.v1 as tf


@gin.configurable
class AdamWithSeparateLRSchedule(optimize.AdamWeightDecayOptimizer):
  """An  Adam optimizer that includes "correct" L2 weight decay.

  Uses a separate, constant LR for variables matching a specific regex. This is
  used for instance in ProductKeyMemories, where we want to keep a high,
  constant learning rate for the value embeddings that only get sparsely
  updated.
  """

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               constant_learning_rate=0.02,
               variable_constant_lr="values"):
    if exclude_from_weight_decay is None:
      exclude_from_weight_decay = [variable_constant_lr]
    else:
      exclude_from_weight_decay.append(variable_constant_lr)
    super(AdamWithSeparateLRSchedule, self).__init__(
        learning_rate=learning_rate,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        exclude_from_weight_decay=exclude_from_weight_decay,
    )
    self.constant_learning_rate = constant_learning_rate
    self.variable_constant_lr = variable_constant_lr

  def apply_grad(self, grad, var):
    # Modify learning rate for exception variables
    if re.search(self.variable_constant_lr, var.name) is not None:
      old_lr = self.learning_rate
      self.learning_rate = self.constant_learning_rate
      assignments = super(AdamWithSeparateLRSchedule,
                          self).apply_grad(grad, var)
      self.learning_rate = old_lr
    else:
      assignments = super(AdamWithSeparateLRSchedule,
                          self).apply_grad(grad, var)
    return assignments


def compute_lr_for_step(schedules, learning_rate, batch_size, sequence_length):
  """Get actual LR for step."""
  actual_lr_rates = []
  for lr_schedule in schedules:
    if lr_schedule is None:
      actual_lr_rates.append(learning_rate)
    else:
      converted_schedule = functools.partial(
          learning_rate_schedules.product_learning_rate,
          factors=lr_schedule)
      train_steps = utils.auto_train_steps(batch_size,
                                           sequence_length)
      converted_schedule = functools.partial(
          converted_schedule, total_train_steps=train_steps)
      if callable(converted_schedule):
        # the following happens on CPU since TPU can't handle summaries.
        with mtf.utils.outside_all_rewrites():
          converted_schedule = converted_schedule(
              step=tf.train.get_global_step())
          tf.summary.scalar("alt_learning_rate", converted_schedule)
      actual_lr_rates.append(converted_schedule)
  return actual_lr_rates


@gin.configurable
def learning_rate_cut_off(step,
                          total_train_steps,
                          cut_off_step=gin.REQUIRED):
  """Cuts off the learning rate after a certain number of steps.

  This sets LR to zero for some parameters. Instead of not updating
  the parameters altogether, this method was a little cleaner because
  its composable with other factors and does not branch off to not
  update certain parameters.

  This is used in Weighted Transformer to freeze weights.

  TODO(yitay): Consider moving into MTF later

  Args:
    step: a tf.scalar representing the step we want the learning rate for.
    total_train_steps: total train steps
    cut_off_step: a number, the cut off to reduce LR to zero.

  Returns:
    a tf.Scalar, the learning rate for the step.
  """
  del total_train_steps
  step = tf.cast(step, tf.float32)
  return tf.cond(tf.math.greater_equal(step, cut_off_step),
                 lambda: tf.constant(0.0, dtype=tf.float32), lambda: step)
