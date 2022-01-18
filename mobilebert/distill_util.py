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

"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def layer_wise_learning_rate(layer_id,
                             steps_per_phase,
                             binary=False,
                             background_lr=1.0):
  """Calculate the learning rate for progressive knowledge transfer."""
  global_step = tf.train.get_or_create_global_step()
  start_warmup_steps = steps_per_phase * layer_id
  boundaries = [start_warmup_steps, start_warmup_steps + steps_per_phase]
  values = [0.0, 1.0, 0.1]
  layer_wise_lr = background_lr * tf.train.piecewise_constant(
      global_step, boundaries, values)
  if not binary:
    return 3.0 * layer_wise_lr
  else:
    layer_wise_gate = tf.where(tf.math.greater(layer_wise_lr, 0.5), 1.0, 0.0)
    return layer_wise_gate


def get_background_lr(global_step, steps_per_phase):
  """A periodic linear-warmp background learning rate schedule."""
  truncated_global_step = tf.mod(global_step, steps_per_phase)
  if steps_per_phase > 1:
    background = tf.minimum(
        tf.train.polynomial_decay(
            1.0,
            truncated_global_step,
            steps_per_phase,
            end_learning_rate=0.9,
            power=0.5),
        tf.train.polynomial_decay(
            0.0,
            truncated_global_step,
            steps_per_phase,
            end_learning_rate=10.0,
            power=1.0))
  else:
    background = 1.0
  return background
