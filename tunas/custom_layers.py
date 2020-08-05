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

# Lint as: python2, python3
"""Custom neural network layers built on top of TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def update_exponential_moving_average(tensor, momentum, name=None):
  """Returns an exponential moving average of `tensor`.

  We will update the moving average every time the returned `tensor` is
  evaluated. A zero-debias will be applied, so we will return unbiased
  estimates during the first few training steps.

  Args:
    tensor: A floating point tensor.
    momentum: A scalar floating point Tensor with the same dtype as `tensor`.
    name: Optional string, the name of the operation in the TensorFlow graph.

  Returns:
    A Tensor with the same shape and dtype as `tensor`.
  """
  with tf.variable_scope(name,
                         'update_exponential_moving_average',
                         [tensor, momentum]):
    # NOTE: We force `numerator` and `denominator` to be resource variables.
    # This ensures that any writes performed on those variables will be
    # reflected by subsequent calls to read_value().
    numerator = tf.get_variable(
        'numerator', initializer=0.0, trainable=False, use_resource=True)
    denominator = tf.get_variable(
        'denominator', initializer=0.0, trainable=False, use_resource=True)
    update_ops = [
        numerator.assign(momentum*numerator + (1-momentum)*tensor),
        denominator.assign(momentum*denominator + (1-momentum)),
    ]
    with tf.control_dependencies(update_ops):
      return numerator.read_value() / denominator.read_value()


def cosine_decay_with_linear_warmup(peak_learning_rate,
                                    global_step,
                                    max_global_step,
                                    warmup_steps=0):
  """linearly warmup the learning rate before switching to cosine decay.

  Args:
    peak_learning_rate: Initial learning rate at the beginning of cosine
        decay (also at the end of the linear warmup).
    global_step: A scalar int32 or int64 Tensor or a Python number. Global
        step to use for the decay computation.
    max_global_step: Maximum global step for model training.
    warmup_steps: Number of warmup steps at the beginning of training.

  Returns:
    Learning rate following the linear warmup and cosine decay schedule.
  """
  cosine_lr = tf.train.cosine_decay(
      learning_rate=peak_learning_rate,
      global_step=global_step - warmup_steps,
      decay_steps=max_global_step - warmup_steps)
  if warmup_steps > 0:
    warmup_lr = peak_learning_rate * tf.cast(
        global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    learning_rate = tf.cond(
        global_step < warmup_steps, lambda: warmup_lr, lambda: cosine_lr)
  else:
    learning_rate = cosine_lr
  return learning_rate


def linear_warmup(global_step, warmup_steps, name=None):
  """"Linearly increase the output from 0 to 1 over `warmup_steps` steps.

  Args:
    global_step: A scalar int32 or int64 Tensor or a Python integer.
    warmup_steps: Non-negative integer, number of warmup steps at the beginning
        of training.
    name: Optional name for the TensorFlow scope.

  Returns:
    A scalar float Tensor between 0 and 1.
  """
  with tf.name_scope(name, 'linear_warmup', [global_step, warmup_steps]):
    if warmup_steps < 0:
      raise ValueError(
          'Invalid warmup_steps, must be a non-negative integer: {}'
          .format(warmup_steps))
    elif warmup_steps == 0:
      return tf.constant(1.0, dtype=tf.float32)
    else:  # warmup_steps > 0
      global_step_float = tf.cast(global_step, tf.float32)
      warmup_steps_float = tf.cast(warmup_steps, tf.float32)
      result = global_step_float / warmup_steps_float
      result = tf.where_v2(global_step >= warmup_steps, 1.0, result)
      result = tf.where_v2(global_step <= 0, 0.0, result)
      return result


def linear_decay(global_step, decay_steps, name=None):
  """Output linearly decreases from 1 to 0 over this many steps of training."""
  with tf.name_scope(name, 'linear_decay', [global_step, decay_steps]) as scope:
    return 1.0 - linear_warmup(global_step, decay_steps, name=scope)


class TransposedInitializer(tf.keras.initializers.Initializer):
  """Create an initializer for the transposed tensor."""

  def __init__(self, initializer):
    self._initializer = initializer

  def __call__(self, shape, dtype=None, partition_info=None):
    original_shape = [shape[0], shape[1], shape[3], shape[2]]
    output = self._initializer(original_shape, dtype, partition_info)
    transposed_output = tf.transpose(output, [0, 1, 3, 2])
    return transposed_output

  def get_config(self):
    return self._initializer.get_config()
