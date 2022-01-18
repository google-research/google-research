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

# Lint as: python2, python3
"""Utilities for optimizing TensorFlow network architectures on TPU.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, List, Optional, Sequence, Text, Tuple, Union

from six.moves import zip
import tensorflow.compat.v1 as tf


def _cross_replica_mean(x):
  number_of_shards = tf.tpu.cross_replica_sum(tf.ones(dtype=x.dtype, shape=()))
  return tf.tpu.cross_replica_sum(x) / number_of_shards


_TensorAndVarList = List[Tuple[tf.Tensor, tf.Variable]]


def apply_adam(
    loss,
    global_step = None,
    var_list = None,
    learning_rate = 1e-3,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,
    regularization_loss = None,
    transform_grads_fn = None,
    name = None):
  """Alternate implementation of ADAM optimizer for TPU.

  There are a few subtle but important differences between this optimizer and
  what you'd get if you used TF's default CrossShardOptimizer(AdamOptimizer()).
    * In the standard TF implementation, we'd update the second moment estimate
      of the gradients (`v`) by first averaging the gradients across all
      replicas and then squaring the result. In our implementation, we square
      the gradients first, then average. This leads to much lower effective
      learning rates when there are large gradient updates in random directions.
      It's also closer to the behavior we'd get by applying updates
      sequentially.
    * We support a separate `regularization_loss` argument. If provided, we
      behave as if two separate optimizers were applied at each training step:
      an ADAM optimizer for `loss` and an SGD optimizer for
      `regularization_loss`. This makes it possible to apply decoupled
      AdamW-style weight decay, as suggested by Loschilov and Hutter.

  References:
    * Kingma and Ba. "Adam: A Method for Stochastic Optimization."
      https://arxiv.org/pdf/1412.6980.pdf
    * Loshchilov and Hutter. "Fixing Weight Decay Regularization in Adam."
      https://arxiv.org/pdf/1711.05101.pdf

  Args:
    loss: Tensor containing the value to minimize.
    global_step: Optional `Variable` to increment by one after the
        variables have been updated.
    var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in
        the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
    learning_rate: Tensor or float. The learning rate.
    beta1: Float. The exponential decay rate for the 1st moment estimates.
    beta2: Float. The exponential decay rate for the 2nd moment estimates.
    epsilon: Float. A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
    regularization_loss: Optional float tensor. If provided, we will act as if
        the model had two separate optimizers: an ADAM optimizer for `loss` and
        a separate SGD optimizer (without moving average accumulators) for
        `regularization_loss`.
    transform_grads_fn: Optional function to apply to the gradients before using
        them. The input should be a list `grads_and_vars` containing
        `(tensor, variable)` tuples, and the output should have the same type.
    name: Optional name for the resulting TensorFlow subgraph.

  Returns:
    A tf.Operation which will update the model's trainable variables and moving
    average accumulators.
  """
  if var_list is None:
    var_list = tf.trainable_variables()

  with tf.variable_scope('tpu_apply_adam', name,
                         [loss, var_list, learning_rate, beta1, beta2, epsilon],
                         use_resource=True):
    grad_list = tf.gradients(loss, var_list)
    if transform_grads_fn is not None:
      grad_list, _ = list(
          zip(*transform_grads_fn(list(zip(grad_list, var_list)))))

    if regularization_loss is not None:
      reg_grad_list = tf.gradients(regularization_loss, var_list)
    else:
      reg_grad_list = [None] * len(var_list)

    # Convert arguments to tensors
    learning_rate_t = tf.convert_to_tensor(learning_rate, name='learning_rate')
    beta1_t = tf.convert_to_tensor(beta1, name='beta1')
    beta2_t = tf.convert_to_tensor(beta2, name='beta2')
    epsilon_t = tf.convert_to_tensor(epsilon, name='epsilon')

    # Create zero-debias accumulators. See Section 2 of Kingma and Ba's
    # paper (https://arxiv.org/pdf/1412.6980.pdf).
    beta1_power = tf.get_variable(
        'beta1_power',
        initializer=beta1,
        trainable=False)
    beta2_power = tf.get_variable(
        'beta2_power',
        initializer=beta2,
        trainable=False)
    adjusted_lr = learning_rate_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power)

    # Compute a value update and a moving-average accumulator update for each
    # trainable variable.
    update_ops = []
    for grad, reg_grad, var in zip(grad_list, reg_grad_list, var_list):
      # Create shadow variables to keep track of moving-average statistics
      # related to the gradients.
      assert isinstance(var, tf.Variable), var
      var_name = var.name[:var.name.index(':')]
      m = tf.get_variable(
          var_name + '/m',
          shape=var.shape,
          initializer=tf.initializers.zeros(),
          trainable=False,
          dtype=var.dtype.base_dtype)
      v = tf.get_variable(
          var_name + '/v',
          shape=var.shape,
          initializer=tf.initializers.zeros(),
          trainable=False,
          dtype=var.dtype.base_dtype)

      # Aggregate information about gradients across the TPU replicas.
      grad_mean = _cross_replica_mean(grad)
      grad_square_mean = _cross_replica_mean(tf.square(grad))
      if reg_grad is not None:
        reg_grad_mean = _cross_replica_mean(reg_grad)

      # Update moving averages
      beta1_cast = tf.cast(beta1_t, var.dtype.base_dtype)
      beta2_cast = tf.cast(beta2_t, var.dtype.base_dtype)
      m_update_op = m.assign(beta1_cast*m + (1-beta1_cast)*grad_mean)
      v_update_op = v.assign(beta2_cast*v + (1-beta2_cast)*grad_square_mean)

      # Update the values of the trainable variables.
      with tf.control_dependencies([m_update_op, v_update_op]):
        # Apply an ADAM update to the variables.
        adjusted_lr_cast = tf.cast(adjusted_lr, var.dtype.base_dtype)
        epsilon_cast = tf.cast(epsilon_t, var.dtype.base_dtype)

        numerator = m.read_value()
        denominator = tf.sqrt(v.read_value()) + epsilon_cast
        var_update_op = var.assign_sub(
            adjusted_lr_cast * numerator / denominator)

        # Optionally apply a regularization update that bypasses the moving
        # average accumulators.
        if reg_grad is not None:
          with tf.control_dependencies([var_update_op]):
            lr_cast = tf.cast(learning_rate_t, var.dtype.base_dtype)
            var_update_op = var.assign_sub(lr_cast * reg_grad_mean)
        update_ops.append(var_update_op)

    # Update the beta1 and beta2 zero-debias accumulators.
    with tf.control_dependencies(update_ops):
      beta_update_op = tf.group(
          beta1_power.assign(beta1_t * beta1_power),
          beta2_power.assign(beta2_t * beta2_power))

    # Update the global_step (if provided) before returning.
    with tf.control_dependencies([beta_update_op]):
      if global_step is not None:
        with tf.control_dependencies([global_step.assign_add(1)]):
          return tf.no_op()
      else:
        return tf.no_op()
