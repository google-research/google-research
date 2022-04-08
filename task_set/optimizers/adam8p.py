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

"""Adam with extra hyper parameters for l1, l2 reg and lr schedules."""

import re
from typing import Text, List, Dict, Any
import numpy as np

from task_set import registry
from task_set.optimizers import base
from task_set.optimizers import utils
import tensorflow.compat.v1 as tf


class Adam8POptimizer(base.BaseOptimizer):
  r"""8 hyper parameter Adam.

  This is the Adam optimizer[1] with the addition of l1 and l2 regularization
  and a combination of linear and exponential learning rate decay.

  Note the l1 and l2 regularization is added to the loss. See AdamW[2] for a
  discussion of why this might be a bad idea.

  The update is as follows:

  # initialize variables
  m <- 0
  v <- 0
  beta1p <- beta1
  beta2p <- beta2

  # updating x \in R^N:
  g = d/dx(f(x) + l2*||x||^2_2 + l1*||x||_1)

  m <- beta1 * m + (1.0 - beta1)*g
  v <- beta2 * v + (1.0 - beta2)*g^2

  mh <- m / (1 - beta1p)
  vh <- v / (v - beta2p)

  update <- mh / (sqrt(vh+1e-10) + epsilon)

  beta1p <- beta1 * beta1p
  beta2p <- beta2 * beta2p

  linear_factor <- max(1 - linear_decay * global_step, 0.0)
  exp_factor <- exp(-exponential_decay * global_step)
  lr = exp_factor * linear_factor * learning_rate

  x <- lr * linear_factor * exp_factor * update

  [1] https://arxiv.org/abs/1412.6980
  [2] https://arxiv.org/abs/1711.05101
  """

  def __init__(
      self,
      learning_rate = 1e-3,
      beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 1e-8,
      l1 = 1e-7,
      l2 = 1e-7,
      linear_decay = 0.0,
      exponential_decay = 0.0,
      reg_factor = 1.0,
      training_steps = 10000,
  ):
    """Initialize the optimizer. See class documentation for equations."""
    self._learning_rate = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._l1 = l1
    self._l2 = l2
    self._linear_decay = linear_decay
    self._exponential_decay = exponential_decay
    self._reg_factor = reg_factor
    self._training_steps = training_steps

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

  def minimize(self, loss, global_step,
               var_list):
    """Create op that applies Adam8p step."""
    if not var_list:
      raise ValueError("Explicitly pass var_list!")
    if not global_step:
      raise ValueError("Explicitly pass global_step!")

    # Add regularization to the loss
    grads_and_vars = self.compute_gradients(loss, var_list=var_list)
    return self.apply_gradients(grads_and_vars, global_step=global_step)

  def apply_gradients(self, grads_and_vars, global_step, name=None):
    """Perform an update with the parameters."""

    # we meta-train with 10k steps. When applying to longer problems we want to
    # have a reasonable schedule so we rescale.

    rescale_global_step = float(10000) / self._training_steps * tf.to_float(
        global_step)

    beta1_power = tf.get_variable(
        dtype=tf.float32, name="beta1_power", initializer=self._beta1)
    beta2_power = tf.get_variable(
        dtype=tf.float32, name="beta2_power", initializer=self._beta2)

    exp_factor = tf.exp(-self._exponential_decay *
                        tf.to_float(rescale_global_step))

    # lr reduction per step.
    linear_factor = tf.maximum(
        1 - self._linear_decay * tf.to_float(rescale_global_step), 0.0)

    lr = exp_factor * linear_factor * self._learning_rate

    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue
      # sparse to dense conversion
      grad = tf.convert_to_tensor(grad)

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      next_m = (self._beta1 * m + (1.0 - self._beta1) * grad)
      next_v = (self._beta2 * v + (1.0 - self._beta2) * tf.square(grad))
      next_m_hat = next_m / (1 - beta1_power)
      next_v_hat = next_v / (1 - beta2_power)
      update = next_m_hat / (tf.sqrt(next_v_hat + 1e-10) + self._epsilon)

      next_param = param - lr * update

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])

    # Do this after all other assignments are done to prevent a race condition.
    with tf.control_dependencies(assignments):
      assignments.extend([
          beta1_power.assign(beta1_power * self._beta1),
          beta2_power.assign(beta2_power * self._beta2),
          global_step.assign_add(1),
      ])
    return tf.group(*assignments, name=name)

  def compute_gradients(self, loss, var_list=None, **kwargs):
    if not var_list:
      var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    if self._l1:
      l1 = tf.add_n(
          [tf.reduce_sum(tf.abs(p)) * self._reg_factor for p in var_list])
      loss = loss + l1 * self._l1
    if self._l2:
      l2 = tf.add_n(
          [tf.reduce_sum(tf.square(p)) * self._reg_factor for p in var_list])
      loss = loss + l2 * self._l2

    grads_and_vars = zip(
        tf.gradients(loss, var_list, colocate_gradients_with_ops=True),
        var_list)
    return grads_and_vars


Adam8PConfig = Dict[Text, Any]


@registry.optimizers_registry.register_sampler("adam8p_wide_grid")
def sample_adam8p_wide_grid(seed):
  """Sample a random configuration from a wide grid for adam8p."""
  rng = np.random.RandomState(seed)
  cfg = {
      "learning_rate": utils.sample_log_float(rng, 1e-8, 1e1),
      "beta1": 1 - utils.sample_log_float(rng, 1e-4, 1e0),
      "beta2": 1 - utils.sample_log_float(rng, 1e-6, 1e0),
      "epsilon": utils.sample_log_float(rng, 1e-10, 1e3),
      "l1": utils.sample_log_float(rng, 1e-8, 1e1),
      "l2": utils.sample_log_float(rng, 1e-8, 1e1),
      "linear_decay": utils.sample_log_float(rng, 1e-7, 1e-4),
      "exponential_decay": utils.sample_log_float(rng, 1e-3, 1e-6),
  }
  return cfg


@registry.optimizers_registry.register_getter("adam8p_wide_grid")
def get_adam8p(
    cfg,
    training_steps = 10000  # pylint: disable=unused-argument
):
  return Adam8POptimizer(**cfg)


@registry.optimizers_registry.register_sampler("adam6p_wide_grid")
def sample_adam6p_wide_grid(seed):
  """Sample a random configuration from a wide grid for adam6p."""
  rng = np.random.RandomState(seed + 123455)
  cfg = {
      "learning_rate": utils.sample_log_float(rng, 1e-8, 1e1),
      "beta1": 1 - utils.sample_log_float(rng, 1e-4, 1e0),
      "beta2": 1 - utils.sample_log_float(rng, 1e-6, 1e0),
      "epsilon": utils.sample_log_float(rng, 1e-10, 1e3),
      "linear_decay": utils.sample_log_float(rng, 1e-7, 1e-4),
      "exponential_decay": utils.sample_log_float(rng, 1e-3, 1e-6),
  }
  return cfg


@registry.optimizers_registry.register_getter("adam6p_wide_grid")
def get_adam6p(cfg, training_steps = 10000):
  return Adam8POptimizer(l1=0.0, l2=0.0, training_steps=training_steps, **cfg)
