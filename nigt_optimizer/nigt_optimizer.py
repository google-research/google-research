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

"""NIGT Optimizer.

See the paper https://arxiv.org/abs/2002.03305
This optimizer uses uses Taylor expansions to approximate variance reduction
algorithms while using only a single gradient evaluation per iteration.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six
import tensorflow.compat.v1 as tf


class NIGTOptimizer(tf.train.Optimizer):
  """NIGTOptimizer."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta=0.9,
               gamma=1e-3,
               use_igt=True,
               use_adaptive=False,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               name="NIGTptimizer"):
    """Constructs an optimizer."""
    super(NIGTOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta = beta
    self.gamma = gamma
    self.use_igt = use_igt
    self.use_adaptive = use_adaptive
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.exclude_from_layer_adaptation = exclude_from_layer_adaptation

  def compute_x(self, param_name, param, m, prev_w_norm, prev_eta, prev_beta):
    """Compute prev x value on the fly.

    Alternatively, we can store this as a slot but that would double the
    memory usage of our parameters. We don't like that!

    Args:
      param_name: Name of the parameter. Used to check whether to normalize the
        gradients for this layer.
      param: The parameter `Tensor`.
      m: Accumulated momentum `Tensor` of shape same as param.
      prev_w_norm: Scalar tracking norm of the param tensor at previous
        iteration.
      prev_eta: Scalar tracking the learning rate applied at previous iteration.
      prev_beta: Scalar tracking momentum applied at previous iteration.

    Returns:
      x: An intermediate `Tensor` of shape same as param. Will be used for the
        final update.
    """
    prev_ratio = 1.0
    if self._do_layer_adaptation(param_name):
      prev_g_norm = tf.norm(m, ord=2)
      prev_ratio = self.gamma * tf.where(
          tf.math.greater(prev_w_norm, 0),
          tf.where(
              tf.math.greater(prev_g_norm, 0),
              (prev_w_norm / prev_g_norm), 1.0), 1.0)
    prev_normalized_m_with_lr = prev_ratio * prev_eta * m

    x = param - tf.divide(
        tf.multiply(prev_beta, prev_normalized_m_with_lr), prev_beta - 1.0)
    return x

  def swap_to_optimal_params(self, params, name=None):
    """Swaps weights to be more optimal after training.

    NIGT evaluates gradients at points that are *different* than the points
    that we expect to have lower loss values. During training, the network
    weights are set to the be points where we evaluate gradients. This function
    returns an operation that will change the network weights to be the points
    that the NIGT believes to be more optimal, and should be used before
    evaluation.

    Note that once this function is called, the parameter values need to be
    swapped *back* in order to continue training. This should not be a
    concern if the function is only used in eval jobs.

    Args:
      params: list of parameters to update.
      name: name for operation.

    Returns:
      an operation that changes the parameters in params to better values.
    """
    switch_ops = []
    for param in params:
      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=six.ensure_str(param_name) + "/m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      prev_w_norm = tf.get_variable(
          name=six.ensure_str(param_name) + "/prev_w_norm",
          dtype=tf.float32,
          trainable=False,
          initializer=lambda w=param: tf.norm(w.initialized_value(), ord=2))

      prev_eta = tf.get_variable(
          name=six.ensure_str(param_name) + "/prev_eta",
          shape=[],
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      prev_beta = tf.get_variable(
          name=six.ensure_str(param_name) + "/prev_beta",
          shape=[],
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      x = self.compute_x(param_name, param, m, prev_w_norm, prev_eta, prev_beta)

      switch_ops.append(param.assign(x))
    return tf.group(*switch_ops, name=name)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=six.ensure_str(param_name) + "/m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Note: shape is not passed here explicitly since tf.get_variable
      # complains when you do that while passing a Tensor as an initializer.
      prev_w_norm = tf.get_variable(
          name=six.ensure_str(param_name) + "/prev_w_norm",
          dtype=tf.float32,
          trainable=False,
          initializer=lambda w=param: tf.norm(w.initialized_value(), ord=2))

      prev_eta = tf.get_variable(
          name=six.ensure_str(param_name) + "/prev_eta",
          shape=[],
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      prev_beta = tf.get_variable(
          name=six.ensure_str(param_name) + "/prev_beta",
          shape=[],
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      if self._do_use_weight_decay(param_name):
        grad += self.weight_decay_rate * param

      if self.use_adaptive:
        grad_squared_sum = tf.get_variable(
            name=six.ensure_str(param_name) + "/grad_squared_sum",
            shape=[],
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())

        max_grad = tf.get_variable(
            name=six.ensure_str(param_name) + "/max_grad",
            shape=[],
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())

        iteration = tf.get_variable(
            name=six.ensure_str(param_name) + "/iteration",
            shape=[],
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())

        next_grad_squared_sum = grad_squared_sum + tf.norm(grad, 2)
        next_iteration = iteration + 1
        next_max_grad = tf.maximum(max_grad, tf.norm(grad, 2))
        assignments.extend([
            grad_squared_sum.assign(next_grad_squared_sum),
            iteration.assign(next_iteration),
            max_grad.assign(next_max_grad)
        ])

        # Intuitively we should be able to leave g_sum=next_grad_squared_sum,
        # but current theory needs this extra t^1/4 max_grad term.
        g_sum = next_grad_squared_sum + tf.pow(next_iteration,
                                               0.25) * next_max_grad

        eta = self.learning_rate / tf.pow(
            tf.pow(next_iteration, 3.0) * tf.pow(g_sum, 2.0), 1.0 / 7.0)
        a = tf.minimum(1.0, 1.0 / (next_iteration * tf.pow(eta, 2.0) * g_sum))
        beta = 1.0 - a
      else:
        eta = self.learning_rate
        beta = self.beta

      next_m = (tf.multiply(beta, m) + tf.multiply(1.0 - beta, grad))

      ratio = 1.0
      w_norm = tf.norm(param, ord=2)
      if self._do_layer_adaptation(param_name):
        g_norm = tf.norm(next_m, ord=2)
        ratio = self.gamma * tf.where(
            tf.math.greater(w_norm, 0),
            tf.where(tf.math.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)
      normalized_m_with_lr = ratio * eta * next_m

      if self.use_igt:
        prev_x = self.compute_x(param_name, param, m, prev_w_norm, prev_eta,
                                prev_beta)
        next_x = prev_x - normalized_m_with_lr
        next_param = next_x + tf.divide(
            tf.multiply(beta, normalized_m_with_lr), beta - 1.0)
      else:
        next_param = param - normalized_m_with_lr
      assignments.extend([
          param.assign(next_param),
          m.assign(next_m),
          prev_w_norm.assign(w_norm),
          prev_eta.assign(eta),
          prev_beta.assign(beta)
      ])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", six.ensure_str(param_name))
    if m is not None:
      param_name = m.group(1)
    return param_name
