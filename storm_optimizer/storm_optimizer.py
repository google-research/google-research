# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""STOchastic Recursive Momentum Optimizer.

Applies variance reduction without need for large batch sizes or checkpoints
to obtain faster convergence to critical points in smooth non-convex problems.
See paper: https://arxiv.org/abs/1905.10018
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.optimizer_v2 import optimizer_v2

GATE_OP = 1

PREVIOUS_ITERATE = "previous_iterate"
GRAD_ESTIMATE = "grad_estimate"
SUM_GRAD_SQUARED = "sum_grad_squared"
MAXIMUM_GRADIENT = "maximum_gradient"
SUM_ESTIMATES_SQUARED = "sum_estimates_squared"


class StormOptimizer(optimizer_v2.OptimizerV2):
  """StormOptimizer implementation."""

  def __init__(self,
               lr=1.0,
               g_max=0.01,
               momentum=100.0,
               eta=10.0,
               output_summaries=False,
               use_locking=False,
               name="StormOptimizer"):
    """Construct new StormOptimizer.

    Args:
      lr: learning rate scaling (called k in the original paper).
      g_max: initial value of gradient squared accumulator. In theory should be
        an estimate of the maximum gradient size.
      momentum: Momentum scaling.
      eta: initial value of denominator in adaptive learning rate (called w in
        the original paper).
      output_summaries: Whether to output scalar_summaries of some internal
        variables. Note that this may significantly impact the number of
        iterations per second.
      use_locking: whether to use locks for update operations.
      name: name for optimizer.
    """
    super(StormOptimizer, self).__init__(use_locking, name)
    self.lr = lr
    self.g_max = g_max
    self.momentum = momentum
    self.eta = eta
    self.output_summaries = output_summaries

  def _find_read_tensors(self, outputs, target):
    """identify tensors in graph that come from reading target variable."""
    read_tensors = set()
    visited = set([])

    def dfs_dependency_tree(parent):
      for x in parent.op.inputs:
        if x.name not in visited:
          if x.name == target.name:
            read_tensors.add(parent)
          visited.add(x.name)
          dfs_dependency_tree(x)

    for output in outputs:
      dfs_dependency_tree(output)
    return read_tensors

  def _make_replace_dict(self, state, grads, var_list):
    """map tensors in graph to values at previous iterate."""
    replace_dict = {}
    for var in var_list:
      # This is inefficient because we call _find_read_tensors to DFS the
      # computation graph once for each var. Ideally we would only need
      # to DFS once. However this is not a big deal because this is a one-time
      # cost and is not repeated every iteration.
      previous_iterate = tf.convert_to_tensor(
          state.get_slot(var, PREVIOUS_ITERATE))
      read_tensors = self._find_read_tensors(grads, var)
      for t in read_tensors:
        replace_dict[t] = previous_iterate
    return replace_dict

  def _recompute_gradients(self, state):
    """recomputes gradient of loss at current example and previous iterate."""

    replace_dict = self._make_replace_dict(state, self.grads, self.vars)

    recomputed_grads = tf.contrib.graph_editor.graph_replace(
        self.grads, replace_dict)

    return recomputed_grads

  def _create_slot_with_value(self, state, var, value, name):
    state.create_slot(
        var, tf.constant(value, shape=var.shape, dtype=var.dtype.base_dtype),
        name)

  def _create_vars(self, var_list, state):
    for var in var_list:
      state.create_slot(var, var.initialized_value(), PREVIOUS_ITERATE)
      self._create_slot_with_value(state, var, self.g_max**3, SUM_GRAD_SQUARED)
      state.zeros_slot(var, GRAD_ESTIMATE)
      state.create_slot(var,
                        tf.constant(self.g_max, dtype=var.dtype.base_dtype),
                        MAXIMUM_GRADIENT)
      self._create_slot_with_value(state, var, 0.01, SUM_ESTIMATES_SQUARED)

  def _prepare(self, state):
    # These are dicts to hold per-variable intermediate values
    # that are recomputed from scratch every iteration.
    self.grads = []
    self.vars = []

  def _resource_apply_dense(self, grad, var, state):
    return self._apply_dense(grad, var, state)

  def _apply_dense(self, grad, var, state):
    # We actually apply grads in _finish. This function is used only to
    # store all the variables and gradients so we can access them all in one
    # function.
    self.grads.append(grad)
    self.vars.append(var)

    return tf.no_op()

  def _finish(self, state):

    update_ops = []

    grads_at_prev_iterate = self._recompute_gradients(state)

    for var, grad, grad_at_prev_iterate in zip(self.vars, self.grads,
                                               grads_at_prev_iterate):
      sum_grad_squared = state.get_slot(var, SUM_GRAD_SQUARED)
      previous_iterate = state.get_slot(var, PREVIOUS_ITERATE)
      maximum_gradient = state.get_slot(var, MAXIMUM_GRADIENT)
      sum_estimates_squared = state.get_slot(var, SUM_ESTIMATES_SQUARED)

      maximum_gradient_updated = tf.assign(
          maximum_gradient, tf.maximum(maximum_gradient, tf.norm(grad)))
      update_ops.append(maximum_gradient_updated)

      sum_grad_squared_updated = tf.assign_add(sum_grad_squared,
                                               tf.pow(tf.abs(grad), 2.0))
      update_ops.append(sum_grad_squared_updated)

      smoothness = tf.norm(grad - grad_at_prev_iterate) / (
          0.0001 + tf.norm(var - previous_iterate))
      eta = self.lr * tf.pow(self.eta + sum_grad_squared_updated, -1.0 / 3.0)

      beta = tf.minimum(1.0, self.momentum * tf.square(eta))

      grad_estimate = state.get_slot(var, GRAD_ESTIMATE)

      new_grad_estimate = grad + (1.0 - beta) * (
          grad_estimate - grad_at_prev_iterate)
      new_grad_estimate = tf.clip_by_value(new_grad_estimate,
                                           -maximum_gradient_updated,
                                           maximum_gradient_updated)

      if self.output_summaries:
        tf.summary.scalar(self._name + "/smoothness/" + var.name, smoothness)
        tf.summary.scalar(self._name + "/max_grad/" + var.name,
                          maximum_gradient_updated)
        tf.summary.scalar(self._name + "/average_beta/" + var.name,
                          tf.reduce_mean(beta))
        tf.summary.scalar(self._name + "/iterate_diff/" + var.name,
                          tf.norm(var - previous_iterate))
        tf.summary.scalar(self._name + "/grad_diff/" + var.name,
                          tf.norm(grad - grad_at_prev_iterate))
        tf.summary.scalar(self._name + "/vr_grad_estimate_norm/" + var.name,
                          tf.norm(new_grad_estimate))
        tf.summary.scalar(self._name + "/grad_norm/" + var.name, tf.norm(grad))

      grad_estimate_updated = tf.assign(grad_estimate, new_grad_estimate)
      update_ops.append(grad_estimate_updated)

      sum_estimates_squared_updated = tf.assign_add(
          sum_estimates_squared, tf.square(new_grad_estimate))
      update_ops.append(sum_estimates_squared_updated)

      with tf.control_dependencies([grad_at_prev_iterate]):
        previous_iterate_updated = tf.assign(previous_iterate, var)
        update_ops.append(previous_iterate_updated)

      step = -eta * grad_estimate_updated

      with tf.control_dependencies([previous_iterate_updated]):
        var_updated = tf.assign_add(var, step)
        update_ops.append(var_updated)

    return tf.group(*update_ops)

  # Add colocate_gradients_with_ops argument to compute_gradients for
  # compatibility with tensor2tensor.
  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        grad_loss=None,
                        stop_gradients=None,
                        colocate_gradients_with_ops=False,
                        scale_loss_by_num_replicas=None):
    return super(StormOptimizer,
                 self).compute_gradients(loss, var_list, gate_gradients,
                                         aggregation_method, grad_loss,
                                         stop_gradients,
                                         scale_loss_by_num_replicas)
