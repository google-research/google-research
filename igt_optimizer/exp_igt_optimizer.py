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

"""A exponential any time implicit gradient transport (IGT) optimizer.

An implicit gradient transport (IGT) optimizer centers on the idea of deriving
corrections to past gradients so they may be reused.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow.compat.v1 as tf  # tf

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.training import slot_creator
# pylint:enable=g-direct-tensorflow-import


def _var_key(var):
  """Returns a variable key for use in a slot dictionary."""
  return (var.op.graph, var.op.name)


def get_gamma_t(step, tail_fraction):
  """Returns an increasing momentum: step / (step + 1)."""
  s = tf.to_float(step)
  tail_fraction = tf.to_float(tail_fraction)
  c = 1. / tail_fraction

  gamma_t = c * (s - 1.) / (1. + c * (s - 1.))
  gamma_t *= (1. - tf.sqrt((1 - c) / (s * (s - 1.))) / c)

  return tf.cond(tf.equal(step, 1), lambda: 0., lambda: gamma_t)


class ExpIgtOptimizer(object):
  """An IGT Optimizer using the Exponential Anytime Tail Averaging."""

  def __init__(self,
               learning_rate,
               tail_fraction=2.,
               optimizer='gd',
               adam_epsilon=None,
               use_locking=True,
               name='ExpIgtOptimizer'):
    """Inits an ExpIgtOptimizer.

    Args:
      learning_rate: A Tensor or float, the learning rate.
      tail_fraction: A float, the tail fraction of data to use.
      optimizer: A string, the optimizer to use to apply the update ('sg', 'mom'
        or 'adam').
      adam_epsilon: A float or None, an optional value to specify for Adam's
        epsilong parameter.
      use_locking: Bool. If True use locks for update operations.
      name: A string. The name to use for accumulators created for the
        optimizer.

    Raises:
      ValueError: If name is malformed.
    """
    self.name = name
    self.learning_rate = learning_rate
    self.tail_fraction = tail_fraction

    if optimizer == 'gd':
      tf.logging.info('Using INNER GradientDescentOptimizer.')
      self._optimizer = tf.train.GradientDescentOptimizer(
          learning_rate,
          use_locking=use_locking,
          name=six.ensure_str(name) + '_gd_inner')
    elif optimizer == 'mom':
      tf.logging.info('Using INNER MomentumOptimizer: mom 0.9.')
      self._optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=0.9,
          use_locking=use_locking,
          name='mom_inner')
    elif optimizer == 'adam':
      if adam_epsilon is None:
        adam_epsilon = 1e-8
      tf.logging.info('Using INNER AdamOptimizer (epsilon %d).', adam_epsilon)
      self._optimizer = tf.train.AdamOptimizer(
          learning_rate=learning_rate,
          use_locking=use_locking,
          epsilon=adam_epsilon,
          name=six.ensure_str(name) + '_adam_inner')
    else:
      raise ValueError(
          '{} is not a supported apply-optimizer.'.format(optimizer))

    # Used to apply a specific update to variables.
    self._slots = {}
    self.step = tf.get_variable(
        'step',
        shape=(),
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False)

    self.relevant_vars = set()

  def _create_slots(self, var_list):
    """Creates all slots needed by the variables.

    Args:
      var_list: A list of `Variable` objects.
    """
    # We're currently using 3 slots, we could use less.
    for var in var_list:
      self.relevant_vars.add(var)

      # The gradient estimate.
      estimate = slot_creator.create_zeros_slot(var, 'estimate')
      estimate_slots = self._slots.setdefault('estimate', {})
      estimate_slots[_var_key(var)] = estimate

      # The true parameter values (the variables contain shifted parameters).
      true_param = slot_creator.create_slot(var, var.initialized_value(),
                                            'true_param')
      true_slots = self._slots.setdefault('true_param', {})
      true_slots[_var_key(var)] = true_param

      # Storage for the update of the "apply" optimizer.
      update = slot_creator.create_zeros_slot(var, 'update')
      update_slots = self._slots.setdefault('update', {})
      update_slots[_var_key(var)] = update

  def swap_true_and_shifted(self):
    """Swap the actual parameters (shifted) and the true parameters."""
    swap_ops = []
    for var in self.relevant_vars:
      true_param = self._slots['true_param'][_var_key(var)]
      update = self._slots['update'][_var_key(var)]

      set_update = update.assign(var)
      with tf.control_dependencies([set_update]):
        set_var = var.assign(true_param)
        with tf.control_dependencies([set_var]):
          swap_ops.append(true_param.assign(set_update))
    return tf.group(swap_ops, name='swap')

  def apply_gradients(self, grads_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    Args:
      grads_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
      RuntimeError: If you should use `_distributed_apply()` instead.
    """
    if name is None:
      name = self.name

    grads_vars = [(g, v) for g, v in grads_vars if g is not None]
    var_list = [v for _, v in grads_vars]
    self._create_slots(var_list)

    step = self.step.assign_add(1)

    # Update the estimate and clear the slots that will receive the updates.
    clear_deps = []
    apply_optimizer_grads_vars = []
    for g, var in grads_vars:
      # Update the estimate.
      estimate = self._slots['estimate'][_var_key(var)]
      gamma_t = get_gamma_t(step, self.tail_fraction)
      estimate = estimate.assign(gamma_t * estimate + (1. - gamma_t) * g)

      # Clear the slot that will hold the inner optimizer's update.
      update = self._slots['update'][_var_key(var)]
      clear_deps.append(update.assign(tf.zeros(update.shape, update.dtype)))

      # Give the estimate to the apply optimizer.
      apply_optimizer_grads_vars.append((estimate, update))

    # Compute the update once the clearing has completed.
    with tf.control_dependencies(clear_deps):
      # Note: the global step, if any, is not passed here.
      apply_op = self._optimizer.apply_gradients(apply_optimizer_grads_vars,
                                                 global_step, name)

    # Update the true parameters and set up the shift for the next epoch.
    shift_vars_deps = []
    next_gamma_t = get_gamma_t(step + 1, self.tail_fraction)
    with tf.control_dependencies([apply_op]):
      for g, var in grads_vars:
        update = self._slots['update'][_var_key(var)]
        true_param = self._slots['true_param'][_var_key(var)]

        # Update the true parameter.
        true_param = true_param.assign_add(update)

        # Update the shifted parameter.
        multiplier = next_gamma_t / (1. - next_gamma_t)
        shift_vars_deps.append(var.assign(true_param + multiplier * update))
    return tf.group(shift_vars_deps, name=name)

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Computes the gradients of `loss` for variables in `var_list`.

    Args:
      loss: A tensor, the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph under
        the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with the
        corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
    """
    return self._optimizer.compute_gradients(
        loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

  def minimize(self,
               loss,
               global_step=None,
               var_list=None,
               gate_gradients=tf.train.Optimizer.GATE_OP,
               aggregation_method=None,
               colocate_gradients_with_ops=False,
               name=None,
               grad_loss=None):
    """Add operations to minimize `loss` by updating `var_list`.

    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in the
        graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with the
        corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.
    """
    grads_and_vars = self.compute_gradients(
        loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          'No gradients provided for any variable, check your graph for ops'
          ' that do not support gradients, between variables %s and loss %s.' %
          ([str(v) for _, v in grads_and_vars], loss))

    return self.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)

  def get_slot(self, var, name):
    """Returns a slot named `name` created for `var`.

    Args:
      var: A variable passed to `minimize()` or `apply_gradients()`.
      name: A string.

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    named_slots = self._slots.get(name, None)
    if named_slots:
      return named_slots.get(_var_key(var), None)
    return self._optimizer.get_slot(var, name)

  def get_slot_names(self):
    """Returns the list of slot names.

    Returns:
      A list of strings.
    """
    inner = self._optimizer.get_slot_names()
    outer = sorted(self._slots.keys())
    combined = outer + inner
    if combined:
      # Guard agains key duplication.
      assert len(combined) == len(set(combined))
    return combined
