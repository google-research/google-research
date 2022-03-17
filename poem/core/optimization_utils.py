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

"""Defines functions and classes related to optimization."""

import re
import tensorflow.compat.v1 as tf


class AdamWeightDecayOptimizer(tf.train.AdamOptimizer):
  """A basic Adam optimizer that includes correct L2 weight decay.

  Reference:
    Ilya Loshchilov & Frank Hutter. Decoupled Weight Decay Regularization.
    https://arxiv.org/abs/1711.05101.

  IMPORTANT: The current version of this optimizer only supports training with a
  single GPU. Training with multiple GPUs will use default optimization rules of
  Adam (without L2 weight decay).
  """

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               use_locking=False,
               name='AdamWeightDecayOptimizer'):
    """Constructs a AdamWeightDecayOptimizer.

    Args:
      learning_rate: A Tensor or a floating point value for the learning rate.
      weight_decay_rate: A float value for the weight decay rate.
      beta1: A float value for the exponential decay rate of the 1st moment
        estimates.
      beta2: A float value for the exponential decay rate for the 2nd moment
        estimates.
      epsilon: A small constant for numerical stability.
      exclude_from_weight_decay: A list of strings for the names of parameters
        should be excluded from the weight decay.
      use_locking: A boolean for whether to use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "AdamWeightDecayOptimizer".
    """
    super(AdamWeightDecayOptimizer, self).__init__(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        use_locking=use_locking,
        name=name)
    self._weight_decay_rate = weight_decay_rate
    self._exclude_from_weight_decay = exclude_from_weight_decay
    self._decay_vars = None

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Applies gradients to variables.

    Args:
      grads_and_vars: A list of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """
    self._collect_decay_vars(grads_and_vars)
    return super().apply_gradients(
        grads_and_vars, global_step=global_step, name=name)

  def _apply_dense(self, grad, var):
    """Adds ops to apply dense gradients to `var`."""
    if var.ref() in self._decay_vars:
      beta1_power, beta2_power = self._get_beta_accumulators()
      beta1_power = tf.cast(beta1_power, var.dtype.base_dtype)
      beta2_power = tf.cast(beta2_power, var.dtype.base_dtype)
      lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
      lr = lr_t * tf.math.sqrt(1 - beta2_power) / (1 - beta1_power)
      update_op = var.assign_sub(lr * self._weight_decay_rate * var,
                                 self._use_locking)
      with tf.control_dependencies([update_op]):
        return super()._apply_dense(grad, var)

    return super()._apply_dense(grad, var)

  def _collect_decay_vars(self, grads_and_vars):
    """Collects the variables for weight decay."""
    if self._decay_vars is not None:
      return

    self._decay_vars = []
    for (_, var) in grads_and_vars:
      if var is None:
        continue
      var_name = self._get_var_name(var.name)
      if self._do_use_weight_decay(var_name):
        self._decay_vars.append(var.ref())
    self._decay_vars = set(self._decay_vars)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self._weight_decay_rate:
      return False
    if self._exclude_from_weight_decay:
      for r in self._exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_var_name(self, param_name):
    """Gets the variable name from the tensor name."""
    m = re.search('^(.*):\\d+$', param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
