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

"""Parameter-free optimization inside parameter-free optimization.

Learns a preconditioned optimal direction using a two-layer algorithm
structure. An inner un-preconditioned online learning algorithm runs on each
coordinate of the problem. This inner optimizer is used to find an optimal
pre-conditioned direction for the outer optimization algorithm.

See the paper
Ashok Cutkosky, and Tamas Sarlos.
"Matrix-Free Preconditioning in Online Learning", ICML 2019.
http://proceedings.mlr.press/v97/cutkosky19b.html

Args:
  epsilon: regret at 0 of outer optimizer
  epsilon_v: regret at 0 of each coordinate of inner optimizer
  g_max: guess for maximum L1 norm of gradients
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

GATE_OP = 1

OUTER_BETTING_FRACTION = "outer_betting_fraction"
INNER_BETTING_FRACTION = "inner_betting_fraction"

OUTER_WEALTH = "outer_wealth"
INNER_WEALTH = "inner_wealth"
INNER_REWARD = "inner_reward"

INNER_SUM_GRAD_SQUARED = "inner_sum_grad_squared"
INNER_SUM_GRAD = "inner_sum_grad"

MAXIMUM_GRADIENT = "maximum_gradient"
INNER_MAXIMUM_GRADIENT = "inner_maximum_gradient"

# used if add_average flag is true
AVERAGE_OFFSET = "average_offset"
SUM_GRAD_NORM_SQUARED = "sum_grad_norm_squared"
PREVIOUS_OFFSET = "previous_offset"

EPSILON = "epsilon"
EPSILON_V = "epsilon_v"
G_MAX = "g_max"
BETTING_DOMAIN = "betting_domain"
ETA = "eta"
LR = "learning_rate"

ONSBET = "ONSBET"
SCINOL = "SCINOL"
INNER_OPTIMIZERS = [ONSBET, SCINOL]

INITIAL_VALUE = "initial_value"

SMALL_VALUE = 0.00000001


class RecursiveOptimizer(tf.train.Optimizer):
  """RecursiveOptimizer implementation."""

  def __init__(self,
               lr=1.0,
               epsilon=1.0,
               epsilon_v=1.0,
               g_max=SMALL_VALUE,
               betting_domain=0.5,
               tau=SMALL_VALUE,
               eta=None,
               rescale_inner=True,
               inner_optimizer="SCINOL",
               add_average=False,
               beta=0.9,
               output_summaries=False,
               use_locking=False,
               name="RecursiveOptimizer"):
    """Construct new RecursiveOptimizer.

    Args:
      lr: ''learning rate'' - a scale factor on the predictions. Should have
        identical performance to changing epsilon (initial wealth).
      epsilon: regret at 0 of outer optimizer (this is the initial wealth).
      epsilon_v: regret at 0 of each coordinate of inner optimizer
        (per-coordinate initial wealth)
      g_max: guess for maximum L1 norm of gradients. In theory, this guess needs
        to be an over-estimate, otherwise all bounds are invalid in the worst
        case. In stochastic problems we shouldn't expect worst-case behavior and
        so violations of the bound are not bad. Larger values lead to more a
        conservative algorithm, so we opt for an aggressive default.
      betting_domain: maximum betting fraction.
      tau: initial value for denominator in inner optimizer update.
      eta: If inner optimizer is ONS, manually overrides the ONS learning rate.
        If inner optimizer is SCINOL, sets maximum betting fraction of SCINOL in
        first several iterations.
      rescale_inner: Modifies the behavior of the inner optimizer to adapt to
        gradient scaling. For ONS, rescale the gradients supplied to the inner
        optimizer by their maximum value. For SCINOL, scale the initial wealth
        epsilon_v by the maximum gradient value.
      inner_optimizer: which optimizer to use as inner optimizer. ONSBET
        corresponds to using coin-betting reduction with ONS as base optimizer.
        SCINOL corresponds to scale-invariant online learning algorithm
        (https://arxiv.org/pdf/1902.07528.pdf).
      add_average: Whether to add the weighted average of past iterates to the
        current iterate as described in (section 6 of
        https://arxiv.org/abs/1802.06293). This is "morally" similar to momentum
          term in other SGD variants in that it pushes the iterates further in
          direction they have been moving.
      beta: only relevent when add_average=True. Uses an exponentially weighted
        average with exponential parameter beta when computing the average
        iterate.
      output_summaries: Whether to output scalar_summaries of some internal
        variables. Note that this will significantly impact the number of
        iterations per second.
      use_locking: whether to use locks for update operations.
      name: name for optimizer.
    """
    super(RecursiveOptimizer, self).__init__(use_locking, name)
    self.output_summaries = output_summaries
    self.g_max = max(g_max, SMALL_VALUE)
    self.epsilon = max(epsilon, SMALL_VALUE)
    self.epsilon_v = max(epsilon_v, SMALL_VALUE)
    self.tau = max(tau, SMALL_VALUE)
    self.rescale_inner = rescale_inner
    self.inner_optimizer = inner_optimizer
    self.add_average = add_average

    if self.inner_optimizer not in INNER_OPTIMIZERS:
      raise ValueError("Invalid inner optimizer!")

    if eta is None or eta == 0:
      if inner_optimizer == ONSBET:
        # Set learning rate for the online newton step update.
        # This is the maximum eta such that:
        # f(x) - f(u) < f'(x) * (x-u) - ((x-u) f'(x))^2/(2*eta)
        # for all x,u in [-betting_domain, betting_domain]
        # where f(x) = -log(1+x)
        eta = 0.5 / (
            betting_domain -
            betting_domain**2 * np.log(1 + 1.0 / betting_domain))
      elif inner_optimizer == SCINOL:
        eta = 0.1
    self.eta = eta

    self.lr = lr
    self.eta = eta
    self.beta = beta
    self.betting_domain = betting_domain

    self.non_slot_dict = {}

  # Propagates use_locking from constructor to
  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/assign
  def _assign(self, ref, value):
    return tf.assign(ref, value, use_locking=self._use_locking)

  # Propagates use_locking from constructor to
  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/assign_add
  def _assign_add(self, ref, value):
    return tf.assign_add(ref, value, use_locking=self._use_locking)

  def _create_slot_with_value(self, var, value, name, dtype=None):
    if dtype is None:
      dtype = var.dtype.base_dtype
    self._get_or_make_slot(var,
                           tf.constant(value, shape=var.shape,
                                       dtype=dtype), name, name + "_slot")

  def _create_non_slot_with_value(self, value, name, dtype):
    non_slot = tf.get_variable(
        name=self.get_name() + "/non_slot_variables/" + name,
        dtype=dtype,
        trainable=False,
        initializer=value)
    self.non_slot_dict[name] = non_slot

  def _get_non_slot(self, name):
    return self.non_slot_dict[name]

  def _create_zeros_slot(self, var, name):
    self._zeros_slot(var, name, name + "_slot")

  def _create_slots(self, var_list):
    for var in var_list:
      # TODO(cutkosky): See if any of these can be eliminated, and if this
      # improves performance.
      self._create_zeros_slot(var, OUTER_BETTING_FRACTION)

      self._get_or_make_slot(var, var.initialized_value(), INITIAL_VALUE,
                             INITIAL_VALUE + "_slot")
      self._create_slot_with_value(var, self.tau, INNER_SUM_GRAD_SQUARED)
      self._create_slot_with_value(var, self.g_max, INNER_MAXIMUM_GRADIENT)

      if self.inner_optimizer == SCINOL:
        self._create_zeros_slot(
            var,
            INNER_SUM_GRAD,
        )
        self._create_zeros_slot(var, INNER_REWARD)

      if self.inner_optimizer == ONSBET:
        self._create_zeros_slot(var, INNER_BETTING_FRACTION)
        self._create_slot_with_value(var, self.epsilon_v, INNER_WEALTH)

      if self.add_average:
        self._create_zeros_slot(var, AVERAGE_OFFSET)

    dtype = var_list[0].dtype.base_dtype
    self._create_non_slot_with_value(self.epsilon, OUTER_WEALTH, dtype)
    self._create_non_slot_with_value(self.g_max, MAXIMUM_GRADIENT, dtype)
    self._create_non_slot_with_value(0.0, SUM_GRAD_NORM_SQUARED, dtype)

  def _prepare(self):
    # These are dicts to hold per-variable intermediate values
    # that are recomputed from scratch every iteration.
    self.grads = {}

    # These dicts store increments that will be added up to obtain the
    # correct global value once all variables have been processed.
    self.betting_fraction_dot_product_deltas = {}
    self.wealth_deltas = {}
    self.grad_norms = {}

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense(grad, var)

  def _apply_dense(self, grad, var):
    # We actually apply grads in _finish. This function is used
    # to record intermediate variables related to the individual gradients
    # which we eventually combine in _finish to obtain global statistics
    # (e.g. the L1 norm of the full gradient).

    self.grads[var] = grad

    betting_fraction = self.get_slot(var, OUTER_BETTING_FRACTION)
    self.betting_fraction_dot_product_deltas[var] = tf.reduce_sum(
        betting_fraction * grad)

    # Wealth increases by -g \cdot w where w is the parameter value.
    # Since w = Wealth * v with betting fraction v, we can write
    # the wealth increment as -(g \cdot v) Wealth.
    # TODO(cutkosky): at one point there was a bug in which epsilon
    # was not added here. It seemed performance may have degraded
    # somewhat after fixing this. Find out why this would be.
    wealth_delta = -self.betting_fraction_dot_product_deltas[
        var] * self._get_non_slot(OUTER_WEALTH)
    self.wealth_deltas[var] = wealth_delta

    self.grad_norms[var] = tf.norm(grad, 1)

    return tf.no_op()

  def _compute_inner_update(self, var, grad):
    if self.inner_optimizer == ONSBET:
      return self._compute_inner_update_onsbet(var, grad)
    if self.inner_optimizer == SCINOL:
      return self._compute_inner_update_scinol(var, grad)

    raise TypeError("Unknown inner_optimizer: " + self.inner_optimizer)

  def _compute_inner_update_scinol(self, var, grad):
    update_ops = []

    betting_domain = tf.cast(self.betting_domain, var.dtype.base_dtype)

    reward = self.get_slot(var, INNER_REWARD)
    betting_fraction = self.get_slot(var, OUTER_BETTING_FRACTION)
    sum_grad_squared = self.get_slot(var, INNER_SUM_GRAD_SQUARED)
    sum_grad = self.get_slot(var, INNER_SUM_GRAD)
    inner_maximum_gradient = self.get_slot(var, INNER_MAXIMUM_GRADIENT)

    # clip inner gradient to respect previous inner_maximum_gradient value
    # This introduces at most an additive constant overhead in the regret
    # since the inner betting fraction lies in a bounded domain.
    clipped_grad = tf.clip_by_value(grad, -inner_maximum_gradient,
                                    inner_maximum_gradient)

    with tf.control_dependencies([clipped_grad]):
      inner_maximum_gradient_updated = self._assign(
          inner_maximum_gradient,
          tf.maximum(inner_maximum_gradient, tf.abs(grad)))
      update_ops.append(inner_maximum_gradient_updated)

    clipped_old_betting_fraction = tf.clip_by_value(betting_fraction,
                                                    -betting_domain,
                                                    betting_domain)

    # Process grad to respect truncation to [-betting_domain, betting_domain]
    truncated_grad = tf.where(
        tf.greater_equal(
            clipped_grad * (betting_fraction - clipped_old_betting_fraction),
            0.0), clipped_grad, tf.zeros(tf.shape(clipped_grad)))

    reward_delta = -betting_fraction * truncated_grad
    reward_updated = self._assign_add(reward, reward_delta)
    update_ops.append(reward_updated)

    sum_grad_squared_updated = self._assign_add(sum_grad_squared,
                                                tf.square(truncated_grad))
    update_ops.append(sum_grad_squared_updated)

    sum_grad_updated = self._assign_add(sum_grad, truncated_grad)
    update_ops.append(sum_grad_updated)

    # The second term in this maximum, inner_maximum_gradient_updated / self.eta
    # is a hack to force the betting fraction to not be too big at first.
    scaling = tf.minimum(
        tf.rsqrt(sum_grad_squared_updated +
                 tf.square(inner_maximum_gradient_updated)),
        self.eta / inner_maximum_gradient_updated)
    theta = -sum_grad_updated * scaling

    # rescale inner flag is a hack that rescales the epsilon_v by the
    # maximum inner gradient.
    if self.rescale_inner:
      epsilon_scaling = inner_maximum_gradient_updated
    else:
      epsilon_scaling = 1.0

    inner_betting_fraction = tf.sign(theta) * tf.minimum(tf.abs(theta),
                                                         1.0) * scaling / 2.0
    new_betting_fraction = inner_betting_fraction * (
        reward_updated + epsilon_scaling * self.epsilon_v)

    betting_fraction_updated = self._assign(betting_fraction,
                                            new_betting_fraction)
    update_ops.append(betting_fraction_updated)

    clipped_betting_fraction = tf.clip_by_value(betting_fraction_updated,
                                                -betting_domain, betting_domain)

    if self.output_summaries:
      mean_unclipped_betting_fraction_summary = tf.reduce_mean(
          tf.abs(betting_fraction_updated))
      max_unclipped_betting_fraction_summary = tf.reduce_max(
          tf.abs(betting_fraction_updated))

      mean_clipped_betting_fraction_summary = tf.reduce_mean(
          tf.abs(clipped_betting_fraction))
      max_clipped_betting_fraction_summary = tf.reduce_max(
          tf.abs(clipped_betting_fraction))

      max_abs_gradient = tf.reduce_max(tf.abs(grad))
      max_truncated_grad = tf.reduce_max(tf.abs(truncated_grad))

      tf.summary.scalar(self._name + "/mean_unclipped_bet/" + var.name,
                        mean_unclipped_betting_fraction_summary)
      tf.summary.scalar(self._name + "/max_unclipped_bet/" + var.name,
                        max_unclipped_betting_fraction_summary)
      tf.summary.scalar(self._name + "/mean_clipped_bet/" + var.name,
                        mean_clipped_betting_fraction_summary)
      tf.summary.scalar(self._name + "/max_clipped_bet/" + var.name,
                        max_clipped_betting_fraction_summary)

      tf.summary.scalar(self._name + "/max_abs_inner_grad/" + var.name,
                        max_abs_gradient)
      tf.summary.scalar(
          self._name + "/max_abs_truncated_inner_grad/" + var.name,
          max_truncated_grad)
    return clipped_betting_fraction, tf.group(*update_ops)

  def _compute_inner_update_onsbet(self, var, grad):
    update_ops = []

    eta = tf.cast(self.eta, var.dtype.base_dtype)
    betting_domain = tf.cast(self.betting_domain, var.dtype.base_dtype)

    wealth = self.get_slot(var, INNER_WEALTH)
    betting_fraction = self.get_slot(var, OUTER_BETTING_FRACTION)
    inner_betting_fraction = self.get_slot(var, INNER_BETTING_FRACTION)
    sum_grad_squared = self.get_slot(var, INNER_SUM_GRAD_SQUARED)
    inner_maximum_gradient = self.get_slot(var, INNER_MAXIMUM_GRADIENT)

    inner_maximum_gradient_updated = self._assign(
        inner_maximum_gradient, tf.maximum(inner_maximum_gradient,
                                           tf.abs(grad)))
    update_ops.append(inner_maximum_gradient_updated)

    clipped_old_betting_fraction = tf.clip_by_value(betting_fraction,
                                                    -betting_domain,
                                                    betting_domain)

    # Process grad to respect truncation to [-betting_domain, betting_domain]
    truncated_grad = tf.where(
        tf.greater_equal(
            grad * (betting_fraction - clipped_old_betting_fraction), 0), grad,
        tf.zeros(tf.shape(grad)))

    wealth_delta = -betting_fraction * truncated_grad
    wealth_updated = self._assign_add(wealth, wealth_delta)
    update_ops.append(wealth_updated)

    # This is the gradient with respect to the betting fraction v
    # use by the ONS algorithm - a kind of "inner inner grad".
    # Hueristic: We also scale v_grad down by the inner maximum gradient so as
    # to make it ``unitless''. This is helpful because the learning rate for
    # ONS is proportional to sum v_grad**2, and so the scale of the learning
    # rate and of v_grad are unlikely to be properly matched without this.
    if self.rescale_inner:
      v_grad = truncated_grad / (
          (1.0 - inner_betting_fraction * truncated_grad) *
          inner_maximum_gradient_updated)
    else:
      v_grad = truncated_grad / (
          (1.0 - inner_betting_fraction * truncated_grad))

    sum_grad_squared_updated = self._assign_add(sum_grad_squared,
                                                tf.square(v_grad))
    update_ops.append(sum_grad_squared_updated)

    new_inner_betting_fraction = inner_betting_fraction - eta * v_grad / (
        sum_grad_squared_updated)
    new_inner_betting_fraction = tf.clip_by_value(new_inner_betting_fraction,
                                                  -betting_domain,
                                                  betting_domain)
    inner_betting_fraction_updated = self._assign(inner_betting_fraction,
                                                  new_inner_betting_fraction)
    update_ops.append(inner_betting_fraction_updated)

    if self.output_summaries:
      mean_inner_betting_fraction_summary = tf.reduce_mean(
          tf.abs(inner_betting_fraction_updated))
      max_inner_betting_fraction_summary = tf.reduce_max(
          tf.abs(inner_betting_fraction_updated))
      inner_maximum_gradient_summary = tf.reduce_max(
          inner_maximum_gradient_updated)
      tf.summary.scalar(self._name + "/mean_inner_betting/" + var.name,
                        mean_inner_betting_fraction_summary)
      tf.summary.scalar(self._name + "/max_inner_betting/" + var.name,
                        max_inner_betting_fraction_summary)
      tf.summary.scalar(self._name + "/inner_maximum_gradient/" + var.name,
                        inner_maximum_gradient_summary)

    betting_fraction_updated = self._assign(
        betting_fraction, inner_betting_fraction_updated * wealth_updated)
    update_ops.append(betting_fraction_updated)

    clipped_betting_fraction = tf.clip_by_value(betting_fraction_updated,
                                                -betting_domain, betting_domain)

    return clipped_betting_fraction, tf.group(*update_ops)

  def _finish(self, update_ops, name):

    outer_wealth = self._get_non_slot(OUTER_WEALTH)
    betting_domain = self.betting_domain
    maximum_gradient = self._get_non_slot(MAXIMUM_GRADIENT)

    wealth_increment = sum(self.wealth_deltas.values())
    betting_fraction_dot_product = sum(
        self.betting_fraction_dot_product_deltas.values())
    grad_norm = sum(self.grad_norms.values())

    maximum_gradient_updated = self._assign(
        maximum_gradient, tf.maximum(maximum_gradient, grad_norm))
    update_ops.append(maximum_gradient_updated)

    gradient_scaling = 1.0 / maximum_gradient_updated
    # We will replace gradient with gradient/maximum_gradient_updated in order
    # to ensure ||gradient||_1 \le 1.
    # Since betting_fraction_dot_product and wealth_increment were calculated
    # using the original gradient, we also scale them by the same amount.
    betting_fraction_dot_product = betting_fraction_dot_product * gradient_scaling
    wealth_increment = wealth_increment * gradient_scaling

    outer_wealth_updated = self._assign_add(outer_wealth, wealth_increment)
    update_ops.append(outer_wealth_updated)

    inner_grad_scaling = (1.0 - betting_domain) / (1.0 -
                                                   betting_fraction_dot_product)

    if self.output_summaries:
      tf.summary.scalar(self._name + "/total_wealth", outer_wealth_updated)
      tf.summary.scalar(self._name + "/maximum_gradient_norm",
                        maximum_gradient_updated)
      tf.summary.scalar(self._name + "/gradient_L1_norm", grad_norm)

    if self.add_average:
      grad_norm_squared = tf.square(grad_norm)
      sum_grad_norm_squared = self._get_non_slot(SUM_GRAD_NORM_SQUARED)
      sum_grad_norm_squared_updated = self._assign(
          sum_grad_norm_squared,
          self.beta * sum_grad_norm_squared + grad_norm_squared)

    for var in self.grads:

      grad = self.grads[var]

      if self.inner_optimizer == SCINOL:
        inner_grad = grad * inner_grad_scaling
      else:
        # Rescale gradient to have L1 norm at most 1.0
        scaled_grad = grad * gradient_scaling
        inner_grad = scaled_grad * inner_grad_scaling

      betting_fraction, inner_update_op = self._compute_inner_update(
          var, inner_grad)
      update_ops.append(inner_update_op)

      if self.output_summaries:
        betting_fraction_summary = tf.reduce_mean(tf.abs(betting_fraction))
        tf.summary.scalar(self._name + "/mean_abs_betting_fraction/" + var.name,
                          betting_fraction_summary)
        max_betting_fraction_summary = tf.reduce_max(tf.abs(betting_fraction))
        tf.summary.scalar(self._name + "/max_abs_betting_fraction/" + var.name,
                          max_betting_fraction_summary)

      next_offset = self.lr * betting_fraction * outer_wealth_updated
      initial_value = self.get_slot(var, INITIAL_VALUE)

      if self.add_average:
        average_offset = self.get_slot(var, AVERAGE_OFFSET)
        average_offset_updated = self._assign_add(
            average_offset,
            (grad_norm_squared *
             (next_offset - average_offset)) / (sum_grad_norm_squared_updated))
        update_ops.append(average_offset_updated)

        var_updated = self._assign(
            var, next_offset + average_offset_updated + initial_value)
      else:
        var_updated = self._assign(var, next_offset + initial_value)
      update_ops.append(var_updated)

    return tf.group(*update_ops, name=name)
