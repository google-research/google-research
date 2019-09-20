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

"""Neural network approximation of density ratio using DualDICE.

Based on the paper `DualDICE: Behavior-Agnostic Estimation of Discounted
Stationary Distribution Corrections' by Ofir Nachum, Yinlam Chow, Bo Dai,
and Lihong Li. See https://arxiv.org/abs/1906.04733
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow as tf
from typing import Callable, Optional, Text

from dual_dice import policy as policy_lib
from dual_dice.algos import base as base_algo


class NeuralSolverParameters(object):
  """Set of parameters common to neural network solvers."""

  def __init__(self,
               state_dim,
               action_dim,
               gamma,
               discrete_actions = True,
               deterministic_env = False,
               hidden_dim = 64,
               hidden_layers = 1,
               activation = tf.nn.tanh,
               nu_learning_rate = 0.0001,
               zeta_learning_rate = 0.001,
               batch_size = 512,
               num_steps = 10000,
               log_every = 500,
               smooth_over = 4,
               summary_writer = None,
               summary_prefix = ''):
    """Initializes the parameters.

    Args:
      state_dim: Dimension of state observations.
      action_dim: If the environment uses continuous actions, this should be
        the dimension of the actions. If the environment uses discrete actions,
        this should be the number of discrete actions.
      gamma: The discount to use.
      discrete_actions: Whether the environment uses discrete actions or not.
      deterministic_env: Whether to take advantage of a deterministic
        environment. If this and average_next_nu are both True, the optimization
        for nu is performed agnostic to zeta (in the primal form).
      hidden_dim: The internal dimension of the neural networks.
      hidden_layers: Number of internal layers in the neural networks.
      activation: Activation to use in the neural networks.
      nu_learning_rate: Learning rate for nu.
      zeta_learning_rate: Learning rate for zeta.
      batch_size: Batch size.
      num_steps: Number of steps (batches) to train for.
      log_every: Log progress and debug information every so many steps.
      smooth_over: Number of iterations to smooth over for final value estimate.
      summary_writer: An optional summary writer to log information to.
      summary_prefix: A prefix to prepend to the summary tags.
    """
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.gamma = gamma
    self.discrete_actions = discrete_actions
    self.deterministic_env = deterministic_env
    self.hidden_dim = hidden_dim
    self.hidden_layers = hidden_layers
    self.activation = activation
    self.nu_learning_rate = nu_learning_rate
    self.zeta_learning_rate = zeta_learning_rate
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.log_every = log_every
    self.smooth_over = smooth_over
    self.summary_writer = summary_writer
    self.summary_prefix = summary_prefix


class NeuralDualDice(base_algo.BaseAlgo):
  """Approximate the density ratio using neural networks."""

  def __init__(self,
               parameters,
               solve_for_state_action_ratio = True,
               average_next_nu = True,
               average_samples = 1,
               function_exponent = 1.5):
    """Initializes the solver.

    Args:
      parameters: An object holding the common neural network parameters.
      solve_for_state_action_ratio: Whether to solve for state-action density
        ratio. Defaults to True, which is recommended, since solving for the
        state density ratio requires importance weights which can introduce
        training instability.
      average_next_nu: Whether to take an empirical expectation over next nu.
        This can improve stability of training.
      average_samples: Number of empirical samples to average over for next nu
        computation (only relevant in continuous environments).
      function_exponent: The form of the function f(x). We use a polynomial
        f(x)=|x|^p / p where p is function_exponent.

    Raises:
      ValueError: If function_exponent is less than or equal to 1.
      NotImplementedError: If actions are continuous.
    """
    self._parameters = parameters
    self._solve_for_state_action_ratio = solve_for_state_action_ratio
    self._average_next_nu = average_next_nu
    self._average_samples = average_samples

    if not self._parameters.discrete_actions:
      raise NotImplementedError('Continuous actions are not fully supported.')

    if function_exponent <= 1:
      raise ValueError('Exponent for f must be at least 1.')

    # Conjugate of f(x) = |x|^p / p is f*(x) = |x|^q / q where q = p / (p - 1).
    conjugate_exponent = function_exponent / (function_exponent - 1)
    self._f = lambda x: tf.abs(x) ** function_exponent / function_exponent
    self._fstar = lambda x: tf.abs(x) ** conjugate_exponent / conjugate_exponent

    # Build and initialize graph.
    self._build_graph()
    self._session = tf.Session()
    self._session.run(tf.global_variables_initializer())

  def _build_graph(self):
    self._create_placeholders()

    # Convert discrete actions to one-hot vectors.
    action = tf.one_hot(self._action, self._parameters.action_dim)
    next_action = tf.one_hot(self._next_action, self._parameters.action_dim)
    initial_action = tf.one_hot(self._initial_action,
                                self._parameters.action_dim)

    nu, next_nu, initial_nu, zeta = self._compute_values(
        action, next_action, initial_action)

    # Density ratio given by approximated zeta values.
    self._density_ratio = zeta

    if self._solve_for_state_action_ratio:
      delta_nu = nu - next_nu * self._parameters.gamma
    else:
      delta_nu = nu - next_nu * self._parameters.gamma * self._policy_ratio

    unweighted_zeta_loss = (delta_nu * zeta - self._fstar(zeta) -
                            (1 - self._parameters.gamma) * initial_nu)
    self._zeta_loss = -(tf.reduce_sum(self._weights * unweighted_zeta_loss) /
                        tf.reduce_sum(self._weights))

    if self._parameters.deterministic_env and self._average_next_nu:
      # Dont use Fenchel conjugate trick and instead optimize primal.
      unweighted_nu_loss = (self._f(delta_nu) -
                            (1 - self._parameters.gamma) * initial_nu)
      self._nu_loss = (tf.reduce_sum(self._weights * unweighted_nu_loss) /
                       tf.reduce_sum(self._weights))
    else:
      self._nu_loss = -self._zeta_loss

    self._train_nu_op = tf.train.AdamOptimizer(
        self._parameters.nu_learning_rate).minimize(
            self._nu_loss, var_list=tf.trainable_variables('nu'))
    self._train_zeta_op = tf.train.AdamOptimizer(
        self._parameters.zeta_learning_rate).minimize(
            self._zeta_loss, var_list=tf.trainable_variables('zeta'))
    self._train_op = tf.group(self._train_nu_op, self._train_zeta_op)

    # Debug quantity (should be close to 1).
    self._debug = (
        tf.reduce_sum(self._weights * self._density_ratio) /
        tf.reduce_sum(self._weights))

  def _create_placeholders(self):
    self._state = tf.placeholder(
        tf.float32, [None, self._parameters.state_dim], 'state')
    self._next_state = tf.placeholder(
        tf.float32, [None, self._parameters.state_dim], 'next_state')
    self._initial_state = tf.placeholder(
        tf.float32, [None, self._parameters.state_dim], 'initial_state')

    self._action = tf.placeholder(tf.int32, [None], 'action')
    self._next_action = tf.placeholder(tf.int32, [None], 'next_action')
    self._initial_action = tf.placeholder(tf.int32, [None], 'initial_action')

    # Ratio of policy sampling probabilities of self._action.
    self._policy_ratio = tf.placeholder(tf.float32, [None], 'policy_ratio')

    # Policy sampling probabilities associated with next state.
    self._target_policy_next_probs = tf.placeholder(
        tf.float32, [None, self._parameters.action_dim])

    self._weights = tf.placeholder(tf.float32, [None], 'weights')

  def _compute_values(self, action, next_action, initial_action):
    nu = self._nu_network(self._state, action)
    initial_nu = self._nu_network(self._initial_state, initial_action)

    if self._average_next_nu:
      # Average next nu over all actions weighted by target policy
      # probabilities.
      all_next_actions = [
          tf.one_hot(act * tf.ones_like(self._next_action),
                     self._parameters.action_dim)
          for act in range(self._parameters.action_dim)]
      all_next_nu = [self._nu_network(self._next_state, next_action_i)
                     for next_action_i in all_next_actions]
      next_nu = sum(
          self._target_policy_next_probs[:, act_index] * all_next_nu[act_index]
          for act_index in range(self._parameters.action_dim))
    else:
      next_nu = self._nu_network(self._next_state, next_action)

    zeta = self._zeta_network(self._state, action)

    return nu, next_nu, initial_nu, zeta

  def _nu_network(self, state, action):
    with tf.variable_scope('nu', reuse=tf.AUTO_REUSE):
      if self._solve_for_state_action_ratio:
        inputs = tf.concat([state, action], -1)
      else:
        inputs = state
      outputs = self._network(inputs)
    return outputs

  def _zeta_network(self, state, action):
    with tf.variable_scope('zeta', reuse=tf.AUTO_REUSE):
      if self._solve_for_state_action_ratio:
        inputs = tf.concat([state, action], -1)
      else:
        inputs = state
      outputs = self._network(inputs)
    return outputs

  def _network(self, inputs):
    with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
      input_dim = int(inputs.shape[-1])
      prev_dim = input_dim
      prev_outputs = inputs
      # Hidden layers.
      for layer in range(self._parameters.hidden_layers):
        with tf.variable_scope('layer%d' % layer, reuse=tf.AUTO_REUSE):
          weight = tf.get_variable(
              'weight', [prev_dim, self._parameters.hidden_dim],
              initializer=tf.glorot_uniform_initializer())
          bias = tf.get_variable(
              'bias', initializer=tf.zeros([self._parameters.hidden_dim]))
          pre_activation = tf.matmul(prev_outputs, weight) + bias
          post_activation = self._parameters.activation(pre_activation)
        prev_dim = self._parameters.hidden_dim
        prev_outputs = post_activation

      # Final layer.
      weight = tf.get_variable(
          'weight_final', [prev_dim, 1],
          initializer=tf.glorot_uniform_initializer())
      bias = tf.get_variable(
          'bias_final', [1], initializer=tf.zeros_initializer())
      output = tf.matmul(prev_outputs, weight) + bias
      return output[Ellipsis, 0]

  def solve(self, data, target_policy, baseline_policy=None):
    """Solves for density ratios and then approximates target policy value.

    Args:
      data: The transition data store to use.
      target_policy: The policy whose value we want to estimate.
      baseline_policy: The policy used to collect the data. If None,
        we default to data.policy.

    Returns:
      Estimated average per-step reward of the target policy.

    Raises:
      ValueError: If NaNs encountered in policy ratio computation.
    """
    if baseline_policy is None:
      baseline_policy = data.policy

    value_estimates = []
    for step in range(self._parameters.num_steps):
      batch = data.sample_batch(self._parameters.batch_size)
      feed_dict = {
          self._state: batch.state,
          self._action: batch.action,
          self._next_state: batch.next_state,
          self._initial_state: batch.initial_state,
          self._weights: self._parameters.gamma ** batch.time_step,
      }

      # On-policy next action and initial action.
      feed_dict[self._next_action] = target_policy.sample_action(
          batch.next_state)
      feed_dict[self._initial_action] = target_policy.sample_action(
          batch.initial_state)

      if self._average_next_nu:
        next_probabilities = target_policy.get_probabilities(batch.next_state)
        feed_dict[self._target_policy_next_probs] = next_probabilities

      policy_ratio = policy_lib.get_policy_ratio(baseline_policy, target_policy,
                                                 batch.state, batch.action)

      if np.any(np.isnan(policy_ratio)):
        raise ValueError('NaNs encountered in policy ratio: %s.' % policy_ratio)
      feed_dict[self._policy_ratio] = policy_ratio

      self._session.run(self._train_op, feed_dict=feed_dict)

      if step % self._parameters.log_every == 0:
        debug = self._session.run(self._debug, feed_dict=feed_dict)
        tf.logging.info('At step %d' % step)
        tf.logging.info('Debug: %s' % debug)
        value_estimate = self.estimate_average_reward(data, target_policy)
        tf.logging.info('Estimated value: %s' % value_estimate)
        value_estimates.append(value_estimate)
        tf.logging.info(
            'Estimated smoothed value: %s' %
            np.mean(value_estimates[-self._parameters.smooth_over:]))

        if self._parameters.summary_writer:
          summary = tf.Summary(value=[
              tf.Summary.Value(
                  tag='%sdebug' % self._parameters.summary_prefix,
                  simple_value=debug),
              tf.Summary.Value(
                  tag='%svalue_estimate' % self._parameters.summary_prefix,
                  simple_value=value_estimate)])
          self._parameters.summary_writer.add_summary(summary, step)

    value_estimate = self.estimate_average_reward(data, target_policy)
    tf.logging.info('Estimated value: %s' % value_estimate)
    value_estimates.append(value_estimate)
    tf.logging.info('Estimated smoothed value: %s' %
                    np.mean(value_estimates[-self._parameters.smooth_over:]))

    # Return estimate that is smoothed over last few iterates.
    return np.mean(value_estimates[-self._parameters.smooth_over:])

  def _state_action_density_ratio(self, state, action):
    batched = len(np.shape(state)) > 1
    if not batched:
      state = np.expand_dims(state, 0)
      action = np.expand_dims(action, 0)
    density_ratio = self._session.run(
        self._density_ratio,
        feed_dict={
            self._state: state,
            self._action: action
        })
    if not batched:
      return density_ratio[0]
    return density_ratio

  def _state_density_ratio(self, state):
    batched = len(np.shape(state)) > 1
    if not batched:
      state = np.expand_dims(state, 0)
    density_ratio = self._session.run(
        self._density_ratio, feed_dict={self._state: state})
    if not batched:
      return density_ratio[0]
    return density_ratio

  def estimate_average_reward(self, data, target_policy):
    """Estimates value (average per-step reward) of policy.

    The estimation is based on solved values of zeta, so one should call
    solve() before calling this function.

    Args:
      data: The transition data store to use.
      target_policy: The policy whose value we want to estimate.

    Returns:
      Estimated average per-step reward of the target policy.
    """
    if self._solve_for_state_action_ratio:
      return base_algo.estimate_value_from_state_action_ratios(
          data, self._parameters.gamma, self._state_action_density_ratio)
    else:
      return base_algo.estimate_value_from_state_ratios(
          data, target_policy, self._parameters.gamma,
          self._state_density_ratio)

  def close(self):
    tf.reset_default_graph()
    self._session.close()
