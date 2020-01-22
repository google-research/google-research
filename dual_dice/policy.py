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

"""Utilities for working with policies.

A policy is an object which provides mechanisms for sampling actions conditioned
on state. It should also provide the sampling probabilities associated with
state-action pairs.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import numpy as np
import six

from typing import Any, Callable, Sequence


@six.add_metaclass(abc.ABCMeta)
class Policy(object):
  """Abstract class representing a policy."""

  @abc.abstractmethod
  def sample_action(self, state):
    """Randomly samples action according to policy at state."""

  @abc.abstractmethod
  def get_probability(self, state, action):
    """Gets probability of sampling action at state."""

  @abc.abstractmethod
  def get_probabilities(self, state):
    """Gets probability distribution over actions at state."""

  @abc.abstractmethod
  def get_log_probability(self, state, action):
    """Gets log-probability of sampling action at state."""


class DiscreteFunctionPolicy(Policy):
  """Policy based on function of states-->categorical distribution."""

  def __init__(self, policy_fn):
    """Initializes the policy.

    Args:
      policy_fn: A function which takes in a state and returns a distribution
        over a finite number of actions. The distribution should be in the
        form of a NumPy array. The function should be able to handle batched
        states.
    """
    self._policy_fn = policy_fn

  def sample_action(self, state):
    probabilities = self._policy_fn(state)
    if len(np.shape(probabilities)) == 1:  # Unbatched inputs.
      return np.random.choice(len(probabilities), p=probabilities)
    else:  # Batched inputs.
      cum_probabilities = probabilities.cumsum(axis=1)
      uniform_samples = np.random.rand(len(cum_probabilities), 1)
      actions = (uniform_samples < cum_probabilities).argmax(axis=1)
      return actions

  def get_probability(self, state, action):
    probabilities = self._policy_fn(state)
    if len(np.shape(probabilities)) == 1:  # Unbatched inputs.
      return probabilities[action]
    else:  # Batched inputs.
      return probabilities[np.arange(len(probabilities)), action]

  def get_probabilities(self, state):
    return self._policy_fn(state)

  def get_log_probability(self, state, action):
    return np.log(1e-8 + self.get_probability(state, action))


class TabularPolicy(DiscreteFunctionPolicy):
  """Policy based on table of states-->probability distribution."""

  def __init__(self, probability_table,
               obs_to_index_fn = lambda obs: obs):
    """Initializes the policy.

    Args:
      probability_table: An array-like object of dimension [num_states,
        num_actions]. Each row should be a valid probability distribution.
      obs_to_index_fn: An optional function for mapping raw environment
        states to indices into the probability table. Should be usable on
        batches of states.
    """
    probability_table = np.array(probability_table)
    policy_fn = lambda state: probability_table[obs_to_index_fn(state)]

    super(TabularPolicy, self).__init__(policy_fn)


class MixturePolicy(Policy):
  """Policy based on mixture of multiple policies.

  Takes some finite number of policies with corresponding weights and produces
  a policy that at each state is a mixture distribution of these policies at
  that state. That is,
  mixture(a|s) = weight1 * policy1(a|s) + weight2 * policy2(a|s).

  """

  def __init__(self, policies, weights):
    """Initializes the policy.

    Args:
      policies: A list of policy objects.
      weights: A list of floats, corresponding to the weights that should be
        used in the mixture of the policies.

    Raises:
      ValueError if policies and weights are empty or of different length.
    """
    if not policies:
      raise ValueError('Input policies is empty.')
    if len(policies) != len(weights):
      raise ValueError('Policies and weights have different lengths: %d and %d.'
                       % (len(policies), len(weights)))

    self._policies = policies
    self._weights = np.array(weights) / np.sum(weights)

  def sample_action(self, state):
    policy_choice = np.random.choice(len(self._weights), p=self._weights)
    return self._policies[policy_choice].sample_action(state)

  def get_probability(self, state, action):
    policy_probabilities = [p.get_probability(state, action)
                            for p in self._policies]
    return np.dot(self._weights, policy_probabilities)

  def get_probabilities(self, state):
    policy_probabilities = [p.get_probabilities(state) for p in self._policies]
    return sum(w * p for w, p in zip(self._weights, policy_probabilities))

  def get_log_probability(self, state, action):
    policy_log_probabilities = [p.get_log_probability(state, action)
                                for p in self._policies]
    return np.log(np.dot(self._weights, np.exp(policy_log_probabilities)))


def get_policy_ratio(policy0, policy1, state, action,
                     epsilon = 0.0):
  """Compute the ratio of two policies sampling an action at a state.

  Args:
    policy0: The first (baseline) policy object.
    policy1: The second (target) policy object.
    state: The state.
    action: The action.
    epsilon: An optional small constant to add to both the numerator and
      denominator.  Defaults to 0.

  Returns:
    The ratio of probability of the second policy selecting the action compared
      to the probability of the first policy selecting the action.
  """
  probability0 = policy0.get_probability(state, action)
  probability1 = policy1.get_probability(state, action)
  return (probability1 + epsilon) / (probability0 + epsilon)
