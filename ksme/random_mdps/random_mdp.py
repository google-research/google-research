# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""This class creates a random MDP.

It must respect the interface expected by metric.py.
"""

import gin
import numpy as np


@gin.configurable
class RandomMDP(object):
  """Class to create random MDPs."""

  def __init__(self, num_states, num_actions, policy_type='stochastic',
               reward_variance=1.0):
    assert num_states > 0, 'Number of states must be positive.'
    assert num_actions > 0, 'Number of actions must be positive.'
    self.num_states = num_states
    self.num_actions = num_actions
    if policy_type == 'stochastic':
      self.policy = np.random.dirichlet(np.ones(num_actions), size=num_states)
    elif policy_type == 'deterministic':
      self.policy = np.zeros((num_states, num_actions))
      np.put_along_axis(
          self.policy,
          np.random.randint(num_actions, size=num_states)[:, None],
          values=1., axis=1)
    else:
      raise ValueError(f'Unknown policy type: {policy_type}')
    # We start with a fully unnormalized SxAxS matrix.
    self.transition_probs = np.random.rand(num_states, num_actions, num_states)
    for x in range(num_states):
      for a in range(num_actions):
        # Pick the number of states with zero mass.
        num_non_next = np.random.randint(1, num_states)
        non_next_idx = np.random.choice(np.arange(num_states),
                                        size=num_non_next, replace=False)
        # Zero out the chosen states.
        self.transition_probs[x, a, non_next_idx] = 0.
        # Normalize to make them sum to one.
        self.transition_probs[x, a, :] /= np.sum(self.transition_probs[x, a, :])
    # Reward mean and stddev are picked randomly.
    self.rewards = np.random.normal(loc=0.5, scale=reward_variance,
                                    size=(num_states, num_actions))
    # Clip rewards to lie in [0., 1.]
    self.rewards = np.clip(self.rewards, 0.0, 1.0)
    self.policy_transition_probs = np.einsum('ijk,ij->ik',
                                             self.transition_probs,
                                             self.policy)
    self.policy_rewards = np.einsum('ij,ij->i', self.rewards, self.policy)
