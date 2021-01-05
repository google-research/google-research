# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

  def __init__(self, num_states, num_actions):
    assert num_states > 0, 'Number of states must be positive.'
    assert num_actions > 0, 'Number of actions must be positive.'
    self.num_states = num_states
    self.num_actions = num_actions
    # We start with a fully unnormalized SxAxS matrix.
    self.transition_probs = np.random.rand(num_states, num_actions, num_states)
    for s in range(num_states):
      for a in range(num_actions):
        # Pick the number of states with zero mass.
        num_non_next = np.random.randint(1, num_states)
        non_next_idx = np.random.choice(np.arange(num_states),
                                        size=num_non_next, replace=False)
        # Zero out the chosen states.
        self.transition_probs[s, a, non_next_idx] = 0.
        # Normalize to make them sum to one.
        self.transition_probs[s, a, :] /= np.sum(self.transition_probs[s, a, :])
    # Reward mean and stddev are picked randomly.
    self.rewards = np.random.rand(num_states, num_actions)
    self.reward_stddevs = np.random.rand(num_states, num_actions)

  def reset(self):
    pass

  def render_custom_observation(self, unused_obs, unused_d, unused_cmap,
                                boundary_values=None):
    del boundary_values
    return None

  def _sample_next_state(self, s, a):
    return np.random.choice(self.num_states, p=self.transition_probs[s, a, :])

  def _sample_reward(self, s, a):
    r = np.random.normal(self.rewards[s, a], self.reward_stddevs[s, a])
    # We restrict rewards to [0, 1].
    return max(0, min(r, 1))
