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

r"""Given an environment, compute, in closed form, the SR wrt a given policy.

The SR was originally introduced by Dayan (1993).
"""

import gin
import numpy as np


@gin.configurable
def sr_closed_form(env, policy, gamma):
  r"""Computes the successor representation in closed form.

  Args:
    env: Gym environment wrapped with our MDP wrapper.
    policy: S \times A matrix whose entry policy[s, a] encodes the probability
    of the agent taking action a in state s.
    gamma: Discount factor to be used when computing the SR.

  Returns:
    sr: successor representation, which is a matrix S x S.
  """
  num_states = env.num_states
  num_actions = env.num_actions
  transition_prob = np.copy(env.transition_probs)
  trans_matrix = np.zeros((num_states, num_states))
  # The transition probability is S x A x S. We need to make it S x S here.
  for s in range(num_states):
    for a in range(num_actions):
      for s_next in range(num_states):
        trans_matrix[s][s_next] += transition_prob[s][a][s_next] * policy[s][a]

  sr = np.linalg.inv(np.identity(num_states) - gamma * trans_matrix)
  return sr, trans_matrix
