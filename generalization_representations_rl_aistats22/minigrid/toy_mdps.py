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

"""Classes to create several toy MDPs.
"""

import numpy as np

from generalization_representations_rl_aistats22.minigrid import rl_basics


class StarMDP(object):
  """Class to create the StarMDP."""

  def __init__(self, num_states, terminal=False):
    assert num_states > 0, 'Number of states must be positive.'
    self.num_states = num_states
    self.num_actions = 1
    if not terminal:  # if the nth state is not terminal
      self.transition_probs = np.zeros(
          (num_states, self.num_actions, num_states))
      for s in range(num_states - 1):
        # action 0 goes to the star center from star corners
        self.transition_probs[s, 0, num_states-1] = 1
        # action 0 goes to star corners uniformly from star center
        self.transition_probs[num_states-1, 0, s] = 1 / (num_states - 1)
    else:
      self.transition_probs = np.zeros(
          (num_states, self.num_actions, num_states))
      for s in range(num_states - 1):
        # action 0 goes to the star center
        self.transition_probs[s, 0, num_states-1] = 1
      self.transition_probs[num_states-1, 0, num_states-1] = 1


class DisconnectedMDP(object):
  """Class to create a disconnected MDP."""

  def __init__(self, num_states):
    assert num_states > 0, 'Number of states must be positive.'
    self.num_states = num_states
    self.num_actions = 1
    self.transition_probs = np.zeros(
        (num_states, self.num_actions, num_states))
    for s in range(num_states):
      self.transition_probs[s, 0, s] = 1


class FullyConnectedMDP(object):
  """Class to create a fully connected MDP."""

  def __init__(self, num_states):
    assert num_states > 0, 'Number of states must be positive.'
    self.num_states = num_states
    self.num_actions = num_states - 1
    self.transition_probs = np.zeros(
        (num_states, self.num_actions, num_states))
    for s in range(self.num_states):
      for a in range(self.num_actions):
        # action i goes to state i+1
        self.transition_probs[s, a, a+1] = 1


class Torus1dMDP(object):
  """Class to create a cycle MDP."""

  def __init__(self, num_states):
    assert num_states > 0, 'Number of states must be positive.'
    self.num_states = num_states
    self.num_actions = 2  # action 0 (resp 1) goes to state n-1 (resp n+1)
    self.transition_probs = np.zeros(
        (num_states, self.num_actions, num_states))
    for s in range(self.num_states):
      self.transition_probs[s, 0, (s-1) % num_states] = 1
      self.transition_probs[s, 1, (s+1) % num_states] = 1


class Torus2dMDP(object):
  """Class to create a 2d torus MDP."""

  def __init__(self, num_states):
    assert num_states > 0, 'Number of states must be positive.'
    self.num_states = num_states
    self.num_actions = 4
    # action 0 goes to state (i + 1 mod n, j)
    # action 1 goes to state (i - 1 mod n, j)
    # action 2 goes to state (i, j+1 mod n)
    # action 3 goes to state (i, j-1 mod n)
    self.transition_probs = np.zeros(
        (num_states, self.num_actions, num_states))
    n = int(np.sqrt(num_states))
    for s in range(self.num_states):
      x, y = rl_basics.get_state_xy(s, n)
      nn1 = rl_basics.get_state_idx((x+1)%n, y, n)
      nn2 = rl_basics.get_state_idx((x-1)%n, y, n)
      nn3 = rl_basics.get_state_idx(x, (y+1)%n, n)
      nn4 = rl_basics.get_state_idx(x, (y-1)%n, n)
      self.transition_probs[s, 0, nn1] = 1
      self.transition_probs[s, 1, nn2] = 1
      self.transition_probs[s, 2, nn3] = 1
      self.transition_probs[s, 3, nn4] = 1


class ChainMDP(object):
  """Class to create a chain MDP."""

  def __init__(self, num_states):
    assert num_states > 0, 'Number of states must be positive.'
    self.num_states = num_states
    self.num_actions = 2  # action 0 means left and 1 means right
    self.transition_probs = np.zeros((num_states, self.num_actions, num_states))
    for s in range(1, self.num_states - 1):
      self.transition_probs[s, 0, s - 1] = 1
      self.transition_probs[s, 1, s + 1] = 1
    self.transition_probs[0, 0, 0] = 1
    self.transition_probs[0, 1, 1] = 1
    self.transition_probs[num_states-1, 1, num_states-1] = 1
    self.transition_probs[num_states-1, 0, num_states-2] = 1


class LatticeMDP(object):
  """Class to create a chain MDP."""

  def __init__(self, num_states):
    assert num_states > 0, 'Number of states must be positive.'
    self.num_states = num_states
    self.num_actions = 4  # action (0 = left), (1= right), (2=up), (3= down)
    self.transition_probs = np.zeros((num_states, self.num_actions, num_states))
    for s in range(1, self.num_states - 1):
      self.transition_probs[s, 0, s - 1] = 1
      self.transition_probs[s, 1, s + 1] = 1
    self.transition_probs[0, 0, 0] = 1
    self.transition_probs[0, 1, 1] = 1
    self.transition_probs[num_states-1, 1, num_states-1] = 1
    self.transition_probs[num_states-1, 0, num_states-2] = 1
