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

# Lint as: python3
"""Replay buffer."""

import itertools as it
import pickle
import numpy as np


class Memory(object):
  """Container of episodes."""

  def __init__(self, data_keys=()):
    self.observations = []
    self.actions = []
    self.rewards = []
    self.data = {k: [] for k in data_keys}

  def log_init(self, obs):
    """Call this to begin logging a new episode."""
    self.observations.append([obs])
    self.actions.append([])
    self.rewards.append([])
    for value in self.data.values():
      value.append([])

  def log_experience(self, obs, act, reward, next_obs, data):
    """Add experience to the current episode."""
    assert (self.observations[-1][-1] == obs).all()
    self.observations[-1].append(next_obs)
    self.actions[-1].append(act)
    self.rewards[-1].append(reward)
    for key in data:
      self.data[key][-1].append(data[key])

  def entered_states(self):
    return np.array(list(it.chain.from_iterable(self.observations)))

  def exited_states(self):
    return np.array(list(it.chain.from_iterable(
        map(lambda obslist: obslist[0:-1], self.observations))))

  def attempted_actions(self):
    return np.array(list(it.chain.from_iterable(self.actions)))

  def observed_rewards(self):
    return np.array(list(it.chain.from_iterable(self.rewards)))

  def executed_state_action_pairs(self):
    return np.array(list(zip(self.exited_states(), self.attempted_actions())))

  def serialize(self):
    return pickle.dumps(
        (np.array(self.observations),
         np.array(self.actions),
         np.array(self.rewards),
         {key: np.array(value) for key, value in self.data.items()})
    )

  def unserialize(self, s):
    stuff = pickle.loads(s)
    obs, act, rew = stuff[:3]
    self.observations = obs
    self.actions = act
    self.rewards = rew
    if len(stuff) > 3:
      self.data = stuff[3]
