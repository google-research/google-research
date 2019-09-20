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

"""Utilities for generating and storing experience transition data.

Provides mechanisms for generating off-policy transition data given a behavior
policy as well as storing and sampling from this off-policy data.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc

import attr
import numpy as np
import six
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import dual_dice.policy as policy_lib


@attr.s
class TransitionTuple(object):
  state = attr.ib()
  action = attr.ib()
  reward = attr.ib()
  next_state = attr.ib()
  discount = attr.ib()
  initial_state = attr.ib()
  time_step = attr.ib()
  action_sampled_prob = attr.ib()


@six.add_metaclass(abc.ABCMeta)
class TransitionData(object):
  """Abstract class representing a store of off-policy data."""

  @abc.abstractmethod
  def sample_batch(self, batch_size):
    """Samples batch of sarsa data."""

  @abc.abstractmethod
  def get_all(self):
    """Gets all stored transition data."""

  @abc.abstractmethod
  def iterate_once(self):
    """Iterate over all data once."""

  @property
  @abc.abstractmethod
  def policy(self):
    """Get policy used to sample off-policy data, if it exists."""


class TrajectoryData(TransitionData):
  """Transition data storage based on contiguous trajectories."""

  def __init__(self,
               trajectories,
               policy = None):
    """Initializes the data with a set of trajectories.

    Args:
      trajectories: A list of trajectories. Each trajectory is a list of
        (state, action, reward, next_state) tuples.
      policy: The behavior policy, if known.  Defaults to None (unknown).
    """
    self._trajectories = trajectories
    self._policy = policy
    self._num_data = 0

    data = TransitionTuple([], [], [], [], [], [], [], [])
    for trajectory in trajectories:
      if not np.size(trajectory):
        continue
      initial_state, _, _, _ = trajectory[0]
      for step, (state, action, reward, next_state) in enumerate(trajectory):
        data.state.append(state)
        data.action.append(action)
        data.reward.append(reward)
        data.next_state.append(next_state)

        # Discount is always 1 (infinite-horizon setting).
        data.discount.append(1.0)

        data.initial_state.append(initial_state)
        data.time_step.append(step)
        action_sampled_prob = (None if policy is None else
                               policy.get_probability(state, action))
        data.action_sampled_prob.append(action_sampled_prob)

        self._num_data += 1

    # Convert data to NumPy array.
    self._data = TransitionTuple(*map(np.array, attr.astuple(data)))

  def sample_batch(self, batch_size):
    """Samples batch of sarsa data."""
    indices = np.random.choice(self._num_data, batch_size, replace=True)
    return TransitionTuple(
        *[np.take(arr, indices, axis=0) for arr in attr.astuple(self._data)])

  def get_all(self):
    """Gets all stored sarsa data."""
    return self._data

  def iterate_once(self):
    """Iterates over all data once."""
    for idx in range(self._num_data):
      yield TransitionTuple(
          *[np.take(arr, idx, axis=0) for arr in attr.astuple(self._data)])

  @property
  def policy(self):
    """Gets policy used to sample off-policy data, if it exists."""
    return self._policy

  @property
  def trajectories(self):
    """Gets original trajectories used to create off-policy data."""
    return self._trajectories


def collect_data(
    env, policy,
    num_trajectories, trajectory_length, gamma = 0.99,
    reward_fn = None):
  """Creates off-policy dataset by running a behavior policy in an environment.

  Args:
    env: An environment.
    policy: A behavior policy.
    num_trajectories: Number of trajectories to collect.
    trajectory_length: Desired length of each trajectory; how many steps to run
      behavior policy in the environment before resetting.
    gamma: Discount used for total and average reward calculation.
    reward_fn: A function (default None) in case the environment reward
      should be overwritten. This function should take in the environment
      reward and the environment's `done' flag and should return a new reward
      to use. A new reward function must be passed in for environments that
      terminate, since the code assumes an infinite-horizon setting.

  Returns:
    data: A TrajectoryData object containing the collected experience.
    avg_episode_rewards: Compute per-episode discounted rewards averaged over
      the trajectories.
    avg_step_rewards: Computed per-step average discounted rewards averaged
      over the trajectories.

  Raises:
    ValueError: If the environment terminates and a reward_fn is not passed in.
  """
  trajectories = []
  trajectory_rewards = []
  total_mass = 0  # For computing average per-step reward.
  for _ in range(num_trajectories):
    trajectory = []
    total_reward = 0
    discount = 1.0
    state = env.reset()
    for _ in range(trajectory_length):
      action = policy.sample_action(state)
      next_state, reward, done, _ = env.step(action)
      if reward_fn is not None:
        reward = reward_fn(reward, done)
      elif done:
        raise ValueError(
            'Environment terminated but reward_fn is not specified.')

      trajectory.append((state, action, reward, next_state))
      total_reward += reward * discount
      total_mass += discount

      state = next_state
      discount *= gamma

    trajectories.append(trajectory)
    trajectory_rewards.append(total_reward)
    avg_step_rewards = np.sum(trajectory_rewards) / total_mass

  avg_episode_rewards = np.mean(trajectory_rewards)
  avg_step_rewards = np.sum(trajectory_rewards) / total_mass

  return (TrajectoryData(trajectories, policy=policy),
          avg_episode_rewards, avg_step_rewards)
