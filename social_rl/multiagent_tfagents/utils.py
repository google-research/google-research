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

"""Utlity functions for multigrid environments."""
from typing import Tuple

import gym
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.drivers import tf_driver
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types


class LSTMStateWrapper(gym.ObservationWrapper):
  """Wrapper to add LSTM state to observation dicts."""

  def __init__(self, env, lstm_size):
    super(LSTMStateWrapper, self).__init__(env)
    self.lstm_size = lstm_size
    observation_space_dict = self.env.observation_space.spaces.copy()
    multiagent = len(observation_space_dict['image'].shape) == 4
    if multiagent:
      n_agents = observation_space_dict['image'].shape[0]
      self.policy_state_shape = (n_agents,) + self.lstm_size
    else:
      self.policy_state_shape = (self.lstm_size)
    observation_space_dict['policy_state'] = gym.spaces.Tuple(
        (gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.policy_state_shape,
            dtype='float32'),
         gym.spaces.Box(
             low=-np.inf,
             high=np.inf,
             shape=self.policy_state_shape,
             dtype='float32')))
    self.observation_space = gym.spaces.Dict(observation_space_dict)

  def observation(self, observation):
    observation['policy_state'] = (np.zeros(
        self.policy_state_shape,
        dtype=np.float32), np.zeros(self.policy_state_shape, dtype=np.float32))
    return observation


class StateTFDriver(tf_driver.TFDriver):
  """A TFDriver that adds policy state to observations.

  These policy states are used to compute attention weights in the attention
  architecture.
  """

  def run(
      self, time_step, policy_state = ()
  ):
    """Run policy in environment given initial time_step and policy_state.

    Args:
      time_step: The initial time_step.
      policy_state: The initial policy_state.

    Returns:
      A tuple (final time_step, final policy_state).
    """
    num_steps = tf.constant(0.0)
    num_episodes = tf.constant(0.0)

    while num_steps < self._max_steps and num_episodes < self._max_episodes:
      action_step = self.policy.action(time_step, policy_state)
      next_time_step = self.env.step(action_step.action)
      next_time_step.observation['policy_state'] = (
          policy_state['actor_network_state'][0],
          policy_state['actor_network_state'][1])

      traj = trajectory.from_transition(time_step, action_step, next_time_step)
      for observer in self._transition_observers:
        observer((time_step, action_step, next_time_step))
      for observer in self.observers:
        observer(traj)

      num_episodes += tf.math.reduce_sum(
          tf.cast(traj.is_boundary(), tf.float32))
      num_steps += tf.math.reduce_sum(tf.cast(~traj.is_boundary(), tf.float32))

      time_step = next_time_step
      policy_state = action_step.state

    return time_step, policy_state
