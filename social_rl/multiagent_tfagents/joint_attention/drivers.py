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

"""Environment drivers for multigrid environments."""
from typing import Tuple

import numpy as np
import tensorflow as tf
from tf_agents.drivers import py_driver
from tf_agents.drivers import tf_driver
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types


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


class StatePyDriver(py_driver.PyDriver):
  """A PyDriver that adds policy state to observations.

  These policy states are used to compute attention weights in the attention
  architecture.
  """

  def run(
      self,
      time_step,
      policy_state = ()
  ):
    """Run policy in environment given initial time_step and policy_state.

    Args:
      time_step: The initial time_step.
      policy_state: The initial policy_state.

    Returns:
      A tuple (final time_step, final policy_state).
    """
    num_steps = 0
    num_episodes = 0
    while num_steps < self._max_steps and num_episodes < self._max_episodes:
      action_step = self.policy.action(time_step, policy_state)
      next_time_step = self.env.step(action_step.action)
      if next_time_step.step_type == 0:
        policy_state = self.policy.get_initial_state(self.env.batch_size or 1)
      next_time_step.observation['policy_state'] = (
          policy_state['actor_network_state'][0][0].numpy(),
          policy_state['actor_network_state'][1][0].numpy())

      traj = trajectory.from_transition(time_step, action_step, next_time_step)
      for observer in self._transition_observers:
        observer((time_step, action_step, next_time_step))
      for observer in self.observers:
        observer(traj)

      num_episodes += np.sum(traj.is_boundary())
      num_steps += np.sum(~traj.is_boundary())

      time_step = next_time_step
      policy_state = action_step.state

    return time_step, policy_state
