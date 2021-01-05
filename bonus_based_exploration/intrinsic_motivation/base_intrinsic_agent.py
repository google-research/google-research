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

"""Base class for agent with an intrinsic reward.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.agents.dqn import dqn_agent as base_dqn_agent
import numpy as np
import tensorflow.compat.v1 as tf

NATURE_DQN_OBSERVATION_SHAPE = base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = base_dqn_agent.NATURE_DQN_DTYPE
NATURE_DQN_STACK_SIZE = base_dqn_agent.NATURE_DQN_STACK_SIZE


class IntrinsicDQNAgent(base_dqn_agent.DQNAgent):
  """Abstract class for a DQN agent with an intrinsic reward."""

  def _add_intrinsic_reward(self, observation, extrinsic_reward):
    """Compute the intrinsic reward."""
    if not hasattr(self, 'intrinsic_model'):
      raise NotImplementedError
    intrinsic_reward = self.intrinsic_model.compute_intrinsic_reward(
        observation, self.training_steps, self.eval_mode)
    reward = np.clip(intrinsic_reward + extrinsic_reward, -1., 1.)

    if (self.summary_writer is not None and
        self.training_steps % self.summary_writing_frequency == 0):
      summary = tf.Summary(value=[
          tf.Summary.Value(tag='Train/ExtrinsicReward',
                           simple_value=extrinsic_reward),
          tf.Summary.Value(tag='Train/IntrinsicReward',
                           simple_value=np.clip(intrinsic_reward, -1., 1.)),
          tf.Summary.Value(tag='Train/TotalReward',
                           simple_value=reward)
      ])
      self.summary_writer.add_summary(summary, self.training_steps)

    return reward

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    total_reward = self._add_intrinsic_reward(self._observation, reward)
    return base_dqn_agent.DQNAgent.step(self, total_reward, observation)

  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    total_reward = self._add_intrinsic_reward(self._observation, reward)
    base_dqn_agent.DQNAgent.end_episode(self, total_reward)
