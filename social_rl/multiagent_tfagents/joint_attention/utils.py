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

"""Utility functions for multigrid environments."""

import gym
import numpy as np
import tensorflow as tf
from tf_agents.train import actor
from tf_agents.utils import common
from social_rl.multiagent_tfagents.joint_attention import drivers


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


class StateActor(actor.Actor):
  """An Actor that adds uses the StatePyDriver."""

  def __init__(self,
               *args,
               steps_per_run=None,
               episodes_per_run=None,
               **kwargs):
    """Initializes a StateActor.

    Args:
      *args: See superclass.
      steps_per_run: Number of steps evaluated per run call.
      episodes_per_run: Number of episodes evaluated per run call.
      **kwargs: See superclass.
    """
    super(StateActor, self).__init__(*args,
                                     steps_per_run,
                                     episodes_per_run,
                                     **kwargs)

    self._driver = drivers.StatePyDriver(
        self._env,
        self._policy,
        self._observers,
        max_steps=steps_per_run,
        max_episodes=episodes_per_run)

    self.reset()

  def write_metric_summaries(self):
    """Generates scalar summaries for the actor metrics."""
    super().write_metric_summaries()
    if self._metrics is None:
      return
    with self._summary_writer.as_default(), \
         common.soft_device_placement(), \
         tf.summary.record_if(lambda: True):
      # Generate summaries against the train_step
      for m in self._metrics:
        tag = m.name
        if 'Multiagent' in tag:
          for a in range(m.n_agents):
            tf.compat.v2.summary.scalar(name=tag + '_agent' + str(a),
                                        data=m.result_for_agent(a),
                                        step=self._train_step)
