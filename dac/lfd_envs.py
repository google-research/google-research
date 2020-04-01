# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A wrapper for Lfd environments that converts them into OpenAI Gym format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types
import gym
import numpy as np


class AbsorbingWrapper(gym.ObservationWrapper):
  """Wraps an environment to have an indicator dimension.

  The indicator dimension is used to represent absorbing states of MDP.

  If the last dimension is 0. It corresponds to a normal state of the MDP,
  1 corresponds to an absorbing state.

  The environment itself returns only normal states, absorbing states are added
  later.

  This wrapper is used mainly for GAIL, since we need to have explicit
  absorbing states in order to be able to assign rewards.
  """

  def __init__(self, env):
    super(AbsorbingWrapper, self).__init__(env)
    obs_space = self.observation_space
    self.observation_space = gym.spaces.Box(
        shape=(obs_space.shape[0] + 1,),
        low=obs_space.low[0],
        high=obs_space.high[0])

  def observation(self, observation):
    return self.get_non_absorbing_state(observation)

  def get_non_absorbing_state(self, obs):
    """Converts an original state of the environment into a non-absorbing state.

    Args:
      obs: a numpy array that corresponds to a state of unwrapped environment.

    Returns:
      A numpy array corresponding to a non-absorbing state obtained from input.
    """
    return np.concatenate([obs, [0]], -1)

  def get_absorbing_state(self):
    """Returns an absorbing state that corresponds to the environment.

    Returns:
      A numpy array that corresponds to an absorbing state.
    """
    obs = np.zeros(self.observation_space.shape)
    obs[-1] = 1
    return obs

  @property
  def _max_episode_steps(self):
    return self.env._max_episode_steps  # pylint: disable=protected-access


class LfdWrapper(gym.ObservationWrapper):
  """Wraps an Lfd environment to match OpenAI Gym format.
  """

  def __init__(self, env):
    super(LfdWrapper, self).__init__(env)
    self.action_space = env.act_space

    np_random = np.random.RandomState()
    self.unwrapped.np_random = np_random

    def sample_action(self):
      return np_random.uniform(
          size=self.shape, low=self.act_min,
          high=self.act_max).astype('float32')

    self.action_space.sample = types.MethodType(sample_action,
                                                self.action_space)
    self.observation_space = env.obs_space['state']

  def observation(self, observation):
    return observation['state']


class MultitaskWrapper(gym.ObservationWrapper):
  """Wraps an multitask environment to match OpenAI Gym format.
  """

  def __init__(self, env):
    super(MultitaskWrapper, self).__init__(env)
    self.action_space = env.act_space

    np_random = np.random.RandomState()
    self.unwrapped.np_random = np_random

    def sample_action(self):
      return np_random.uniform(
          size=self.shape, low=self.act_min,
          high=self.act_max).astype('float32')

    self.action_space.sample = types.MethodType(sample_action,
                                                self.action_space)

    self.observation_space = gym.spaces.Box(
        -np.inf,
        np.inf, [
            env.obs_space['state'].shape[0] +
            env.obs_space['task_embeddings'].shape[0]
        ],
        dtype=env.obs_space['state'].dtype)

  def observation(self, observation):
    return np.concatenate(
        [observation['state'], observation['task_embeddings']], -1)


class BulletWrapper(gym.Wrapper):
  """Wraps a Bullet environment to match OpenAI Gym format.

  In particular, it changed the type of the rewards.
  """

  def step(self, action):
    observation, reward, done, info = self.env.step(action)

    if info is None:
      info = {'original_done': str(done)}
    else:
      info['original_done'] = str(done)

    if str(done) == 'DoneType.FALSE':
      done = False
    else:
      done = True

    return observation, reward, done, info
