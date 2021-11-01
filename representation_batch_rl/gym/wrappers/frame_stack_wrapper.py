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

"""Wrapper to stack multiple gym.spaces.Box observations.

Note: Unlike atari_wrappers.py we do not use LazyFrames to ensure observations
don't repeat memory, and we only support 1D observations.
"""

import collections

import gym
from gym import spaces
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class FrameStackWrapper(gym.Wrapper):
  """Env wrapper to Stack k last frames."""

  def __init__(self, env, k):
    if not isinstance(env.observation_space, spaces.Box):
      raise ValueError('env %s does not use spaces.Box.' % str(env))
    super(FrameStackWrapper, self).__init__(env)

    if not isinstance(k, int):
      raise ValueError('Expected integer k')
    self.k = k
    self.frames = collections.deque([], maxlen=k)
    obs = env.observation_space
    if len(obs.shape) != 1:
      raise ValueError('Only 1D observations supported.')
    self.observation_space = spaces.Box(
        low=np.tile(obs.low.tolist(), k),
        high=np.tile(obs.high.tolist(), k),
        dtype=obs.dtype)
    self._max_episode_steps = env._max_episode_steps  # pylint: disable=protected-access

  def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.k
    return np.concatenate(self.frames)


def stack_expert_frames(expert_data, k):
  """Stack exert (s, a, s', d, r) tuples from the flattened data format."""
  if not isinstance(expert_data, dict):
    raise ValueError('Expected dict expert data.')

  stacked_data = {key: [] for key in expert_data}

  def _reset_to_frame(i):
    state = collections.deque([], maxlen=k)
    for _ in range(k):
      state.append(expert_data['states'][i])
    return state

  state = _reset_to_frame(0)
  nsteps = expert_data['dones'].shape[0]

  for i, done in enumerate(expert_data['dones']):
    # Calculate the stacked tuple and append.
    stacked_data['states'].append(np.concatenate(state))
    stacked_data['actions'].append(expert_data['actions'][i])
    state.append(expert_data['next_states'][i])
    stacked_data['next_states'].append(np.concatenate(state))
    stacked_data['dones'].append(done)
    if 'rewards' in stacked_data:
      stacked_data['rewards'].append(expert_data['rewards'][i])

    if done and (i + 1) < nsteps:
      state = _reset_to_frame(i + 1)

  for key, value in stacked_data.items():
    stacked_data[key] = np.stack(value)
    assert stacked_data[key].shape[0] == nsteps

  return stacked_data


class FrameStackWrapperTfAgents(wrappers.PyEnvironmentBaseWrapper):
  """Env wrapper to stack k last frames.

  Maintains a circular buffer of the last k frame observations and returns
  TimeStep including a concatenated state vector (with the last frames action,
  reward, etc). Used to train models with multi-state context.

  Note, the first frame's state is replicated k times to produce a state vector
  sequence of [s_0, s_0, ..., s_0], [s_0, s_0, ..., s_1], etc.
  """

  def __init__(self, env, k):
    super(FrameStackWrapperTfAgents, self).__init__(env)

    obs_spec: array_spec.ArraySpec = self._env.observation_spec()
    if not isinstance(obs_spec, array_spec.ArraySpec):
      raise ValueError('Unsupported observation_spec %s' % str(obs_spec))
    if len(obs_spec.shape) != 1 and len(obs_spec.shape) != 3:
      raise ValueError(
          'Only 1D or 3D observations supported (found shape %s)' % (
              str(obs_spec.shape)))

    if len(obs_spec.shape) == 1:
      self._stacked_observation_spec = array_spec.ArraySpec(
          shape=(obs_spec.shape[0] * k,),
          dtype=obs_spec.dtype,
          name=obs_spec.name + '_stacked')
    else:
      self._stacked_observation_spec = array_spec.ArraySpec(
          shape=(obs_spec.shape[0:2] + (obs_spec.shape[2] * k,)),
          dtype=obs_spec.dtype,
          name=obs_spec.name + '_stacked')

    self._k: int = k
    self._timesteps = collections.deque([], maxlen=k)
    if hasattr(env, '_max_episode_steps'):
      self._max_episode_steps = env._max_episode_steps  # pylint: disable=protected-access

  def _reset(self):
    timestep = self._env.reset()
    assert isinstance(timestep, ts.TimeStep), (
        'Expected TimeStep, got %s' % type(timestep))
    for _ in range(self._k):
      self._timesteps.append(timestep)
    return self._get_timestep(timestep)

  def _step(self, action):
    timestep = self._env.step(action)
    assert isinstance(timestep, ts.TimeStep), (
        'Expected TimeStep, got %s' % type(timestep))
    self._timesteps.append(timestep)
    return self._get_timestep(timestep)

  def _get_timestep(self, time_step):
    assert len(self._timesteps) == self._k
    time_step = time_step._asdict()
    time_step['observation'] = np.concatenate([
        frame.observation for frame in self._timesteps], axis=-1)
    # assert self.observation_spec().shape == time_step['observation'].shape
    return ts.TimeStep(**time_step)

  def observation_spec(self):
    return self._stacked_observation_spec
