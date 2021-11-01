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

"""A wrapper that scales and shifts observations."""
import gym
import numpy as np
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step


class NormalizeStateWrapper(gym.ObservationWrapper):
  """Wraps an environment to shift and scale observations.
  """

  def __init__(self, env, shift, scale):
    super(NormalizeStateWrapper, self).__init__(env)
    self.shift = shift
    self.scale = scale

  def observation(self, observation):
    return (observation + self.shift) * self.scale

  @property
  def _max_episode_steps(self):
    if hasattr(self.env, '_max_episode_steps'):
      return self.env._max_episode_steps  # pylint: disable=protected-access
    else:
      return None


class NormalizeStateWrapperTFAgents(wrappers.PyEnvironmentBaseWrapper):
  """Wraps an environment and scales and shifts state (observations)."""

  def __init__(self, env, shift, scale):
    super(NormalizeStateWrapperTFAgents, self).__init__(env)
    self._shift = shift
    self._scale = scale

    # We don't support bounded observation specs (otherwise we'd have to scale
    # them).
    assert isinstance(env.observation_spec(), array_spec.ArraySpec), (
        'Expected ArraySpec, found %s' % str(type(env.observation_spec())))

  def _transform_state(self, state):
    return ((state + self._shift) * self._scale).astype(np.float32)

  def _reset(self):
    timestep = self._env.reset()._asdict()
    timestep['observation'] = self._transform_state(timestep['observation'])
    return time_step.TimeStep(**timestep)

  def _step(self, action):
    timestep = self._env.step(action)._asdict()
    timestep['observation'] = self._transform_state(timestep['observation'])
    return time_step.TimeStep(**timestep)
