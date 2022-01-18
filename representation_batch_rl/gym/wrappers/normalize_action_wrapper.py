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

"""Wrapper to normalize gyn.spaces.Box actions in [-1, 1].
"""

from absl import logging
import gym
from gym import spaces
import numpy as np
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.typing import types


class NormalizeBoxActionWrapper(gym.ActionWrapper):
  """Rescale the action space of the environment."""

  def __init__(self, env):
    if not isinstance(env.action_space, spaces.Box):
      raise ValueError('env %s does not use spaces.Box.' % str(env))
    super(NormalizeBoxActionWrapper, self).__init__(env)
    self._max_episode_steps = env._max_episode_steps  # pylint: disable=protected-access

  def action(self, action):
    # rescale the action
    low, high = self.env.action_space.low, self.env.action_space.high
    scaled_action = low + (action + 1.0) * (high - low) / 2.0
    scaled_action = np.clip(scaled_action, low, high)

    return scaled_action

  def reverse_action(self, scaled_action):
    low, high = self.env.action_space.low, self.env.action_space.high
    action = (scaled_action - low) * 2.0 / (high - low) - 1.0
    return action


def check_and_normalize_box_actions(env):
  """Wrap env to normalize actions if [low, high] != [-1, 1]."""
  low, high = env.action_space.low, env.action_space.high

  if isinstance(env.action_space, spaces.Box):
    if (np.abs(low + np.ones_like(low)).max() > 1e-6 or
        np.abs(high - np.ones_like(high)).max() > 1e-6):
      logging.info('Normalizing environment actions.')
      return NormalizeBoxActionWrapper(env)

  # Environment does not need to be normalized.
  return env


class NormalizeActionWrapperTFAgents(wrappers.PyEnvironmentBaseWrapper):
  """Wraps an environment and scales and shifts actions before applying."""

  def __init__(self, env, shift, scale):
    super(NormalizeActionWrapperTFAgents, self).__init__(env)
    self._shift = shift
    self._scale = scale

    act_spec = env.action_spec()
    assert isinstance(act_spec, array_spec.BoundedArraySpec), (
        'Expected BoundedArraySpec, found %s' % str(type(act_spec)))

    self._action_spec = array_spec.BoundedArraySpec(
        act_spec.shape, act_spec.dtype,
        self._transform_action(act_spec.minimum),
        self._transform_action(act_spec.maximum))

  def _transform_action(self, action):
    return ((action + self._shift) * self._scale).astype(np.float32)

  def _step(self, action):
    """Steps the environment after scaling and shifting the actions.

    Args:
      action: Action to take.

    Returns:
      The next time_step from the environment.
    """
    return self._env.step(self._transform_action(action))

  def action_spec(self):
    return self._action_spec
