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

"""A wrapper that adds explicit absorbing states."""
from absl import flags
import gym
import numpy as np


FLAGS = flags.FLAGS


class RandomActionReset(gym.Wrapper):
  """A wrapper that executes random action after resets."""

  def __init__(self, env, max_random_actions=5):
    """Sample a random number of random actions after each reset.

    Args:
      env: A gym enviornment to wrap.
      max_random_actions: A maximum number of random actions to perform.
    """
    gym.Wrapper.__init__(self, env)
    self.max_random_actions = max_random_actions
    self._max_episode_steps = env._max_episode_steps  # pylint: disable=protected-access

  def reset(self, **kwargs):
    """Reset and perform a random number of random actions."""
    self.env.reset(**kwargs)
    num_random = self.unwrapped.np_random.randint(1,
                                                  self.max_random_actions + 1)
    for _ in range(num_random):
      action = self.action_space.sample()
      obs, _, done, _ = self.env.step(action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs

  def step(self, action):
    return self.env.step(action)


class StickyActions(gym.Wrapper):
  """A wrapper that executes random action after resets."""

  def __init__(self, env, sticky_action_frequency=5):
    """Sample a random number of random actions after each reset.

    Args:
      env: A gym enviornment to wrap.
      sticky_action_frequency: A frequency of sticky action.
    """
    gym.Wrapper.__init__(self, env)
    self.sticky_action_frequency = sticky_action_frequency
    self.prev_action = self.action_space.sample()
    self._max_episode_steps = env._max_episode_steps  # pylint: disable=protected-access

  def reset(self, **kwargs):
    """Reset and perform a random number of random actions."""
    obs = self.env.reset(**kwargs)
    self.prev_action = self.action_space.sample()
    return obs

  def step(self, action):
    freq = self.unwrapped.np_random.randint(0, self.sticky_action_frequency)
    if freq == 0:
      action = self.prev_action
    self.prev_action = np.copy(action)
    return self.env.step(action)


def maybe_make_stochastic(env_name):
  """Create an environment instance (with optional stochastic wrappers)."""
  if env_name.startswith('Stochastic-'):
    root_env_name = env_name.lstrip('Stochastic-')
  else:
    root_env_name = env_name

  env = gym.make(root_env_name)

  if env_name.startswith('Stochastic-'):
    env = RandomActionReset(env)
    env = StickyActions(env)

  return env
