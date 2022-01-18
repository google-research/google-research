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

"""Combine multiple environments to step them in batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class BatchEnv(object):
  """Combine multiple environments to step them in batch."""

  def __init__(self, envs, blocking):
    """Combine multiple environments to step them in batch.

    To step environments in parallel, environments must support a
    `blocking=False` argument to their step and reset functions that makes them
    return callables instead to receive the result at a later time.

    Args:
      envs: List of environments.
      blocking: Step environments after another rather than in parallel.

    Raises:
      ValueError: Environments have different observation or action spaces.
    """
    self._envs = envs
    self._blocking = blocking
    observ_space = self._envs[0].observation_space
    if not all(env.observation_space == observ_space for env in self._envs):
      raise ValueError('All environments must use the same observation space.')
    action_space = self._envs[0].action_space
    if not all(env.action_space == action_space for env in self._envs):
      raise ValueError('All environments must use the same observation space.')

  def __len__(self):
    """Number of combined environments."""
    return len(self._envs)

  def __getitem__(self, index):
    """Access an underlying environment by index."""
    return self._envs[index]

  def __getattr__(self, name):
    """Forward unimplemented attributes to one of the original environments.

    Args:
      name: Attribute that was accessed.

    Returns:
      Value behind the attribute name one of the wrapped environments.
    """
    return getattr(self._envs[0], name)

  def set_attribute(self, name, values, single=False):
    """Set attributes of the original environments.

    Args:
      name: Attribute that was accessed.
      values: List of values (one element per env) or one value for all envs.
      single: Use the same value for all environments.
    """
    if single:
      for env in self._envs:
        env.set_attribute(name, values)
    else:
      for env, value in zip(self._envs, values):
        env.set_attribute(name, value)

  def step(self, action):
    """Forward a batch of actions to the wrapped environments.

    Args:
      action: Batched action to apply to the environment.

    Raises:
      ValueError: Invalid actions.

    Returns:
      Batch of observations, rewards, and done flags.
    """
    actions = action
    for index, (env, action) in enumerate(zip(self._envs, actions)):
      if not env.action_space.contains(action):
        message = 'Invalid action at index {}: {}'
        raise ValueError(message.format(index, action))
    if self._blocking:
      transitions = [
          env.step(action)
          for env, action in zip(self._envs, actions)]
    else:
      transitions = [
          env.step(action, blocking=False)
          for env, action in zip(self._envs, actions)]
      transitions = [transition() for transition in transitions]
    observs, rewards, dones, infos = zip(*transitions)
    observ = np.stack(observs)
    reward = np.stack(rewards)
    done = np.stack(dones)
    info = tuple(infos)
    return observ, reward, done, info

  def reset(self, indices=None):
    """Reset the environment and convert the resulting observation.

    Args:
      indices: The batch indices of environments to reset; defaults to all.

    Returns:
      Batch of observations.
    """
    if indices is None:
      indices = np.arange(len(self._envs))
    if self._blocking:
      observs = [self._envs[index].reset() for index in indices]
    else:
      observs = [self._envs[index].reset(blocking=False) for index in indices]
      observs = [observ() for observ in observs]
    observ = np.stack(observs)
    return observ

  def close(self):
    """Send close messages to the external process and join them."""
    for env in self._envs:
      if hasattr(env, 'close'):
        env.close()
