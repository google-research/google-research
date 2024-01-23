# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Utilities for the contrastive RL agent."""
import functools
from typing import Dict
from typing import Optional, Sequence

from acme import types
from acme.agents.jax import actors
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.utils.observers import base as observers_base
from acme.wrappers import base
from acme.wrappers import canonical_spec
from acme.wrappers import gym_wrapper
from acme.wrappers import step_limit
import dm_env
import env_utils
import jax
import numpy as np


def obs_to_goal_1d(obs, start_index, end_index):
  assert len(obs.shape) == 1
  return obs_to_goal_2d(obs[None], start_index, end_index)[0]


def obs_to_goal_2d(obs, start_index, end_index):
  assert len(obs.shape) == 2
  if end_index == -1:
    return obs[:, start_index:]
  else:
    return obs[:, start_index:end_index]


class SuccessObserver(observers_base.EnvLoopObserver):
  """Measures success by whether any of the rewards in an episode are positive.
  """

  def __init__(self):
    self._rewards = []
    self._success = []

  def observe_first(self, env, timestep
                    ):
    """Observes the initial state."""
    if self._rewards:
      success = np.sum(self._rewards) >= 1
      self._success.append(success)
    self._rewards = []

  def observe(self, env, timestep,
              action):
    """Records one environment step."""
    assert timestep.reward in [0, 1]
    self._rewards.append(timestep.reward)

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    return {
        'success': float(np.sum(self._rewards) >= 1),
        'success_1000': np.mean(self._success[-1000:]),
    }


class DistanceObserver(observers_base.EnvLoopObserver):
  """Observer that measures the L2 distance to the goal."""

  def __init__(self, obs_dim, start_index, end_index,
               smooth = True):
    self._distances = []
    self._obs_dim = obs_dim
    self._obs_to_goal = functools.partial(
        obs_to_goal_1d, start_index=start_index, end_index=end_index)
    self._smooth = smooth
    self._history = {}

  def _get_distance(self, env,
                    timestep):
    if hasattr(env, '_dist'):
      assert env._dist  # pylint: disable=protected-access
      return env._dist[-1]  # pylint: disable=protected-access
    else:
      # Note that the timestep comes from the environment, which has already
      # had some goal coordinates removed.
      obs = timestep.observation[:self._obs_dim]
      goal = timestep.observation[self._obs_dim:]
      dist = np.linalg.norm(self._obs_to_goal(obs) - goal)
      return dist

  def observe_first(self, env, timestep
                    ):
    """Observes the initial state."""
    if self._smooth and self._distances:
      for key, value in self._get_current_metrics().items():
        self._history[key] = self._history.get(key, []) + [value]
    self._distances = [self._get_distance(env, timestep)]

  def observe(self, env, timestep,
              action):
    """Records one environment step."""
    self._distances.append(self._get_distance(env, timestep))

  def _get_current_metrics(self):
    metrics = {
        'init_dist': self._distances[0],
        'final_dist': self._distances[-1],
        'delta_dist': self._distances[0] - self._distances[-1],
        'min_dist': min(self._distances),
    }
    return metrics

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    metrics = self._get_current_metrics()
    if self._smooth:
      for key, vec in self._history.items():
        for size in [10, 100, 1000]:
          metrics['%s_%d' % (key, size)] = np.nanmean(vec[-size:])
    return metrics


class ObservationFilterWrapper(base.EnvironmentWrapper):
  """Wrapper that exposes just the desired goal coordinates."""

  def __init__(self, environment,
               idx):
    """Initializes a new ObservationFilterWrapper.

    Args:
      environment: Environment to wrap.
      idx: Sequence of indices of coordinates to keep.
    """
    super().__init__(environment)
    self._idx = idx
    observation_spec = environment.observation_spec()
    spec_min = self._convert_observation(observation_spec.minimum)
    spec_max = self._convert_observation(observation_spec.maximum)
    self._observation_spec = dm_env.specs.BoundedArray(
        shape=spec_min.shape,
        dtype=spec_min.dtype,
        minimum=spec_min,
        maximum=spec_max,
        name='state')

  def _convert_observation(self, observation):
    return observation[self._idx]

  def step(self, action):
    timestep = self._environment.step(action)
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def reset(self):
    timestep = self._environment.reset()
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def observation_spec(self):
    return self._observation_spec


def make_environment(env_name, start_index, end_index,
                     seed):
  """Creates the environment.

  Args:
    env_name: name of the environment
    start_index: first index of the observation to use in the goal.
    end_index: final index of the observation to use in the goal. The goal
      is then obs[start_index:goal_index].
    seed: random seed.
  Returns:
    env: the environment
    obs_dim: integer specifying the size of the observations, before
      the start_index/end_index is applied.
  """
  np.random.seed(seed)
  gym_env, obs_dim, max_episode_steps = env_utils.load(env_name)
  goal_indices = obs_dim + obs_to_goal_1d(np.arange(obs_dim), start_index,
                                          end_index)
  indices = np.concatenate([
      np.arange(obs_dim),
      goal_indices
  ])
  env = gym_wrapper.GymWrapper(gym_env)
  env = step_limit.StepLimitWrapper(env, step_limit=max_episode_steps)
  env = ObservationFilterWrapper(env, indices)
  if env_name.startswith('ant_'):
    env = canonical_spec.CanonicalSpecWrapper(env)
  return env, obs_dim


class InitiallyRandomActor(actors.GenericActor):
  """Actor that takes actions uniformly at random until the actor is updated.
  """

  def select_action(self,
                    observation):
    if (self._params['mlp/~/linear_0']['b'] == 0).all():
      shape = self._params['Normal/~/linear']['b'].shape
      rng, self._state = jax.random.split(self._state)
      action = jax.random.uniform(key=rng, shape=shape,
                                  minval=-1.0, maxval=1.0)
    else:
      action, self._state = self._policy(self._params, observation,
                                         self._state)
    return utils.to_numpy(action)
