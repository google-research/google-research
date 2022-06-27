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

"""Environment wrappers for policy training with functional distance.
"""

import collections
import copy
import os
import pickle
from typing import Callable, Mapping, Optional, Tuple

from acme import types as acme_types
from acme import wrappers
from acme.jax import utils
from acme.utils import counting
import dm_env
from dm_env import specs as dm_env_specs
import jax.numpy as jnp
import numpy as np
import tree
from tensorflow.io import gfile
from func_dist.data_utils import image_utils

KeyedSpec = Mapping[str, acme_types.NestedSpec]
KeyedRewardSpec = Mapping[str, dm_env_specs.Array]
DistanceFnOutput = Tuple[jnp.ndarray, jnp.ndarray]


class DistanceFn:
  """Given a state and a goal, returns a distance and intermediate features."""

  def __init__(
      self,
      distance_fn,
      history_length):
    self._distance_fn = distance_fn
    self.history_length = history_length

  def __call__(self, state, goal):
    return self._distance_fn(state, goal)


class RecordEpisodesWrapper(wrappers.EnvironmentWrapper):
  """Record full episodes of a gym Environment, including info fields."""

  def __init__(self, environment, counter,
               logdir, record_every = 1000, num_to_record = 1,
               eval_mode = False):
    self._environment = environment
    self._logdir = logdir
    self._counter = counter
    print('Initializing RecordEpisodesWrapper with counters',
          self._counter.get_counts())
    self._record_every = record_every
    self._num_to_record = num_to_record
    self._eval_mode = eval_mode
    self._episodes_to_record = []
    self._episode = collections.defaultdict(list)

  def _get_episode_idx(self):
    counts = self._counter.get_counts()
    if self._eval_mode and 'eval_episodes' in counts:
      episode_idx = counts['eval_episodes']
    elif not self._eval_mode and 'episodes' in counts:
      episode_idx = counts['episodes']
    else:
      episode_idx = 0
    return episode_idx

  def _should_record_episode(self):
    # Always records self._num_to_record episodes per file, including the
    # first file, when first episode_idx = 1 instead of 0.
    return (self._get_episode_idx() % self._record_every < self._num_to_record
            or (self._episodes_to_record
                and len(self._episodes_to_record) < self._num_to_record))

  def _write_episodes(self):
    """Write recorded episodes to pickle."""
    # Inclusive first and last.
    episode_idx = self._get_episode_idx()
    if self._eval_mode:
      counts = self._counter.get_counts()
      train_steps = counts['steps'] if 'steps' in counts else 0
      filename = f'evaluation_{train_steps}.pkl'
    else:
      first_ep = episode_idx - len(self._episodes_to_record) + 1
      last_ep = episode_idx
      filename = f'episodes_{first_ep}-{last_ep}.pkl'
    log_path = os.path.join(self._logdir, filename)
    print('Episode', episode_idx, ': flushing to', log_path)
    with gfile.GFile(log_path, 'wb') as f:
      pickle.dump(self._episodes_to_record, f)
    self._episodes_to_record = []

  def reset(self):
    """Reset the episode."""
    timestep = self._environment.reset()
    if self._should_record_episode():
      self._episode['observations'].append(
          self._compress_images(timestep.observation))
    return timestep

  def _compress_images(
      self, observation):
    """Compress 3D images / 4D stacks of images and leave other fields as-is."""
    compressed_observation = copy.deepcopy(observation)
    for k, v in compressed_observation.items():
      if len(v.shape) >= 3:
        v = image_utils.img_to_uint(v)
        if len(v.shape) == 3:
          compressed_observation[k] = image_utils.compress_image(v)
        elif len(v.shape) == 4:
          compressed_observation[k] = [
              image_utils.compress_image(img) for img in v]
    return compressed_observation

  def step(self, action):
    """Step the environment, and record time step outputs when applicable."""
    timestep = self._environment.step(action)

    if self._should_record_episode():
      self._episode['observations'].append(
          self._compress_images(timestep.observation))
      self._episode['actions'].append(action)
      self._episode['rewards'].append(timestep.reward)
      self._episode['terminals'].append(timestep.last())
      if timestep.last():
        self._episodes_to_record.append(self._episode)
        self._episode = collections.defaultdict(list)
        if len(self._episodes_to_record) == self._num_to_record:
          self._write_episodes()
    return timestep


class EndOnSuccessWrapper(wrappers.EnvironmentWrapper):
  """End episode early if success criteria is reached (reward = 1)."""

  def __init__(self, environment):
    self._environment = environment

  def step(self, action):
    timestep = self._environment.step(action)
    if timestep.reward == 1:
      timestep = dm_env.termination(timestep.reward, timestep.observation)
    return timestep


class ReshapeImageWrapper(wrappers.EnvironmentWrapper):
  """Reshape flattened image observation."""

  def __init__(self, environment):
    self._environment = environment
    self._observation_spec = tree.map_structure(
        self._update_spec, self._environment.observation_spec())

  def observation_spec(self):
    return self._observation_spec

  def _update_spec(self, spec):
    flat_image = np.zeros(spec.shape)
    new_shape = self._wrap_observation(flat_image).shape
    return dm_env_specs.Array(shape=new_shape, dtype=spec.dtype, name=spec.name)

  def _wrap_observation(
      self, observation):
    observation = image_utils.shape_img(observation)
    return observation

  def reset(self):
    timestep = self._environment.reset()
    observation = self._wrap_observation(timestep.observation)
    return timestep._replace(observation=observation)

  def step(self, action):
    timestep = self._environment.step(action)
    observation = self._wrap_observation(timestep.observation)
    return timestep._replace(observation=observation)


class GoalConditionedWrapper(wrappers.EnvironmentWrapper):
  """Add a goal image to the observed state."""

  def __init__(self, environment, goal):
    self._environment = environment
    self._goal = goal
    original_spec = self.environment.observation_spec()
    self._observation_spec = {
        'state': dm_env_specs.Array(
            shape=original_spec.shape, dtype=original_spec.dtype, name='state'),
        'goal': dm_env_specs.Array(
            shape=self._goal.shape, dtype=self._goal.dtype, name='goal')}

  def observation_spec(self):
    return self._observation_spec

  def _update_spec(self, spec):
    new_shape = (
        spec.shape[0], spec.shape[1] + self._goal.shape[1], *spec.shape[2:])
    return dm_env_specs.Array(shape=new_shape, dtype=spec.dtype, name=spec.name)

  def _wrap_observation(
      self, observation):
    observation = {'state': observation, 'goal': self._goal}
    return observation

  def reset(self):
    timestep = self._environment.reset()
    observation = self._wrap_observation(timestep.observation)
    return timestep._replace(observation=observation)

  def step(self, action):
    timestep = self._environment.step(action)
    observation = self._wrap_observation(timestep.observation)
    return timestep._replace(observation=observation)


class DistanceModelWrapper(wrappers.EnvironmentWrapper):
  """Embed observation images and add predicted distance reward."""

  def __init__(self,
               environment,
               distance_fn,
               max_episode_steps = None,
               baseline_distance = None,
               distance_reward_weight = 1.,
               environment_reward_weight = 0.):
    self._environment = environment
    self._distance_fn = distance_fn
    self._max_episode_steps = max_episode_steps
    self._baseline_distance = baseline_distance
    self._distance_reward_weight = distance_reward_weight
    self._environment_reward_weight = environment_reward_weight

    self._observation_spec = self._update_spec(
        self._environment.observation_spec())
    self._reward_spec = {
        'distance': dm_env_specs.Array((), np.float32),
        'environment': self._environment.reward_spec(),
    }

  def observation_spec(self):
    return self._observation_spec

  def reward_spec(self):
    return self._reward_spec

  def _update_spec(self, base_spec
                   ):
    dummy_obs = utils.zeros_like(base_spec)
    emb, _ = self._distance_fn(dummy_obs['state'], dummy_obs['goal'])
    full_spec = dict(base_spec)
    full_spec['embeddings'] = (
        dm_env_specs.Array(shape=emb.shape, dtype=emb.dtype))
    return full_spec

  def _distance_to_reward(self, distance, env_reward):
    reward = -distance * self._distance_reward_weight
    if self._max_episode_steps is not None:
      reward /= self._max_episode_steps
    reward += self._environment_reward_weight * env_reward
    return reward

  def _wrap_timestep(self, timestep):
    embeddings, distance = self._distance_fn(
        timestep.observation['state'], timestep.observation['goal'])
    if self._baseline_distance is not None:
      distance -= self._baseline_distance
      distance = np.maximum(0, distance)
    full_observation = dict(timestep.observation)
    full_observation['embeddings'] = embeddings
    timestep = timestep._replace(observation=full_observation)

    if timestep.reward is not None:
      distance_reward = self._distance_to_reward(distance, timestep.reward)
      full_reward = {
          'environment': timestep.reward, 'distance': distance_reward}
      timestep = timestep._replace(reward=full_reward)
    return timestep

  def reset(self):
    timestep = self._environment.reset()
    return self._wrap_timestep(timestep)

  def step(self, action):
    timestep = self._environment.step(action)
    return self._wrap_timestep(timestep)


class RewardWrapper(wrappers.EnvironmentWrapper):
  """Define oracle reward."""

  def __init__(self, environment):
    self._environment = environment

  def step(self, action):
    timestep = self._environment.step(action)
    state_obs = self._environment._get_obs()  # pylint: disable=protected-access
    state = state_obs['state_observation']
    goal = state_obs['state_desired_goal']
    hand_xy = state[:2]
    puck_xy = state[3:]
    goal_xy = goal[3:]
    true_distance = (
        np.linalg.norm(hand_xy - puck_xy) + np.linalg.norm(puck_xy - goal_xy))
    max_distance = 1.0  # 2x the max distance from the arm to the goal.
    oracle_reward = -true_distance / max_distance
    timestep.reward['distance_pred'] = timestep.reward['distance']
    timestep.reward['distance'] = oracle_reward
    return timestep


class VisibleStateWrapper(wrappers.EnvironmentWrapper):
  """Expose only time step fields used for policy learning."""

  def __init__(self,
               environment,
               eval_mode = False):
    self._environment = environment
    self._eval_mode = eval_mode
    self._observation_spec = self._environment.observation_spec()['embeddings']
    self._reward_type = 'environment' if self._eval_mode else 'distance'
    self._reward_spec = self._environment.reward_spec()[self._reward_type]

  def observation_spec(self):
    return self._observation_spec

  def reward_spec(self):
    return self._reward_spec

  def _wrap_timestep(self, timestep):
    if timestep.reward is None:
      reward = timestep.reward
    else:
      reward = timestep.reward[self._reward_type]
    return timestep._replace(
        observation=timestep.observation['embeddings'],
        reward=reward)

  def reset(self):
    timestep = self._environment.reset()
    return self._wrap_timestep(timestep)

  def step(self, action):
    timestep = self._environment.step(action)
    return self._wrap_timestep(timestep)
