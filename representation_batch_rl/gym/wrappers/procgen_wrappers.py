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

# pytype: disable=wrong-arg-types
"""Wrappers for ProcGen, modified for current Acme + TFAgents setup."""

from baselines.common import vec_env
import gym
import numpy as np
from procgen import env as procgen_env
import tensorflow as tf

from tf_agents import environments as tf_agents_environments
from tf_agents.environments import gym_wrapper as tf_agents_gym_wrapper
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils

ENV_NAMES = procgen_env.ENV_NAMES


class ObsToFloat(gym.ObservationWrapper):
  """Converts observations to floating point values, and changes observation space to dtype float.

  Also can scale values.
  """

  def __init__(self, env, divisor=1.0):
    self.env = env
    self.divisor = divisor
    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space
    self.metadata = self.env.metadata
    self.observation_space.dtype = np.float32

  def observation(self, observation):
    if isinstance(observation, dict):
      # dict space
      scalar_obs = {}
      for k, v in observation.items():
        scalar_obs[k] = v.astype(np.float32)
      return scalar_obs / self.divisor
    else:
      return observation.astype(np.float32) / self.divisor


def vector_wrap_environment(venv,
                            normalize_obs=False,
                            normalize_ret=False,
                            monitor=True):
  """Converts observation dicts with key 'rgb' into just array.

  Can also log files and normalize observations.

  Args:
    venv: Original venv.
    normalize_obs: Normalizes the observation using (x-mean)/std.
    normalize_ret: Normalizes the reward using x/std. Should generally NOT be
      used.
    monitor: Monitors metadata (usually only used inside OpenAI PPO).

  Returns:
    Wrapped venv.

  """
  venv = vec_env.VecExtractDictObs(venv, "rgb")

  if monitor:
    venv = vec_env.VecMonitor(
        venv=venv,
        filename=None,
        keep_buf=100,
    )
  if normalize_obs or normalize_ret:
    # TODO(xingyousong): Check gamma arg if using ret=True.
    # Using default arg clipvalues, but showing them explicitly here.
    venv = vec_env.VecNormalize(
        venv=venv, ob=normalize_obs, ret=normalize_ret, clipob=10., cliprew=10.)
  return venv


class TFAgentsParallelProcGenEnv(tf_agents_environments.PyEnvironment):
  """Wrapped ProcGen environment for TF_agents algorithms."""

  def __init__(self,
               num_envs,
               discount=1.0,
               spec_dtype_map=None,
               simplify_box_bounds=True,
               flatten=False,
               normalize_rewards=False,
               **procgen_kwargs):
    """Uses Native C++ Environment Vectorization, which reduces RAM usage.

    Except the num_envs and **procgen_kwargs, all of the other __init__
    args come from the original TF-Agents GymWrapper and
    ParallelPyEnvironment wrappers.

    Args:
      num_envs: List of callables that create environments.
      discount: Discount rewards automatically (also done in algorithms).
      spec_dtype_map: A dict from spaces to dtypes to use as the default dtype.
      simplify_box_bounds: Whether to replace bounds of Box space that are
        arrays with identical values with one number and rely on broadcasting.
      flatten: Boolean, whether to use flatten action and time_steps during
        communication to reduce overhead.
      normalize_rewards: Use VecNormalize to normalize rewards. Should be used
        for collect env only.
      **procgen_kwargs: Keyword arguments passed into the native ProcGen env.
    """
    super(TFAgentsParallelProcGenEnv, self).__init__()

    self._num_envs = num_envs

    parallel_env = procgen_env.ProcgenEnv(num_envs=num_envs, **procgen_kwargs)
    parallel_env = vector_wrap_environment(
        parallel_env,
        normalize_obs=False,
        normalize_ret=normalize_rewards,
        monitor=False)
    parallel_env = ObsToFloat(parallel_env, divisor=255.0)

    self._parallel_env = parallel_env

    self._observation_spec = tf_agents_gym_wrapper.spec_from_gym_space(
        self._parallel_env.observation_space, spec_dtype_map,
        simplify_box_bounds, "observation")

    self._action_spec = tf_agents_gym_wrapper.spec_from_gym_space(
        self._parallel_env.action_space, spec_dtype_map, simplify_box_bounds,
        "action")
    self._time_step_spec = ts.time_step_spec(self._observation_spec,
                                             self.reward_spec())

    self._flatten = flatten
    self._discount = discount

    self._dones = [True] * num_envs  # Contains "done"s for all subenvs.

  @property
  def parallel_env(self):
    return self._parallel_env

  @property
  def batched(self):
    return True

  @property
  def batch_size(self):
    return self._num_envs

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def time_step_spec(self):
    return self._time_step_spec

  def close(self):
    self._parallel_env.close()

  def _step(self, actions):
    if tf.is_tensor(actions):
      actions = actions.numpy()
    observations, rewards, temp_dones, self._infos = self._parallel_env.step(
        actions)
    timesteps = []

    for i, done in enumerate(temp_dones):
      if done:
        time_step = ts.termination(observations[i], rewards[i])
      else:
        if self._dones[i]:
          time_step = ts.restart(observations[i])
        else:
          time_step = ts.transition(observations[i], rewards[i], self._discount)
      timesteps.append(time_step)

    self._dones = temp_dones

    return self._stack_time_steps(timesteps)

  def _reset(self):
    observations = self._parallel_env.reset()
    self._dones = [False] * self._num_envs

    timesteps = ts.restart(observations, batch_size=self._num_envs)
    return timesteps

  def _stack_time_steps(self, time_steps):
    """Given a list of TimeStep, combine to one with a batch dimension."""
    if self._flatten:
      return nest_utils.fast_map_structure_flatten(
          lambda *arrays: np.stack(arrays), self._time_step_spec, *time_steps)
    else:
      return nest_utils.fast_map_structure(lambda *arrays: np.stack(arrays),
                                           *time_steps)
