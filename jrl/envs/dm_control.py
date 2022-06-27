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

"""DM Control env creators."""

import typing

from dm_control import suite
import dm_env
from dm_env import specs

from acme import wrappers
import numpy as np


class FlatObservationWrapper(wrappers.EnvironmentWrapper):
  def _convert_obs(self, obs):
    flat_obs = [v.flatten() for v in obs.values()]
    return np.concatenate(flat_obs, axis=-1)


  def reset(self):
    ts = self._environment.reset()
    return ts._replace(observation=self._convert_obs(ts.observation))


  def step(self, action):
    ts = self._environment.step(action)
    return ts._replace(observation=self._convert_obs(ts.observation))


  def observation_spec(self):
    original_obs_spec = self._environment.observation_spec()
    types = []
    sizes = []
    for k, v in original_obs_spec.items():
      types.append(v.dtype)
      sizes.append(np.prod(v.shape))

    assert all(x == types[0] for x in types), 'All types not the same!'
    total_size = sum(sizes)
    return specs.Array(
        shape=(total_size,),
        dtype=types[0],
        name='flat_obs_spec')


def create_dm_control_env(
    task_name,
):
  """Create the environment for the dm_control task.

  Args:
    task_name: Name of d4rl task.
  Returns:
    dm env.
  """
  split_name = task_name.split('__')
  domain_name, task_name = split_name[0], split_name[1]

  env = suite.load(domain_name=domain_name, task_name=task_name)
  env = FlatObservationWrapper(env)

  return env
