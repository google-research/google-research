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

"""Wrapper providing a multiagent adapter for Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tf_agents.environments import gym_wrapper
from tf_agents.specs import array_spec
from tf_agents.specs.tensor_spec import BoundedTensorSpec
from tf_agents.trajectories import time_step as ts_lib
from tf_agents.utils import nest_utils


class MultiagentGymWrapper(gym_wrapper.GymWrapper):
  """Wrapper implementing PyEnvironment interface for multiagent gym envs.

  Reward spec is generated based on the number of agents.

  Action and observation specs are automatically generated from the action and
  observation spaces of the underlying environment. The expectation is that the
  first dimension of the environment specs will be the number of agents.
  """

  def __init__(self,
               gym_env,
               n_agents,
               discount=1.0,
               spec_dtype_map=None,
               match_obs_space_dtype=True,
               auto_reset=True,
               simplify_box_bounds=True):
    self.n_agents = n_agents

    super(MultiagentGymWrapper, self).__init__(
        gym_env, discount, spec_dtype_map, match_obs_space_dtype, auto_reset,
        simplify_box_bounds)

    # Create a single-agent version of the action spec and then tile it to
    # comply with tf-agents spec requirements.
    single_action_spec = BoundedTensorSpec(
        shape=(), dtype=self._action_spec.dtype, name=self._action_spec.name,
        minimum=self._action_spec.minimum, maximum=self._action_spec.maximum)
    self._action_spec = (single_action_spec,) * n_agents

  def reward_spec(self):
    """Defines a vector reward based on the number of agents.

    Returns:
      An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
    """
    if self._gym_env.minigrid_mode:
      return array_spec.ArraySpec(shape=(), dtype=np.float32, name='reward')
    else:
      return array_spec.ArraySpec(shape=(self.n_agents,), dtype=np.float32,
                                  name='reward')

  def _reset(self):
    observation = self._gym_env.reset()
    self._info = None
    self._done = False

    if self._match_obs_space_dtype:
      observation = self._to_obs_space_dtype(observation)
    reset_step = ts_lib.restart(observation, reward_spec=self.reward_spec())
    return reset_step

  def _step(self, action):
    # Automatically reset the environments on step if they need to be reset.
    if self._handle_auto_reset and self._done:
      return self.reset()

    # Some environments (e.g. FrozenLake) use the action as a key to the
    # transition probability so it has to be hashable. In the case of discrete
    # actions we have a numpy scalar (e.g array(2)) which is not hashable
    # in this case, we simply pull out the scalar value which will be hashable.
    try:
      action = action.item() if self._action_is_discrete else action
    except AttributeError:
      action = action[0]  # Remove ListWrapper for single-agent compatibility

    observation, reward, self._done, self._info = self._gym_env.step(action)

    if self._match_obs_space_dtype:
      observation = self._to_obs_space_dtype(observation)

    reward = np.asarray(reward, dtype=self.reward_spec().dtype)
    outer_dims = nest_utils.get_outer_array_shape(reward, self.reward_spec())

    if self._done:
      return ts_lib.termination(observation, reward, outer_dims=outer_dims)
    else:
      return ts_lib.transition(observation, reward, self._discount,
                               outer_dims=outer_dims)
