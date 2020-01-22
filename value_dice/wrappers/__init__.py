# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A collection of gym wrappers."""

import gym
from value_dice.wrappers.absorbing_wrapper import AbsorbingWrapper
from value_dice.wrappers.normalize_action_wrapper import check_and_normalize_box_actions
from value_dice.wrappers.normalize_action_wrapper import NormalizeBoxActionWrapper
from value_dice.wrappers.normalize_state_wrapper import NormalizeStateWrapper


def create_il_env(env_name, seed, shift, scale):
  """Create a gym environment for imitation learning.

  Args:
    env_name: an environment name.
    seed: a random seed.
    shift: a numpy vector to shift observations.
    scale: a numpy vector to scale observations.

  Returns:
    An initialized gym environment.
  """
  env = gym.make(env_name)
  env = check_and_normalize_box_actions(env)
  env.seed(seed)

  if shift is not None:
    env = NormalizeStateWrapper(env, shift=shift, scale=scale)

  return AbsorbingWrapper(env)
