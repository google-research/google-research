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

"""Register MultiGrid environments with OpenAI gym."""

import gym
from gym.envs.registration import register as gym_register

env_list = []


def register(env_id, entry_point, reward_threshold=0.95):
  """Register a new environment with OpenAI gym based on id."""
  assert env_id.startswith("MultiGrid-")
  if env_id in env_list:
    del gym.envs.registry.env_specs[id]
  else:
    # Add the environment to the set
    env_list.append(id)

  # Register the environment with OpenAI gym
  gym_register(
      id=env_id, entry_point=entry_point, reward_threshold=reward_threshold)
