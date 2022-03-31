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

"""D4RL env creators."""

import typing

from acme import wrappers
import d4rl
import dm_env
import gym


def create_d4rl_env(
    task_name,
):
  """Create the environment for the d4rl task.

  Args:
    task_name: Name of d4rl task.
  Returns:
    dm env.
  """
  env = gym.make(task_name)
  env = wrappers.GymWrapper(env)

  return env
