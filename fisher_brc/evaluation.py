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

"""Policy evaluation."""
from typing import Tuple


def evaluate(env, policy, num_episodes = 10):
  """Evaluates the policy.

  Args:
    env: Environment to evaluate the policy on.
    policy: Policy to evaluate.
    num_episodes: A number of episodes to average the policy on.

  Returns:
    Averaged reward and a total number of steps.
  """
  total_timesteps = 0
  total_returns = 0.0

  for _ in range(num_episodes):
    episode_return = 0
    timestep = env.reset()

    while not timestep.is_last():
      action = policy.act(timestep.observation)
      timestep = env.step(action)

      total_returns += timestep.reward[0]
      episode_return += timestep.reward[0]
      total_timesteps += 1

  return total_returns / num_episodes, total_timesteps / num_episodes
