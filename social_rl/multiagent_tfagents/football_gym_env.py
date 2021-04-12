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

"""Multiagent Google Research Football env for gym.
"""
from gfootball import env as football_env
import gin
import gym
from social_rl.multiagent_tfagents import multiagent_gym_suite


class DictWrapper(gym.ObservationWrapper):
  """Wrapper to convert observations into dicts, and actions from dicts."""

  def __init__(self, env):
    super().__init__(env)
    self.observation_space = gym.spaces.Dict(
        {'image': self.env.observation_space})

  def observation(self, observation):
    return {'image': observation}

  def step(self, action):
    action = [i.item() for i in action]
    observation, reward, done, info = self.env.step(action)
    return self.observation(observation), reward, done, info


@gin.configurable
def load(env_name='academy_3_vs_1_with_keeper',
         channel_dimensions=(48, 36),
         gym_env_wrappers=(),
         env_wrappers=(),
         **unused_gym_kwargs):
  """Loads the selected environment and wraps it with the specified wrappers.

  Note that by default a TimeLimit wrapper is used to limit episode lengths
  to the default benchmarks defined by the registered environments.

  Args:
    env_name: Name for the environment to load.
    channel_dimensions: Dimensions for observations.
    gym_env_wrappers: Iterable with references to wrapper classes to use
      directly on the gym environment.
    env_wrappers: Iterable with references to wrapper classes to use on the
      gym_wrapped environment.
    **unused_gym_kwargs: For compatibility with multiagent_gym_env.

  Returns:
    A PyEnvironment instance.
  """
  if env_name in [
      'academy_run_pass_and_shoot_with_keeper', 'academy_3_vs_1_with_keeper'
  ]:
    n_agents = 3
  else:
    raise ValueError('Env not supported')
  env = football_env.create_environment(
      env_name=env_name,
      stacked=False,
      logdir='/tmp/football',
      write_goal_dumps=False,
      write_full_episode_dumps=False,
      render=False,
      representation='extracted',
      channel_dimensions=channel_dimensions,
      number_of_left_players_agent_controls=n_agents)
  env.minigrid_mode = False
  env = multiagent_gym_suite.wrap_env(
      env,
      n_agents=n_agents,
      gym_env_wrappers=[DictWrapper] + list(gym_env_wrappers),
      env_wrappers=env_wrappers)
  return env
