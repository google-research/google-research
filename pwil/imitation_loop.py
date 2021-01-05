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

"""Imitation loop for PWIL."""

import time

import acme
from acme.utils import counting
from acme.utils import loggers
import dm_env


class TrainEnvironmentLoop(acme.core.Worker):
  """PWIL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = TrainEnvironmentLoop(environment, actor, rewarder)
    loop.run(num_steps)

  The `Rewarder` overwrites the timestep from the environment to define
  a custom reward.

  The runner stores episode rewards and a series of statistics in the provided
  `Logger`.
  """

  def __init__(
      self,
      environment,
      actor,
      rewarder,
      counter=None,
      logger=None
  ):
    self._environment = environment
    self._actor = actor
    self._rewarder = rewarder
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger()

  def run(self, num_steps):
    """Perform the run loop.

    Args:
      num_steps: number of steps to run the loop for.
    """
    current_steps = 0
    while current_steps < num_steps:

      # Reset any counts and start the environment.
      start_time = time.time()
      self._rewarder.reset()

      episode_steps = 0
      episode_return = 0
      episode_imitation_return = 0
      timestep = self._environment.reset()

      self._actor.observe_first(timestep)

      # Run an episode.
      while not timestep.last():
        action = self._actor.select_action(timestep.observation)
        obs_act = {'observation': timestep.observation, 'action': action}
        imitation_reward = self._rewarder.compute_reward(obs_act)
        timestep = self._environment.step(action)
        imitation_timestep = dm_env.TimeStep(step_type=timestep.step_type,
                                             reward=imitation_reward,
                                             discount=timestep.discount,
                                             observation=timestep.observation)

        self._actor.observe(action, next_timestep=imitation_timestep)
        self._actor.update()

        # Book-keeping.
        episode_steps += 1
        episode_return += timestep.reward
        episode_imitation_return += imitation_reward

      # Collect the results and combine with counts.
      counts = self._counter.increment(episodes=1, steps=episode_steps)
      steps_per_second = episode_steps / (time.time() - start_time)
      result = {
          'episode_length': episode_steps,
          'episode_return': episode_return,
          'episode_return_imitation': episode_imitation_return,
          'steps_per_second': steps_per_second,
      }
      result.update(counts)

      self._logger.write(result)
      current_steps += episode_steps


class EvalEnvironmentLoop(acme.core.Worker):
  """PWIL evaluation environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = EvalEnvironmentLoop(environment, actor, rewarder)
    loop.run(num_episodes)

  The `Rewarder` overwrites the timestep from the environment to define
  a custom reward. The evaluation environment loop does not update the agent,
  and computes the wasserstein distance with expert demonstrations.

  The runner stores episode rewards and a series of statistics in the provided
  `Logger`.
  """

  def __init__(
      self,
      environment,
      actor,
      rewarder,
      counter=None,
      logger=None
  ):
    self._environment = environment
    self._actor = actor
    self._rewarder = rewarder
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger()

  def run(self, num_episodes):
    """Perform the run loop.

    Args:
      num_episodes: number of episodes to run the loop for.
    """
    for _ in range(num_episodes):
      # Reset any counts and start the environment.
      start_time = time.time()
      self._rewarder.reset()

      episode_steps = 0
      episode_return = 0
      episode_imitation_return = 0
      timestep = self._environment.reset()

      # Run an episode.
      trajectory = []
      while not timestep.last():
        action = self._actor.select_action(timestep.observation)
        obs_act = {'observation': timestep.observation, 'action': action}
        trajectory.append(obs_act)
        imitation_reward = self._rewarder.compute_reward(obs_act)

        timestep = self._environment.step(action)

        # Book-keeping.
        episode_steps += 1
        episode_return += timestep.reward
        episode_imitation_return += imitation_reward

      counts = self._counter.increment(episodes=1, steps=episode_steps)
      w2_dist = self._rewarder.compute_w2_dist_to_expert(trajectory)

      # Collect the results and combine with counts.
      steps_per_second = episode_steps / (time.time() - start_time)
      result = {
          'episode_length': episode_steps,
          'episode_return': episode_return,
          'episode_wasserstein_distance': w2_dist,
          'episode_return_imitation': episode_imitation_return,
          'steps_per_second': steps_per_second,
      }
      result.update(counts)

      self._logger.write(result)
