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

# python3
"""A simple agent-environment training loop."""

import itertools

from acme import core
from acme.utils import counting
from acme.utils import loggers

import dm_env


class EnvironmentLoop:
  """A simple RL environment loop which handles logging for mime use case."""

  def __init__(
      self,
      environment,
      actor,
      counter = None,
      logger = None,
      label = 'environment_loop',
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label)

  def run(self, num_episodes = None):
    """Perform the run loop."""

    iterator = range(num_episodes) if num_episodes else itertools.count()

    for i in iterator:
      # Reset the environment.
      timestep = self._environment.reset()
      episode_steps = 0
      episode_return = 0

      # Run an episode.
      while not timestep.last():
        # Generate an action from the agent's policy.
        action = self._actor.policy(timestep.observation)

        # Step the environment.
        next_timestep = self._environment.step(action, summary_step=i)

        # Tell the agent about what just happened.
        self._actor.observe(timestep.observation, action, next_timestep.reward,
                            next_timestep.discount)

        # Request that the agent updates itself.
        self._actor.update()

        # Book-keeping.
        timestep = next_timestep
        episode_steps += 1
        episode_return += timestep.reward

      # Tell the agent about the last observation and allow the agent to make
      # one final update.
      self._actor.observe_last(timestep.observation)
      self._actor.update()

      # Record counts.
      counts = self._counter.increment(episodes=1, steps=episode_steps)

      # Collect the results and combine with counts.
      result = {'episode_return': episode_return}
      result.update(counts)

      # Log the given results.
      self._logger.write(result)
