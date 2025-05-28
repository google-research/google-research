# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Class to make it easier to specify MiniGrid environments.

This class will read a .grid file that specifies the grid environment and
dynamics. It offers the option of specifying MDPs with four directional actions
instead of rotate/fwd.

Gym-MiniGrid is built on the assumption that the agent is always facing a
specific direction. Thus, the design decision was to always rotate the agent
towards the direction it wants to go and then try to go. As such, the arrow is
always changing its direction. It doesn't matter in the tabular case. In the
setting in which one looks at the raw pixels, the agent direction encodes its
previous action. This might be something to keep in mind.

Moreover, we decided to not allow other actions such as pick up object or
toggle.  This setting should be the standard one. If one wants to add objects,
we might need to do something more here, like automatically activate 'pick up
object' if the agent ends up at the same tile as the object.

The constructor parameter `stochasticity` controls the amount of probability
mass distributed to transitioning to any of the other neighbouring states
(including staying in the current state).
"""

import enum
import random

from absl import logging
import gin
import gym
from gym_minigrid import minigrid
from gym_minigrid import register
import numpy as np


GIN_FILES_PREFIX = 'minigrid_basics/envs'
_ENTRY_POINT = 'minigrid_basics.envs.mon_minigrid:MonMiniGridEnv'



ASCII_TO_OBJECT = {
    '*': minigrid.Wall,
    's': None,
    'g': minigrid.Goal
}


def parse_ascii_grid(ascii_grid):
  raw_grid = []
  for l in ascii_grid.split('\n'):
    if not l:
      continue
    raw_grid.append(list(l))
  return np.array(raw_grid)


@gin.configurable
def register_environment(env_id):
  """This needs to be called before `gym.make` to register the environment."""
  register.register(
      id=env_id,
      entry_point=_ENTRY_POINT
  )
  return env_id


@gin.configurable
class MonMiniGridEnv(minigrid.MiniGridEnv):
  """Overrides MiniGridEnv to get 4 directional actions instead of rotate/fwd.
  """
  # Overriding action definitions to allow for only four directional actions

  class DirectionalActions(enum.IntEnum):
    # Right, down, left, up.
    right = 0
    down = 1
    left = 2
    up = 3

  def __init__(self, ascii_grid, directional=False,
               agent_pos=None, goal_pos=None,
               mission='Reach the goal', custom_rewards=None,
               max_steps=100, see_through_walls=True, seed=1337,
               agent_view_size=7, stochasticity=0.0, episodic=True):
    """Constructor for MonMinigrid.

    The specifics of the environment are specified through gin files, and
    `register_environment` should be called before `gym.make` to ensure the
    appropriate environment(s) are registered.

    Args:
      ascii_grid: str, ASCII specification of the GridWorld layout.
      directional: bool, whether we use 4 directional or traditional actions.
      agent_pos: pair of ints or None, user-specified start position, if any.
      goal_pos: pair of ints or None, user-specified goal position, if any.
      mission: str, mission for this task.
      custom_rewards: list or None, user can specify a list of triples
        (x, y, r), where `(x, y)` is the coordinate and `r` is the reward.
        If None, will assume all goal states yield a reward of 1.
      max_steps: int, maximum steps per episode.
      see_through_walls: bool, whether agent can see through walls.
      seed: int, seed used for randomization.
      agent_view_size: int, range of agent visibility.
      stochasticity: float, stochasticity in the environment.
      episodic: bool, whether the task is episodic.
    """
    # The constructor parameters defined here can be set via gin_bindings. See
    # the python files in the examples directory for samples.
    self._ascii_grid = ascii_grid
    self._directional = directional
    self._agent_default_pos = agent_pos
    self._goal_default_pos = goal_pos
    self._mission = mission
    self._custom_rewards = custom_rewards
    self._build_raw_grid()
    super().__init__(width=self.width, height=self.height,
                     max_steps=max_steps, see_through_walls=see_through_walls,
                     seed=seed, agent_view_size=agent_view_size)
    if self._directional:
      # Action enumeration for this environment
      self.actions = MonMiniGridEnv.DirectionalActions
      self.action_space = gym.spaces.Discrete(len(self.actions))
    self.stochasticity = stochasticity
    self.episodic = episodic  # If False, reaching the goal doesn't end the ep.

  def _build_raw_grid(self):
    """ASCII specification of grid layout, must be specified in .gin file."""
    self._raw_grid = parse_ascii_grid(self._ascii_grid)
    self.width = self._raw_grid.shape[0]
    self.height = self._raw_grid.shape[1]
    # If a start position has been specified, add it to grid.
    if self._agent_default_pos is not None:
      assert len(self._agent_default_pos) == 2
      x, y = self._agent_default_pos
      self._raw_grid[x, y] = 's'
    # If a goal position has been specified, add it to the grid.
    if self._goal_default_pos is not None:
      assert len(self._goal_default_pos) == 2
      x, y = self._goal_default_pos
      self._raw_grid[x, y] = 'g'

  def _gen_grid(self, width, height):
    self.grid = minigrid.Grid(self.width, self.height)
    for x in range(self.width):
      for y in range(self.height):
        if self._raw_grid[x, y] != ' ':
          if self._raw_grid[x, y] == 's':
            self.agent_pos = (x, y)
            self.agent_dir = self._rand_int(0, 4)
          obj = ASCII_TO_OBJECT[self._raw_grid[x, y]]
          obj = obj if obj is None else obj()
          self.grid.set(x, y, obj)
    # If a start position has not been specified, place agent randomly.
    if 's' not in self._raw_grid:
      self.place_agent()
    # If no goal has been specified, place goal randomly.
    if 'g' not in self._raw_grid:
      self.place_obj(minigrid.Goal())
    self.mission = self._mission

  def step(self, action):
    if not self._directional:
      return super().step(action)

    # The action space is simplified here. The agent can't pick up an object,
    # it can't drop it, it can't toggle nor call it a day (done).
    self.step_count += 1

    reward = 0
    done = False

    assert action < 4, 'unknown action'
    self.agent_dir = int(action)

    # If we have stochasticity, we may alter the action performed.
    if self.stochasticity > 0.:
      p = random.random()
      if p < self.stochasticity:
        # We choose an action randomly. If the result matches the chosen action,
        # the agent "slips" and stays in place.
        random_action = random.randint(0, 3)
        if self.agent_dir == random_action:
          return self.gen_obs(), reward, done, {}
        # Switch to randomly sampled action.
        self.agent_dir = random_action

    # Get the position in front of the agent
    fwd_pos = self.front_pos
    # Get the contents of the cell in front of the agent
    fwd_cell = self.grid.get(*fwd_pos)
    if fwd_cell is None or fwd_cell.can_overlap():
      self.agent_pos = fwd_pos
    if fwd_cell is not None and fwd_cell.type == 'goal':
      if self.episodic:
        done = True
      reward = self._reward()
    if fwd_cell is not None and fwd_cell.type == 'lava':
      done = True

    if self.step_count >= self.max_steps:
      done = True

    obs = self.gen_obs()

    return obs, reward, done, {}

  def _reward(self):
    if self._custom_rewards is None:
      # Deterministic reward upon arrival
      return 1
    for custom_reward in self._custom_rewards:
      if (self.agent_pos[0] == custom_reward[0] and
          self.agent_pos[1] == custom_reward[1]):
        if self._raw_grid[custom_reward[0], custom_reward[1]] != 'g':
          logging.warning('Non-goal state (%d, %d) has reward.',
                          custom_reward[0], custom_reward[1])
        return custom_reward[2]
    # Default to a reward of 1.
    return 1
