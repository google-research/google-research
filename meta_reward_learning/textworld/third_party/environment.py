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

# Lint as: python3
"""Environment Class.

Code adapted from the github repository:
https://github.com/spro/practical-pytorch/blob/master/reinforce-gridworld/
"""

from .seed_helpers import np_random
import numpy as np


MIN_PLANT_VALUE = -1.0
MAX_PLANT_VALUE = 0.0
GOAL_VALUE = 1
EDGE_VALUE = -1
VISIBLE_RADIUS = 1

START_HEALTH = 1
STEP_VALUE = -1 / 15  # The Agent Dies in 15 Steps
ACTION_MAP = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}


def path_to_str(path):
  return ''.join([ACTION_MAP[a] for a in path])


class Grid(object):
  # pylint: disable=line-too-long
  """The Grid class keeps track of the grid world: a 2d array of empty squares, plants, and the goal.

  Poisonous plants are randomly placed with value -1  and if the agent lands
  on one, that value is added to the agent's health. The agent's goal is to
  reach the goal square, placed in one of the corners. As the agent moves around
  it gradually loses health so it has to move with purpose.

  The agent can see a surrounding area `VISIBLE_RADIUS` squares out from its
  position, so the edges of the grid are padded by that much with negative
  values.If the agent "falls off the edge" it dies instantly.
  """

  # pylint: enable=line-too-long

  def __init__(self, goal_id=None, grid_size=8, n_plants=15, seed=None):
    self.rand_seed = self.seed(seed)
    self.grid_size = grid_size
    self.n_plants = n_plants
    if goal_id is None:
      self.goal_id = self.np_random.randint(0, 4)
    else:
      if goal_id < 0 or goal_id > 3:
        raise ValueError('{} not a valid goal id.'.format(goal_id))
      self.goal_id = goal_id

  def reset(self):
    """Resets the environment to the start state."""
    padded_size = self.grid_size + 2 * VISIBLE_RADIUS
    # Padding for edges
    self.grid = np.zeros((padded_size, padded_size), dtype=np.float32)

    # Edges
    self.grid[0:VISIBLE_RADIUS, :] = EDGE_VALUE
    self.grid[-VISIBLE_RADIUS:, :] = EDGE_VALUE
    self.grid[:, 0:VISIBLE_RADIUS] = EDGE_VALUE
    self.grid[:, -VISIBLE_RADIUS:] = EDGE_VALUE

    # Goal in one of the corners
    s1 = VISIBLE_RADIUS
    e1 = self.grid_size + VISIBLE_RADIUS - 1
    gps = [(e1, e1), (s1, e1), (e1, s1), (s1, s1)]
    # gp = gps[self.np_random.randint(0, len(gps)-1)]
    gp = gps[self.goal_id]
    self.goal_pos = gp
    self.grid[gp] = GOAL_VALUE

    mid_point = (self.grid_size // 2, self.grid_size // 2)
    # Add the goal and agent pos so that a plant is not placed there
    placed_plants = set([self.goal_pos, mid_point])
    # Randomly placed plants at unique positions
    for _ in range(self.n_plants):
      while True:
        ry = self.np_random.randint(0, self.grid_size - 1) + VISIBLE_RADIUS
        rx = self.np_random.randint(0, self.grid_size - 1) + VISIBLE_RADIUS
        plant_pos = (ry, rx)
        if plant_pos not in placed_plants:
          placed_plants.add(plant_pos)
          break
      self.grid[plant_pos] = MIN_PLANT_VALUE

  def visible(self, pos):
    y, x = pos
    return self.grid[y - VISIBLE_RADIUS:y + VISIBLE_RADIUS + 1, x -
                     VISIBLE_RADIUS:x + VISIBLE_RADIUS + 1]

  def seed(self, seed=None):
    self.np_random, seed = np_random(seed)
    self.rand_state = self.np_random.get_state()
    return seed


class Environment(object):
  # pylint: disable=line-too-long
  """The Environment encapsulates the Grid and Agent.

  The environment handles the bulk of the logic of assigning rewards when the
  agent acts. If an agent lands on a plant or goal or edge, its health is
  updated accordingly. Plants are removed from the grid (set to 0) when "eaten"
  by the agent. Every time step there is also a slight negative health penalty
  so that the agent must keep finding plants or reach the goal to survive.

  The Environment's main function is step(action) -> (state, reward, done),
  which updates the world state with a chosen action and returns the resulting
  state,
  and also returns a reward and whether the episode is done. The state it
  returns is what the agent will use to make its action predictions, which in
  this case
  is the visible grid area (flattened into one dimension) and the current agent
  health (to give it some "self awareness").

  The episode is considered done if won or lost - won if the agent reaches the
  goal (agent.health >= GOAL_VALUE) and lost if the agent dies from falling off
  the edge, eating too many poisonous plants, or getting too hungry
  (agent.health
  <= 0).

  In this experiment the environment only returns a single reward at the end of
  the episode (to make it more challenging). Values from plants and the step
  penalty are implicit - they might cause the agent to live longer or die
  sooner,
  but they aren't included in the final reward.

  The Environment also keeps track of the grid and agent states for each step of
  an episode, for visualization.
  """

  # pylint: enable=line-too-long

  def __init__(self, name=None, **kwargs):
    self.grid = Grid(**kwargs)
    self.agent = HealthAgent()
    self._render = False
    self._reset_grid = True
    self._name = name

  def reset(self, render=False):
    """Start a new episode by resetting grid and agent."""
    if self._reset_grid:
      self.grid.reset()
      self._reset_grid = False
    self._render = render
    c = self.grid.grid_size // 2
    self.agent.reset(pos=(c, c))

    self.t = 0
    self.history = []
    if self._render:
      self.record_step()
    return

  @property
  def num_actions(self):
    return self.agent.num_actions

  @property
  def dummy_state(self):
    return np.ones(1, dtype=np.float32)

  @property
  def name(self):
    return self._name

  @property
  def rand_seed(self):
    return self.grid.rand_seed

  @property
  def rand_state(self):
    return self.grid.rand_state

  @property
  def np_random(self):
    return self.grid.np_random

  def record_step(self):
    """Add the current state to history for display later."""
    grid = np.array(self.grid.grid, dtype=np.float32)
    # Agent marker faded by health
    grid[self.agent.pos] = self.agent.health * 0.5
    visible = np.array(self.grid.visible(self.agent.pos), dtype=np.float32)
    self.history.append((grid, visible, self.agent.health))

  @property
  def visible_state(self):
    """Return visible area surrounding the agent and current agent health."""
    visible = self.grid.visible(self.agent.pos)
    y, x = self.agent.pos
    yg, xg = self.grid.goal_pos
    pos_arr = [y, x, yg, xg]
    for i, pos in enumerate(pos_arr):
      pos_arr[i] = (pos - VISIBLE_RADIUS) / self.grid.grid_size
    extras = np.array([self.agent.health] + pos_arr, dtype=np.float32)
    return np.concatenate((visible.flatten(), extras), 0)

  def step(self, action):
    """Update state (grid and agent) based on an action."""
    self.agent.act(action)

    # Get reward from where agent landed, add to agent health
    value = self.grid.grid[self.agent.pos]
    # if self._render:
    #  self.grid.grid[self.agent.pos] = 0
    self.agent.health += value

    # Check if agent won (reached the goal) or lost (health reached 0)
    won = self.agent.pos == self.grid.goal_pos
    lost = self.agent.health <= 0
    done = won or lost

    # Rewards at end of episode
    if won:
      reward = 1
    else:
      reward = 0  # Reward will only come at the end

    # Save in history
    if self._render:
      self.record_step()

    return reward, done

  def seed(self, seed=None):
    self.grid.seed(seed)


class GridAgent(object):
  """Sprite that can move in the GridWorld."""

  def __init__(self, pos=None):
    self.pos = pos

  def act(self, action):
    """Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT."""
    y, x = self.pos
    if action == 0:
      y -= 1
    elif action == 1:
      x += 1
    elif action == 2:
      y += 1
    elif action == 3:
      x -= 1
    self.pos = (y, x)

  def reset(self, pos):
    self.pos = pos

  @property
  def num_actions(self):
    return 4


class HealthAgent(GridAgent):
  """The HealthAgent has a current position and a health.

  All this class does is update the position based on an action (up, right, down
  or left) and decrement a small `STEP_VALUE` at every time step, so that it
  eventually starves if it doesn't reach the goal.
  """

  def reset(self, pos):
    super(HealthAgent, self).reset(pos)
    self.health = START_HEALTH

  def act(self, action):
    super(HealthAgent, self).act(action)
    self.health += STEP_VALUE  # Gradually getting hungrier


class TextEnvironment(Environment):
  """TextEnvironment Class."""

  def __init__(self, **kwargs):
    super(TextEnvironment, self).__init__(**kwargs)
    # Text context which corresponds to the goal in the environment
    self._context = None

  @property
  def context(self):
    return self._context

  @context.setter
  def context(self, context):
    self._context = np.array(context, dtype=np.int8)

  @property
  def text_context(self):
    return path_to_str(self._context)
