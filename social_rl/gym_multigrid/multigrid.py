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

"""Implements the multi-agent version of the Grid and MultiGridEnv classes.

Note that at each step, the environment expects an array of actions equal to the
number of agents with which the class was initialized. Similarly, it will return
an array of observations, and an array of rewards.

In the competitive version, as soon as one agent finds the goal, the game is
over.

In the non-competitive case, all episodes have a fixed length based on the
maximum number of steps. To avoid issues with some agents finishing early and
therefore requiring support for non-scalar step types, if an agent finishes
before the step deadline it will be respawned in a new location. To make the
single-agent case comparable to this design, it should also run for a fixed
number of steps and allow the agent to find the goal as many times as possible
within this step budget.

Unlike Minigrid, Multigrid does not include the string text of the 'mission'
with each observation.
"""
import math

import gym
import gym_minigrid.minigrid as minigrid
import gym_minigrid.rendering as rendering
import numpy as np

# Map of color names to RGB values
AGENT_COLOURS = [
    np.array([60, 182, 234]),  # Blue
    np.array([229, 52, 52]),  # Red
    np.array([144, 32, 249]),  # Purple
    np.array([69, 196, 60]),  # Green
    np.array([252, 227, 35]),  # Yellow
]


class WorldObj(minigrid.WorldObj):
  """Override MiniGrid base class to deal with Agent objects."""

  def __init__(self, obj_type, color=None):
    assert obj_type in minigrid.OBJECT_TO_IDX, obj_type
    self.type = obj_type
    if color:
      assert color in minigrid.COLOR_TO_IDX, color
      self.color = color

    self.contains = None

    # Initial position of the object
    self.init_pos = None

    # Current position of the object
    self.cur_pos = None

  @staticmethod
  def decode(type_idx, color_idx, state):
    """Create an object from a 3-tuple state description."""

    obj_type = minigrid.IDX_TO_OBJECT[type_idx]
    if obj_type != 'agent':
      color = minigrid.IDX_TO_COLOR[color_idx]

    if obj_type == 'empty' or obj_type == 'unseen':
      return None

    if obj_type == 'wall':
      v = minigrid.Wall(color)
    elif obj_type == 'floor':
      v = minigrid.Floor(color)
    elif obj_type == 'ball':
      v = minigrid.Ball(color)
    elif obj_type == 'key':
      v = minigrid.Key(color)
    elif obj_type == 'box':
      v = minigrid.Box(color)
    elif obj_type == 'door':
      # State, 0: open, 1: closed, 2: locked
      is_open = state == 0
      is_locked = state == 2
      v = Door(color, is_open, is_locked)
    elif obj_type == 'goal':
      v = minigrid.Goal()
    elif obj_type == 'lava':
      v = minigrid.Lava()
    elif obj_type == 'agent':
      v = Agent(color_idx, state)
    else:
      assert False, "unknown object type in decode '%s'" % obj_type

    return v


class Door(minigrid.Door):
  """Extends minigrid Door class to multiple agents possibly carrying keys."""

  def toggle(self, env, pos, carrying):
    # If the player has the right key to open the door
    if self.is_locked:
      if isinstance(carrying, minigrid.Key) and carrying.color == self.color:
        self.is_locked = False
        self.is_open = True
        return True
      return False

    self.is_open = not self.is_open
    return True


class Agent(WorldObj):
  """Class to represent other agents existing in the world."""

  def __init__(self, agent_id, state):
    super(Agent, self).__init__('agent')
    self.agent_id = agent_id
    self.dir = state

  def can_contain(self):
    """Can this contain another object?"""
    return True

  def encode(self):
    """Encode the a description of this object as a 3-tuple of integers."""
    return (minigrid.OBJECT_TO_IDX[self.type], self.agent_id, self.dir)

  def render(self, img):
    tri_fn = rendering.point_in_triangle(
        (0.12, 0.19),
        (0.87, 0.50),
        (0.12, 0.81),
    )

    # Rotate the agent based on its direction
    tri_fn = rendering.rotate_fn(
        tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
    color = AGENT_COLOURS[self.agent_id]
    rendering.fill_coords(img, tri_fn, color)


class Grid(minigrid.Grid):
  """Extends Grid class, overrides some functions to cope with multi-agent case."""

  @classmethod
  def render_tile(cls,
                  obj,
                  highlight=None,
                  tile_size=minigrid.TILE_PIXELS,
                  subdivs=3,
                  cell_type=None):
    """Render a tile and cache the result."""
    # Hash map lookup key for the cache
    if isinstance(highlight, list):
      key = (tuple(highlight), tile_size)
    else:
      key = (highlight, tile_size)
    key = obj.encode() + key if obj else key

    if key in cls.tile_cache:
      return cls.tile_cache[key]

    img = np.zeros(
        shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    # Draw the grid lines (top and left edges)
    rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1),
                          (100, 100, 100))
    rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031),
                          (100, 100, 100))

    if obj is not None and obj.type != 'agent':
      obj.render(img)

    # Highlight the cell if needed (do not highlight walls)
    if highlight and not (cell_type is not None and cell_type == 'wall'):
      if isinstance(highlight, list):
        for a, agent_highlight in enumerate(highlight):
          if agent_highlight:
            rendering.highlight_img(img, color=AGENT_COLOURS[a])
      else:
        # Default highlighting for agent's partially observed views
        rendering.highlight_img(img)

    # Render agents after highlight to avoid highlighting agent triangle (the
    # combination of colours makes it difficult to ID agent)
    if obj is not None and obj.type == 'agent':
      obj.render(img)

    # Downsample the image to perform supersampling/anti-aliasing
    img = rendering.downsample(img, subdivs)

    # Cache the rendered tile
    cls.tile_cache[key] = img

    return img

  def render(self,
             tile_size,
             highlight_mask=None):
    """Render this grid at a given scale.

    Args:
      tile_size: Tile size in pixels.
      highlight_mask: An array of binary masks, showing which part of the grid
        should be highlighted for each agent. Can also be used in partial
        observation for single agent, which must be handled differently.

    Returns:
      An image of the rendered Grid.
    """
    if highlight_mask is None:
      highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

    # Compute the total grid size
    width_px = self.width * tile_size
    height_px = self.height * tile_size

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    # Render the grid
    for y in range(0, self.height):
      for x in range(0, self.width):
        cell = self.get(x, y)
        cell_type = cell.type if cell else None

        if isinstance(highlight_mask, list):
          # Figure out highlighting for each agent
          n_agents = len(highlight_mask)
          highlights = [highlight_mask[a][x, y] for a in range(n_agents)]
        else:
          highlights = highlight_mask[x, y]

        tile_img = Grid.render_tile(
            cell,
            highlight=highlights,
            tile_size=tile_size,
            cell_type=cell_type,
        )

        ymin = y * tile_size
        ymax = (y + 1) * tile_size
        xmin = x * tile_size
        xmax = (x + 1) * tile_size
        img[ymin:ymax, xmin:xmax, :] = tile_img

    return img

  @staticmethod
  def decode(array):
    """Decode an array grid encoding back into a grid."""

    width, height, channels = array.shape
    assert channels == 3

    vis_mask = np.ones(shape=(width, height), dtype=np.bool)

    grid = Grid(width, height)
    for i in range(width):
      for j in range(height):
        type_idx, color_idx, state = array[i, j]
        v = WorldObj.decode(type_idx, color_idx, state)
        grid.set(i, j, v)
        vis_mask[i, j] = (type_idx != minigrid.OBJECT_TO_IDX['unseen'])

    return grid, vis_mask

  def rotate_left(self):
    """Rotate the grid counter-clockwise, including agents within it."""
    grid = Grid(self.height, self.width)

    for i in range(self.width):
      for j in range(self.height):
        v = self.get(i, j)

        # Directions are relative to the agent so must be modified
        if v is not None and v.type == 'agent':
          # Make a new agent so original grid isn't modified
          v = Agent(v.agent_id, v.dir)
          v.dir -= 1
          if v.dir < 0:
            v.dir += 4

        grid.set(j, grid.height - 1 - i, v)

    return grid

  def slice(self, top_x, top_y, width, height, agent_pos=None):
    """Get a subset of the grid for agents' partial observations."""

    grid = Grid(width, height)

    for j in range(0, height):
      for i in range(0, width):
        x = top_x + i
        y = top_y + j

        if x >= 0 and x < self.width and \
           y >= 0 and y < self.height:
          v = self.get(x, y)
        else:
          v = minigrid.Wall()

        grid.set(i, j, v)

    return grid


class MultiGridEnv(minigrid.MiniGridEnv):
  """2D grid world game environment with multi-agent support."""

  def __init__(
      self,
      grid_size=None,
      width=None,
      height=None,
      max_steps=100,
      see_through_walls=False,
      seed=52,
      agent_view_size=7,
      n_agents=3,
      competitive=False,
      fixed_environment=False,
      minigrid_mode=False,
      fully_observed=False
  ):
    """Constructor for multi-agent gridworld environment generator.

    Args:
      grid_size: Number of tiles for the width and height of the square grid.
      width: Number of tiles across grid width.
      height: Number of tiles in height of grid.
      max_steps: Number of environment steps before the episode end (max
        episode length).
      see_through_walls: True if agents can see through walls.
      seed: Random seed used in generating environments.
      agent_view_size: Number of tiles in the agent's square, partially
        observed view of the world.
      n_agents: The number of agents playing in the world.
      competitive: If True, as soon as one agent locates the goal, the episode
        ends for all agents. If False, if one agent locates the goal it is
        respawned somewhere else in the grid, and the episode continues until
        max_steps is reached.
      fixed_environment: If True, will use the same random seed each time the
        environment is generated, so it will remain constant / be the same
        environment each time.
      minigrid_mode: Set to True to maintain backwards compatibility with
        minigrid in the single agent case.
      fully_observed: If True, each agent will receive an observation of the
        full environment state, rather than a partially observed, ego-centric
        observation.
    """
    self.fully_observed = fully_observed

    # Can't set both grid_size and width/height
    if grid_size:
      assert width is None and height is None
      width = grid_size
      height = grid_size

    # Set the number of agents
    self.n_agents = n_agents

    # If competitive, only one agent is allowed to reach the goal.
    self.competitive = competitive

    if self.n_agents == 1:
      self.competitive = True

    # Action enumeration for this environment
    self.actions = MultiGridEnv.Actions

    # Number of cells (width and height) in the agent view
    self.agent_view_size = agent_view_size
    if self.fully_observed:
      self.agent_view_size = max(width, height)

    # Range of possible rewards
    self.reward_range = (0, 1)

    # Compute observation and action spaces
    # Direction always has an extra dimension for tf-agents compatibility
    self.direction_obs_space = gym.spaces.Box(
        low=0, high=3, shape=(self.n_agents,), dtype='uint8')

    # Maintain for backwards compatibility with minigrid.
    self.minigrid_mode = minigrid_mode
    if self.fully_observed:
      obs_image_shape = (width, height, 3)
    else:
      obs_image_shape = (self.agent_view_size, self.agent_view_size, 3)

    if self.minigrid_mode:
      msg = 'Backwards compatibility with minigrid only possible with 1 agent'
      assert self.n_agents == 1, msg

      # Single agent case
      # Actions are discrete integer values
      self.action_space = gym.spaces.Discrete(len(self.actions))
      # Images have three dimensions
      self.image_obs_space = gym.spaces.Box(
          low=0,
          high=255,
          shape=obs_image_shape,
          dtype='uint8')
    else:
      # First dimension of all observations is the agent ID
      self.action_space = gym.spaces.Box(low=0, high=len(self.actions)-1,
                                         shape=(self.n_agents,), dtype='int64')

      self.image_obs_space = gym.spaces.Box(
          low=0,
          high=255,
          shape=(self.n_agents,) + obs_image_shape,
          dtype='uint8')

    # Observations are dictionaries containing an encoding of the grid and the
    # agent's direction
    observation_space = {'image': self.image_obs_space,
                         'direction': self.direction_obs_space}
    if self.fully_observed:
      self.position_obs_space = gym.spaces.Box(low=0,
                                               high=max(width, height),
                                               shape=(self.n_agents, 2),
                                               dtype='uint8')
      observation_space['position'] = self.position_obs_space
    self.observation_space = gym.spaces.Dict(observation_space)

    # Window to use for human rendering mode
    self.window = None

    # Environment configuration
    self.width = width
    self.height = height
    self.max_steps = max_steps
    self.see_through_walls = see_through_walls

    # Current position and direction of the agent
    self.agent_pos = [None] * self.n_agents
    self.agent_dir = [None] * self.n_agents

    # Maintain a done variable for each agent
    self.done = [False] * self.n_agents

    # Initialize the RNG
    self.seed_value = seed
    self.seed(seed=seed)
    self.fixed_environment = fixed_environment

    # Initialize the state
    self.reset()

  def reset(self):
    if self.fixed_environment:
      self.seed(self.seed_value)

    # Current position and direction of the agent
    self.agent_pos = [None] * self.n_agents
    self.agent_dir = [None] * self.n_agents
    self.done = [False] * self.n_agents

    # Generate the grid. Will be random by default, or same environment if
    # 'fixed_environment' is True.
    self._gen_grid(self.width, self.height)

    # These fields should be defined by _gen_grid
    for a in range(self.n_agents):
      assert self.agent_pos[a] is not None
      assert self.agent_dir[a] is not None

      # Check that the agent doesn't overlap with an object
      start_cell = self.grid.get(*self.agent_pos[a])
      assert (start_cell.type == 'agent' or
              start_cell is None or start_cell.can_overlap())

    # Item picked up, being carried, initially nothing
    self.carrying = [None] * self.n_agents

    # Step count since episode start
    self.step_count = 0

    # Return first observation
    obs = self.gen_obs()

    return obs

  def __str__(self):
    """Produce a pretty string of the environment's grid along with the agent.

    A grid cell is represented by 2-character string, the first one for
    the object and the second one for the color.

    Returns:
      String representation of the grid.
    """

    # Map of object types to short string
    obj_to_str = {
        'wall': 'W',
        'floor': 'F',
        'door': 'D',
        'key': 'K',
        'ball': 'A',
        'box': 'B',
        'goal': 'G',
        'lava': 'V',
    }

    # Map agent's direction to short string
    agent_dir_to_str = {0: '>', 1: 'V', 2: '<', 3: '^'}

    text = ''

    for j in range(self.grid.height):
      for i in range(self.grid.width):
        # Draw agents
        agent_here = False
        for a in range(self.n_agents):
          if self.agent_pos[a] is not None and (i == self.agent_pos[a][0] and
                                                j == self.agent_pos[a][1]):
            text += str(a) + agent_dir_to_str[self.agent_dir[a]]
            agent_here = True
        if agent_here:
          continue

        c = self.grid.get(i, j)

        if c is None:
          text += '  '
          continue

        if c.type == 'door':
          if c.is_open:
            text += '__'
          elif c.is_locked:
            text += 'L' + c.color[0].upper()
          else:
            text += 'D' + c.color[0].upper()
          continue

        text += obj_to_str[c.type] + c.color[0].upper()

      if j < self.grid.height - 1:
        text += '\n'

    return text

  def place_obj(self,
                obj,
                top=None,
                size=None,
                reject_fn=None,
                max_tries=math.inf):
    """Place an object at an empty position in the grid.

    Args:
      obj: Instance of Minigrid WorldObj class (such as Door, Key, etc.).
      top: (x,y) position of the top-left corner of rectangle where to place.
      size: Size of the rectangle where to place.
      reject_fn: Function to filter out potential positions.
      max_tries: Throw an error if a position can't be found after this many
        tries.

    Returns:
      Position where object was placed.
    """
    if top is None:
      top = (0, 0)
    else:
      top = (max(top[0], 0), max(top[1], 0))

    if size is None:
      size = (self.grid.width, self.grid.height)

    num_tries = 0

    while True:
      # This is to handle with rare cases where rejection sampling
      # gets stuck in an infinite loop
      if num_tries > max_tries:
        raise gym.error.RetriesExceededError(
            'Rejection sampling failed in place_obj')

      num_tries += 1

      pos = np.array((self._rand_int(top[0],
                                     min(top[0] + size[0], self.grid.width)),
                      self._rand_int(top[1],
                                     min(top[1] + size[1], self.grid.height))))

      # Don't place the object on top of another object
      if self.grid.get(*pos) is not None:
        continue

      # Don't place the object where the agent is
      pos_no_good = False
      for a in range(self.n_agents):
        if np.array_equal(pos, self.agent_pos[a]):
          pos_no_good = True
      if pos_no_good:
        continue

      # Check if there is a filtering criterion
      if reject_fn and reject_fn(self, pos):
        continue

      break

    self.grid.set(pos[0], pos[1], obj)

    if obj is not None:
      obj.init_pos = pos
      obj.cur_pos = pos

    return pos

  def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
    """Set the starting point of all agents in the world.

    Name chosen for backwards compatibility.

    Args:
      top: (x,y) position of the top-left corner of rectangle where agents can
        be placed.
      size: Size of the rectangle where to place.
      rand_dir: Choose a random direction for agents.
      max_tries: Throw an error if a position can't be found after this many
        tries.
    """
    for a in range(self.n_agents):
      self.place_one_agent(
          a, top=top, size=size, rand_dir=rand_dir, max_tries=math.inf)

  def place_one_agent(self,
                      agent_id,
                      top=None,
                      size=None,
                      rand_dir=True,
                      max_tries=math.inf,
                      agent_obj=None):
    """Set the agent's starting point at an empty position in the grid."""

    self.agent_pos[agent_id] = None
    pos = self.place_obj(None, top, size, max_tries=max_tries)

    self.place_agent_at_pos(agent_id, pos, agent_obj=agent_obj,
                            rand_dir=rand_dir)

    return pos

  def place_agent_at_pos(self, agent_id, pos, agent_obj=None, rand_dir=True):
    self.agent_pos[agent_id] = pos
    if rand_dir:
      self.agent_dir[agent_id] = self._rand_int(0, 4)

    # Place the agent object into the grid
    if not agent_obj:
      agent_obj = Agent(agent_id, self.agent_dir[agent_id])
      agent_obj.init_pos = pos
    else:
      agent_obj.dir = self.agent_dir[agent_id]
    agent_obj.cur_pos = pos
    self.grid.set(pos[0], pos[1], agent_obj)

  @property
  def dir_vec(self):
    """Get the direction vector for the agent (points toward forward movement).

    Returns:
      An array of directions that each agent is facing.
    """

    for a in range(self.n_agents):
      assert self.agent_dir[a] >= 0 and self.agent_dir[a] < 4
    return [
        minigrid.DIR_TO_VEC[self.agent_dir[a]] for a in range(self.n_agents)
    ]

  @property
  def right_vec(self):
    """Get the vector pointing to the right of the agents."""
    return [np.array((-dy, dx)) for (dx, dy) in self.dir_vec]

  @property
  def front_pos(self):
    """Get the position of the cell that is right in front of the agent."""
    front_pos = [None] * self.n_agents
    for a in range(self.n_agents):
      assert self.agent_pos[a] is not None and self.dir_vec[a] is not None
      front_pos[a] = self.agent_pos[a] + self.dir_vec[a]
    return front_pos

  def get_view_coords(self, i, j, agent_id):
    """Convert grid coordinates into agent's partially observed view.

    Translate and rotate absolute grid coordinates (i, j) into the agent's
    partially observable view (sub-grid).

    Note that the resulting coordinates may be negative or outside of the
    agent's view size.

    Args:
      i: Integer x coordinate.
      j: Integer y coordinate.
      agent_id: ID of the agent.

    Returns:
      Agent-centric coordinates.
    """

    ax, ay = self.agent_pos[agent_id]
    dx, dy = self.dir_vec[agent_id]
    rx, ry = self.right_vec[agent_id]

    # Compute the absolute coordinates of the top-left view corner
    sz = self.agent_view_size
    hs = self.agent_view_size // 2
    tx = ax + (dx * (sz - 1)) - (rx * hs)
    ty = ay + (dy * (sz - 1)) - (ry * hs)

    lx = i - tx
    ly = j - ty

    # Project the coordinates of the object relative to the top-left
    # corner onto the agent's own coordinate system
    vx = (rx * lx + ry * ly)
    vy = -(dx * lx + dy * ly)

    return vx, vy

  def get_view_exts(self, agent_id):
    """Get the extents of the square set of tiles visible to the agent.

    Note: the bottom extent indices are not included in the set

    Args:
      agent_id: Integer ID of the agent.

    Returns:
      Top left and bottom right (x,y) coordinates of set of visible tiles.
    """
    # Facing right
    if self.agent_dir[agent_id] == 0:
      top_x = self.agent_pos[agent_id][0]
      top_y = self.agent_pos[agent_id][1] - self.agent_view_size // 2
    # Facing down
    elif self.agent_dir[agent_id] == 1:
      top_x = self.agent_pos[agent_id][0] - self.agent_view_size // 2
      top_y = self.agent_pos[agent_id][1]
    # Facing left
    elif self.agent_dir[agent_id] == 2:
      top_x = self.agent_pos[agent_id][0] - self.agent_view_size + 1
      top_y = self.agent_pos[agent_id][1] - self.agent_view_size // 2
    # Facing up
    elif self.agent_dir[agent_id] == 3:
      top_x = self.agent_pos[agent_id][0] - self.agent_view_size // 2
      top_y = self.agent_pos[agent_id][1] - self.agent_view_size + 1
    else:
      assert False, 'invalid agent direction'

    bot_x = top_x + self.agent_view_size
    bot_y = top_y + self.agent_view_size

    return (top_x, top_y, bot_x, bot_y)

  def relative_coords(self, x, y, agent_id):
    """Check if a grid position belongs to the agent's field of view.

    Args:
      x: Integer x coordinate.
      y: Integer y coordinate.
      agent_id: ID of the agent.

    Returns:
      The corresponding agent-centric coordinates of the grid position.
    """
    vx, vy = self.get_view_coords(x, y, agent_id)

    if (vx < 0 or vy < 0 or vx >= self.agent_view_size or
        vy >= self.agent_view_size):
      return None

    return vx, vy

  def in_view(self, x, y, agent_id):
    """Check if a grid position is visible to the agent."""
    return self.relative_coords(x, y, agent_id) is not None

  def agent_sees(self, x, y, agent_id):
    """Check if a non-empty grid position is visible to the agent."""
    coordinates = self.relative_coords(x, y, agent_id)
    if coordinates is None:
      return False
    vx, vy = coordinates

    obs = self.gen_obs()
    obs_grid, _ = Grid.decode(obs['image'][agent_id])
    obs_cell = obs_grid.get(vx, vy)
    world_cell = self.grid.get(x, y)

    return obs_cell is not None and obs_cell.type == world_cell.type

  def agent_is_done(self, agent_id):
    # Remove correspnding agent object from the grid
    pos = self.agent_pos[agent_id]
    agent_obj = self.grid.get(pos[0], pos[1])
    self.grid.set(pos[0], pos[1], None)

    self.done[agent_id] = True

    # If an agent finishes the level while carrying an object, it is randomly
    # respawned in a new position. Warning: this may break dependencies for the
    # level (e.g. if a key is spawned on the wrong side of a door).
    # TODO(natashajaques): environments can define respawn behavior
    if self.carrying[agent_id]:
      self.place_obj(obj=self.carrying[agent_id])
      self.carrying[agent_id] = None

    # Respawn agent in new location
    self.place_one_agent(agent_id, agent_obj=agent_obj)

  def move_agent(self, agent_id, new_pos):
    # Retrieve agent object
    old_pos = self.agent_pos[agent_id]
    agent_obj = self.grid.get(old_pos[0], old_pos[1])
    assert agent_obj.agent_id == agent_id
    assert (agent_obj.cur_pos == old_pos).all()

    # Update the agent position in grid and environment
    self.grid.set(old_pos[0], old_pos[1], None)
    self.agent_pos[agent_id] = new_pos
    agent_obj.cur_pos = new_pos
    self.grid.set(new_pos[0], new_pos[1], agent_obj)
    assert (self.grid.get(
        new_pos[0], new_pos[1]).cur_pos == self.agent_pos[agent_id]).all()

  def rotate_agent(self, agent_id):
    # Retrieve agent object
    pos = self.agent_pos[agent_id]
    agent_obj = self.grid.get(pos[0], pos[1])
    assert agent_obj.agent_id == agent_id

    # Update the dir
    agent_obj.dir = self.agent_dir[agent_id]
    self.grid.set(pos[0], pos[1], agent_obj)
    assert self.grid.get(pos[0], pos[1]).dir == self.agent_dir[agent_id]

  def step_one_agent(self, action, agent_id):
    reward = 0

    # Get the position in front of the agent
    fwd_pos = self.front_pos[agent_id]

    # Rotate left
    if action == self.actions.left:
      self.agent_dir[agent_id] -= 1
      if self.agent_dir[agent_id] < 0:
        self.agent_dir[agent_id] += 4
      self.rotate_agent(agent_id)

    # Rotate right
    elif action == self.actions.right:
      self.agent_dir[agent_id] = (self.agent_dir[agent_id] + 1) % 4
      self.rotate_agent(agent_id)

    # Move forward
    elif action == self.actions.forward:
      successful_forward = self._forward(agent_id, fwd_pos)
      fwd_cell = self.grid.get(*fwd_pos)
      if successful_forward and fwd_cell is not None and fwd_cell.type == 'goal':
        reward = self._reward()

    # Pick up an object
    elif action == self.actions.pickup:
      self._pickup(agent_id, fwd_pos)

    # Drop an object
    elif action == self.actions.drop:
      self._drop(agent_id, fwd_pos)

    # Toggle/activate an object
    elif action == self.actions.toggle:
      self._toggle(agent_id, fwd_pos)

    # Done action -- by default acts as no-op.
    elif action == self.actions.done:
      pass

    else:
      assert False, 'unknown action'

    return reward

  def _forward(self, agent_id, fwd_pos):
    """Attempts to move the forward one cell, returns True if successful."""
    fwd_cell = self.grid.get(*fwd_pos)
    # Make sure agents can't walk into each other
    agent_blocking = False
    for a in range(self.n_agents):
      if a != agent_id and np.array_equal(self.agent_pos[a], fwd_pos):
        agent_blocking = True

    # Deal with object interactions
    if not agent_blocking:
      if fwd_cell is not None and fwd_cell.type == 'goal':
        self.agent_is_done(agent_id)
      elif fwd_cell is not None and fwd_cell.type == 'lava':
        self.agent_is_done(agent_id)
      elif fwd_cell is None or fwd_cell.can_overlap():
        self.move_agent(agent_id, fwd_pos)
      return True
    return False

  def _pickup(self, agent_id, fwd_pos):
    """Attempts to pick up object, returns True if successful."""
    fwd_cell = self.grid.get(*fwd_pos)
    if fwd_cell and fwd_cell.can_pickup():
      if self.carrying[agent_id] is None:
        self.carrying[agent_id] = fwd_cell
        self.carrying[agent_id].cur_pos = np.array([-1, -1])
        self.grid.set(fwd_pos[0], fwd_pos[1], None)
        a_pos = self.agent_pos[agent_id]
        agent_obj = self.grid.get(a_pos[0], a_pos[1])
        agent_obj.contains = fwd_cell
        return True
    return False

  def _drop(self, agent_id, fwd_pos):
    """Attempts to drop object, returns True if successful."""
    fwd_cell = self.grid.get(*fwd_pos)
    if not fwd_cell and self.carrying[agent_id]:
      self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying[agent_id])
      self.carrying[agent_id].cur_pos = fwd_pos
      self.carrying[agent_id] = None
      a_pos = self.agent_pos[agent_id]
      agent_obj = self.grid.get(a_pos[0], a_pos[1])
      agent_obj.contains = None
      return True
    return False

  def _toggle(self, agent_id, fwd_pos):
    """Attempts to toggle object, returns True if successful."""
    fwd_cell = self.grid.get(*fwd_pos)
    if fwd_cell:
      if fwd_cell.type == 'door':
        return fwd_cell.toggle(self, fwd_pos, self.carrying[agent_id])
      else:
        return fwd_cell.toggle(self, fwd_pos)
    return False

  def step(self, actions):
    # Maintain backwards compatibility with MiniGrid when there is one agent
    if not isinstance(actions, list) and self.n_agents == 1:
      actions = [actions]

    self.step_count += 1

    rewards = [0] * self.n_agents

    # Randomize order in which agents act for fairness
    agent_ordering = np.arange(self.n_agents)
    np.random.shuffle(agent_ordering)

    # Step each agent
    for a in agent_ordering:
      rewards[a] = self.step_one_agent(actions[a], a)

    obs = self.gen_obs()

    # Backwards compatibility
    if self.minigrid_mode:
      rewards = rewards[0]

    collective_done = False
    # In competitive version, if one agent finishes the episode is over.
    if self.competitive:
      collective_done = np.sum(self.done) >= 1

    # Running out of time applies to all agents
    if self.step_count >= self.max_steps:
      collective_done = True

    return obs, rewards, collective_done, {}

  def gen_obs_grid(self, agent_id):
    """Generate the sub-grid observed by the agent.

    This method also outputs a visibility mask telling us which grid cells
    the agent can actually see.

    Args:
      agent_id: Integer ID of the agent for which to generate the grid.

    Returns:
      Sub-grid and visibility mask.
    """

    top_x, top_y, _, _ = self.get_view_exts(agent_id)

    grid = self.grid.slice(top_x, top_y, self.agent_view_size,
                           self.agent_view_size)

    for _ in range(self.agent_dir[agent_id] + 1):
      grid = grid.rotate_left()

    # Process occluders and visibility
    # Note that this incurs some performance cost
    if not self.see_through_walls:
      vis_mask = grid.process_vis(
          agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1))
    else:
      vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

    # Make it so the agent sees what it's carrying
    # We do this by placing the carried object at the agent's position
    # in the agent's partially observable view
    agent_pos = grid.width // 2, grid.height - 1
    if self.carrying[agent_id]:
      grid.set(agent_pos[0], agent_pos[1], self.carrying[agent_id])
    else:
      grid.set(agent_pos[0], agent_pos[1], None)

    return grid, vis_mask

  def gen_obs(self):
    """Generate the stacked observation for all agents."""
    images = []
    dirs = []
    positions = []
    for a in range(self.n_agents):
      if self.fully_observed:
        image = self.grid.encode()
        direction = self.agent_dir[a]
      else:
        image, direction = self.gen_agent_obs(a)
      images.append(image)
      dirs.append(direction)
      positions.append(self.agent_pos[a])

    # Backwards compatibility: if there is a single agent do not return an array
    if self.minigrid_mode:
      images = images[0]

    # Observations are dictionaries containing:
    # - an image (partially observable view of the environment)
    # - the agent's direction/orientation (acting as a compass)
    # Note direction has shape (1,) for tfagents compatibility
    obs = {
        'image': images,
        'direction': dirs
    }
    if self.fully_observed:
      obs['position'] = positions

    return obs

  def gen_agent_obs(self, agent_id):
    """Generate the agent's view (partially observed, low-resolution encoding).

    Args:
      agent_id: ID of the agent for which to generate the observation.

    Returns:
      3-dimensional partially observed agent-centric view, and int direction
    """
    grid, vis_mask = self.gen_obs_grid(agent_id)

    # Encode the partially observable view into a numpy array
    image = grid.encode(vis_mask)

    return image, self.agent_dir[agent_id]

  def get_obs_render(self, obs, tile_size=minigrid.TILE_PIXELS // 2):
    """Render an agent observation for visualization."""

    grid, vis_mask = Grid.decode(obs)

    # Render the whole grid
    img = grid.render(
        tile_size,
        # agent_pos=self.agent_pos,
        # agent_dir=self.agent_dir,
        highlight_mask=vis_mask)

    return img

  def compute_agent_visibility_mask(self, agent_id):
    # Mask of which cells to highlight
    highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

    # Compute which cells are visible to the agent
    _, vis_mask = self.gen_obs_grid(agent_id)

    # Compute the world coordinates of the bottom-left corner
    # of the agent's view area
    f_vec = self.dir_vec[agent_id]
    r_vec = self.right_vec[agent_id]
    top_left = self.agent_pos[agent_id] + f_vec * (self.agent_view_size-1) - \
        r_vec * (self.agent_view_size // 2)

    # For each cell in the visibility mask
    for vis_j in range(0, self.agent_view_size):
      for vis_i in range(0, self.agent_view_size):
        # If this cell is not visible, don't highlight it
        if not vis_mask[vis_i, vis_j]:
          continue

        # Compute the world coordinates of this cell
        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

        if abs_i < 0 or abs_i >= self.width:
          continue
        if abs_j < 0 or abs_j >= self.height:
          continue

        # Mark this cell to be highlighted
        highlight_mask[abs_i, abs_j] = True

    return highlight_mask

  def render(self,
             mode='human',
             close=False,
             highlight=True,
             tile_size=minigrid.TILE_PIXELS):
    """Render the whole-grid human view."""

    if close:
      if self.window:
        self.window.close()
      return None

    if mode == 'human' and not self.window:
      self.window = minigrid.window.Window('gym_minigrid')
      self.window.show(block=False)

    if highlight:
      highlight_mask = []
      for a in range(self.n_agents):
        if self.agent_pos[a] is not None:
          highlight_mask.append(self.compute_agent_visibility_mask(a))
    else:
      highlight_mask = None

    # Render the whole grid
    img = self.grid.render(tile_size, highlight_mask=highlight_mask)

    if mode == 'human':
      self.window.show_img(img)
      self.window.set_caption(self.mission)

    return img
