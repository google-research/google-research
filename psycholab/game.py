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

"""Creates grid social games from ASCII art diagrams."""

from __future__ import print_function

import collections

from gym import spaces
import numpy as np

# Careful with convention 'up' is displayed as -1 on the grid plot.
MOVING_ACTIONS = {
    'stand': np.array([0, 0]),
    'up': np.array([0, -1]),
    'down': np.array([0, 1]),
    'left': np.array([-1, 0]),
    'right': np.array([1, 0])
}


class Element(object):
  """An element can be either an item object or a player.

  It has a color.
  It can be visible or not, pushable or not, passable or not.
  In future (by adding physics), it will have a speed
  and will be bouncing or not.
  """

  def __init__(self, color=(254, 254, 254), visible=True, pushable=False,
               passable=False):
    self.color = color
    self.visible = visible
    self.pushable = pushable
    self.passable = passable


class Item(Element):
  """An item "on the floor" that can be collected or pushed by players."""

  def __init__(self, color=(254, 254, 254), visible=True, pushable=True,
               passable=True, force_collect=True):
    super(Item, self).__init__(color=color,
                               visible=visible,
                               pushable=pushable,
                               passable=passable)
    # If force_collect is True,
    # the item is automatically collected when the agent runs into it:
    self.force_collect = force_collect


class Player(Element):
  """A player agent."""

  def __init__(self, color=(254, 254, 254), visible=True, pushable=False,
               passable=False):
    super(Player, self).__init__(color=color,
                                 visible=visible,
                                 pushable=pushable,
                                 passable=passable)


class Game(object):
  """Generates a Gym-based multi-player grid world markov game.

  Attributes:
    ascii_art: a list of strings, representing the map of the game. Ex:
                art = ['#####',
                       '#   #',
                       '#a=A#',
                       '#####',]
                This represents a room surrounded by absolute walls ('#')
                containing one item ('a'), one player ('A') and one player
                wall ('=') that only stops players (items can pass).

    items: items is a dictionary mapping items to their names (lowercase)
            in the ascii-art ex, we would have:
            items = {Item():'a'}.

    players: a dictionary mapping players to their names (uppercase) in the
             ascii-art: players = {Player():'A'}.

    tabular: a boolean that specifies if observations are images (False)
             or integers (True).

    max_steps: an integer that represents the maximal number of steps
               before the game ends.

    actions: dictionary mapping integers (representing actions)
             to strings (textual descriptions of actions)

    last_events: list of strings describing events. Events are textual
                 descriptions of what occurs in the game.
                 The default events that can appear are:

      - 'A_moves' (when player A takes a moving action (1,2,3 or 4))
      - 'A_goes_to_the_wall' (player A is blocked by a wall)
      - 'A_is_blocked_by_X' (when player A is blocked by another player or item)
      - 'A_lost_the_drawn' (when several players try to reach the same cell,
                            one player will be picked randomly)
      - 'A_collects_x' (when player A colletcs an item)

    rewards: dictionary mapping events (strings) to rewards (floats)

    terminaisons: list of events (strings) that causes the end of the game.

    effects: dictionary mapping events (strings) with functions that
                  modifies the game's attributes.

    steps: counts the number of steps done during an episode.

    done: boolean specifying if the game has ended.

    height: vertical dimensions of the game map.

    width: horizontal dimensions of the game map.

    players_cells: dictionary mapping players names to their
                       (x,y) positions in the game.

    players_items: dictionary mapping players names to dictionaries mapping
                       item names to integers, representing the quantity
                       of each item collected by each player.

    items_cells: dictionary mapping items names to their
                      (x,y) positions in the game.

    player_walls: dictionary mapping player walls to their (x,y) positions.

    absolute_walls: dictionary mapping absolute walls
                    to their (x,y) positions.

    content: 2d table of dictionaries mapping elements
                  (player or items names) to integers (their quantities)
                  at each (x,y) position.

    players_order: sorted list of players names,
                   used to attribute indexes (intergers) to players.

    items_order: sorted list of items names,
                 used to attribute indexes (interger) to items.

    players_idx: dictionary mapping players names to players indexes (int).

    items_idx: dictionary mapping items names to their indexes (int).

    num_players: number of players.

    num_items: number of items.

    num_actions: number of actions for each player.

    num_states: number of states (upper bound of the number
                of combinations of players positions).

    action_space: gym-like action space (a MultiDiscrete space).
  """

  def __init__(self, ascii_art, items, players, tabular=False, max_steps=100):
    """Init a game and read the ascii-art map."""

    # Arguments:
    self.ascii_art = ascii_art
    self.items = items
    self.players = players
    self.tabular = tabular
    self.max_steps = max_steps

    # Default:
    self.actions = {0: 'stand', 1: 'up', 2: 'down', 3: 'left', 4: 'right'}
    self.rewards = {}
    self.terminaisons = []
    self.effects = {}
    self.steps = 0
    self.done = False
    self.last_events = []

    # To be determined from the ascii-art:
    self.height = None
    self.width = None
    self.players_cells = None
    self.players_items = None
    self.items_cells = None
    self.player_walls = None
    self.absolute_walls = None
    self.content = None

    # Initialize attributes from the ascii-art game map:
    self.read_ascii_art()

    # Define ordering indexes for players and items:
    self.players_order = sorted(self.players.keys())
    self.players_idx = {
        player: i for i, player in enumerate(self.players_order)}
    self.items_order = sorted(self.items.keys())
    self.items_idx = {item: i for i, item in enumerate(self.items_order)}

  def read_ascii_art(self):
    """Generates the list of items and player and the map from the ascii art."""

    self.height = len(self.ascii_art)
    self.width = max([len(line) for line in self.ascii_art])
    self.players_cells = {}
    self.players_items = {}
    self.items_cells = {}
    self.player_walls = np.zeros((self.width, self.height))
    self.absolute_walls = np.zeros((self.width, self.height))
    self.content = []
    for _ in range(self.width):
      cell_content = []
      for _ in range(self.height):
        cell_content.append(collections.defaultdict(int))
      self.content.append(cell_content)

    for y, line in enumerate(self.ascii_art):
      for x, char in enumerate(line):
        if char == '#':
          self.absolute_walls[x, y] = 1
        elif char == '=':
          self.player_walls[x, y] = 1
        else:
          self.content[x][y][char] += 1

          if char.isupper():
            self.players_cells[char] = np.array([x, y])
            self.players_items[char] = collections.defaultdict(int)
          if char.islower():
            if char in self.items_cells:
              self.items_cells[char].append(np.array([x, y]))
            else:
              self.items_cells[char] = [np.array([x, y])]

    assert set(self.players.keys()) >= set(self.players_cells.keys()), (
        'some players may have no description')

    assert set(self.items.keys()) >= set(self.items_cells.keys()), (
        'some items may have no description')

  def display(self):
    print('game map:')
    print('---------')
    for line in self.ascii_art:
      print(line)

  def reset(self):
    self.read_ascii_art()
    self.done = False
    self.last_events = []
    self.steps = 0
    return self.generate_observations()

  def add_action(self, action_name, conditions=None, consequences=None):
    """Add an action, its condition to be doable and its consequences.

    Eventual associated rewards should be defined using the method 'add_reward'.

    Args:
      action_name: name of the action (str).
      conditions: event (str) or list of events. All must hold, can be empty.

      consequences: event (str) or list of events, can be empty.
    """

    action_idx = len(self.actions)
    self.actions[action_idx] = action_name
    if conditions is not None and not isinstance(conditions, list):
      conditions = [conditions]
    self.actions_conditions[action_name] = conditions
    if consequences is not None and not isinstance(consequences, list):
      consequences = [consequences]
    self.actions_consequences[action_name] = consequences

  def add_effect(self, event, effect):
    """Add a function (effect) that modifies the game if event is held.

    Args:
      event: the event that causes the effect (str).
      effect: function that modifies the game. It should take a
          game as argument and modify its public attributes
          or call its public methods.
          For example,
          reaching a new game level could be implemented as follow:

          def my_effect(my_game):
            my_game.art = new_ascii_art
            my_game.read_ascii_art()
            return my_game.reset() # return the new initial state
    """
    self.effects[event] = effect

  def add_reward(self, event, targets_rewards):
    for target_player, reward in targets_rewards.items():
      if event in self.rewards:
        self.rewards[event].append((target_player, reward))
      else:
        self.rewards[event] = [(target_player, reward)]

  def add_terminaison(self, conditions=None):
    """Add a list of conditions (events) that defines a terminal state.

    Args:
      conditions: event (str) or list of events. All must hold, can be empty.
    """
    if conditions is not None and not isinstance(conditions, list):
      conditions = [conditions]
    self.terminaisons.append(conditions)

  @property
  def num_players(self):
    return len(self.players_order)

  @property
  def num_actions(self):
    return len(self.actions)

  @property
  def action_space(self):
    return spaces.MultiDiscrete([self.num_actions] * self.num_players)

  @property
  def num_states(self):
    # num_states is an upper bound of the number of possible
    # tuples containing each player's position.
    # So far, this number does not take collectable objects into account,
    # so a game with collectable objects is partially observed.
    return (self.height * self.width) ** self.num_players

  @property
  def num_items(self):
    return len(self.items)

  def step(self, actions):
    """Applies a gym-based environement step.

    Args:
      actions: list of integers (or numpy array)
      containing the actions of each agent.

    Returns:
      observations: image or list of integers, depending on the game setting.
    """
    events = []
    # If you want an order that depends on some fitness,
    # you must add a condition here.
    actions, conflict_events = self.solve_conflicts(actions)
    events += conflict_events
    random_order = np.random.permutation(self.num_players)
    for player_idx in random_order:
      player = self.players_order[player_idx]
      action = actions[player_idx]
      step_events = self.single_player_step(player, action)
      events += step_events

    self.last_events = events
    self.steps += 1

    if self.terminal_step():
      self.done = True

    self.apply_effects()
    self.apply_physics()  # Using players/items speed and directions attributes.

    observations = self.generate_observations()
    rewards = self.reward_events()
    done = self.is_done()
    infos = self.infos()
    return observations, rewards, done, infos

  def single_player_step(self, player, action):
    """Process separately the step of one player.

    Args:
      player: (str) name of player doing the action.
      action: (int) action number.

    Returns:
      events: list of event caused by the player's action.
    """

    events = []
    if action in self.actions:
      action_name = self.actions[action]
      if action_name in MOVING_ACTIONS:
        new_position = self.players_cells[player] + MOVING_ACTIONS[action_name]
        for element in self.content[new_position[0]][new_position[1]]:
          if element in self.items:
            item = self.items[element]
            if item.force_collect:
              action_events = self.player_collects_item(
                  player, element)
              events += action_events
          # If element is a pushable player -> push it.
          # TODO(alexisjacq) if it is a pushable item, must affect item_cells.
          elif element in self.players and self.players[element].pushable:
            other = self.players[element]
            other_new_position = (
                self.players_cells[element] + MOVING_ACTIONS[action_name])
            for other_element in (
                self.content[other_new_position[0]][other_new_position[1]]):
              if other_element in self.items:
                other_item = self.items[other_element]
                if other_item.force_collect:
                  action_events = self.player_collects_item(
                      other, other_element)
                  events += action_events

            del self.content[self.players_cells[element][0]][
                self.players_cells[element][1]][element]
            self.players_cells[element] = other_new_position
            self.content[other_new_position[0]][
                other_new_position[1]][element] = 1

        del self.content[self.players_cells[player][0]][
            self.players_cells[player][1]][player]
        self.players_cells[player] = new_position
        self.content[new_position[0]][new_position[1]][player] = 1

      else:
        action_events = self.player_acts(action)
        events += action_events

    return events

  def player_collects_item(self, player, item):
    """add the picked item to the list of player's possesions."""

    self.players_items[player][item] += 1

    # TODO(alexisjacq): if item limited quantity, must affect item_cells:
    # Add argument 'new_position' to this function.
    # Then add something like:
    # self.content[new_position[0]][new_position[1]][item] -= 1
    # if self.content[new_position[0]][new_position[1]][item] == 0:
    #   del self.content[new_position[0]][new_position[1]][item]

    events = [player+'_collects_'+item]
    return events

  def player_acts(self, action):
    """Check if action meets conditions for being possible.

    If action is possible, return consequences.

    Args:
      action: action of one agent.

    Returns:
      events: consequences of action if possible.
    """

    conditions = self.actions_conditions[action]
    ok = True
    for condition in conditions:
      if condition not in self.last_events:
        ok = False
        break

    events = []
    if ok:  # All conditions are in self.last_events.
      consequences = self.action_consequences[action]
      for consequence in consequences:
        events.append(consequence)

    return events

  def solve_conflicts(self, actions):
    """This method removes forbidden/conflictual actions.

    For example: if two or more agents agents reach the same cell,
    only one -- randomly chosen -- does the move.

    Args:
      actions: list of integers (or numpy array)
      containing the actions of each agent

    Returns:
      actions: corrected list of actions.
      events: describes the conflicts encountered.
    """
    events = []
    future_cells = collections.defaultdict(list)

    # When player runs into walls and blocking items:
    for player in self.players.keys():
      player_idx = self.players_idx[player]
      player_passable = self.players[player].passable
      action = actions[player_idx]
      if action > 0:
        events.append(player+'_moves')
      action_name = self.actions[action]
      if action_name in MOVING_ACTIONS and action > 0 and not player_passable:
        new_position = self.players_cells[player] + MOVING_ACTIONS[action_name]
        # Check if no wall or (unpassable + unpushable) object:
        player_wall = self.player_walls[new_position[0]][new_position[1]]
        absolute_wall = self.absolute_walls[new_position[0]][new_position[1]]
        if player_wall + absolute_wall > 0:
          actions[player_idx] = 0  # stand
          events.append(player+'_goes_in_walls')
        else:
          for element in self.content[new_position[0]][new_position[1]]:
            # When player runs into unpassable/pushable item:
            if element in self.items:
              passable = self.items[element].passable
              pushable = self.items[element].pushable
              if not passable and not pushable:
                actions[player_idx] = 0
                events.append(player+'_blocked_by_'+element)

    # When players cross each others, both are stoped:
    for player in self.players.keys():
      player_idx = self.players_idx[player]
      player_passable = self.players[player].passable
      action = actions[player_idx]
      action_name = self.actions[action]
      if action_name in MOVING_ACTIONS and action > 0 and not player_passable:
        new_position = self.players_cells[player] + MOVING_ACTIONS[action_name]
        for element in self.content[new_position[0]][new_position[1]]:
          if element in self.players:
            other_action = actions[self.players_idx[element]]
            passable = self.players[element].passable
            pushable = self.players[element].pushable
            other_action_name = self.actions[other_action]
            cross = ((MOVING_ACTIONS[action_name] +
                      MOVING_ACTIONS[other_action_name])**2).sum()
            if cross == 0 and not passable and not pushable:
              actions[player_idx] = 0
              actions[self.players_idx[element]] = 0
              events.append(player+'_blocked_by_'+element)
              events.append(element+'_blocked_by_'+player)

    # When players blocked by standing player:
    for player in self.players.keys():
      player_idx = self.players_idx[player]
      player_passable = self.players[player].passable
      action = actions[player_idx]
      action_name = self.actions[action]
      if action_name in MOVING_ACTIONS and action > 0 and not player_passable:
        new_position = self.players_cells[player] + MOVING_ACTIONS[action_name]
        for element in self.content[new_position[0]][new_position[1]]:
          if element in self.players:
            other_action = actions[self.players_idx[element]]
            passable = self.players[element].passable
            pushable = self.players[element].pushable
            # Check if it is an unpassable/unpushable player who stays:
            if other_action == 0 and not passable and not pushable:
              actions[player_idx] = 0
              events.append(player+'_blocked_by_'+element)

    # Predict futur positions:
    for player in self.players.keys():
      player_idx = self.players_idx[player]
      player_passable = self.players[player].passable
      action = actions[player_idx]
      action_name = self.actions[action]
      if action_name in MOVING_ACTIONS and action > 0 and not player_passable:
        new_position = self.players_cells[player] + MOVING_ACTIONS[action_name]
        future_cells[tuple(new_position)].append(player)

    # When several players try to reach the same cell,
    # one player is picked randomly:
    for content in future_cells.values():
      if len(content) > 1:
        losers = np.random.choice(content, len(content)-1, replace=False)
        for player in losers:
          player_idx = self.players_idx[player]
          actions[player_idx] = 0
          events.append(player+'_lost_the_drawn')

    return actions, events

  def terminal_step(self):
    """Detect if the state is terminal."""

    if self.max_steps is not None and self.steps > self.max_steps:
      return True

    for conditions in self.terminaisons:
      ok = True
      for condition in conditions:
        if condition not in self.last_events:
          ok = False
          break

      if ok:
        return True

    return False

  def is_done(self):
    return self.done

  def reward_events(self):
    rewards = np.zeros(self.num_players)
    for event in self.last_events:
      if event in self.rewards:
        for player_target, reward in self.rewards[event]:
          target_idx = self.players_idx[player_target]
          rewards[target_idx] += reward

    return rewards

  def apply_physics(self):
    # TODO(alexisjacq)
    pass

  def apply_effects(self):
    for event in self.last_events:
      if event in self.effects:
        self.effects[event](self)

  def render(self):
    """Returns the image of the map with elements and wall colors.

    By default, walls > agents > items
    and alphabetical order between superposed agents or item
    """
    image = np.zeros((self.height, self.width, 3), dtype='uint8')
    for item_name in self.items_order:
      item = self.items[item_name]
      item_color = np.array(item.color, dtype=int)
      for x, y in self.items_cells[item_name]:
        image[x, y, :] = item_color

    for player_name in self.players_order:
      player = self.players[player_name]
      player_color = np.array(player.color, dtype='uint8')
      x, y = self.players_cells[player_name]
      for channel in range(3):
        image[x, y, channel] = player_color[channel]

    for channel in range(3):
      image[:, :, channel][self.player_walls > 0] = 100

    for channel in range(3):
      image[:, :, channel][self.absolute_walls > 0] = 150

    return image

  def discrete_state(self, obs):
    """Converts an x,y position into a discrete state.

    Args:
      obs: list of discrete (x,y) positions of players.

    Returns:
      state: a unique discrete number associated with the list of positions.
    """
    state = 0
    for i, (x, y) in enumerate(obs):
      state += (x * self.width + y) * ((self.width * self.height) ** i)
    return state

  def one_hot_state(self, obs):
    """Converts a list of x,y positions into a "one-hot" vector.

    Args:
      obs: list of discrete (x,y) positions of players.

    Returns:
      state: numpy array of size (1, (width + height) * num_players).
      The first 'width' elements encode the column for the first player.
      They are all zeros except the x-th which is 1.
      (similar for second part about encoding the row for the first player
      and then for all other players).
      This is not exactly a one-hot encoding since multiple ones are
      set (two by player).

    Ex: in a 2-players 3x3 grid, obs = ((x1, y1), (x2, y2)) = ((2, 3), (1, 1))
    one_hot_state(obs) = ((0,1,0 , 0,0,1 , 1,0,0 , 1,0,0))
    """
    state = np.zeros((1, (self.width + self.height) * self.num_players))
    for i, (x, y) in enumerate(obs):
      state[0, i * (self.width + self.height) + x] = 1
      state[0, i * (self.width + self.height) + self.width + y] = 1
    return state

  def generate_observations(self):
    if self.tabular:
      obs = []
      for player in self.players_order:
        x, y = self.players_cells[player]
        obs.append((x, y))
      return obs
    else:
      return self.render()

  def infos(self):
    infos = {'event_list': str(self.last_events)}
    return infos


