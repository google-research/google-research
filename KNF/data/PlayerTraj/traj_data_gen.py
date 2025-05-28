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

"""Preprocess basketball trajectory dataset."""
import os
import numpy as np
import pandas as pd

# Download NBA baseketball player trajectory data
# !git clone https://github.com/linouk23/NBA-Player-Movements
# unzip all .7z files and move them into the "json_data" folder
os.mkdir('json_data')


class Player:
  """Helper function."""

  def __init__(self, player_dict):
    """attributes: playerid, firstname, lastname, jersey, position."""
    for key in player_dict:
      setattr(self, key, player_dict[key])

  def __repr__(self):
    return f'{self.firstname} {self.lastname}, #{self.jersey}, {self.position}'


class Team:
  """Helper function."""

  def __init__(self, team_dict):
    """attributes: teamid, name, abbreviation, players."""
    for key in team_dict:
      setattr(self, key, team_dict[key])
    self.players = [Player(player) for player in self.players]

    self.players_mapping = {}
    for player in self.players:
      self.players_mapping[
          f'{player.firstname} {player.lastname}'] = player.playerid

  def __repr__(self):
    descr = f'{self.name}\n\tPlayers:\n'
    for player in self.players:
      descr += f'\t{repr(player)}\n'
    return descr


class Game:
  """Helper function."""

  def __init__(self, game_json_file):
    self.game_json_file = game_json_file
    data_frame = pd.read_json(game_json_file)
    self.events = data_frame['events']
    self.num_events = len(self.events)

    self.visitor = Team(self.events[0]['visitor'])
    self.home = Team(self.events[0]['home'])

    # name to id
    self.players_mapping = {
        **self.visitor.players_mapping,
        **self.home.players_mapping
    }

  def __repr__(self):
    descr = 'Visitor: ' + repr(self.visitor)
    descr += 'Home: ' + repr(self.home)
    return descr

  def get_player_list(self):
    names = list(self.visitor.players_mapping.keys()) + list(
        self.home.players_mapping.keys())
    return names

  def get_player_movement(self, player_name, event_id):
    """extracts position and time."""
    playerid = self.players_mapping[player_name]
    event = self.events[event_id]
    moments = event['moments']
    x, y, t = [], [], []
    for moment in moments:
      for player in moment[5]:
        if player[1] == playerid:
          x.append(player[2])
          y.append(player[3])
          t.append(moment[2])
    return x, y, t


def is_connected(x, y):
  """checks if the trajectory is connected."""
  for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:]):
    if np.abs(x0 - x1) > 3 or np.abs(y0 - y1) > 3:
      # check disconnectedness if the neighbouring position is larger than 3 ft.
      return False
  return True


def sample_movement_from_games(game_json_list, rngkey):
  """Sample trajs from data files.

  Args:
    game_json_list: json file list.
    rngkey: random key.

  Returns:
    The first returning argument is a position set used for the tracking model
    [[x_t1, y_t2], ...];
    The second returning argument is a combo of the original coordinate set and
    the game clock set [x, y, t].
  """
  pos_set = []
  ori_movements = []
  for i, g in enumerate(game_json_list):
    game = Game(game_json_file='json_data/' + g)
    print('Game:', i)
    print(game)
    names = game.get_player_list()
    x = []
    i = 0
    # sample 5 long trajectories from each game
    while i < 5:

      # repeat until get a player with movement information
      name = rngkey.choice(names, replace=False)
      event_id = rngkey.choice(game.num_events, replace=False)
      x, y, t = game.get_player_movement(name, event_id)

      # The trajectory has to be connected and length has to be bigger than 300
      if not is_connected(x, y) or len(x) < 300:
        print('not connected')
        continue
      else:
        pos = np.array([[x_1, x_2] for x_1, x_2 in zip(x, y)])
        print('Sampled player, event id, and shape:', name, event_id, pos.shape,
              len(pos_set))
        pos_set.append(pos)
        ori_movements.append([x, y, t])
        i += 1
  return pos_set, ori_movements


game_list = os.listdir('json_data')

rng = np.random.RandomState(seed=1234)

tr = rng.choice(game_list, size=60, replace=False)

te_avail_game_list = [g for g in game_list if g not in tr]
te = rng.choice(te_avail_game_list, size=10, replace=False)

print('Training set')
tr_pos_set, _ = sample_movement_from_games(tr, rng)

print('Testing set')
te_pos_set, _ = sample_movement_from_games(te, rng)

np.save('train.npy', tr_pos_set)
np.save('test.npy', te_pos_set)
