# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Temptation game.

this example comes from the game introduced in paper
Foolproof Cooperative Learning
by Alexis Jacq, Julien Perolat, Matthieu Geist and Olivier Pietquin.
"""

import numpy as np
from psycholab import game
from psycholab import visualizer


def create_game():
  """Creates the temptation game."""

  art = ['########',
         'a  AB  a',
         'b      b',
         'c      c',
         'd      d',
         'e      e',
         'f      f',
         'g      g',
         'h      h',
         'i      i',
         'j      j',
         '########'
        ]

  item_a = game.Item(color=(60, 254, 254))
  item_b = game.Item(color=(80, 254, 254))
  item_c = game.Item(color=(100, 254, 254))
  item_d = game.Item(color=(120, 254, 254))
  item_e = game.Item(color=(140, 254, 254))
  item_f = game.Item(color=(160, 254, 254))
  item_g = game.Item(color=(180, 254, 254))
  item_h = game.Item(color=(200, 254, 254))
  item_i = game.Item(color=(220, 254, 254))
  item_j = game.Item(color=(254, 254, 254))

  items = {'a': item_a,
           'b': item_b,
           'c': item_c,
           'd': item_d,
           'e': item_e,
           'f': item_f,
           'g': item_g,
           'h': item_h,
           'i': item_i,
           'j': item_j,
          }

  player_a = game.Player(color=(0, 100, 254))
  player_b = game.Player(color=(254, 100, 0))

  players = {'A': player_a, 'B': player_b}

  env = game.Game(art, items, players, tabular=True)
  env.display()

  env.add_reward('A_moves', {'A': -1})
  env.add_reward('B_moves', {'B': -1})
  for i, item in enumerate('abcdefghij'):
    env.add_reward('A_collects_' + item, {'A': (i + 1) * 10})
    env.add_reward('B_collects_' + item, {'B': (i + 1) * 10})
    env.add_terminaison('A_collects_' + item)
    env.add_terminaison('B_collects_' + item)

  # for frame-by-frame visualization:
  env = visualizer.Visualizer(env, fps=2, by_episode=False)

  # for fast visualization:
  # env = visualizer.Visualizer(env, fps=1000, by_episode=True)

  return env


def run_game(env, max_step):
  """Runs `max_step` iterations of the game `env` and print players returns."""

  obs = env.reset()
  # obs2state converts observations into states
  # 'obs' contains all agent x, y positions.
  # 'state' is an integer representing the combination of
  # all agents x, y positions.
  state = env.obs2state(obs)
  transitions = []
  returns = 0
  episode = 0

  for _ in range(max_step):
    # Pick a random action for all agents:
    actions = np.random.choice(range(env.num_actions), env.num_players)
    # Environment step:
    obs, rewards, done, info = env.step(actions)
    new_state = env.obs2state(obs)
    transitions.append((state, new_state, rewards, actions, done, info))
    state = new_state
    # Sum rewards:
    returns += rewards

    if done:
      # The last episode is finished:
      episode += 1
      print('episode', episode, 'returns', returns)
      # Reset env for new episode
      obs = env.reset()
      # state = env.obs2state(obs)
      returns = 0

  # Close visualizer:
  env.finish()

if __name__ == '__main__':
  game_env = create_game()
  run_game(game_env, max_step=200000)

