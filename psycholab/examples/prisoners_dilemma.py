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

"""prisoners dilemma grid game.

this example comes from the games introduced in paper
A Polynomial-time Nash Equilibrium Algorithm for Repeated Stochastic Games
by Enrique Munoz de Cote and Michael L. Littman
"""

import numpy as np
from psycholab import game
from psycholab import visualizer


def create_game():
  """Create the prisoners dilemma game."""

  art = ['####d####',
         'a  A B  b',
         '#########'
        ]

  item_a = game.Item(color=(0, 254, 254))
  item_b = game.Item(color=(254, 254, 0))
  item_d = game.Item(color=(0, 254, 254))
  items = {'a': item_a, 'b': item_b, 'd': item_d}

  player_a = game.Player(color=(0, 100, 254))
  player_b = game.Player(color=(254, 100, 0))

  players = {'A': player_a, 'B': player_b}

  env = game.Game(art, items, players, tabular=True)
  env.display()

  env.add_reward('A_moves', {'A': -1})
  env.add_reward('B_moves', {'B': -1})
  env.add_reward('A_collects_a', {'A': 100})
  env.add_reward('B_collects_b', {'B': 100})
  env.add_reward('A_collects_d', {'A': 100})
  env.add_reward('B_collects_d', {'B': 100})

  env.add_terminaison('A_collects_d')
  env.add_terminaison('B_collects_d')
  env.add_terminaison('A_collects_a')
  env.add_terminaison('B_collects_b')

  # for frame-by-frame visualization:
  env = visualizer.Visualizer(env, fps=2, by_episode=False)

  # for fast visualization:
  # env = visualizer.Visualizer(env, fps=1000, by_episode=True)

  return env


def run_game(env, max_step):
  """Runs `max_step` iterations of the game `env` and print players returns."""

  obs = env.reset()
  # discrete_state converts observations into states
  # 'obs' contains all agent x, y positions.
  # 'state' is an integer representing the combination of
  # all agents x, y positions.
  state = env.discrete_state(obs)
  transitions = []
  returns = 0
  episode = 0

  for _ in range(max_step):
    # Pick a random action for all agents:
    actions = np.random.choice(range(env.num_actions), env.num_players)
    # Environment step:
    obs, rewards, done, info = env.step(actions)
    new_state = env.discrete_state(obs)
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
      # state = env.discrete_state(obs)
      returns = 0

  # Close visualizer:
  env.finish()

if __name__ == '__main__':
  game_env = create_game()
  run_game(game_env, max_step=200000)
