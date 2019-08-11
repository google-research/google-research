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

"""Test psycholab games."""


import unittest
from psycholab import game


class TestGame(unittest.TestCase):
  """Test the game environement as defined in game.py."""

  def test_transition(self):

    # (this is the prisonners dilemma grid game from examples)
    art = ['####d####',
           'a  A B  b',
           '#########'
          ]

    item_a = game.Item(color=(0, 254, 254))
    item_b = game.Item(color=(254, 254, 0))
    item_d = game.Item(color=(254, 254, 254))

    items = {'a': item_a, 'b': item_b, 'd': item_d}

    player_a = game.Player(color=(0, 100, 254))
    player_b = game.Player(color=(254, 100, 0))

    players = {'A': player_a, 'B': player_b}

    env = game.Game(art, items, players, tabular=True, max_steps=50)

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

    _ = env.reset()

    # A goes in wall
    actions = [1, 0]
    _, rewards, _, infos = env.step(actions)
    self.assertIn('A_moves', infos['event_list'])
    self.assertIn('A_goes_in_walls', infos['event_list'])
    self.assertEqual(rewards[0], -1)
    _ = env.reset()

    # A goes in B
    actions = [4, 0]
    _ = env.step(actions)
    actions = [4, 0]
    _, rewards, _, infos = env.step(actions)
    self.assertIn('A_moves', infos['event_list'])
    self.assertIn('A_blocked_by_B', infos['event_list'])
    self.assertEqual(rewards[0], -1)
    _ = env.reset()

    # A and B reach same cell
    actions = [4, 3]
    _, rewards, _, infos = env.step(actions)
    self.assertIn('A_moves', infos['event_list'])
    self.assertIn('B_moves', infos['event_list'])
    self.assertTrue(
        ('A_lost_the_drawn' in infos['event_list'])
        or ('B_lost_the_drawn' in infos['event_list']))
    self.assertEqual(rewards[0], -1)
    self.assertEqual(rewards[1], -1)
    _ = env.reset()

    # A and B block each other
    actions = [4, 0]
    _ = env.step(actions)
    actions = [4, 3]
    _, rewards, _, infos = env.step(actions)
    self.assertIn('A_moves', infos['event_list'])
    self.assertIn('B_moves', infos['event_list'])
    self.assertIn('A_blocked_by_B', infos['event_list'])
    self.assertIn('B_blocked_by_A', infos['event_list'])
    self.assertEqual(rewards[0], -1)
    self.assertEqual(rewards[1], -1)
    _ = env.reset()

    # A reaches reward
    actions = [3, 0]
    _ = env.step(actions)
    actions = [3, 0]
    _ = env.step(actions)
    actions = [3, 0]
    _, rewards, _, infos = env.step(actions)
    self.assertIn('A_moves', infos['event_list'])
    self.assertIn('A_collects_a', infos['event_list'])
    self.assertEqual(rewards[0], 100-1)
    _ = env.reset()


if __name__ == '__main__':
  unittest.main()
