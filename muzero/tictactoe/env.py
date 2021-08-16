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

# Lint as: python3
"""TicTacToe env factory."""

from absl import flags
from absl import logging
import gym
import numpy as np

from muzero import core as mzcore

flags.DEFINE_integer('env_board_size', 3, 'TicTacToe board size (one side).')
flags.DEFINE_integer(
    'env_opponent_level', 2, 'Opponent difficulty level.'
    '0 for random, 1 for try to win, 2 for try to win and block player.')
flags.DEFINE_bool('env_compact', False, 'Use compact board representation.')

FLAGS = flags.FLAGS

# Environment constants.
PLAYER = 1
OPPONENT = -1
NEUTRAL = 0


def get_descriptor():
  if FLAGS.env_compact:
    observation_space = gym.spaces.Box(
        -1, 1, (FLAGS.env_board_size, FLAGS.env_board_size, 1), np.float32)
  else:
    observation_space = gym.spaces.Box(
        0, 1, (FLAGS.env_board_size, FLAGS.env_board_size, 3), np.float32)

  return mzcore.EnvironmentDescriptor(
      observation_space=observation_space,
      action_space=gym.spaces.Discrete(FLAGS.env_board_size**2),
      reward_range=mzcore.Range(-2., 1.),
      value_range=mzcore.Range(-2., 1.),
  )


class TicTacToeEnv(gym.Env):
  """TicTacToe environment."""

  def __init__(
      self,
      compact,
      board_size,
      opponent_level,
      board=None,
      random_state=None,
  ):
    super().__init__()
    self.descriptor = get_descriptor()
    self._board = board
    self._rand = np.random.RandomState()
    if random_state:
      self._rand.set_state(random_state)
    self._compact = compact
    self._board_size = board_size
    self._opponent_level = opponent_level

  def _obs(self):
    if self._compact:
      return self._board[:, :, None].astype(np.float32)
    else:
      return np.stack([
          self._board == PLAYER,
          self._board == NEUTRAL,
          self._board == OPPONENT,
      ], -1).astype(np.float32) * 2. - 1.

  def _check_win(self, player):
    target = self._board_size * player
    if (np.any(np.sum(self._board, 0) == target) or
        np.any(np.sum(self._board, 1) == target) or
        np.sum(np.diag(self._board)) == target or
        np.sum(np.diag(self._board[::-1])) == target):
      return True
    else:
      return False

  def _action_to_rc(self, action):
    return action // self._board_size, action % self._board_size

  def _possible_actions(self):
    return np.where(np.ravel(self._board) == NEUTRAL)[0]

  def _is_done(self):
    return self._check_win(PLAYER) or self._check_win(OPPONENT) or np.sum(
        self._board == NEUTRAL) == 0

  def _apply_action(self, action, player=PLAYER):
    ar, ac = self._action_to_rc(action)
    self._board[ar, ac] = player

  def _get_opponent_action(self):
    possible_actions = self._possible_actions()
    # pylint: disable=protected-access
    if self._opponent_level >= 1:
      # Look for winning move.
      for a in possible_actions:
        g = self._copy()
        g._apply_action(a, OPPONENT)
        if g._check_win(OPPONENT):
          return a
    if self._opponent_level >= 2:
      # Look for losing move.
      for a in possible_actions:
        g = self._copy()
        g._apply_action(a, PLAYER)
        if g._check_win(PLAYER):
          return a
    # pylint: enable=protected-access
    return self._rand.choice(possible_actions)

  def _copy(self):
    new_env = TicTacToeEnv(
        board=np.copy(self._board),
        random_state=self._rand.get_state(),
        compact=self._compact,
        board_size=self._board_size,
        opponent_level=self._opponent_level,
    )
    return new_env

  def seed(self, seed=None):
    self._rand = np.random.RandomState(seed)

  def reset(self):
    self._board = np.ones(
        (self._board_size, self._board_size), np.int32) * NEUTRAL
    return self._obs(), {}

  def step(self, action, training_steps=0):
    if self._is_done():
      return self._obs(), 0., True, {}
    if action not in self._possible_actions():
      return self._obs(), -2., True, {}
    self._apply_action(action, PLAYER)
    if self._check_win(PLAYER):
      return self._obs(), 1., True, {}
    if self._is_done():
      return self._obs(), 0., True, {}
    opponent_action = self._get_opponent_action()
    self._apply_action(opponent_action, OPPONENT)
    if self._check_win(OPPONENT):
      return self._obs(), -1., True, {}
    if self._is_done():
      return self._obs(), 0., True, {}
    return self._obs(), 0., False, {}

  def render(self, mode='human'):
    if mode == 'human':
      print(self._board)
      return
    elif mode == 'rgb_array':
      return np.stack([self._board] * 3, -1)
    elif mode == 'ansi':
      return str(self._board)
    else:
      raise ValueError('mode not found: {}'.format(mode))


def create_environment(task, training=True):  # pylint: disable=missing-docstring,unused-argument
  logging.info('Creating environment: tictactoe')
  env = TicTacToeEnv(
      compact=FLAGS.env_compact,
      board_size=FLAGS.env_board_size,
      opponent_level=FLAGS.env_opponent_level,
  )
  env.seed(task)
  return env
