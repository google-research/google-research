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

"""A matplotlib based visualization tool for psycholab environments.

This module introduces a matplotlib based visualization of a set of
agents on a psycholab environment. It takes as input an environment,
and wraps it for a step function that updates a visualization of the
environment's map and the rewards of all agents.
"""

import time
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import pandas as pd


class Visualizer(object):
  """Takes as input a psycholab env and visualize its map and rewards."""

  UNIT_SIZE = 15
  ROWS = 2

  def __init__(
      self, env, fps=1000, by_episode=False, save_video=False, directory=''):
    self.env = env
    self.fps = fps
    self.by_episode = by_episode
    self.rewards = np.zeros(self.env.num_players)
    # Num_players + 1 for the average reward:
    self.rewards_data = [[[], [], []] for _ in range(self.env.num_players + 1)]

    # Fig 1 = env map
    # Fig 2 = players rewards
    self.fig = plt.figure(figsize=(self.UNIT_SIZE, self.UNIT_SIZE * self.ROWS))
    # self.fig.subplots_adjust(
    # left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    self.save_video = save_video
    if self.save_video:
      self.dir = directory
      self.metadata = dict(title='Movie Test', artist='Matplotlib',
                           comment='Movie support!')
      self.writer = FFMpegWriter(fps=10, metadata=self.metadata)
      self.writer.setup(self.fig, self.dir + 'game.mp4', 300)

    # player colors for plot:
    self.average_color = (0, 0, 0)
    self.players_colors = []
    for player_name in self.env.players_order:
      player_color = [
          (255 + c) / (2 * 255)  for c in self.env.players[player_name].color]
      self.players_colors.append(player_color)

    self._init_game()
    self._init_rewards()

    self._eps = 0.1
    self._min = 0
    self._max = 0

    self.env_episodes = 0
    self.freq = 1 / self.fps

    self.fig.tight_layout()
    plt.ion()
    plt.show(block=False)

  @property
  def steps(self):
    return self.env.steps

  @property
  def num_players(self):
    return self.env.num_players

  @property
  def num_actions(self):
    return self.env.num_actions

  @property
  def num_states(self):
    return self.env.num_states

  def _init_game(self):
    """Initializes the current game map."""

    self.game_axes = plt.subplot(211)
    self.game_image = self.game_axes.imshow(
        self.env.render(), interpolation='nearest')

  def _init_rewards(self):
    """Initializes the plots of the evolution of all players rewards."""

    self.reward_axes = plt.subplot(212)
    self.reward_axes.title.set_text('players returns')
    self.reward_axes.set_xlabel('env episodes')
    self.reward_plots = []
    for player_color in self.players_colors:
      self.reward_plots.append(
          self.reward_axes.plot([], [], lw=2, color=player_color))

    # Average reward plot:
    self.reward_plots.append(
        self.reward_axes.plot([], [], lw=2, color=self.average_color))

  def _smooth_data(self, data, percentage=10):
    """Uses percentage% last episodes moving average to smooth."""

    win_size = int(self.env_episodes / percentage) + 1
    reward_array = np.array(data)
    average = pd.Series(reward_array).rolling(
        window=win_size).mean().iloc[win_size-1:].values
    smoothed = np.zeros(self.env_episodes)
    smoothed[-len(average):] = average
    return smoothed

  def _update(self, done, infos):
    """Update the current visualization of the game."""

    # The game map:
    self.game_axes.title.set_text(infos)
    self.game_image.set_data(self.env.render())

    # Players rewards:
    if done:
      self.env_episodes += 1

      for player, reward in enumerate(self.rewards):
        self.rewards_data[player][0].append(self.env_episodes)
        self.rewards_data[player][1].append(reward)
        self.rewards_data[player][2] = self._smooth_data(
            self.rewards_data[player][1])

      self.rewards_data[-1][0].append(self.env_episodes)
      self.rewards_data[-1][1].append(self.rewards.mean())
      self.rewards_data[-1][2] = self._smooth_data(self.rewards_data[-1][1])

      # Update min and max values for reward plots:
      big_array = np.stack([data[2] for data in self.rewards_data])
      self._max = np.max(big_array)
      self._min = np.min(big_array)

      for player_plot, reward_data in zip(self.reward_plots, self.rewards_data):
        player_plot[0].set_xdata(reward_data[0])
        player_plot[0].set_ydata(reward_data[2])  # uses smoothed value
        self.reward_axes.set_xlim(0, np.max(reward_data[0]) + 1)
        self.reward_axes.set_ylim(self._min - 1, self._max + 1)

      self.rewards *= 0

    self.fig.tight_layout()
    self.fig.canvas.draw()

    if self.save_video:
      self.writer.grab_frame()

  def reset(self):
    self.rewards *= 0
    observation = self.env.reset()
    return observation

  def step(self, action):
    """Do an environment step, update the visualization."""

    observation, rewards, done, infos = self.env.step(action)
    self.rewards += rewards
    if self.by_episode:
      if done:
        self._update(done, infos)
    else:
      self._update(done, infos)

    time.sleep(self.freq)
    return observation, rewards, done, infos

  def discrete_state(self, obs):
    return self.env.discrete_state(obs)

  def finish(self):
    if self.save_video:
      self.writer.finish()
    else:
      pass

