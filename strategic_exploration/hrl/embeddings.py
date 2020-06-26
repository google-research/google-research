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

import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from strategic_exploration.hrl import abstract_state as AS
from strategic_exploration.hrl.worker import Goal
from gtd.ml.torch.utils import GPUVariable, try_gpu


class StateEmbedder(nn.Module):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def forward(self, states):
    """Embeds a batch of states and returns a batch of embeddings.

        Args:
            states (list[State]): batch_size State objects

        Returns:
            Variable[FloatTensor]: (batch_size, embed_dim)
        """
    raise NotImplementedError()

  @abc.abstractproperty
  def embed_dim(self):
    raise NotImplementedError()


class PixelStateEmbedder(StateEmbedder):
  """Implements the DQN Pixel state embedder.

  Expects an 84 x 84 x 4 image.
    Returns a 512 dim embedding.
    """

  def __init__(self):
    super(PixelStateEmbedder, self).__init__()
    self._layer1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
    # Pad so that layer2 outputs 10 x 10 x 64
    self._layer2_pad = nn.ZeroPad2d((1, 2, 1, 2))
    self._layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self._layer3 = nn.Conv2d(64, 64, padding=(1, 1), kernel_size=3, stride=1)
    self._layer4 = nn.Linear(10 * 10 * 64, self.embed_dim)

  def forward(self, states):
    states = GPUVariable(
        torch.FloatTensor(np.stack(state.pixel_state for state in states)))
    states = states / 255.
    hidden = F.relu(self._layer1(states))
    hidden = F.relu(self._layer2(self._layer2_pad(hidden)))
    hidden = F.relu(self._layer3(hidden))
    hidden = hidden.view((-1, 10 * 10 * 64))  # flatten
    return F.relu(self._layer4(hidden))

  @property
  def embed_dim(self):
    return 512


class LocalPixelStateEmbedder(StateEmbedder):
  """Expects an 84 x 84 x 4 image.

  Crops the image to a horizontal strip
    centered around the agent's y position. Returns a 512 dim embedding.
    """

  def __init__(self):
    super(LocalPixelStateEmbedder, self).__init__()
    self._layer1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
    # Pad so that layer2 outputs 10 x 10 x 64
    self._layer2_pad = nn.ReflectionPad2d((1, 1, 0, 0))
    self._layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self._layer3 = nn.Conv2d(64, 64, padding=(1, 1), kernel_size=3, stride=1)
    self._layer4 = nn.Linear(10 * 64, self.embed_dim)

  def forward(self, states):

    def crop(state):
      abstract_state = AS.AbstractState(state)
      y = int(abstract_state.pixel_y * 84. / 210.)
      cropped = state.pixel_state[:, y - 10:y + 10, :]
      return cropped

    states = GPUVariable(
        torch.FloatTensor(np.stack(crop(state) for state in states)))
    states = states / 255.
    hidden = F.relu(self._layer1(states))
    hidden = F.relu(self._layer2(self._layer2_pad(hidden)))
    hidden = F.relu(self._layer3(hidden))
    hidden = hidden.view((-1, 10 * 64))  # flatten
    return F.relu(self._layer4(hidden))

  @property
  def embed_dim(self):
    return 512


class ExtraLocalPixelStateEmbedder(StateEmbedder):
  """Expects an 84 x 84 x 4 image.

  Crops the image to a (20 x 60) rectangle
    centered around the agent's position. Returns a 512 dim embedding.
    """

  def __init__(self):
    super(ExtraLocalPixelStateEmbedder, self).__init__()
    self._layer1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
    self._layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self._layer3 = nn.Conv2d(64, 64, padding=(1, 0), kernel_size=3, stride=1)
    self._layer4 = nn.Linear(4 * 64, self.embed_dim)

  def forward(self, states):

    def crop(state):
      abstract_state = AS.AbstractState(state)
      y = int(abstract_state.pixel_y * 84. / 210.)
      x = int(abstract_state.pixel_x * 84. / 160.)
      cropped = state.pixel_state[:, y - 10:y + 10, max(0, x - 30):x + 30]
      padding = (0, 0)
      if x - 30 < 0:
        padding = (30 - x, 0)
      elif x + 30 > 160:
        padding = (0, 190 - x)
      cropped = torch.FloatTensor(cropped)
      cropped = try_gpu(
          torch.nn.functional.pad(cropped, padding, mode="reflect"))
      return cropped

    states = torch.stack([crop(state) for state in states])
    states = states / 255.
    hidden = F.relu(self._layer1(states))
    hidden = F.relu(self._layer2(hidden))
    hidden = F.relu(self._layer3(hidden))
    hidden = hidden.view((-1, 4 * 64))  # flatten
    return F.relu(self._layer4(hidden))

  @property
  def embed_dim(self):
    return 512


"""Embeds RAM Atari state into a 128 dim Tensor following:

   https://arxiv.org/pdf/1605.01335.pdf.
"""


class RAMStateEmbedder(StateEmbedder):

  def __init__(self):
    super(RAMStateEmbedder, self).__init__()
    self._layer1 = nn.Linear(128, 128)
    self._layer2 = nn.Linear(128, 128)
    self._layer3 = nn.Linear(128, 128)
    self._layer4 = nn.Linear(128, self.embed_dim)

  def forward(self, states):
    states = GPUVariable(
        torch.FloatTensor(np.stack(state.ram_state for state in states)))
    states /= 255.

    hidden = F.relu(self._layer1(states))
    hidden = F.relu(self._layer2(hidden))
    hidden = F.relu(self._layer3(hidden))
    hidden = F.relu(self._layer4(hidden))
    return hidden

  @property
  def embed_dim(self):
    return 128


class GoalRewardEmbedder(StateEmbedder):
  """Embeds AbstractState as goal, and cumulative reward separately."""

  def __init__(self):
    super(GoalRewardEmbedder, self).__init__()
    self._reward_embedder = nn.Linear(5, 32)
    self._layer1 = nn.Linear(Goal.size() - 1, 96)
    self._layer2 = nn.Linear(128, self.embed_dim)

  def forward(self, states):
    cum_rewards = torch.LongTensor(
        np.stack(state.goal.cum_reward for state in states))
    cum_rewards = cum_rewards.view(-1, 1)
    states = GPUVariable(
        torch.FloatTensor(
            np.stack(state.goal.all_but_cum_reward for state in states)))
    reward_one_hot = torch.FloatTensor(cum_rewards.shape[0], 5)
    reward_one_hot.zero_()
    reward_one_hot.scatter_(1, cum_rewards, 1)
    reward_one_hot = GPUVariable(reward_one_hot)
    reward_embed = F.relu(self._reward_embedder(reward_one_hot))
    state_embed = F.relu(self._layer1(states))
    hidden = torch.cat([state_embed, reward_embed], dim=1)
    output = F.relu(self._layer2(hidden))
    return output

  @property
  def embed_dim(self):
    return 64


class IgnoreRewardGoalEmbedder(StateEmbedder):
  """Embeds goals, but ignored the cumulative reward."""

  def __init__(self):
    super(IgnoreRewardGoalEmbedder, self).__init__()
    self._layer1 = nn.Linear(Goal.size() - 1, 128)
    self._layer2 = nn.Linear(128, self.embed_dim)

  def forward(self, states):
    states = GPUVariable(
        torch.FloatTensor(
            np.stack(state.goal.all_but_cum_reward for state in states)))
    hidden = F.relu(self._layer1(states))
    output = F.relu(self._layer2(hidden))
    return output

  @property
  def embed_dim(self):
    return 32


class GoalEmbedder(StateEmbedder):
  """Embeds AbstractStates as goals."""

  def __init__(self):
    super(GoalEmbedder, self).__init__()
    self._layer1 = nn.Linear(Goal.size(), 128)
    self._layer2 = nn.Linear(128, self.embed_dim)

  def forward(self, states):
    # Normal embeddings
    states = GPUVariable(
        torch.FloatTensor(np.stack(state.goal.numpy() for state in states)))
    hidden = F.relu(self._layer1(states))
    output = F.relu(self._layer2(hidden))
    return output

  @property
  def embed_dim(self):
    return 32


class ConcatenateEmbedder(StateEmbedder):
  """Embeds states by:

        concatenate(embedder1(states), embedder2(states))
    """

  def __init__(self, embedder1, embedder2):
    super(ConcatenateEmbedder, self).__init__()
    self._embedder1 = embedder1
    self._embedder2 = embedder2

  def forward(self, states):
    return torch.cat((self._embedder1(states), self._embedder2(states)), dim=1)

  @property
  def embed_dim(self):
    return self._embedder1.embed_dim + self._embedder2.embed_dim
