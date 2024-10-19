# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Reward model."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def gen_net(in_size=1024, out_size=1, h_size=512, n_layers=3, activation='sig'):
  """Gets NN."""
  net = []
  for _ in range(n_layers):
    net.append(nn.Linear(in_size, h_size))
    net.append(nn.LeakyReLU())
    in_size = h_size
  net.append(nn.Linear(in_size, out_size))
  if activation == 'tanh':
    net.append(nn.Tanh())
  elif activation == 'sig':
    net.append(nn.Sigmoid())
  elif activation == 'relu':
    net.append(nn.ReLU())
  return net


class RewardModel(nn.Module):
  """Reward model."""

  def __init__(self, in_size=1024, h_size=512, n_layers=3, activation='sig'):
    super().__init__()

    self.in_size = in_size
    self.h_size = h_size
    self.n_layers = n_layers
    self.activation = activation

    self.model = nn.Sequential(
        *gen_net(
            in_size=in_size,
            out_size=1,
            h_size=h_size,
            n_layers=n_layers,
            activation=activation,
        )
    )

  def forward(
      self, text, img
  ):
    input_f = torch.cat([text, img], axis=-1)
    score = self.model(input_f)

    return score


class RewardModelAdapter(nn.Module):
  """Reward model adapter."""

  def __init__(
      self,
      vis_in_size,
      txt_in_size,
      out_size,
      h_size=512,
      n_layers=3,
      activation='no',
  ):
    super().__init__()

    self.vision_model = nn.Sequential(
        *gen_net(
            in_size=vis_in_size,
            out_size=out_size,
            h_size=h_size,
            n_layers=n_layers,
            activation=activation,
        )
    )
    self.text_model = nn.Sequential(
        *gen_net(
            in_size=txt_in_size,
            out_size=out_size,
            h_size=h_size,
            n_layers=n_layers,
            activation=activation,
        )
    )

  def forward(
      self, text, img
  ):
    vis_f = self.vision_model(img)
    txt_f = self.text_model(text)
    return vis_f, txt_f


def leaky_relu(p=0.2):
  return nn.LeakyReLU(p, inplace=True)


class ConditionalLinear(nn.Module):
  """Conditional linear."""

  def __init__(self, num_in, num_out, n_steps):
    super(ConditionalLinear, self).__init__()
    self.num_out = num_out
    self.lin = nn.Linear(num_in, num_out)
    self.embed = nn.Embedding(n_steps, num_out)
    self.embed.weight.data.uniform_()
    torch.nn.init.xavier_normal_(self.lin.weight)

  def forward(self, x, y):
    out = self.lin(x)
    gamma = self.embed(y)
    out = gamma.view(-1, self.num_out) * out
    return out


class Value(nn.Module):
  """Value."""

  def __init__(self, num_steps, img_shape):
    super(Value, self).__init__()
    self.lin1 = ConditionalLinear(int(np.prod(img_shape)), 4096, num_steps)
    self.lin2 = ConditionalLinear(4096, 1024, num_steps)
    self.lin3 = ConditionalLinear(1024, 256, num_steps)
    self.lin4 = nn.Linear(256, 1)
    torch.nn.init.xavier_normal_(self.lin4.weight)

  def forward(self, img, t):
    x = img.view(img.shape[0], -1)
    x = F.relu(self.lin1(x, t))
    x = F.relu(self.lin2(x, t))
    x = F.relu(self.lin3(x, t))
    return self.lin4(x)


class ValueMulti(nn.Module):
  """ValueMulti."""

  def __init__(self, num_steps, img_shape):
    super(ValueMulti, self).__init__()
    self.lin1 = ConditionalLinear(int(np.prod(img_shape)) + 768, 256, num_steps)
    self.lin2 = ConditionalLinear(256, 256, num_steps)
    self.lin3 = ConditionalLinear(256, 256, num_steps)
    self.lin4 = nn.Linear(256, 1)
    torch.nn.init.xavier_normal_(self.lin4.weight)

  def forward(self, img, txt_emb, t):
    x = img.view(img.shape[0], -1)
    x = torch.cat([x, txt_emb], dim=1)
    # x = torch.cat([x, txt_emb], dim=1)
    x = F.relu(self.lin1(x, t))
    x = F.relu(self.lin2(x, t))
    x = F.relu(self.lin3(x, t))
    return self.lin4(x)


class TimeEmbedding(nn.Module):
  """Time embedding."""

  def __init__(self, max_time, embed_dim):
    super(TimeEmbedding, self).__init__()
    self.max_time = max_time
    self.embed_dim = embed_dim
    self.embedding = nn.Embedding(max_time, embed_dim)

  def forward(self, time):
    # time is of shape [batch_size, 1]
    time_embed = self.embedding(time)
    time_embed = time_embed.view(-1, self.embed_dim)
    return time_embed


class TDCNN(nn.Module):
  """TDCNN."""

  def __init__(self, time_dim, max_time, embed_dim, num_classes):
    super(TDCNN, self).__init__()
    self.time_dim = time_dim
    self.max_time = max_time
    self.embed_dim = embed_dim
    self.time_embed = TimeEmbedding(max_time, embed_dim)
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
    self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
    self.fc1 = nn.Linear(32 * ((time_dim - 4) // 2) ** 2 + embed_dim, 64)
    self.fc2 = nn.Linear(64, num_classes)

  def forward(self, x, time):
    # x is of shape [batch_size, 1, time_dim]
    # time is of shape [batch_size, 1]
    x = F.relu(self.conv1(x))
    x = F.max_pool1d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool1d(x, 2)
    x = x.view(-1, 32 * ((self.time_dim - 4) // 2) ** 2)
    time_embed = self.time_embed(time)
    x = torch.cat((x, time_embed), dim=1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
