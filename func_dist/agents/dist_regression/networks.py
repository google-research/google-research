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

"""Neural net definitions for distance regression from images.
"""

from typing import Sequence
import flax.linen as nn


class Encoder(nn.Module):
  """Convolutional network which encodes images into an embedding space."""
  features: Sequence[int]
  kernel_size: int

  @nn.compact
  def __call__(self, x):
    for feat in self.features:
      x = nn.Conv(
          features=feat, kernel_size=(self.kernel_size, self.kernel_size))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = x.reshape((x.shape[0], -1))  # flatten
    return x


class FullyConnectedNet(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x


class CNN(nn.Module):
  """An Encoder and a FullyConnectedNet stacked together."""
  conv_features: Sequence[int]
  kernel_size: int
  dense_features: Sequence[int]

  def setup(self):
    self.encoder = Encoder(self.conv_features, self.kernel_size)
    self.fcs = FullyConnectedNet(self.dense_features)

  def encode(self, x):
    return self.encoder(x)

  def predict_distance(self, x):
    return self.fcs(x)
