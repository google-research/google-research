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

"""Common building blocks for various models."""


from typing import Sequence

import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from hct.common import typing


class MLP(nn.Module):
  """Basic MLP."""
  features: Sequence[int]
  activate_final: bool = False
  activation: typing.ActivationFunction = nn.relu

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.activation(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    if self.activate_final:
      x = self.activation(x)
    return x


class ResNetDenseBlock(nn.Module):
  """ResNet block with dense layers."""
  features: int
  activation: typing.ActivationFunction = nn.relu

  def setup(self):
    self.dense0 = nn.Dense(self.features // 4)
    self.dense1 = nn.Dense(self.features // 4)
    self.dense2 = nn.Dense(self.features)
    self.dense3 = nn.Dense(self.features)

  def __call__(self, x):
    y = self.dense0(x)
    y = self.activation(y)
    y = self.dense1(y)
    y = self.activation(y)
    y = self.dense2(y)
    if x.shape != y.shape:
      x = self.dense3(x)
    return self.activation(x + y)


class ResNetDense(nn.Module):
  """ResNet dense module."""
  features: int  # Number of features for each layer.
  depth: int = 8  # Number of layers, equivalent to (N-2)//3 blocks.
  out_dim: int = 1  # Dimension of output.

  activation: typing.ActivationFunction = nn.relu

  def setup(self):
    assert self.depth >= 5
    assert (self.depth - 2) % 3 == 0
    self.num_blocks = (self.depth - 2) // 3
    self.dense0 = nn.Dense(self.features)
    self.blocks = tuple([
        ResNetDenseBlock(self.features, self.activation)
        for _ in range(self.num_blocks)
    ])
    self.dense1 = nn.Dense(self.out_dim)

  def __call__(self, x):
    x = self.dense0(x)
    x = self.activation(x)
    for block in self.blocks:
      x = block(x)
    x = self.dense1(x)
    return x


class ResNetConvBlock(nn.Module):
  """ResNet block (modified to match Transporter Nets implementation)."""

  features: int  # number of filters
  stride: int = 1
  activation: typing.ActivationFunction = nn.relu

  def setup(self):

    self.conv0 = nn.Conv(self.features // 4, (1, 1), (self.stride, self.stride))
    self.conv1 = nn.Conv(self.features // 4, (3, 3))
    self.conv2 = nn.Conv(self.features, (1, 1))
    self.conv3 = nn.Conv(self.features, (1, 1), (self.stride, self.stride))

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.activation(self.conv0(x))
    y = self.activation(self.conv1(y))
    y = self.activation(self.conv2(y))
    if residual.shape != y.shape:
      residual = self.conv3(residual)
    assert y.shape == residual.shape
    return self.activation(residual + y)


class ResNetBatchNorm(nn.Module):
  """ResNet-based Image encoder w/ BatchNorm.

  This models the map: zs = Fs(s).
  """
  embed_dim: int = 64  # dim(zs)
  width: int = 128

  def setup(self):
    def image_map(images):
      zs = nn.Conv(self.width, (3, 3), (1, 1))(images)
      zs = ResNetConvBlock(self.width, stride=2)(zs)
      zs = ResNetConvBlock(self.width)(zs)
      zs = ResNetConvBlock(self.width, stride=2)(zs)
      zs = ResNetConvBlock(self.width)(zs)
      zs = ResNetConvBlock(self.width, stride=2)(zs)
      zs = ResNetConvBlock(self.width)(zs)
      zs = ResNetConvBlock(self.width, stride=2)(zs)
      zs = ResNetConvBlock(self.width)(zs)
      zs = ResNetConvBlock(self.width, stride=2)(zs)
      zs = ResNetConvBlock(self.width)(zs)
      zs = nn.avg_pool(zs, zs.shape[1:-1])
      zs = zs.reshape((zs.shape[0], -1))
      return nn.Dense(self.embed_dim)(zs)
    self.image_map = image_map

  @nn.compact
  def __call__(self, images, train=False):
    """Encode images.

    Args:
      images: ndarray of shape (batch, image_size, image_size, image_channels).
      train: boolean for batchnorm.

    Returns:
      zs: image embedding, ndarray w/ shape: (batch, embed_dim)
    """
    assert len(images.shape) == 4
    image_norm = nn.BatchNorm(use_running_average=not train,
                              momentum=0.9,
                              epsilon=1e-5)(images)
    return self.image_map(image_norm)
