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
"""ResNet architectures."""
from flax.deprecated import nn

import jax.nn
from jax.nn import initializers
import jax.numpy as jnp


class BottleneckBlock(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self, x, filters, strides=(1, 1), groups=1, base_width=64,
            train=True):
    needs_projection = x.shape[-1] != filters * 4 or strides != (1, 1)
    width = int(filters * (base_width / 64.)) * groups
    batch_norm = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5)
    y = nn.Conv(x, width, (1, 1), (1, 1), bias=False, name='conv1')
    y = batch_norm(y, name='bn1')
    y = jax.nn.relu(y)
    y = nn.Conv(y, width, (3, 3), strides, bias=False,
                feature_group_count=groups, name='conv2')
    y = batch_norm(y, name='bn2')
    y = jax.nn.relu(y)
    y = nn.Conv(y, filters * 4, (1, 1), (1, 1), bias=False, name='conv3')
    y = batch_norm(y, name='bn3', scale_init=initializers.zeros)
    if needs_projection:
      x = nn.Conv(x, filters * 4, (1, 1), strides,
                  bias=False, name='proj_conv')
      x = batch_norm(x, name='proj_bn')
    return jax.nn.relu(x + y)


class AbstractResNet(nn.Module):
  """ResNetV1."""

  BLOCK_SIZES = None
  GROUPS = 1
  WIDTH_PER_GROUP = 64
  NUM_FILTERS = 64

  def apply(self, x, num_outputs, train=True):
    x = nn.Conv(x, self.NUM_FILTERS, (7, 7), (2, 2), bias=False,
                name='init_conv')
    x = nn.BatchNorm(x,
                     use_running_average=not train,
                     momentum=0.9, epsilon=1e-5,
                     name='init_bn')
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.BLOCK_SIZES):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = BottleneckBlock(x, self.NUM_FILTERS * 2 ** i,
                            strides=strides, groups=self.GROUPS,
                            base_width=self.WIDTH_PER_GROUP, train=train)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(x, num_outputs, name='clf')
    return x


class ResNet50(AbstractResNet):
  BLOCK_SIZES = [3, 4, 6, 3]


class ResNet101(AbstractResNet):
  BLOCK_SIZES = [3, 4, 23, 3]


class ResNet152(AbstractResNet):
  BLOCK_SIZES = [3, 8, 36, 3]


class ResNet50x2(AbstractResNet):
  BLOCK_SIZES = [3, 4, 6, 3]
  NUM_FILTERS = 128


class ResNet101x2(AbstractResNet):
  BLOCK_SIZES = [3, 4, 23, 3]
  NUM_FILTERS = 128


class ResNet152x2(AbstractResNet):
  BLOCK_SIZES = [3, 8, 36, 3]
  NUM_FILTERS = 128


class ResNext50_32x4d(AbstractResNet):  # pylint: disable=invalid-name
  BLOCK_SIZES = [3, 4, 6, 3]
  GROUPS = 32
  WIDTH_PER_GROUP = 4


class ResNext101_32x8d(AbstractResNet):  # pylint: disable=invalid-name
  BLOCK_SIZES = [3, 4, 23, 3]
  GROUPS = 32
  WIDTH_PER_GROUP = 8


class ResNext152_32x4d(AbstractResNet):  # pylint: disable=invalid-name
  BLOCK_SIZES = [3, 8, 36, 3]
  GROUPS = 32
  WIDTH_PER_GROUP = 4
