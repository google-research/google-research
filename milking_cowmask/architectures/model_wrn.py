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

"""Wide Resnet Model for small image classification."""
from flax import nn
import jax


class WideResnetBlock(nn.Module):
  """Defines a single wide ResNet block."""

  def apply(self, x, channels, strides=(1, 1), dropout_rate=0.0, train=True):
    batch_norm = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5)

    y = batch_norm(x, name='bn1')
    y = jax.nn.relu(y)
    y = nn.Conv(y, channels, (3, 3), strides, padding='SAME', name='conv1')
    y = batch_norm(y, name='bn2')
    y = jax.nn.relu(y)
    if dropout_rate > 0.0:
      y = nn.dropout(y, dropout_rate, deterministic=not train)
    y = nn.Conv(y, channels, (3, 3), padding='SAME', name='conv2')

    # Apply an up projection in case of channel mismatch
    if (x.shape[-1] != channels) or strides != (1, 1):
      x = nn.Conv(x, channels, (3, 3), strides, padding='SAME')
    return x + y


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup."""

  def apply(self,
            x,
            blocks_per_group,
            channels,
            strides=(1, 1),
            dropout_rate=0.0,
            train=True):
    for i in range(blocks_per_group):
      x = WideResnetBlock(
          x,
          channels,
          strides if i == 0 else (1, 1),
          dropout_rate,
          train=train)
    return x


class WideResnet(nn.Module):
  """Defines the WideResnet Model."""

  def apply(self,
            x,
            blocks_per_group,
            channel_multiplier,
            num_outputs,
            dropout_rate=0.0,
            train=True):

    x = nn.Conv(
        x, 16, (3, 3), padding='SAME', name='init_conv')
    x = WideResnetGroup(
        x,
        blocks_per_group,
        16 * channel_multiplier,
        dropout_rate=dropout_rate,
        train=train)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        32 * channel_multiplier, (2, 2),
        dropout_rate=dropout_rate,
        train=train)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        64 * channel_multiplier, (2, 2),
        dropout_rate=dropout_rate,
        train=train)
    x = nn.BatchNorm(
        x,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5)
    x = jax.nn.relu(x)
    x = nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, num_outputs)
    return x
