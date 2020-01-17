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

"""Wide Resnet Model with shake-shake regularization."""
from . import shake
from flax import nn
import jax


class ShakeShakeBlock(nn.Module):
  """Wide ResNet block with shake-shake."""

  def apply(self, x, channels, strides=(1, 1), train=True):
    batch_norm = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5)

    a = b = residual = x

    a = jax.nn.relu(a)
    a = nn.Conv(a, channels, (3, 3), strides, padding='SAME', name='conv_a_1')
    a = batch_norm(a, name='bn_a_1')
    a = jax.nn.relu(a)
    a = nn.Conv(a, channels, (3, 3), padding='SAME', name='conv_a_2')
    a = batch_norm(a, name='bn_a_2')

    b = jax.nn.relu(b)
    b = nn.Conv(b, channels, (3, 3), strides, padding='SAME', name='conv_b_1')
    b = batch_norm(b, name='bn_b_1')
    b = jax.nn.relu(b)
    b = nn.Conv(b, channels, (3, 3), padding='SAME', name='conv_b_2')
    b = batch_norm(b, name='bn_b_2')

    if train and not self.is_initializing():
      ab = shake.shake_shake_train(a, b)
    else:
      ab = shake.shake_shake_eval(a, b)

    # Apply an up projection in case of channel mismatch
    if (residual.shape[-1] != channels) or strides != (1, 1):
      residual = nn.Conv(residual, channels, (3, 3), strides, padding='SAME',
                         name='conv_residual')
      residual = batch_norm(residual, name='bn_residual')

    return residual + ab


class WideResnetShakeShakeGroup(nn.Module):
  """Defines a WideResnetGroup."""

  def apply(self,
            x,
            blocks_per_group,
            channels,
            strides=(1, 1),
            train=True):
    for i in range(blocks_per_group):
      x = ShakeShakeBlock(
          x,
          channels,
          strides if i == 0 else (1, 1),
          train=train)
    return x


class WideResnetShakeShake(nn.Module):
  """Defines the WideResnet Model."""

  def apply(self,
            x,
            blocks_per_group,
            channel_multiplier,
            num_outputs,
            train=True):

    x = nn.Conv(
        x, 16, (3, 3), padding='SAME', name='init_conv')
    x = WideResnetShakeShakeGroup(
        x,
        blocks_per_group,
        16 * channel_multiplier,
        train=train)
    x = WideResnetShakeShakeGroup(
        x,
        blocks_per_group,
        32 * channel_multiplier, (2, 2),
        train=train)
    x = WideResnetShakeShakeGroup(
        x,
        blocks_per_group,
        64 * channel_multiplier, (2, 2),
        train=train)
    x = jax.nn.relu(x)
    x = nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, num_outputs)
    return x
