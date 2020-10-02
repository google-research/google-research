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

# Lint as: python3
"""Wide Resnet Model with ShakeShake regularization.

Reference:

Shake-Shake regularization, Xavier Gastaldi
https://arxiv.org/abs/1705.07485

Initially forked from
github.com/google/flax/blob/master/examples/cifar10/models/wideresnet_shakeshake.py

This implementation mimics the one from
github.com/tensorflow/models/blob/master/research/autoaugment/shake_shake.py
that is widely used as a benchmark.

It uses kaiming normal initialization for convolutional kernels (mode = fan_out
, gain=2.0). The final dense layer use a uniform distribution U[-scale, scale]
where scale = 1 / sqrt(num_classes) as per the autoaugment implementation.

The residual connections follows the implementation of X. Gastaldi (1x1 pooling
with a one pixel offset for half the channels, see 2.1.1 Implementation details
section in the reference above for more details).
"""

from typing import Tuple

from flax import nn
import jax
from jax import numpy as jnp

from flax_models.cifar.models import utils


class Shortcut(nn.Module):
  """Shortcut for residual connections."""

  def apply(self,
            x,
            channels,
            strides = (1, 1),
            train = True):
    """Implements the forward pass in the module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, features]
        where dim is the resolution (width and height if the input is an image).
      channels: How many channels to use in the convolutional layers.
      strides: Strides for the pooling.
      train: If False, will use the moving average for batch norm statistics.

    Returns:
      The output of the resnet block. Will have shape
        [batch_size, dim, dim, channels] if strides = (1, 1) or
        [batch_size, dim/2, dim/2, channels] if strides = (2, 2).
    """

    if x.shape[-1] == channels:
      return x

    # Skip path 1
    h1 = nn.avg_pool(x, (1, 1), strides=strides, padding='VALID')
    h1 = nn.Conv(
        h1,
        channels // 2, (1, 1),
        strides=(1, 1),
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_h1')

    # Skip path 2
    # The next two lines offset the "image" by one pixel on the right and one
    # down (see Shake-Shake regularization, Xavier Gastaldi for details)
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    h2 = jnp.pad(x, pad_arr)[:, 1:, 1:, :]
    h2 = nn.avg_pool(h2, (1, 1), strides=strides, padding='VALID')
    h2 = nn.Conv(
        h2,
        channels // 2, (1, 1),
        strides=(1, 1),
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_h2')
    merged_branches = jnp.concatenate([h1, h2], axis=3)
    return utils.activation(
        merged_branches, apply_relu=False, train=train, name='bn_residual')


class ShakeShakeBlock(nn.Module):
  """Wide ResNet block with shake-shake regularization."""

  def apply(self,
            x,
            channels,
            strides = (1, 1),
            train = True):
    """Implements the forward pass in the module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, features]
        where dim is the resolution (width and height if the input is an image).
      channels: How many channels to use in the convolutional layers.
      strides: Strides for the pooling.
      train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.

    Returns:
      The output of the resnet block. Will have shape
        [batch_size, dim, dim, channels] if strides = (1, 1) or
        [batch_size, dim/2, dim/2, channels] if strides = (2, 2).
    """
    a = b = residual = x

    a = jax.nn.relu(a)
    a = nn.Conv(
        a,
        channels, (3, 3),
        strides,
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_a_1')
    a = utils.activation(a, train=train, name='bn_a_1')
    a = nn.Conv(
        a,
        channels, (3, 3),
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_a_2')
    a = utils.activation(a, apply_relu=False, train=train, name='bn_a_2')

    b = jax.nn.relu(b)
    b = nn.Conv(
        b,
        channels, (3, 3),
        strides,
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_b_1')
    b = utils.activation(b, train=train, name='bn_b_1')
    b = nn.Conv(
        b,
        channels, (3, 3),
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_b_2')
    b = utils.activation(b, apply_relu=False, train=train, name='bn_b_2')

    if train and not self.is_initializing():
      ab = utils.shake_shake_train(a, b)
    else:
      ab = utils.shake_shake_eval(a, b)

    # Apply an up projection in case of channel mismatch.
    residual = Shortcut(residual, channels, strides, train)

    return residual + ab


class WideResnetShakeShakeGroup(nn.Module):
  """Defines a WideResnetGroup."""

  def apply(self,
            x,
            blocks_per_group,
            channels,
            strides = (1, 1),
            train = True):
    """Implements the forward pass in the module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, features]
        where dim is the resolution (width and height if the input is an image).
      blocks_per_group: How many resnet blocks to add to each group (should be
        4 blocks for a WRN28, and 6 for a WRN40).
      channels: How many channels to use in the convolutional layers.
      strides: Strides for the pooling.
      train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.

    Returns:
      The output of the resnet block. Will have shape
        [batch_size, dim, dim, channels] if strides = (1, 1) or
        [batch_size, dim/2, dim/2, channels] if strides = (2, 2).
    """
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
            train = True):
    """Implements a WideResnet with ShakeShake regularization module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, 3]
        where dim is the resolution of the image.
      blocks_per_group: How many resnet blocks to add to each group (should be
        4 blocks for a WRN26 as per standard shake shake implementation).
      channel_multiplier: The multiplier to apply to the number of filters in
        the model (1 is classical resnet, 6 for WRN26-2x6, etc...).
      num_outputs: Dimension of the output of the model (ie number of classes
        for a classification problem).
      train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.

    Returns:
      The output of the WideResnet with ShakeShake regularization, a tensor of
      shape [batch_size, num_classes].
    """
    x = nn.Conv(
        x,
        16, (3, 3),
        padding='SAME',
        kernel_init=utils.conv_kernel_init_fn,
        bias=False,
        name='init_conv')
    x = utils.activation(x, apply_relu=False, train=train, name='init_bn')
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
    return  nn.Dense(x, num_outputs, kernel_init=utils.dense_layer_init_fn)
