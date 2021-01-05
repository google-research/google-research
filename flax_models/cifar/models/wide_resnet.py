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

"""Wide Resnet Model.

Reference:

Wide Residual Networks, Sergey Zagoruyko, Nikos Komodakis
https://arxiv.org/abs/1605.07146

Initially forked from
github.com/google/flax/blob/master/examples/cifar10/models/wideresnet.py

This implementation mimics the one from
github.com/tensorflow/models/blob/master/research/autoaugment/wrn.py
that is widely used as a benchmark.

It uses identity + zero padding skip connections, with kaiming normal
initialization for convolutional kernels (mode = fan_out, gain=2.0).
The final dense layer uses a uniform distribution U[-scale, scale] where
scale = 1 / sqrt(num_classes) as per the autoaugment implementation.

Using the default initialization instead gives error rates approximately 0.5%
greater on cifar100, most likely because the parameters used in the literature
were finetuned for this particular initialization.

Finally, the autoaugment implementation adds more residual connections between
the groups (instead of just between the blocks as per the original paper and
most implementations). It is possible to safely remove those connections without
degrading the performance, which we do by default to match the original
wideresnet paper. Setting `use_additional_skip_connections` to True will add
them back and then reproduces exactly the model used in autoaugment.
"""

from typing import Tuple

from absl import flags
from flax import nn
from jax import numpy as jnp

from flax_models.cifar.models import utils


FLAGS = flags.FLAGS


flags.DEFINE_bool('use_additional_skip_connections', False,
                  'Set to True to use additional skip connections between the '
                  'resnet groups. This reproduces the autoaugment '
                  'implementation, but these connections are not present in '
                  'most implementations. Removing them does not impact '
                  'the performance of the model.')


def _output_add(block_x, orig_x):
  """Add two tensors, padding them with zeros or pooling them if necessary.

  Args:
    block_x: Output of a resnet block.
    orig_x: Residual branch to add to the output of the resnet block.

  Returns:
    The sum of blocks_x and orig_x. If necessary, orig_x will be average pooled
      or zero padded so that its shape matches orig_x.
  """
  stride = orig_x.shape[-2] // block_x.shape[-2]
  strides = (stride, stride)
  if block_x.shape[-1] != orig_x.shape[-1]:
    orig_x = nn.avg_pool(orig_x, strides, strides)
    channels_to_add = block_x.shape[-1] - orig_x.shape[-1]
    orig_x = jnp.pad(orig_x, [(0, 0), (0, 0), (0, 0), (0, channels_to_add)])
  return block_x + orig_x


class WideResnetBlock(nn.Module):
  """Defines a single WideResnetBlock."""

  def apply(self,
            x,
            channels,
            strides = (1, 1),
            activate_before_residual = False,
            train = True):
    """Implements the forward pass in the module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, features]
        where dim is the resolution (width and height if the input is an image).
      channels: How many channels to use in the convolutional layers.
      strides: Strides for the pooling.
      activate_before_residual: True if the batch norm and relu should be
        applied before the residual branches out (should be True only for the
        first block of the model).
      train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.

    Returns:
      The output of the resnet block.
    """
    if activate_before_residual:
      x = utils.activation(x, train, name='init_bn')
      orig_x = x
    else:
      orig_x = x

    block_x = x
    if not activate_before_residual:
      block_x = utils.activation(block_x, train, name='init_bn')

    block_x = nn.Conv(
        block_x,
        channels, (3, 3),
        strides,
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv1')
    block_x = utils.activation(block_x, train=train, name='bn_2')
    block_x = nn.Conv(
        block_x,
        channels, (3, 3),
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv2')

    return _output_add(block_x, orig_x)


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup."""

  def apply(self,
            x,
            blocks_per_group,
            channels,
            strides = (1, 1),
            activate_before_residual = False,
            train = True):
    """Implements the forward pass in the module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, features]
        where dim is the resolution (width and height if the input is an image).
      blocks_per_group: How many resnet blocks to add to each group (should be
        4 blocks for a WRN28, and 6 for a WRN40).
      channels: How many channels to use in the convolutional layers.
      strides: Strides for the pooling.
      activate_before_residual: True if the batch norm and relu should be
        applied before the residual branches out (should be True only for the
        first group of the model).
      train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.

    Returns:
      The output of the resnet block.
    """
    orig_x = x
    for i in range(blocks_per_group):
      x = WideResnetBlock(
          x,
          channels,
          strides if i == 0 else (1, 1),
          activate_before_residual=activate_before_residual and not i,
          train=train)
    if FLAGS.use_additional_skip_connections:
      x = _output_add(x, orig_x)
    return x


class WideResnet(nn.Module):
  """Defines the WideResnet Model."""

  def apply(self,
            x,
            blocks_per_group,
            channel_multiplier,
            num_outputs,
            train = True):
    """Implements a WideResnet module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, 3]
        where dim is the resolution of the image.
      blocks_per_group: How many resnet blocks to add to each group (should be
        4 blocks for a WRN28, and 6 for a WRN40).
      channel_multiplier: The multiplier to apply to the number of filters in
        the model (1 is classical resnet, 10 for WRN28-10, etc...).
      num_outputs: Dimension of the output of the model (ie number of classes
        for a classification problem).
      train: If False, will use the moving average for batch norm statistics.

    Returns:
      The output of the WideResnet, a tensor of shape [batch_size, num_classes].
    """
    first_x = x
    x = nn.Conv(
        x,
        16, (3, 3),
        padding='SAME',
        name='init_conv',
        kernel_init=utils.conv_kernel_init_fn,
        bias=False)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        16 * channel_multiplier,
        activate_before_residual=True,
        train=train)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        32 * channel_multiplier, (2, 2),
        train=train)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        64 * channel_multiplier, (2, 2),
        train=train)
    if FLAGS.use_additional_skip_connections:
      x = _output_add(x, first_x)
    x = utils.activation(x, train=train, name='pre-pool-bn')
    x = nn.avg_pool(x, x.shape[1:3])
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, num_outputs, kernel_init=utils.dense_layer_init_fn)
    return x
