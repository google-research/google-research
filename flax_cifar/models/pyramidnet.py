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

"""PyramidNet model with ShakeDrop regularization.

Reference:

ShakeDrop Regularization for Deep Residual Learning
Yoshihiro Yamada, Masakazu Iwamura, Takuya Akiba, Koichi Kise
https://arxiv.org/abs/1802.02375

Initially forked from
github.com/google/flax/blob/master/examples/cifar10/models/pyramidnet.py

This implementation mimics the one from
https://github.com/tensorflow/models/blob/master/research/autoaugment/shake_drop.py
that is widely used as a benchmark.

We use kaiming normal initialization for convolutional kernels (mode = fan_out,
gain = 2.0). The final dense layer use a uniform distribution U[-scale, scale]
where scale = 1 / sqrt(num_classes) as per the autoaugment implementation.

It is worth noting that this model is slighly different that the one presented
in the Deep Pyramidal Residual Networks paper
(https://arxiv.org/pdf/1610.02915.pdf), as we round instead of trucating when
computing the number of channels in each block. This results in a model with
roughtly 0.2M additional parameters. Rounding is however the method that was
used in follow up work (https://arxiv.org/abs/1905.00397,
https://arxiv.org/abs/2002.12047) so we keep it for consistency.
"""

from typing import Tuple

from flax import nn
import jax.numpy as jnp

from flax_cifar.models import utils


def _shortcut(x, chn_out, strides
              ):
  """Pyramid Net shortcut.

  Use Average pooling to downsample.
  Use zero-padding to increase channels.

  Args:
    x: Input. Should have shape [batch_size, dim, dim, features]
      where dim is the resolution (width and height if the input is an image).
    chn_out: Expected output channels.
    strides: Output stride.

  Returns:
    Shortcut value for Pyramid Net. Shape will be
    [batch_size, dim, dim, chn_out] if strides = (1, 1) (no downsampling) or
    [batch_size, dim/2, dim/2, chn_out] if strides = (2, 2) (downsampling).
  """
  chn_in = x.shape[3]
  if strides != (1, 1):
    x = nn.avg_pool(x, strides, strides)
  if chn_out != chn_in:
    diff = chn_out - chn_in
    x = jnp.pad(x, [[0, 0], [0, 0], [0, 0], [0, diff]])
  return x


class BottleneckShakeDrop(nn.Module):
  """PyramidNet with Shake-Drop Bottleneck."""

  def apply(self,
            x,
            channels,
            strides,
            prob,
            alpha_min,
            alpha_max,
            beta_min,
            beta_max,
            train = True):
    """Implements the forward pass in the module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, features]
        where dim is the resolution (width and height if the input is an image).
      channels: How many channels to use in the convolutional layers.
      strides: Strides for the pooling.
      prob: Probability of dropping the block (see paper for details).
      alpha_min: See paper.
      alpha_max: See paper.
      beta_min: See paper.
      beta_max: See paper.
      train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.

    Returns:
      The output of the bottleneck block.
    """
    y = utils.activation(x, apply_relu=False, train=train, name='bn_1_pre')
    y = nn.Conv(
        y,
        channels, (1, 1),
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='1x1_conv_contract')
    y = utils.activation(y, train=train, name='bn_1_post')
    y = nn.Conv(
        y,
        channels, (3, 3),
        strides,
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='3x3')
    y = utils.activation(y, train=train, name='bn_2')
    y = nn.Conv(
        y,
        channels * 4, (1, 1),
        padding='SAME',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='1x1_conv_expand')
    y = utils.activation(y, apply_relu=False, train=train, name='bn_3')

    if train:
      y = utils.shake_drop_train(y, prob, alpha_min, alpha_max,
                                 beta_min, beta_max)
    else:
      y = utils.shake_drop_eval(y, prob, alpha_min, alpha_max)

    x = _shortcut(x, channels * 4, strides)
    return x + y


def _calc_shakedrop_mask_prob(curr_layer,
                              total_layers,
                              mask_prob):
  """Calculates drop prob depending on the current layer."""
  return 1 - (float(curr_layer) / total_layers) * mask_prob


class PyramidNetShakeDrop(nn.Module):
  """PyramidNet with Shake-Drop."""

  def apply(self,
            x,
            num_outputs,
            pyramid_alpha = 200,
            pyramid_depth = 272,
            train = True):
    """Implements the forward pass in the module.

    Args:
      x: Input to the module. Should have shape [batch_size, dim, dim, 3]
        where dim is the resolution of the image.
      num_outputs: Dimension of the output of the model (ie number of classes
        for a classification problem).
      pyramid_alpha: See paper.
      pyramid_depth: See paper.
      train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.

    Returns:
      The output of the PyramidNet model, a tensor of shape
        [batch_size, num_classes].
    """
    assert (pyramid_depth - 2) % 9 == 0

    # Shake-drop hyper-params
    mask_prob = 0.5
    alpha_min, alpha_max = (-1.0, 1.0)
    beta_min, beta_max = (0.0, 1.0)

    # Bottleneck network size
    blocks_per_group = (pyramid_depth - 2) // 9
    # See Eqn 2 in https://arxiv.org/abs/1610.02915
    num_channels = 16
    # N in https://arxiv.org/abs/1610.02915
    total_blocks = blocks_per_group * 3
    delta_channels = pyramid_alpha / total_blocks

    x = nn.Conv(
        x,
        16, (3, 3),
        padding='SAME',
        name='init_conv',
        bias=False,
        kernel_init=utils.conv_kernel_init_fn)
    x = utils.activation(x, apply_relu=False, train=train, name='init_bn')

    layer_num = 1

    for block_i in range(blocks_per_group):
      num_channels += delta_channels
      layer_mask_prob = _calc_shakedrop_mask_prob(layer_num, total_blocks,
                                                  mask_prob)
      x = BottleneckShakeDrop(
          x,
          int(round(num_channels)), (1, 1),
          layer_mask_prob,
          alpha_min,
          alpha_max,
          beta_min,
          beta_max,
          train=train)
      layer_num += 1

    for block_i in range(blocks_per_group):
      num_channels += delta_channels
      layer_mask_prob = _calc_shakedrop_mask_prob(
          layer_num, total_blocks, mask_prob)
      x = BottleneckShakeDrop(x, int(round(num_channels)),
                              ((2, 2) if block_i == 0 else (1, 1)),
                              layer_mask_prob,
                              alpha_min, alpha_max, beta_min, beta_max,
                              train=train)
      layer_num += 1

    for block_i in range(blocks_per_group):
      num_channels += delta_channels
      layer_mask_prob = _calc_shakedrop_mask_prob(
          layer_num, total_blocks, mask_prob)
      x = BottleneckShakeDrop(x, int(round(num_channels)),
                              ((2, 2) if block_i == 0 else (1, 1)),
                              layer_mask_prob,
                              alpha_min, alpha_max, beta_min, beta_max,
                              train=train)
      layer_num += 1

    assert layer_num - 1 == total_blocks
    x = utils.activation(x, train=train, name='final_bn')
    x = nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, num_outputs, kernel_init=utils.dense_layer_init_fn)
    return x
