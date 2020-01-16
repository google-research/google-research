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

"""PyramidNet with Shake-Drop."""
from architectures import shake

from flax import nn
import jax
import jax.numpy as jnp


def shortcut(x, chn_out, strides):
  """Pyramid Net Shortcut.

  Use Average pooling to downsample
  Use zero-padding to increase channels

  Args:
    x: input
    chn_out: expected number of output channels
    strides: striding applied by block

  Returns:
    shortcut
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

  def apply(self, x, channels, strides, prob, alpha_min, alpha_max,
            beta_min, beta_max, train=True):
    batch_norm = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5)

    y = batch_norm(x, name='bn_1_pre')
    y = nn.Conv(y, channels, (1, 1), padding='SAME', name='1x1_conv_contract')
    y = batch_norm(y, name='bn_1_post')
    y = jax.nn.relu(y)
    y = nn.Conv(y, channels, (3, 3), strides, padding='SAME', name='3x3')
    y = batch_norm(y, name='bn_2')
    y = jax.nn.relu(y)
    y = nn.Conv(y, channels*4, (1, 1), padding='SAME', name='1x1_conv_expand')
    y = batch_norm(y, name='bn_3')

    if train:
      y = shake.shake_drop_train(y, prob, alpha_min, alpha_max,
                                 beta_min, beta_max)
    else:
      y = shake.shake_drop_eval(y, prob, alpha_min, alpha_max)

    x = shortcut(x, channels * 4, strides)
    return x + y


def _calc_shakedrop_mask_prob(curr_layer, total_layers, mask_prob):
  """Calculates drop prob depending on the current layer."""
  return 1 - (float(curr_layer) / total_layers) * mask_prob


class PyramidNetShakeDrop(nn.Module):
  """PyramidNet with Shake-Drop."""

  def apply(self,
            x,
            num_outputs,
            pyramid_alpha=200, pyramid_depth=272,
            train=True):
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

    x = nn.Conv(x, 16, (3, 3), padding='SAME', name='init_conv')
    x = nn.BatchNorm(
        x,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5, name='init_bn')

    layer_num = 1

    for block_i in range(blocks_per_group):
      num_channels += delta_channels
      layer_mask_prob = _calc_shakedrop_mask_prob(
          layer_num, total_blocks, mask_prob)
      x = BottleneckShakeDrop(x, int(num_channels), (1, 1), layer_mask_prob,
                              alpha_min, alpha_max, beta_min, beta_max,
                              train=train)
      layer_num += 1

    for block_i in range(blocks_per_group):
      num_channels += delta_channels
      layer_mask_prob = _calc_shakedrop_mask_prob(
          layer_num, total_blocks, mask_prob)
      x = BottleneckShakeDrop(x, int(num_channels),
                              ((2, 2) if block_i == 0 else (1, 1)),
                              layer_mask_prob,
                              alpha_min, alpha_max, beta_min, beta_max,
                              train=train)
      layer_num += 1

    for block_i in range(blocks_per_group):
      num_channels += delta_channels
      layer_mask_prob = _calc_shakedrop_mask_prob(
          layer_num, total_blocks, mask_prob)
      x = BottleneckShakeDrop(x, int(num_channels),
                              ((2, 2) if block_i == 0 else (1, 1)),
                              layer_mask_prob,
                              alpha_min, alpha_max, beta_min, beta_max,
                              train=train)
      layer_num += 1

    assert layer_num - 1 == total_blocks
    x = nn.BatchNorm(
        x,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5, name='final_bn')
    x = jax.nn.relu(x)
    x = nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, num_outputs)
    return x
