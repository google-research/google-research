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

"""Building blocks for VDVAE."""

from typing import Optional, Tuple

import chex
from flax import linen as nn
import jax

_NUM_CONV_LAYER_PER_BLOCK = 4


def get_vdvae_convolution(output_channels,
                          kernel_shape,
                          weights_scale = 1.,
                          name = None,
                          precision = None):
  """Builds a 2D convolution.

  Args:
    output_channels: number of output channels.
    kernel_shape: shape of convolutional kernel.
    weights_scale: scale of initial weights in the convolution.
    name: name of the module.
    precision: jax precision.

  Returns:
    a nn.Conv2D.
  """
  kernel_init = nn.initializers.variance_scaling(
      scale=weights_scale, mode='fan_in', distribution='truncated_normal')
  return nn.Conv(
      features=output_channels,
      kernel_size=kernel_shape,
      strides=(1, 1),
      padding='SAME',
      use_bias=True,
      kernel_init=kernel_init,
      name=name,
      precision=precision)


class ResBlock(nn.Module):
  """Residual block from the VDVAE paper.

  This block is made of four convolutions, followed by an optional residual
  connection and an optional average pooling to downsample the image.
  Compared to the paper, it uses the same gelu non-linearity but no batch
  normalization.
  It also accepts as an optional input an auxiliary batch of context vectors to
  be processed by 1x1 convolutions. This is typically useful to condition a VAE
  on an embedded context.
  """

  internal_channels: int
  output_channels: int
  downsampling_rate: int = 1
  use_residual_connection: bool = False
  last_weights_scale: float = 1.
  precision: Optional[jax.lax.Precision] = None

  @nn.compact
  def __call__(
      self,
      inputs,
      context_vectors = None,
  ):
    """Applies the res block to input images.

    Args:
      inputs: a rank-4 array of input images of shape (B, H, W, C).
      context_vectors: optional auxiliary inputs, typically used for
        conditioning. If set, they should be of rank 2, and their first (batch)
        dimension should match that of `inputs`. Their number of features is
        arbitrary. They will be reshaped from (B, D) to (B, 1, 1, D) and a 1x1
        convolution will be applied to them.

    Returns:
      a the rank-4 output of the block.
    """
    if self.downsampling_rate < 1:
      raise ValueError('downsampling_rate should be >= 1, but got '
                       f'{self.downsampling_rate}.')

    def build_layers(inputs):
      """Build layers of the ResBlock given a batch of inputs."""
      resolution = inputs.shape[1]
      if resolution > 2:
        kernel_shapes = ((1, 1), (3, 3), (3, 3), (1, 1))
      else:
        kernel_shapes = ((1, 1), (1, 1), (1, 1), (1, 1))

      conv_layers = []
      aux_conv_layers = []
      for layer_idx, kernel_shape in enumerate(kernel_shapes):
        is_last = layer_idx == _NUM_CONV_LAYER_PER_BLOCK - 1
        num_channels = self.output_channels if is_last else self.internal_channels
        weights_scale = self.last_weights_scale if is_last else 1.
        conv_layers.append(
            get_vdvae_convolution(
                num_channels,
                kernel_shape,
                weights_scale,
                name=f'c{layer_idx}',
                precision=self.precision))
        aux_conv_layers.append(
            get_vdvae_convolution(
                num_channels, (1, 1),
                0.,
                name=f'aux_c{layer_idx}',
                precision=self.precision))

      return conv_layers, aux_conv_layers

    chex.assert_rank(inputs, 4)
    if inputs.shape[1] != inputs.shape[2]:
      raise ValueError('VDVAE only works with square images, but got '
                       f'rectangular images of shape {inputs.shape[1:3]}.')
    if context_vectors is not None:
      chex.assert_rank(context_vectors, 2)
      inputs_batch_dim = inputs.shape[0]
      aux_batch_dim = context_vectors.shape[0]
      if inputs_batch_dim != aux_batch_dim:
        raise ValueError('Context vectors batch dimension is incompatible '
                         'with inputs batch dimension. Got '
                         f'{aux_batch_dim} vs {inputs_batch_dim}.')
      context_vectors = context_vectors[:, None, None, :]

    conv_layers, aux_conv_layers = build_layers(inputs)

    outputs = inputs
    for conv, auxiliary_conv in zip(conv_layers, aux_conv_layers):
      outputs = conv(jax.nn.gelu(outputs))
      if context_vectors is not None:
        outputs += auxiliary_conv(context_vectors)

    if self.use_residual_connection:
      in_channels = inputs.shape[-1]
      out_channels = outputs.shape[-1]
      if in_channels != out_channels:
        raise AssertionError('Cannot apply residual connection because the '
                             'number of output channels differs from the '
                             'number of input channels: '
                             f'{out_channels} vs {in_channels}.')
      outputs += inputs
    if self.downsampling_rate > 1:
      shape = (self.downsampling_rate, self.downsampling_rate)
      outputs = nn.avg_pool(
          outputs, window_shape=shape, strides=shape, padding='VALID')
    return outputs
