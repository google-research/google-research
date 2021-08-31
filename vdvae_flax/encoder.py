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

"""Implementation of a VDVAE encoder."""

from typing import Mapping, Optional, Sequence, Tuple

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from vdvae_flax import blocks as blocks_lib


class Encoder(nn.Module):
  """The Encoder of a VDVAE, mapping from images to latents."""

  num_blocks: int
  num_channels: int
  bottlenecked_num_channels: int
  downsampling_rates: Sequence[Tuple[int, int]]
  precision: Optional[jax.lax.Precision] = None
  """Builds a VDVAE encoder.

  Args:
    num_blocks: number of residual blocks in the encoder.
    num_channels: number of channels output by each of the residual blocks.
    bottlenecked_num_channels: number of channels used internally by each
      residual block.
    downsampling_rates: a sequence of tuples (block index, downsampling rate).
      Blocks whose indices are not in this sequence conserve the resolution.
    precision: Optional :class:`jax.lax.Precision` to pass to convolutions.
    name: name of the haiku module.
  """

  def setup(self):
    self._in_conv = blocks_lib.get_vdvae_convolution(
        self.num_channels, (3, 3), name='in_conv', precision=self.precision)

    sampling_rates = sorted(self.downsampling_rates)
    num_blocks = self.num_blocks

    current_sequence_start = 0
    blocks = []
    for block_idx, rate in sampling_rates:
      if rate == 1:
        continue
      sequence_length = block_idx - current_sequence_start
      if sequence_length > 0:
        # Add sequence of non-downsampling blocks as a single layer stack.
        for i in range(current_sequence_start, block_idx):
          blocks.append(
              blocks_lib.ResBlock(
                  self.bottlenecked_num_channels,
                  self.num_channels,
                  downsampling_rate=1,
                  use_residual_connection=True,
                  last_weights_scale=np.sqrt(1.0 / self.num_blocks),
                  precision=self.precision,
                  name=f'res_block_{i}'))

      # Add downsampling block
      blocks.append(
          blocks_lib.ResBlock(
              self.bottlenecked_num_channels,
              self.num_channels,
              downsampling_rate=rate,
              use_residual_connection=True,
              last_weights_scale=np.sqrt(1.0 / self.num_blocks),
              precision=self.precision,
              name=f'res_block_{block_idx}'))
      # Update running parameters
      current_sequence_start = block_idx + 1
    # Add remaining blocks after last downsampling block
    sequence_length = num_blocks - current_sequence_start
    if sequence_length > 0:
      # Add sequence of non-downsampling blocks as a single layer stack.
      for i in range(current_sequence_start, num_blocks):
        blocks.append(
            blocks_lib.ResBlock(
                self.bottlenecked_num_channels,
                self.num_channels,
                downsampling_rate=1,
                use_residual_connection=True,
                last_weights_scale=np.sqrt(1.0 / self.num_blocks),
                precision=self.precision,
                name=f'res_block_{i}'))

    self._blocks = blocks

  def __call__(
      self,
      inputs,
      context_vectors = None,
  ):
    """Encodes a batch of input images.

    Args:
      inputs: a batch of input images of shape [B, H, W, C]. They should be
        centered and of type float32.
      context_vectors: optional batch of shape [B, D]. These are typically used
        to condition the VDVAE.

    Returns:
      a mapping from resolution to encoded image.
    """

    if inputs.dtype != jnp.float32:
      raise ValueError('Expected inputs to be of type float32 but got '
                       f'{inputs.dtype}')
    if len(inputs.shape) != 4 or inputs.shape[1] != inputs.shape[2]:
      raise ValueError('inputs should be a batch of images of shape '
                       f'[B, H, W, C] with H=W, but got {inputs.shape}')
    outputs = self._in_conv(inputs)
    resolution = outputs.shape[1]
    activations = {resolution: outputs}

    for block in self._blocks:
      outputs = block(outputs, context_vectors)
      resolution = outputs.shape[1]
      activations[resolution] = outputs

    return activations
