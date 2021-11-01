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

"""Linen version of unet0. Taken from brain/markov_chains."""

# pytype: disable=wrong-keyword-args,wrong-arg-count

from typing import List, Optional

from absl import logging
from flax import linen as nn
import jax.numpy as jnp

from autoregressive_diffusion.experiments.images.architectures import unet_utils
from autoregressive_diffusion.model.architecture_components import input_embedding

nonlinearity = nn.swish
Normalize = nn.normalization.GroupNorm


def nearest_neighbor_upsample(x):
  B, H, W, C = x.shape  # pylint: disable=invalid-name
  x = x.reshape(B, H, 1, W, 1, C)
  x = jnp.broadcast_to(x, (B, H, 2, W, 2, C))
  return x.reshape(B, H * 2, W * 2, C)


class ResnetBlock(nn.Module):
  """Convolutional residual block."""

  dropout: float
  out_ch: Optional[int] = None

  @nn.compact
  def __call__(self, x, *, temb, y, deterministic):
    B, _, _, C = x.shape  # pylint: disable=invalid-name
    assert temb.shape[0] == B and len(temb.shape) == 2
    out_ch = C if self.out_ch is None else self.out_ch

    h = x
    h = nonlinearity(Normalize(name='norm1')(h))
    h = nn.Conv(
        features=out_ch, kernel_size=(3, 3), strides=(1, 1), name='conv1')(h)

    # add in timestep embedding
    h += nn.Dense(
        features=out_ch, name='temb_proj')(nonlinearity(temb))[:, None, None, :]

    # add in class information
    if y is not None:
      assert y.ndim == 2 and y.shape[0] == B
      h += nn.Dense(features=out_ch, name='y_proj')(y)[:, None, None, :]

    h = nonlinearity(Normalize(name='norm2')(h))
    h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
    h = nn.Conv(
        features=out_ch,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv2')(h)

    if C != out_ch:
      x = nn.Dense(features=out_ch, name='nin_shortcut')(x)

    assert x.shape == h.shape
    logging.info('%s: x=%r temb=%r y=%r', self.name, x.shape, temb.shape,
                 None if y is None else y.shape)
    return x + h


class AttnBlock(nn.Module):
  """Self-attention residual block."""

  num_heads: int
  mode: str

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable
    assert C % self.num_heads == 0

    if self.mode == 'row':
      axis = (2,)  # Select only width axis.
    elif self.mode == 'column':
      axis = (1,)  # Select only height axis.
    elif self.mode == 'full':
      axis = (1, 2)  # Select both axes.
    else:
      raise ValueError()

    h = Normalize(name='norm')(x)
    if self.num_heads == 1:
      q = nn.Dense(features=C, name='q')(h)
      k = nn.Dense(features=C, name='k')(h)
      v = nn.Dense(features=C, name='v')(h)
      h = unet_utils.dot_product_attention(
          q[:, :, :, None, :],
          k[:, :, :, None, :],
          v[:, :, :, None, :],
          axis=axis)[:, :, :, 0, :]
      h = nn.Dense(
          features=C, kernel_init=nn.initializers.zeros, name='proj_out')(h)
    else:
      head_dim = C // self.num_heads
      q = nn.DenseGeneral(features=(self.num_heads, head_dim), name='q')(h)
      k = nn.DenseGeneral(features=(self.num_heads, head_dim), name='k')(h)
      v = nn.DenseGeneral(features=(self.num_heads, head_dim), name='v')(h)
      assert q.shape == k.shape == v.shape == (
          B, H, W, self.num_heads, head_dim)
      h = unet_utils.dot_product_attention(q, k, v, axis=axis)
      h = nn.DenseGeneral(
          features=C,
          axis=(-2, -1),
          kernel_init=nn.initializers.zeros,
          name='proj_out')(h)

    assert h.shape == x.shape
    return x + h


class UNet(nn.Module):
  """A UNet architecture."""

  num_classes: int
  ch: int
  out_ch: int
  ch_mult: List[int]
  num_res_blocks: int
  full_attn_resolutions: List[int]
  num_heads: int
  dropout: float
  max_time: float = 1000.

  @nn.compact
  def __call__(self, x, t, mask, train, context=None):
    assert x.dtype == jnp.int32
    B, H, W, _ = x.shape  # pylint: disable=invalid-name
    assert H == W
    assert context is None

    assert t.shape == (B,)

    h_first, temb = input_embedding.InputProcessingImage(
        num_classes=self.num_classes,
        num_channels=self.ch * self.ch_mult[0],
        max_time=self.max_time)(x, t, mask, train)

    # We don't want to access x, t or mask directly, but only their embeddings
    # via h_first and temb.
    del x, t, mask

    assert h_first.dtype in (jnp.float32, jnp.float64)
    num_resolutions = len(self.ch_mult)
    ch = self.ch
    y = None

    # Downsampling
    hs = [h_first]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = ResnetBlock(
            out_ch=ch * self.ch_mult[i_level],
            dropout=self.dropout,
            name=f'down_{i_level}.block_{i_block}')(
                hs[-1], temb=temb, y=y, deterministic=not train)
        if h.shape[1] in self.full_attn_resolutions:
          h = AttnBlock(
              num_heads=self.num_heads,
              mode='full',
              name=f'down_{i_level}.attn_{i_block}')(h)
        hs.append(h)
      # Downsample
      if i_level != num_resolutions - 1:
        hs.append(self._downsample(hs[-1], name=f'down_{i_level}.downsample'))

    # Middle
    h = hs[-1]
    h = ResnetBlock(dropout=self.dropout, name='mid.block_1')(
        h, temb=temb, y=y, deterministic=not train)
    # This attention is done irrespective of attention resolution configs.
    h = AttnBlock(num_heads=self.num_heads, mode='full', name='mid.attn_1')(h)
    h = ResnetBlock(dropout=self.dropout, name='mid.block_2')(
        h, temb=temb, y=y, deterministic=not train)

    # Upsampling
    for i_level in reversed(range(num_resolutions)):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks + 1):
        h = ResnetBlock(
            out_ch=ch * self.ch_mult[i_level],
            dropout=self.dropout,
            name=f'up_{i_level}.block_{i_block}')(
                jnp.concatenate([h, hs.pop()], axis=-1),
                temb=temb, y=y, deterministic=not train)
        if h.shape[1] in self.full_attn_resolutions:
          h = AttnBlock(
              num_heads=self.num_heads,
              mode='full',
              name=f'up_{i_level}.attn_{i_block}')(h)
      # Upsample
      if i_level != 0:
        h = self._upsample(h, name=f'up_{i_level}.upsample')
    assert not hs

    # End.
    h = nonlinearity(Normalize(name='norm_out')(h))

    h = nn.Conv(
        features=self.out_ch,
        kernel_size=(3, 3),
        strides=(1, 1),
        name='conv_out')(h)

    return h

  def _downsample(self, x, name):
    B, H, W, C = x.shape  # pylint: disable=invalid-name
    x = nn.Conv(features=C, kernel_size=(3, 3), strides=(2, 2), name=name)(x)
    assert x.shape == (B, H // 2, W // 2, C)
    return x

  def _upsample(self, x, name):
    B, H, W, C = x.shape  # pylint: disable=invalid-name
    x = nearest_neighbor_upsample(x)
    x = nn.Conv(features=C, kernel_size=(3, 3), strides=(1, 1), name=name)(x)
    assert x.shape == (B, H * 2, W * 2, C)
    return x

