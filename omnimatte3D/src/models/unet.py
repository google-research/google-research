# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""2d Unet with residual blocks."""

from typing import Sequence

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp


@struct.dataclass
class UnetConfig:
  """Config settings for 2-d residual Unets.

  Attributes:
    out_features: number of output feature maps
    feat_scales:  the values specify a multiplicative factor for the  number of
      feature maps used at a given resolution (relative to 'embed_dim')
    num_res_blocks: number of residual blocks used at every resolution
    embed_dim: initial number of feature maps after the input is processed by a
      single convolutional layer
    dim: number of spatial dimensions processed by the network; supported values
      are 2 and 3
    penultimate_nonlinearity: whether to apply a nonlinearity before the last
      layer; recommended True for new models
  """

  out_features: int = 16
  feat_scales: Sequence[int] = (1, 2, 4, 8, 8)
  num_res_blocks: int = 3
  embed_dim: int = 4
  dim: int = 2
  penultimate_nonlinearity: bool = True


def get_unet_config(config):
  return UnetConfig(
      out_features=config.model.unet_out_dim,
      feat_scales=config.model.unet_feat_scales,
      num_res_blocks=config.model.unet_num_res_blocks,
  )


def _default_init(scale = 1e-10):
  return nn.initializers.variance_scaling(
      scale=scale, mode='fan_avg', distribution='uniform'
  )


class DownsampleBlock(nn.Module):
  """Block for the downsampling ('encoder') pathway."""

  # If true, uses convolution with stride 2. Otherwise uses avg pooling.
  use_conv: bool
  dim: int = 3

  @nn.compact
  def __call__(self, x):
    if self.use_conv:
      x = nn.Conv(
          features=x.shape[-1],
          kernel_size=(3, 3, 3)[: self.dim],
          strides=(2, 2, 2)[: self.dim],
      )(x)
    else:
      x = nn.avg_pool(
          x,
          window_shape=(2, 2, 2)[: self.dim],
          strides=(2, 2, 2)[: self.dim],
          padding='same',
      )

    return x


class UpsampleBlock(nn.Module):
  """Block for the upsampling ('decoder') pathway."""

  use_conv: bool
  dim: int = 3

  @nn.compact
  def __call__(self, x):
    batch = x.shape[0]
    features = x.shape[-1]
    if self.dim == 3:
      x = jax.image.resize(
          x,
          shape=(
              batch,
              x.shape[-4] * 2,
              x.shape[-3] * 2,
              x.shape[-2] * 2,
              features,
          ),
          method='nearest',
      )
    else:
      x = jax.image.resize(
          x,
          shape=(batch, x.shape[-3] * 2, x.shape[-2] * 2, features),
          method='nearest',
      )

    if self.use_conv:
      x = nn.Conv(
          features=features,
          kernel_size=(3, 3, 3)[: self.dim],
          strides=(1, 1, 1)[: self.dim],
      )(x)

    return x


class ResBlock(nn.Module):
  """Residual block."""

  out_channel: int
  dim: int
  upsample: bool = False
  downsample: bool = False
  downsample_conv: bool = False
  upsample_conv: bool = False

  @nn.compact
  def __call__(self, x):
    h = nn.swish(x)

    if self.downsample:
      h = DownsampleBlock(use_conv=self.downsample_conv, dim=self.dim)(h)
      x = DownsampleBlock(use_conv=self.downsample_conv, dim=self.dim)(x)
    elif self.upsample:
      h = UpsampleBlock(use_conv=self.upsample_conv, dim=self.dim)(h)
      x = UpsampleBlock(use_conv=self.upsample_conv, dim=self.dim)(x)

    h = nn.Conv(
        features=self.out_channel,
        kernel_size=(3, 3, 3)[: self.dim],
        strides=(1, 1, 1)[: self.dim],
        kernel_init=_default_init(1.0),
        name='conv_1',
    )(h)

    h = nn.swish(h)
    h = nn.Conv(
        features=self.out_channel,
        kernel_size=(3, 3, 3)[: self.dim],
        strides=(1, 1, 1)[: self.dim],
        kernel_init=_default_init(),
        name='conv_2',
    )(h)

    if x.shape[-1] != self.out_channel:
      x = nn.Conv(
          features=self.out_channel,
          kernel_size=(1, 1, 1)[: self.dim],
          strides=(1, 1, 1)[: self.dim],
          kernel_init=_default_init(1.0),
          name='shortcut',
      )(x)

    return h + x


class DownsampleStack(nn.Module):
  """Downsampling ('encoder') pathway."""

  config: UnetConfig

  @nn.compact
  def __call__(self, x):
    h = x
    h_list = [
        nn.Conv(
            features=self.config.embed_dim,
            kernel_size=(3, 3, 3)[: self.config.dim],
            strides=(1, 1, 1)[: self.config.dim],
            kernel_init=_default_init(1.0),
            name='conv_in',
        )(x)
    ]

    num_resolutions = len(self.config.feat_scales)
    for layer in range(num_resolutions):
      for block in range(self.config.num_res_blocks):
        h = ResBlock(
            out_channel=(
                self.config.embed_dim * self.config.feat_scales[layer]
            ),
            dim=self.config.dim,
            name=f'down_{layer}.block_{block}',
        )(h_list[-1])
        h_list.append(h)

      # Downsample.
      if layer != num_resolutions - 1:
        h_list.append(
            ResBlock(
                out_channel=h_list[-1].shape[-1],
                downsample=True,
                dim=self.config.dim,
                name=f'down_{layer}.downsample',
            )(h_list[-1])
        )

    return h_list


class MiddleStack(nn.Module):
  """Middle part of the network."""

  config: UnetConfig

  @nn.compact
  def __call__(self, x):
    h = ResBlock(
        out_channel=x.shape[-1], dim=self.config.dim, name='mid.block_1'
    )(x)
    h = ResBlock(
        out_channel=x.shape[-1], dim=self.config.dim, name='mid.block_2'
    )(h)
    return h


class UpsampleStack(nn.Module):
  """Upsampling ('decoder') pathway."""

  config: UnetConfig

  @nn.compact
  def __call__(self, x, downsample_stack):
    h = x
    for layer in reversed(range(len(self.config.feat_scales))):
      for block in range(self.config.num_res_blocks + 1):
        h = ResBlock(
            out_channel=(
                self.config.embed_dim * self.config.feat_scales[layer]
            ),
            dim=self.config.dim,
            name=f'up_{layer}.{block}',
        )(jnp.concatenate([h, downsample_stack.pop()], axis=-1))

      # Upsample.
      if layer != 0:
        h = ResBlock(
            out_channel=h.shape[-1] * 2,
            upsample=True,
            dim=self.config.dim,
            name=f'up_{layer}.upsample',
        )(h)

    assert not downsample_stack
    return h


class UNet(nn.Module):
  """Encoder-decoder convolutional network."""

  config: UnetConfig

  @nn.compact
  def __call__(self, x):
    """Applies the U-net to input data.

    Args:
      x: [batch, z, y, x, channels]=shaped input

    Returns:
      U-net output.
    """
    h_list = DownsampleStack(self.config)(x)
    h = h_list[-1]
    h = MiddleStack(self.config)(h)
    h = UpsampleStack(self.config)(h, h_list)

    if self.config.penultimate_nonlinearity:
      h = nn.swish(h)

    h = nn.Conv(
        self.config.out_features,
        (3, 3, 3)[: self.config.dim],
        kernel_init=_default_init(),
        name='output',
    )(h)
    return h
