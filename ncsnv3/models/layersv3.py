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

# pylint: skip-file
"""Layers for defining NCSNv3.
"""
from . import layers
from . import up_or_down_sampling
import flax.nn as nn
import jax
import jax.numpy as jnp
import numpy as np


conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def apply(self, x, embedding_size=256, scale=1.0):
    W = self.param('W', (embedding_size,),
                   jax.nn.initializers.normal(stddev=scale))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def apply(self, x, y, method='cat'):
    h = conv1x1(x, y.shape[-1])
    if method == 'cat':
      return jnp.concatenate([h, y], axis=-1)
    elif method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {method} not recognized.')


class AttnBlockv3(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def apply(self, x, normalize, skip_rescale=False, init_scale=0.):
    B, H, W, C = x.shape
    h = normalize(x, num_groups=min(x.shape[-1] // 4, 32))
    q = NIN(h, C)
    k = NIN(h, C)
    v = NIN(h, C)

    w = jnp.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = jnp.reshape(w, (B, H, W, H * W))
    w = jax.nn.softmax(w, axis=-1)
    w = jnp.reshape(w, (B, H, W, H, W))
    h = jnp.einsum('bhwHW,bHWc->bhwc', w, v)
    h = NIN(h, C, init_scale=init_scale)
    if not skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):

  def apply(self, x, out_ch=None, with_conv=False, fir=False,
            fir_kernel=[1, 3, 3, 1]):
    B, H, W, C = x.shape
    out_ch = out_ch if out_ch else C
    if not fir:
      h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), 'nearest')
      if with_conv:
        h = conv3x3(h, out_ch)
    else:
      if not with_conv:
        h = up_or_down_sampling.upsample_2d(x, fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.Conv2d(
            x,
            out_ch,
            kernel=3,
            up=True,
            resample_kernel=fir_kernel,
            bias=True,
            kernel_init=default_init())

    assert h.shape == (B, 2 * H, 2 * W, out_ch)
    return h


class Downsample(nn.Module):

  def apply(self, x, out_ch=None, with_conv=False, fir=False,
            fir_kernel=[1, 3, 3, 1]):
    B, H, W, C = x.shape
    out_ch = out_ch if out_ch else C
    if not fir:
      if with_conv:
        x = conv3x3(x, out_ch, stride=2)
      else:
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
    else:
      if not with_conv:
        x = up_or_down_sampling.downsample_2d(x, fir_kernel, factor=2)
      else:
        x = up_or_down_sampling.Conv2d(
            x,
            out_ch,
            kernel=3,
            down=True,
            resample_kernel=fir_kernel,
            bias=True,
            kernel_init=default_init())

    assert x.shape == (B, H // 2, W // 2, out_ch)
    return x


class ResnetBlockDDPMv3(nn.Module):
  """ResBlock adapted from DDPM."""

  def apply(self,
            x,
            act,
            normalize,
            temb=None,
            out_ch=None,
            conv_shortcut=False,
            dropout=0.1,
            train=True,
            skip_rescale=False,
            init_scale=0.):
    B, H, W, C = x.shape
    out_ch = out_ch if out_ch else C
    h = act(normalize(x, num_groups=min(x.shape[-1] // 4, 32)))
    h = conv3x3(h, out_ch)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += nn.Dense(
          act(temb), out_ch, kernel_init=default_init())[:, None, None, :]

    h = act(normalize(h, num_groups=min(h.shape[-1] // 4, 32)))
    h = nn.dropout(h, dropout, deterministic=not train)
    h = conv3x3(h, out_ch, init_scale=init_scale)
    if C != out_ch:
      if conv_shortcut:
        x = conv3x3(x, out_ch)
      else:
        x = NIN(x, out_ch)

    if not skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANv3(nn.Module):
  """ResBlock adapted from BigGAN."""

  def apply(self,
            x,
            act,
            normalize,
            up=False,
            down=False,
            temb=None,
            out_ch=None,
            dropout=0.1,
            fir=False,
            fir_kernel=[1, 3, 3, 1],
            train=True,
            skip_rescale=True,
            init_scale=0.):
    B, H, W, C = x.shape
    out_ch = out_ch if out_ch else C
    h = act(normalize(x, num_groups=min(x.shape[-1] // 4, 32)))

    if up:
      if fir:
        h = up_or_down_sampling.upsample_2d(h, fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif down:
      if fir:
        h = up_or_down_sampling.downsample_2d(h, fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    h = conv3x3(h, out_ch)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += nn.Dense(
          act(temb), out_ch, kernel_init=default_init())[:, None, None, :]

    h = act(normalize(h, num_groups=min(h.shape[-1] // 4, 32)))
    h = nn.dropout(h, dropout, deterministic=not train)
    h = conv3x3(h, out_ch, init_scale=init_scale)
    if C != out_ch or up or down:
      x = conv1x1(x, out_ch)

    if not skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
