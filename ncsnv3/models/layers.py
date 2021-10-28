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

# pylint: skip-file
"""Common layers for defining score networks.
"""
import functools
import math
import string

import flax.deprecated.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp


def get_act(config):
  """Get activation functions from the config file."""

  if config.model.nonlinearity.lower() == 'elu':
    return nn.elu
  elif config.model.nonlinearity.lower() == 'relu':
    return nn.relu
  elif config.model.nonlinearity.lower() == 'lrelu':
    return functools.partial(nn.leaky_relu, negative_slope=0.2)
  elif config.model.nonlinearity.lower() == 'swish':
    return nn.swish
  else:
    raise NotImplementedError('activation function does not exist!')


def ncsn_conv1x1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """1x1 convolution with PyTorch initialization. Same as NCSNv1/v2."""
  init_scale = 1e-10 if init_scale == 0 else init_scale
  kernel_init = jnn.initializers.variance_scaling(1 / 3 * init_scale, 'fan_in',
                                                  'uniform')
  kernel_shape = (1, 1) + (x.shape[-1], out_planes)
  bias_init = lambda key, shape: kernel_init(key, kernel_shape)[0, 0, 0, :]
  output = nn.Conv(x, out_planes, kernel_size=(1, 1),
                   strides=(stride, stride), padding='SAME', bias=bias,
                   kernel_dilation=(dilation, dilation),
                   kernel_init=kernel_init,
                   bias_init=bias_init)
  return output


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return jnn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ddpm_conv1x1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """1x1 convolution with DDPM initialization."""
  bias_init = jnn.initializers.zeros
  output = nn.Conv(x, out_planes, kernel_size=(1, 1),
                   strides=(stride, stride), padding='SAME', bias=bias,
                   kernel_dilation=(dilation, dilation),
                   kernel_init=default_init(init_scale),
                   bias_init=bias_init)
  return output


def ncsn_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
  init_scale = 1e-10 if init_scale == 0 else init_scale
  kernel_init = jnn.initializers.variance_scaling(1 / 3 * init_scale, 'fan_in',
                                                  'uniform')
  kernel_shape = (3, 3) + (x.shape[-1], out_planes)
  bias_init = lambda key, shape: kernel_init(key, kernel_shape)[0, 0, 0, :]
  output = nn.Conv(
      x,
      out_planes,
      kernel_size=(3, 3),
      strides=(stride, stride),
      padding='SAME',
      bias=bias,
      kernel_dilation=(dilation, dilation),
      kernel_init=kernel_init,
      bias_init=bias_init)
  return output


def ddpm_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """3x3 convolution with DDPM initialization."""
  bias_init = jnn.initializers.zeros
  output = nn.Conv(
      x,
      out_planes,
      kernel_size=(3, 3),
      strides=(stride, stride),
      padding='SAME',
      bias=bias,
      kernel_dilation=(dilation, dilation),
      kernel_init=default_init(init_scale),
      bias_init=bias_init)
  return output


###########################################################################
# Functions below are ported over from the NCSNv1/NCSNv2 codebase:
# https://github.com/ermongroup/ncsn
# https://github.com/ermongroup/ncsnv2
###########################################################################


class CRPBlock(nn.Module):
  """CRPBlock for RefineNet. Used in NCSNv2."""

  def apply(self, x, features, n_stages, act=nn.relu):
    x = act(x)
    path = x
    for _ in range(n_stages):
      path = nn.max_pool(
          path, window_shape=(5, 5), strides=(1, 1), padding='SAME')
      path = ncsn_conv3x3(path, features, stride=1, bias=False)
      x = path + x
    return x


class CondCRPBlock(nn.Module):
  """Noise-conditional CRPBlock for RefineNet. Used in NCSNv1."""

  def apply(self, x, y, features, n_stages, normalizer, act=nn.relu):
    x = act(x)
    path = x
    for _ in range(n_stages):
      path = normalizer(path, y)
      path = nn.avg_pool(
          path, window_shape=(5, 5), strides=(1, 1), padding='SAME')
      path = ncsn_conv3x3(path, features, stride=1, bias=False)
      x = path + x
    return x


class RCUBlock(nn.Module):
  """RCUBlock for RefineNet. Used in NCSNv2."""

  def apply(self, x, features, n_blocks, n_stages, act=nn.relu):
    for _ in range(n_blocks):
      residual = x
      for _ in range(n_stages):
        x = act(x)
        x = ncsn_conv3x3(x, features, stride=1, bias=False)
      x = x + residual

    return x


class CondRCUBlock(nn.Module):
  """Noise-conditional RCUBlock for RefineNet. Used in NCSNv1."""

  def apply(self, x, y, features, n_blocks, n_stages, normalizer, act=nn.relu):
    for _ in range(n_blocks):
      residual = x
      for _ in range(n_stages):
        x = normalizer(x, y)
        x = act(x)
        x = ncsn_conv3x3(x, features, stride=1, bias=False)
      x += residual
    return x


class MSFBlock(nn.Module):
  """MSFBlock for RefineNet. Used in NCSNv2."""

  def apply(self, xs, shape, features, interpolation='bilinear'):
    sums = jnp.zeros((xs[0].shape[0], *shape, features))
    for i in range(len(xs)):
      h = ncsn_conv3x3(xs[i], features, stride=1, bias=True)
      if interpolation == 'bilinear':
        h = jax.image.resize(h, (h.shape[0], *shape, h.shape[-1]), 'bilinear')
      elif interpolation == 'nearest_neighbor':
        h = jax.image.resize(h, (h.shape[0], *shape, h.shape[-1]), 'nearest')
      else:
        raise ValueError(f'Interpolation {interpolation} does not exist!')
      sums = sums + h
    return sums


class CondMSFBlock(nn.Module):
  """Noise-conditional MSFBlock for RefineNet. Used in NCSNv1."""

  def apply(self, xs, y, shape, features, normalizer, interpolation='bilinear'):
    sums = jnp.zeros((xs[0].shape[0], *shape, features))
    for i in range(len(xs)):
      h = normalizer(xs[i], y)
      h = ncsn_conv3x3(h, features, stride=1, bias=True)
      if interpolation == 'bilinear':
        h = jax.image.resize(h, (h.shape[0], *shape, h.shape[-1]), 'bilinear')
      elif interpolation == 'nearest_neighbor':
        h = jax.image.resize(h, (h.shape[0], *shape, h.shape[-1]), 'nearest')
      else:
        raise ValueError(f'Interpolation {interpolation} does not exist')
      sums = sums + h
    return sums


class RefineBlock(nn.Module):
  """RefineBlock for building NCSNv2 RefineNet."""

  def apply(self,
            xs,
            output_shape,
            features,
            act=nn.relu,
            interpolation='bilinear',
            start=False,
            end=False):
    rcu_block = RCUBlock.partial(n_blocks=2, n_stages=2, act=act)
    rcu_block_output = RCUBlock.partial(
        features=features, n_blocks=3 if end else 1, n_stages=2, act=act)
    hs = []
    for i in range(len(xs)):
      h = rcu_block(xs[i], features=xs[i].shape[-1])
      hs.append(h)

    if not start:
      msf = MSFBlock.partial(features=features, interpolation=interpolation)
      h = msf(hs, output_shape)
    else:
      h = hs[0]

    crp = CRPBlock.partial(features=features, n_stages=2, act=act)
    h = crp(h)
    h = rcu_block_output(h)
    return h


class CondRefineBlock(nn.Module):
  """Noise-conditional RefineBlock for building NCSNv1 RefineNet."""

  def apply(self, xs, y, output_shape,
            features, normalizer, act=nn.relu, interpolation='bilinear',
            start=False, end=False):
    rcu_block = CondRCUBlock.partial(n_blocks=2, n_stages=2, act=act,
                                     normalizer=normalizer)
    rcu_block_output = CondRCUBlock.partial(features=features,
                                            n_blocks=3 if end else 1,
                                            n_stages=2, act=act,
                                            normalizer=normalizer)
    hs = []
    for i in range(len(xs)):
      h = rcu_block(xs[i], y, features=xs[i].shape[-1])
      hs.append(h)

    if not start:
      msf = CondMSFBlock.partial(features=features, interpolation=interpolation,
                                 normalizer=normalizer)
      h = msf(hs, y, output_shape)
    else:
      h = hs[0]

    crp = CondCRPBlock.partial(features=features, n_stages=2, act=act,
                               normalizer=normalizer)
    h = crp(h, y)
    h = rcu_block_output(h, y)
    return h


class ConvMeanPool(nn.Module):
  """ConvMeanPool for building the ResNet backbone."""

  def apply(self, inputs, output_dim, kernel_size=3, biases=True):
    output = nn.Conv(
        inputs,
        features=output_dim,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding='SAME',
        bias=biases)
    output = sum([
        output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
        output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]
    ]) / 4.
    return output


class MeanPoolConv(nn.Module):
  """MeanPoolConv for building the ResNet backbone."""

  def apply(self, inputs, output_dim, kernel_size=3, biases=True):
    output = inputs
    output = sum([
        output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
        output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]
    ]) / 4.
    output = nn.Conv(
        output,
        features=output_dim,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding='SAME',
        bias=biases)
    return output


class ResidualBlock(nn.Module):
  """The residual block for defining the ResNet backbone. Used in NCSNv2."""

  def apply(self, x, output_dim, resample=None, act=nn.elu,
            normalization=None, dilation=1):

    h = normalization(x)
    h = act(h)
    if resample == 'down':
      h = ncsn_conv3x3(h, h.shape[-1], dilation=dilation)
      h = normalization(h)
      h = act(h)
      if dilation > 1:
        h = ncsn_conv3x3(h, output_dim, dilation=dilation)
        shortcut = ncsn_conv3x3(x, output_dim, dilation=dilation)
      else:
        h = ConvMeanPool(h, output_dim)
        shortcut = ConvMeanPool(x, output_dim, kernel_size=1)
    elif resample is None:
      if dilation > 1:
        if output_dim == x.shape[-1]:
          shortcut = x
        else:
          shortcut = ncsn_conv3x3(x, output_dim, dilation=dilation)
        h = ncsn_conv3x3(h, output_dim, dilation=dilation)
        h = normalization(h)
        h = act(h)
        h = ncsn_conv3x3(h, output_dim, dilation=dilation)
      else:
        if output_dim == x.shape[-1]:
          shortcut = x
        else:
          shortcut = ncsn_conv1x1(x, output_dim)
        h = ncsn_conv3x3(h, output_dim)
        h = normalization(h)
        h = act(h)
        h = ncsn_conv3x3(h, output_dim)

    return h + shortcut


class ConditionalResidualBlock(nn.Module):
  """The noise-conditional residual block for building NCSNv1."""

  def apply(self, x, y, output_dim, resample=None, act=nn.elu,
            normalization=None, dilation=1):

    h = normalization(x, y)
    h = act(h)
    if resample == 'down':
      h = ncsn_conv3x3(h, h.shape[-1], dilation=dilation)
      h = normalization(h, y)
      h = act(h)
      if dilation > 1:
        h = ncsn_conv3x3(h, output_dim, dilation=dilation)
        shortcut = ncsn_conv3x3(x, output_dim, dilation=dilation)
      else:
        h = ConvMeanPool(h, output_dim)
        shortcut = ConvMeanPool(x, output_dim, kernel_size=1)
    elif resample is None:
      if dilation > 1:
        if output_dim == x.shape[-1]:
          shortcut = x
        else:
          shortcut = ncsn_conv3x3(x, output_dim, dilation=dilation)
        h = ncsn_conv3x3(h, output_dim, dilation=dilation)
        h = normalization(h, y)
        h = act(h)
        h = ncsn_conv3x3(h, output_dim, dilation=dilation)
      else:
        if output_dim == x.shape[-1]:
          shortcut = x
        else:
          shortcut = ncsn_conv1x1(x, output_dim)
        h = ncsn_conv3x3(h, output_dim)
        h = normalization(h, y)
        h = act(h)
        h = ncsn_conv3x3(h, output_dim)

    return h + shortcut


###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jnp.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


class AttnBlock(nn.Module):
  """Channel-wise self-attention block."""

  def apply(self, x, normalize):
    B, H, W, C = x.shape
    h = normalize(x)
    q = NIN(h, C)
    k = NIN(h, C)
    v = NIN(h, C)

    w = jnp.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = jnp.reshape(w, (B, H, W, H * W))
    w = jax.nn.softmax(w, axis=-1)
    w = jnp.reshape(w, (B, H, W, H, W))
    h = jnp.einsum('bhwHW,bHWc->bhwc', w, v)
    h = NIN(h, C, init_scale=0.)
    return x + h


class Upsample(nn.Module):

  def apply(self, x, with_conv=False):
    B, H, W, C = x.shape
    h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), 'nearest')
    if with_conv:
      h = ddpm_conv3x3(h, C)
    return h


class Downsample(nn.Module):

  def apply(self, x, with_conv=False):
    B, H, W, C = x.shape
    if with_conv:
      x = ddpm_conv3x3(x, C, stride=2)
    else:
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
    assert x.shape == (B, H // 2, W // 2, C)
    return x


def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return jnp.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):

  def apply(self, x, num_units, init_scale=1.):
    in_dim = int(x.shape[-1])
    W = self.param(
        'W',
        shape=(in_dim, num_units),
        initializer=default_init(scale=init_scale))
    b = self.param('b', shape=(num_units,), initializer=jnn.initializers.zeros)
    y = contract_inner(x, W) + b
    assert y.shape == x.shape[:-1] + (num_units,)
    return y


class ResnetBlockDDPM(nn.Module):
  """The ResNet Blocks used in DDPM."""

  def apply(self,
            x,
            act,
            normalize,
            temb=None,
            out_ch=None,
            conv_shortcut=False,
            dropout=0.5,
            train=True):
    B, H, W, C = x.shape
    out_ch = out_ch if out_ch else C
    h = act(normalize(x))
    h = ddpm_conv3x3(h, out_ch)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += nn.Dense(
          act(temb), out_ch, kernel_init=default_init())[:, None, None, :]
    h = act(normalize(h))
    h = nn.dropout(h, dropout, deterministic=not train)
    h = ddpm_conv3x3(h, out_ch, init_scale=0.)
    if C != out_ch:
      if conv_shortcut:
        x = ddpm_conv3x3(x, out_ch)
      else:
        x = NIN(x, out_ch)
    return x + h
