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
"""DDPM model.

This code is the FLAX equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""

import flax.deprecated.nn as nn
import jax.numpy as jnp

from . import utils, layers, normalization


RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ddpm')
class DDPM(nn.Module):
  """DDPM model architecture."""

  def apply(self, x, labels, config, train=True):
    # config parsing
    nf = config.model.nf
    act = get_act(config)
    normalize = get_normalization(config)
    sigmas = utils.get_sigmas(config)

    nf = config.model.nf
    ch_mult = config.model.ch_mult
    num_res_blocks = config.model.num_res_blocks
    attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    num_resolutions = len(ch_mult)

    # timestep/scale embedding
    timesteps = labels  # sigmas[labels] / jnp.max(sigmas)
    temb = layers.get_timestep_embedding(timesteps, nf)
    temb = nn.Dense(temb, nf * 4, kernel_init=default_initializer())
    temb = nn.Dense(act(temb), nf * 4, kernel_init=default_initializer())

    AttnBlock = layers.AttnBlock.partial(normalize=normalize)

    if config.model.conditional:
      # Condition on noise levels.
      ResnetBlock = ResnetBlockDDPM.partial(
          act=act, normalize=normalize, dropout=dropout, temb=temb, train=train)
    else:
      # Do not condition on noise levels explicitly.
      ResnetBlock = ResnetBlockDDPM.partial(
          act=act, normalize=normalize, dropout=dropout, temb=None, train=train)

    if config.data.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    # Downsampling block
    hs = [conv3x3(h, nf)]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        h = ResnetBlock(hs[-1], out_ch=nf * ch_mult[i_level])
        if h.shape[1] in attn_resolutions:
          h = AttnBlock(h)
        hs.append(h)
      if i_level != num_resolutions - 1:
        hs.append(Downsample(hs[-1], with_conv=resamp_with_conv))

    h = hs[-1]
    h = ResnetBlock(h)
    h = AttnBlock(h)
    h = ResnetBlock(h)

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        h = ResnetBlock(
            jnp.concatenate([h, hs.pop()], axis=-1),
            out_ch=nf * ch_mult[i_level])
      if h.shape[1] in attn_resolutions:
        h = AttnBlock(h)
      if i_level != 0:
        h = Upsample(h, with_conv=resamp_with_conv)

    assert not hs

    h = act(normalize(h))
    h = conv3x3(h, x.shape[-1], init_scale=0.)

    if config.model.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = sigmas[labels].reshape((x.shape[0],
                                            *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h
