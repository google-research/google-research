# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""The NCSNv3 model."""

from . import utils, layers, layersv3, normalization
import flax.deprecated.nn as nn
import jax.numpy as jnp
import numpy as np

ResnetBlockDDPM = layersv3.ResnetBlockDDPMv3
ResnetBlockBigGAN = layersv3.ResnetBlockBigGANv3
Combine = layersv3.Combine
conv3x3 = layersv3.conv3x3
conv1x1 = layersv3.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ncsnv3')
class NCSNv3(nn.Module):
  """NCSNv3 model without continuous noise levels."""

  def apply(self, x, labels, y=None, config=None, train=True):
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

    conditional = config.model.conditional  # noise-conditional
    fir = config.model.fir
    fir_kernel = config.model.fir_kernel
    skip_rescale = config.model.skip_rescale
    resblock_type = config.model.resblock_type
    progressive = config.model.progressive
    progressive_input = config.model.progressive_input
    init_scale = config.model.init_scale
    assert progressive.lower() in ['none', 'output_skip', 'residual']
    assert config.model.embedding_type.lower() in ['gaussian', 'positional']
    combine_method = config.model.progressive_combine
    combiner = Combine.partial(method=combine_method)

    # timestep/noise_level embedding
    if config.model.embedding_type == 'gaussian':
      # Gaussian Fourier features embeddings.
      used_sigmas = sigmas[labels]
      temb = layersv3.GaussianFourierProjection(
          jnp.log(used_sigmas),
          embedding_size=nf,
          scale=config.model.fourier_scale)
    elif config.model.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, nf)
    else:
      raise ValueError(f'embedding type {config.model.embedding_type} unknown.')

    temb = nn.Dense(temb, nf * 4, kernel_init=default_initializer())
    temb = nn.Dense(act(temb), nf * 4, kernel_init=default_initializer())

    if y is not None:  # class-conditional image generation
      class_embed = nn.Embed(y, config.data.num_classes, nf * 4)
      class_embed = nn.Dense(
          class_embed, nf * 4, kernel_init=default_initializer())
      class_embed = nn.Dense(
          act(class_embed), nf * 4, kernel_init=default_initializer())

      temb += class_embed

    AttnBlock = layersv3.AttnBlockv3.partial(
        normalize=normalize,
        init_scale=init_scale,
        skip_rescale=skip_rescale)

    Upsample = layersv3.Upsample.partial(
        with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
    if progressive == 'output_skip':
      pyramid_upsample = layersv3.Upsample.partial(
          fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = layersv3.Upsample.partial(
          fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = layersv3.Downsample.partial(
        with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      pyramid_downsample = layersv3.Downsample.partial(
          fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = layersv3.Downsample.partial(
          fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = ResnetBlockDDPM.partial(
          act=act,
          normalize=normalize,
          dropout=dropout,
          temb=temb if conditional else None,
          train=train,
          init_scale=init_scale,
          skip_rescale=skip_rescale)

    elif resblock_type == 'biggan':
      ResnetBlock = ResnetBlockBigGAN.partial(
          act=act,
          normalize=normalize,
          temb=temb if conditional else None,
          train=train,
          dropout=dropout,
          fir=fir,
          fir_kernel=fir_kernel,
          init_scale=init_scale,
          skip_rescale=skip_rescale)

    else:
      raise ValueError(f'resblock_type {resblock_type} unrecognized.')

    if not config.data.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block

    input_pyramid = None
    if progressive_input != 'none':
      input_pyramid = x

    hs = [conv3x3(x, nf)]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        h = ResnetBlock(hs[-1], out_ch=nf * ch_mult[i_level])
        if h.shape[1] in attn_resolutions:
          h = AttnBlock(h)
        hs.append(h)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          h = Downsample(hs[-1])
        else:
          h = ResnetBlock(hs[-1], down=True)

        if progressive_input == 'input_skip':
          input_pyramid = pyramid_downsample(input_pyramid)
          h = combiner(input_pyramid, h)

        elif progressive_input == 'residual':
          input_pyramid = pyramid_downsample(
              input_pyramid, out_ch=h.shape[-1])
          if skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = ResnetBlock(h)
    h = AttnBlock(h)
    h = ResnetBlock(h)

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        h = ResnetBlock(
            jnp.concatenate([h, hs.pop()], axis=-1),
            out_ch=nf * ch_mult[i_level])

      if h.shape[1] in attn_resolutions:
        h = AttnBlock(h)

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            pyramid = conv3x3(
                act(normalize(h, num_groups=min(h.shape[-1] // 4, 32))),
                x.shape[-1],
                bias=True,
                init_scale=init_scale)
          elif progressive == 'residual':
            pyramid = conv3x3(
                act(normalize(h, num_groups=min(h.shape[-1] // 4, 32))),
                h.shape[-1],
                bias=True)
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            pyramid = pyramid_upsample(pyramid)
            pyramid = pyramid + conv3x3(
                act(normalize(h, num_groups=min(h.shape[-1] // 4, 32))),
                x.shape[-1],
                bias=True,
                init_scale=init_scale)
          elif progressive == 'residual':
            pyramid = pyramid_upsample(pyramid, out_ch=h.shape[-1])
            if skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          h = Upsample(h)
        else:
          h = ResnetBlock(h, up=True)

    assert not hs

    if progressive == 'output_skip':
      h = pyramid
    else:
      h = act(normalize(h, num_groups=min(h.shape[-1] // 4, 32)))
      h = conv3x3(h, x.shape[-1], init_scale=init_scale)

    if config.model.scale_by_sigma:
      used_sigmas = sigmas[labels].reshape((x.shape[0],
                                            *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h


@utils.register_model(name='ncsnv3_fourier')
class NCSNv3Fourier(nn.Module):
  """NCSNv3 model with continuous noise levels."""

  def apply(self, x, sigmas, y=None, config=None, train=True):
    # config parsing
    nf = config.model.nf
    act = get_act(config)
    normalize = get_normalization(config)

    nf = config.model.nf
    ch_mult = config.model.ch_mult
    num_res_blocks = config.model.num_res_blocks
    attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    num_resolutions = len(ch_mult)

    conditional = config.model.conditional  # noise-conditional
    fir = config.model.fir
    fir_kernel = config.model.fir_kernel
    skip_rescale = config.model.skip_rescale
    resblock_type = config.model.resblock_type
    progressive = config.model.progressive
    progressive_input = config.model.progressive_input
    init_scale = config.model.init_scale
    assert progressive in ['none', 'output_skip', 'residual']
    combine_method = config.model.progressive_combine
    combiner = Combine.partial(method=combine_method)
    fourier_scale = config.model.fourier_scale

    # timestep/scale embedding
    temb = layersv3.GaussianFourierProjection(jnp.log(sigmas), embedding_size=nf,
                                            scale=fourier_scale)
    temb = nn.Dense(temb, nf * 4, kernel_init=default_initializer())
    temb = nn.Dense(act(temb), nf * 4, kernel_init=default_initializer())

    if y is not None:  # class-conditional image generation.
      class_embed = nn.Embed(y, config.data.num_classes, nf * 4)
      class_embed = nn.Dense(
          class_embed, nf * 4, kernel_init=default_initializer())
      class_embed = nn.Dense(
          act(class_embed), nf * 4, kernel_init=default_initializer())

      temb += class_embed

    AttnBlock = layersv3.AttnBlockv3.partial(
        normalize=normalize,
        init_scale=init_scale,
        skip_rescale=skip_rescale)

    Upsample = layersv3.Upsample.partial(
        with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
    if progressive == 'output_skip':
      pyramid_upsample = layersv3.Upsample.partial(
          fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = layersv3.Upsample.partial(
          fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = layersv3.Downsample.partial(
        with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      pyramid_downsample = layersv3.Downsample.partial(
          fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = layersv3.Downsample.partial(
          fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = ResnetBlockDDPM.partial(
          act=act,
          normalize=normalize,
          dropout=dropout,
          temb=temb if conditional else None,
          train=train,
          init_scale=init_scale,
          skip_rescale=skip_rescale)

    elif resblock_type == 'biggan':
      ResnetBlock = ResnetBlockBigGAN.partial(
          act=act,
          normalize=normalize,
          temb=temb if conditional else None,
          train=train,
          dropout=dropout,
          fir=fir,
          fir_kernel=fir_kernel,
          init_scale=init_scale,
          skip_rescale=skip_rescale)

    else:
      raise ValueError(f'resblock_type {resblock_type} unrecognized.')

    if not config.data.centered:
      x = 2 * x - 1.

    # Downsampling block

    input_pyramid = None
    if progressive_input != 'none':
      input_pyramid = x

    hs = [conv3x3(x, nf)]
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        h = ResnetBlock(hs[-1], out_ch=nf * ch_mult[i_level])
        if h.shape[1] in attn_resolutions:
          h = AttnBlock(h)
        hs.append(h)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          h = Downsample(hs[-1])
        else:
          h = ResnetBlock(hs[-1], down=True)

        if progressive_input == 'input_skip':
          input_pyramid = pyramid_downsample(input_pyramid)
          h = combiner(input_pyramid, h)

        elif progressive_input == 'residual':
          input_pyramid = pyramid_downsample(
              input_pyramid, out_ch=h.shape[-1])
          if skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = ResnetBlock(h)
    h = AttnBlock(h)
    h = ResnetBlock(h)

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        h = ResnetBlock(
            jnp.concatenate([h, hs.pop()], axis=-1),
            out_ch=nf * ch_mult[i_level])

      if h.shape[1] in attn_resolutions:
        h = AttnBlock(h)

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            pyramid = conv3x3(
                act(normalize(h, num_groups=min(h.shape[-1] // 4, 32))),
                x.shape[-1],
                bias=True,
                init_scale=init_scale)
          elif progressive == 'residual':
            pyramid = conv3x3(
                act(normalize(h, num_groups=min(h.shape[-1] // 4, 32))),
                h.shape[-1],
                bias=True)
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            pyramid = pyramid_upsample(pyramid)
            pyramid = pyramid + conv3x3(
                act(normalize(h, num_groups=min(h.shape[-1] // 4, 32))),
                x.shape[-1],
                bias=True,
                init_scale=init_scale)
          elif progressive == 'residual':
            pyramid = pyramid_upsample(pyramid, out_ch=h.shape[-1])
            if skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          h = Upsample(h)
        else:
          h = ResnetBlock(h, up=True)

    assert not hs

    if progressive == 'output_skip':
      h = pyramid
    else:
      h = act(normalize(h, num_groups=min(h.shape[-1] // 4, 32)))
      h = conv3x3(h, x.shape[-1], init_scale=init_scale)

    if config.model.scale_by_sigma:
      used_sigmas = sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h
