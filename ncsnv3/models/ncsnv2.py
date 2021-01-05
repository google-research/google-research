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
"""The NCSNv2 model."""

import flax.nn as nn

from .utils import get_sigmas, register_model
from .layers import (CondRefineBlock, RefineBlock, ResidualBlock, ncsn_conv3x3,
                     ConditionalResidualBlock, get_act)
from .normalization import get_normalization


CondResidualBlock = ConditionalResidualBlock
conv3x3 = ncsn_conv3x3


def get_network(config):
  if config.data.image_size < 96:
    return NCSNv2.partial(config=config)
  elif 96 <= config.data.image_size <= 128:
    return NCSNv2_128.partial(config=config)
  elif 128 < config.data.image_size <= 256:
    return NCSNv2_256.partial(config=config)
  else:
    raise NotImplementedError(
        f'No network suitable for {config.data.image_size}px implemented yet.')


@register_model(name='ncsnv2_64')
class NCSNv2(nn.Module):
  """NCSNv2 model architecture."""

  def apply(self, x, labels, config, train=True):
    # config parsing
    nf = config.model.nf
    act = get_act(config)
    normalizer = get_normalization(config)
    sigmas = get_sigmas(config)
    interpolation = config.model.interpolation

    if not config.data.centered:
      h = 2 * x - 1.
    else:
      h = x

    h = conv3x3(h, nf, stride=1, bias=True)
    # ResNet backbone
    h = ResidualBlock(h, nf, resample=None, act=act, normalization=normalizer)
    layer1 = ResidualBlock(
        h, nf, resample=None, act=act, normalization=normalizer)
    h = ResidualBlock(
        layer1, 2 * nf, resample='down', act=act, normalization=normalizer)
    layer2 = ResidualBlock(
        h, 2 * nf, resample=None, act=act, normalization=normalizer)
    h = ResidualBlock(
        layer2,
        2 * nf,
        resample='down',
        act=act,
        normalization=normalizer,
        dilation=2)
    layer3 = ResidualBlock(
        h, 2 * nf, resample=None, act=act, normalization=normalizer, dilation=2)
    h = ResidualBlock(
        layer3,
        2 * nf,
        resample='down',
        act=act,
        normalization=normalizer,
        dilation=4)
    layer4 = ResidualBlock(
        h, 2 * nf, resample=None, act=act, normalization=normalizer, dilation=4)
    # U-Net with RefineBlocks
    ref1 = RefineBlock([layer4],
                       layer4.shape[1:3],
                       2 * nf,
                       act=act,
                       interpolation=interpolation,
                       start=True)
    ref2 = RefineBlock([layer3, ref1],
                       layer3.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)
    ref3 = RefineBlock([layer2, ref2],
                       layer2.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)
    ref4 = RefineBlock([layer1, ref3],
                       layer1.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act,
                       end=True)

    h = normalizer(ref4)
    h = act(h)
    h = conv3x3(h, x.shape[-1])

    # When using the DDPM loss, no need of normalizing the output
    if config.model.scale_by_sigma:
      used_sigmas = sigmas[labels].reshape(
          (x.shape[0], *([1] * len(x.shape[1:]))))
      return h / used_sigmas
    else:
      return h


@register_model(name='ncsn')
class NCSN(nn.Module):
  """NCSNv1 model architecture."""

  def apply(self, x, labels, config, train=True):
    # config parsing
    nf = config.model.nf
    act = get_act(config)
    normalizer = get_normalization(config, conditional=True)
    sigmas = get_sigmas(config)
    interpolation = config.model.interpolation

    if not config.data.centered:
      h = 2 * x - 1.
    else:
      h = x

    h = conv3x3(h, nf, stride=1, bias=True)
    # ResNet backbone
    h = CondResidualBlock(
        h, labels, nf, resample=None, act=act, normalization=normalizer)
    layer1 = CondResidualBlock(
        h, labels, nf, resample=None, act=act, normalization=normalizer)
    h = CondResidualBlock(
        layer1,
        labels,
        2 * nf,
        resample='down',
        act=act,
        normalization=normalizer)
    layer2 = CondResidualBlock(
        h, labels, 2 * nf, resample=None, act=act, normalization=normalizer)
    h = CondResidualBlock(
        layer2,
        labels,
        2 * nf,
        resample='down',
        act=act,
        normalization=normalizer,
        dilation=2)
    layer3 = CondResidualBlock(
        h,
        labels,
        2 * nf,
        resample=None,
        act=act,
        normalization=normalizer,
        dilation=2)
    h = CondResidualBlock(
        layer3,
        labels,
        2 * nf,
        resample='down',
        act=act,
        normalization=normalizer,
        dilation=4)
    layer4 = CondResidualBlock(
        h,
        labels,
        2 * nf,
        resample=None,
        act=act,
        normalization=normalizer,
        dilation=4)
    # U-Net with RefineBlocks
    ref1 = CondRefineBlock([layer4],
                           labels,
                           layer4.shape[1:3],
                           2 * nf,
                           act=act,
                           normalizer=normalizer,
                           interpolation=interpolation,
                           start=True)
    ref2 = CondRefineBlock([layer3, ref1],
                           labels,
                           layer3.shape[1:3],
                           2 * nf,
                           normalizer=normalizer,
                           interpolation=interpolation,
                           act=act)
    ref3 = CondRefineBlock([layer2, ref2],
                           labels,
                           layer2.shape[1:3],
                           2 * nf,
                           normalizer=normalizer,
                           interpolation=interpolation,
                           act=act)
    ref4 = CondRefineBlock([layer1, ref3],
                           labels,
                           layer1.shape[1:3],
                           nf,
                           normalizer=normalizer,
                           interpolation=interpolation,
                           act=act,
                           end=True)

    h = normalizer(ref4, labels)
    h = act(h)
    h = conv3x3(h, x.shape[-1])

    # When using the DDPM loss, no need of normalizing the output
    if config.model.scale_by_sigma:
      used_sigmas = sigmas[labels].reshape(
          (x.shape[0], *([1] * len(x.shape[1:]))))
      return h / used_sigmas
    else:
      return h


@register_model(name='ncsnv2_128')
class NCSNv2_128(nn.Module):  # pylint: disable=invalid-name
  """NCSNv2 model architecture for 128px images."""

  def apply(self, x, labels, config, train=True):
    # config parsing
    nf = config.model.nf
    act = get_act(config)
    normalizer = get_normalization(config)
    sigmas = get_sigmas(config)
    interpolation = config.model.interpolation

    if not config.data.centered:
      h = 2 * x - 1.
    else:
      h = x

    h = conv3x3(h, nf, stride=1, bias=True)
    # ResNet backbone
    h = ResidualBlock(h, nf, resample=None, act=act, normalization=normalizer)
    layer1 = ResidualBlock(
        h, nf, resample=None, act=act, normalization=normalizer)
    h = ResidualBlock(
        layer1, 2 * nf, resample='down', act=act, normalization=normalizer)
    layer2 = ResidualBlock(
        h, 2 * nf, resample=None, act=act, normalization=normalizer)
    h = ResidualBlock(
        layer2, 2 * nf, resample='down', act=act, normalization=normalizer)
    layer3 = ResidualBlock(
        h, 2 * nf, resample=None, act=act, normalization=normalizer)
    h = ResidualBlock(
        layer3,
        4 * nf,
        resample='down',
        act=act,
        normalization=normalizer,
        dilation=2)
    layer4 = ResidualBlock(
        h, 4 * nf, resample=None, act=act, normalization=normalizer, dilation=2)
    h = ResidualBlock(
        layer4,
        4 * nf,
        resample='down',
        act=act,
        normalization=normalizer,
        dilation=4)
    layer5 = ResidualBlock(
        h, 4 * nf, resample=None, act=act, normalization=normalizer, dilation=4)
    # U-Net with RefineBlocks
    ref1 = RefineBlock([layer5],
                       layer5.shape[1:3],
                       4 * nf,
                       interpolation=interpolation,
                       act=act,
                       start=True)
    ref2 = RefineBlock([layer4, ref1],
                       layer4.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)
    ref3 = RefineBlock([layer3, ref2],
                       layer3.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)
    ref4 = RefineBlock([layer2, ref3],
                       layer2.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act)
    ref5 = RefineBlock([layer1, ref4],
                       layer1.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act,
                       end=True)

    h = normalizer(ref5)
    h = act(h)
    h = conv3x3(h, x.shape[-1])

    if config.model.scale_by_sigma:
      used_sigmas = sigmas[labels].reshape(
          (x.shape[0], *([1] * len(x.shape[1:]))))
      return h / used_sigmas
    else:
      return h


@register_model(name='ncsnv2_256')
class NCSNv2_256(nn.Module):  # pylint: disable=invalid-name
  """NCSNv2 model architecture for 128px images."""

  def apply(self, x, labels, config, train=True):
    # config parsing
    nf = config.model.nf
    act = get_act(config)
    normalizer = get_normalization(config)
    sigmas = get_sigmas(config)
    interpolation = config.model.interpolation

    if not config.data.centered:
      h = 2 * x - 1.
    else:
      h = x

    h = conv3x3(h, nf, stride=1, bias=True)
    # ResNet backbone
    h = ResidualBlock(h, nf, resample=None, act=act, normalization=normalizer)
    layer1 = ResidualBlock(
        h, nf, resample=None, act=act, normalization=normalizer)
    h = ResidualBlock(
        layer1, 2 * nf, resample='down', act=act, normalization=normalizer)
    layer2 = ResidualBlock(
        h, 2 * nf, resample=None, act=act, normalization=normalizer)
    h = ResidualBlock(
        layer2, 2 * nf, resample='down', act=act, normalization=normalizer)
    layer3 = ResidualBlock(
        h, 2 * nf, resample=None, act=act, normalization=normalizer)
    h = ResidualBlock(
        layer3, 2 * nf, resample='down', act=act, normalization=normalizer)
    layer31 = ResidualBlock(
        h, 2 * nf, resample=None, act=act, normalization=normalizer)
    h = ResidualBlock(
        layer31,
        4 * nf,
        resample='down',
        act=act,
        normalization=normalizer,
        dilation=2)
    layer4 = ResidualBlock(
        h, 4 * nf, resample=None, act=act, normalization=normalizer, dilation=2)
    h = ResidualBlock(
        layer4,
        4 * nf,
        resample='down',
        act=act,
        normalization=normalizer,
        dilation=4)
    layer5 = ResidualBlock(
        h, 4 * nf, resample=None, act=act, normalization=normalizer, dilation=4)
    # U-Net with RefineBlocks
    ref1 = RefineBlock([layer5],
                       layer5.shape[1:3],
                       4 * nf,
                       interpolation=interpolation,
                       act=act,
                       start=True)
    ref2 = RefineBlock([layer4, ref1],
                       layer4.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)
    ref31 = RefineBlock([layer31, ref2],
                        layer31.shape[1:3],
                        2 * nf,
                        interpolation=interpolation,
                        act=act)
    ref3 = RefineBlock([layer3, ref31],
                       layer3.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)
    ref4 = RefineBlock([layer2, ref3],
                       layer2.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act)
    ref5 = RefineBlock([layer1, ref4],
                       layer1.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act,
                       end=True)

    h = normalizer(ref5)
    h = act(h)
    h = conv3x3(h, x.shape[-1])

    if config.model.scale_by_sigma:
      used_sigmas = sigmas[labels].reshape(
          (x.shape[0], *([1] * len(x.shape[1:]))))
      return h / used_sigmas
    else:
      return h
