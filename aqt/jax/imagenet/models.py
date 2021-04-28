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

"""Flax implementation of ResNet V1 with optional quantization."""

import functools
import typing
from typing import Any, Optional, Tuple, Type

import dataclasses
from flax import linen as nn
import jax.numpy as jnp

from aqt.jax import flax_layers as aqt_flax_layers
from aqt.jax import quant_config
from aqt.jax.flax import struct as flax_struct


dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""

  @dataclass
  class HParams:
    # 'conv_proj' is only used in between different stages of residual blocks
    #   where the Tensor shape needs to be matched for the next stage.
    #   When it is not needed, it is set to None in hparams.
    conv_proj: Optional[aqt_flax_layers.ConvAqt.HParams]
    conv_1: aqt_flax_layers.ConvAqt.HParams
    conv_2: aqt_flax_layers.ConvAqt.HParams
    conv_3: aqt_flax_layers.ConvAqt.HParams

  hparams: HParams
  filters: int
  quant_context: quant_config.QuantContext
  strides: Tuple[int, int]
  train: bool
  dtype: Type[Any]

  @nn.compact
  def __call__(
      self,
      inputs,
  ):
    """Applies a residual block consisting of Conv-Batch Norm-ReLU chains."""
    filters = self.filters
    strides = self.strides
    hparams = self.hparams
    dtype = self.dtype
    train = self.train
    quant_context = self.quant_context

    needs_projection = inputs.shape[-1] != filters * 4 or strides != (1, 1)
    batch_norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=dtype)

    conv = functools.partial(
        aqt_flax_layers.ConvAqt,
        use_bias=False,
        dtype=dtype,
        quant_context=quant_context,
        paxis_name='batch',
        train=train)

    residual = inputs
    if needs_projection:
      if hparams.conv_proj is None:
        raise ValueError(
            'hparams.conv_proj cannot be None if needs_projection is True.')
      assert hparams.conv_proj is not None
      residual = conv(
          features=filters * 4,
          kernel_size=(1, 1),
          strides=strides,
          name='proj_conv',
          hparams=hparams.conv_proj)(
              residual)
      residual = batch_norm(name='proj_bn')(residual)
    else:
      if hparams.conv_proj is not None:
        raise ValueError(
            'hparams.conv_proj must be None if needs_projection is False.')

    y = conv(
        features=filters,
        kernel_size=(1, 1),
        name='conv1',
        hparams=hparams.conv_1)(
            inputs)
    y = batch_norm(name='bn1')(y)
    y = nn.relu(y)
    y = conv(
        features=filters,
        kernel_size=(3, 3),
        strides=strides,
        name='conv2',
        hparams=hparams.conv_2)(
            y)
    y = batch_norm(name='bn2')(y)
    y = nn.relu(y)
    y = conv(
        features=filters * 4,
        kernel_size=(1, 1),
        name='conv3',
        hparams=hparams.conv_3)(
            y)

    y = batch_norm(name='bn3', scale_init=nn.initializers.zeros)(y)
    output = nn.relu(residual + y)
    return output


class ResNet(nn.Module):
  """ResNetV1 with optional quantization."""

  @dataclass
  class HParams:
    dense_layer: aqt_flax_layers.DenseAqt.HParams
    conv_init: aqt_flax_layers.ConvAqt.HParams
    residual_blocks: Tuple[ResidualBlock.HParams, Ellipsis]
    filter_multiplier: float

  num_classes: int
  hparams: HParams
  quant_context: quant_config.QuantContext
  num_filters: int
  train: bool
  dtype: Type[Any]

  @nn.compact
  def __call__(
      self,
      inputs,
  ):
    """Applies ResNet model. Number of residual blocks inferred from hparams."""
    num_classes = self.num_classes
    hparams = self.hparams
    num_filters = self.num_filters
    dtype = self.dtype

    x = aqt_flax_layers.ConvAqt(
        features=num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        dtype=dtype,
        name='init_conv',
        train=self.train,
        quant_context=self.quant_context,
        paxis_name='batch',
        hparams=hparams.conv_init,
    )(
        inputs)
    x = nn.BatchNorm(
        use_running_average=not self.train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=dtype,
        name='init_bn')(
            x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    filter_multiplier = hparams.filter_multiplier
    for i, block_hparams in enumerate(hparams.residual_blocks):
      proj = block_hparams.conv_proj
      # For projection layers (unless it is the first layer), strides = (2, 2)
      if i > 0 and proj is not None:
        filter_multiplier *= 2
        strides = (2, 2)
      else:
        strides = (1, 1)
      x = ResidualBlock(
          filters=int(num_filters * filter_multiplier),
          hparams=block_hparams,
          quant_context=self.quant_context,
          strides=strides,
          train=self.train,
          dtype=dtype)(
              x)
    x = jnp.mean(x, axis=(1, 2))

    x = aqt_flax_layers.DenseAqt(
        features=num_classes,
        dtype=dtype,
        train=self.train,
        quant_context=self.quant_context,
        paxis_name='batch',
        hparams=hparams.dense_layer,
    )(x, padding_mask=None)

    x = jnp.asarray(x, dtype)
    output = nn.log_softmax(x)
    return output
