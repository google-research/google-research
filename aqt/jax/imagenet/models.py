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

import dataclasses
import functools
import typing
from typing import Any, Optional, Tuple, Type, Iterable

from flax import linen as nn
import jax.numpy as jnp

from aqt.jax import flax_layers as aqt_flax_layers
from aqt.jax import quant_config
from aqt.jax.flax import struct as flax_struct


dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass
PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""

  @dataclass  # pylint: disable=missing-class-docstring
  class HParams:
    # 'conv_proj' is only used in between different stages of residual blocks
    #   where the Tensor shape needs to be matched for the next stage.
    #   When it is not needed, it is set to None in hparams.
    conv_proj: Optional[aqt_flax_layers.ConvAqt.HParams]
    conv_se: aqt_flax_layers.ConvAqt.HParams  # unused in this model
    conv_1: aqt_flax_layers.ConvAqt.HParams
    conv_2: aqt_flax_layers.ConvAqt.HParams
    conv_3: aqt_flax_layers.ConvAqt.HParams
    act_function: str  # unused in this model
    shortcut_ch_shrink_method: str  # unused in this model
    shortcut_ch_expand_method: str  # unused in this model
    shortcut_spatial_method: str  # unused in this model

  hparams: HParams
  filters: int
  quant_context: quant_config.QuantContext
  strides: Tuple[int, int]
  train: bool
  dtype: Type[Any]
  paxis_name: Optional[str] = 'batch'

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
        paxis_name=self.paxis_name,
        train=train)

    needs_projection = inputs.shape[-1] != filters * 4 or strides != (1, 1)
    assert needs_projection != (hparams.conv_proj is None)

    r1 = inputs
    if needs_projection:
      r1 = conv(
          features=filters * 4,
          kernel_size=(1, 1),
          strides=strides,
          name='proj_conv',
          hparams=hparams.conv_proj)(
              r1)
      r1 = batch_norm(name='proj_bn')(r1)

    def conv_block(
        inputs,
        n,
        features,
        kernel_size,
        strides,
        conv_hparams,
        r1=None,
    ):
      y = conv(
          features=features,
          kernel_size=kernel_size,
          name='conv' + n,
          strides=strides,
          hparams=conv_hparams)(inputs)
      scale_init = nn.initializers.zeros if n == '3' else nn.initializers.ones
      y = batch_norm(name='bn' + n, scale_init=scale_init)(y)
      if r1 is not None:
        y = r1 + y
      y = nn.relu(y)
      return y

    y = conv_block(
        inputs,
        n='1',
        features=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        conv_hparams=hparams.conv_1)

    y = conv_block(
        y,
        n='2',
        features=filters,
        kernel_size=(3, 3),
        strides=strides,
        conv_hparams=hparams.conv_2)

    y = conv_block(
        y,
        n='3',
        features=filters * 4,
        kernel_size=(1, 1),
        strides=(1, 1),
        conv_hparams=hparams.conv_3,
        r1=r1)

    return y


class ResNet(nn.Module):
  """ResNetV1 with optional quantization."""

  @dataclass
  class HParams:
    dense_layer: aqt_flax_layers.DenseAqt.HParams
    conv_init: aqt_flax_layers.ConvAqt.HParams
    residual_blocks: Tuple[ResidualBlock.HParams, Ellipsis]
    filter_multiplier: float
    act_function: str  # unused in this model
    se_ratio: float  # unused in this model
    init_group: int  # unused in this model

  num_classes: int
  hparams: HParams
  quant_context: quant_config.QuantContext
  num_filters: int
  train: bool
  dtype: Type[Any]
  paxis_name: Optional[str] = 'batch'

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
        paxis_name=self.paxis_name,
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
          dtype=dtype)(x)

    x = jnp.mean(x, axis=(1, 2))

    x = aqt_flax_layers.DenseAqt(
        features=num_classes,
        dtype=dtype,
        train=self.train,
        quant_context=self.quant_context,
        paxis_name=self.paxis_name,
        hparams=hparams.dense_layer,
    )(x, padding_mask=None)

    x = jnp.asarray(x, dtype)
    # The output of ViT does not have log_softmax.
    # To make resnet50 teacher has the same type of outputs as ViT,
    # comment out the following line
    # output = nn.log_softmax(x)
    return x


def create_resnet(hparams, dtype, train, **kwargs):
  return ResNet(
      num_classes=1000,
      dtype=dtype,
      hparams=hparams,
      quant_context=quant_config.QuantContext(
          update_bounds=False, quantize_weights=True),
      num_filters=64,
      train=train,
      **kwargs)
