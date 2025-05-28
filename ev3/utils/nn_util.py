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

"""Flax implementation of MLP and ResNet.

This file has be adapted from flax.examples.imagenet.models.
"""

import functools as ft
from typing import Any, Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any


class MLP(nn.Module):
  """Multi-layer perceptron."""

  layer_widths: Sequence[int]
  num_labels: int

  @nn.compact
  def __call__(self, inputs, **kwargs):
    x = inputs.reshape(inputs.shape[0], -1)
    for i, n_nodes in enumerate(self.layer_widths):
      x = nn.Dense(n_nodes, name=f'layer_{i}')(x)
      x = nn.relu(x)
    y = nn.Dense(self.num_labels, name='output')(x)
    return y


class ResNetBlock(nn.Module):
  """ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable[[Ellipsis], Any]
  strides: tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(
      self,
      x,
  ):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters, (1, 1), self.strides, name='conv_proj'
      )(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable[[Ellipsis], Any]
  strides: tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters * 4, (1, 1), self.strides, name='conv_proj'
      )(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""

  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable[[Ellipsis], Any] = nn.relu
  conv: ModuleDef = nn.Conv

  @nn.compact
  def __call__(self, x, train = True):
    conv = ft.partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = ft.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        axis_name='batch',
    )

    x = conv(
        self.num_filters,
        (7, 7),
        (2, 2),
        padding=[(3, 3), (3, 3)],
        name='conv_init',
    )(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
        )(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


class ResNetCIFAR(nn.Module):
  """ResNetV1."""

  stage_sizes: Sequence[int]
  num_classes: int
  block_cls: ModuleDef = ResNetBlock
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable[[Ellipsis], Any] = nn.relu
  conv: ModuleDef = nn.Conv

  @nn.compact
  def __call__(self, x, train = True):
    conv = ft.partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = ft.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
    )

    x = conv(
        self.num_filters,
        (3, 3),
        (1, 1),
        padding=[(1, 1), (1, 1)],
        name='conv_init',
    )(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
        )(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNetCFBase = ft.partial(ResNetCIFAR, num_filters=4, stage_sizes=[1, 1, 1])
ResNetCF20 = ft.partial(ResNetCIFAR, num_filters=16, stage_sizes=[3, 3, 3])
ResNetCF32 = ft.partial(ResNetCIFAR, num_filters=16, stage_sizes=[5, 5, 5])
ResNetCF44 = ft.partial(ResNetCIFAR, num_filters=16, stage_sizes=[7, 7, 7])
ResNetCF56 = ft.partial(ResNetCIFAR, num_filters=16, stage_sizes=[9, 9, 9])
ResNetCF110 = ft.partial(ResNetCIFAR, num_filters=16, stage_sizes=[18, 18, 18])

ResNet18 = ft.partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = ft.partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = ft.partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
)
ResNet101 = ft.partial(
    ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock
)
ResNet152 = ft.partial(
    ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock
)
ResNet200 = ft.partial(
    ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock
)


ResNet18Local = ft.partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock, conv=nn.ConvLocal
)


# Used for testing only.
_ResNet1 = ft.partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
_ResNet1Local = ft.partial(
    ResNet, stage_sizes=[1], block_cls=ResNetBlock, conv=nn.ConvLocal
)


ModelNameMapping = {
    'ResNet': ResNet,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet50,
    'ResNet152': ResNet152,
    'ResNet200': ResNet200,
    'ResNetCIFAR': ResNetCIFAR,
    'ResNetCFBase': ResNetCFBase,
    'ResNetCF20': ResNetCF20,
    'ResNetCF32': ResNetCF32,
    'ResNetCF44': ResNetCF44,
    'ResNetCF56': ResNetCF56,
    'ResNetCF110': ResNetCF110,
}


def get_resnet_model(
    model_name,
    num_classes,
    **model_kwargs,
):
  """Helper function to get ResNet model by name.

  Args:
    model_name: name of the model.
    num_classes: Number of classification labels.
    **model_kwargs: Arguments to be pass to the model.

  Returns:
    An nn.module object.
  """
  if model_name not in ModelNameMapping:
    raise NotImplementedError(model_name)
  return ModelNameMapping[model_name](num_classes=num_classes, **model_kwargs)
