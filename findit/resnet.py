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

"""Adopted from Flax/Linen implementation of ResNet V1.

Code path: third_party/py/flax/linen_examples/imagenet/models.py.
"""

import functools
from typing import Any, Callable, Tuple, Optional, Dict, Union

from flax import linen as nn
from flax.linen.initializers import he_normal
import jax.numpy as jnp

from findit import base
from findit import model_utils

ModuleDef = Any
Array = jnp.ndarray
LevelArray = Dict[int, Array]


class ResNetBlock(nn.Module):
  """ResNet block.

  Attributes:
    filters: Number of filters in ResNet block.
    conv: A nn.Module defining the conv layer.
    norm: A nn.Module defining the norm layer.
    act: A callable function for activation.
    strides: A tuple of (int, int) defining the strides of the resnet block.
  """
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable[Ellipsis, Any]
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    """ResNet block.

    Args:
      x: A jnp.ndarray of input features of shape [N, H, W, C]..

    Returns:
      x: A jnp.ndarray of output features of shape [N, H, W, C]..
    """
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides, name='conv1')(x)
    y = self.norm(name=self.name + '_conv1_bn')(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), name='conv2')(y)
    y = self.norm(
        scale_init=nn.initializers.zeros, name=self.name + '_conv2_bn')(
            y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters, (1, 1), self.strides, name=self.name + '_proj_conv')(
              residual)
      residual = self.norm(name=self.name + '_proj_bn')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block.


  Attributes:
    filters: `int` number of filters in ResNet block.
    conv: a nn.Module defining the conv layer.
    norm: a nn.Module defining the norm layer.
    act: a callable function for activation.
    strides: a tuple of (int, int) defining the strides for the conv layer.
    batch_norm_momentum: Batch normalization momentum value.
    batch_norm_epsilon: Batch normalization epsilon value.
    dtype: Type of the jnp.arary, default is jnp.float32.
    train: bool, if the mode is training. Only needed if norm is not specified
  """
  filters: int
  conv: Optional[ModuleDef] = None
  norm: Optional[ModuleDef] = None
  act: Callable[Ellipsis, Any] = nn.relu
  strides: Tuple[int, int] = (1, 1)

  batch_norm_momentum: float = 0.997
  batch_norm_epsilon: float = 1e-4
  dtype: jnp.dtype = jnp.float32
  train: Optional[bool] = None

  @nn.compact
  def __call__(self, x):
    """Bottleneck ResNet block.

    Args:
      x: A jnp.ndarray of input features of shape [N, H, W, C].

    Returns:
      x: A jnp.ndarray of output features of shape [N, H, W, C].
    """
    if self.conv is None:
      conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    else:
      conv = self.conv

    if self.norm is None:
      assert self.train is not None
      norm = functools.partial(
          nn.BatchNorm,
          use_running_average=not self.train,
          momentum=self.batch_norm_momentum,
          epsilon=self.batch_norm_epsilon,
          dtype=self.dtype)
    else:
      norm = self.norm

    residual = x
    y = conv(self.filters, (1, 1), name='conv1')(x)
    y = norm(name=self.name + '_conv1_bn')(y)
    y = self.act(y)
    y = conv(self.filters, (3, 3), self.strides, name='conv2')(y)
    y = norm(name=self.name + '_conv2_bn')(y)
    y = self.act(y)
    y = conv(self.filters * 4, (1, 1), name='conv3')(y)
    y = norm(scale_init=nn.initializers.zeros, name=self.name + '_conv3_bn')(y)

    if residual.shape != y.shape:
      residual = conv(
          self.filters * 4, (1, 1), self.strides, name='proj_conv')(
              residual)
      residual = norm(name='proj_bn')(residual)

    return self.act(residual + y)


class ResNet(base.BaseModel):
  """ResNetV1.

  Attributes:
    stage_sizes: A tuple of int specifying the number of blocks per stage. Use
      tuple here to be jit friendly.
    block_cls: A nn.Module specifying which Resnet block to call.
    num_classes: Number of output classes. When return_features is True,
      num_classes can be None. Otherwise, it has to be set.
    num_filters: Number of filters in the model.
    dtype: Type of the jnp.arary, default is jnp.float32.
    act: A callable function for activation.
    return_features: Return block group features or not.
    batch_norm_momentum: Batch normalization momentum value.
    batch_norm_epsilon: Batch normalization epsilon value.
    batch_norm_group_size: distributed batch norm group size, which means how
      many examples within a batch will be used for batch stats computation. If
      zero, each device will use its own local data.
    width_scale: An int scaling up the channels of each layer.
  """
  stage_sizes: Tuple[int, Ellipsis]
  block_cls: ModuleDef
  num_classes: Optional[int] = None
  num_filters: int = 64
  dtype: jnp.dtype = jnp.float32
  act: Callable[Ellipsis, Any] = nn.relu
  return_features: bool = True
  batch_norm_momentum: float = 0.99
  batch_norm_epsilon: float = 1e-3
  batch_norm_group_size: int = 0
  width_scale: int = 1
  variable_init: Callable[Ellipsis, Array] = he_normal
  norm_layer: Callable[Ellipsis, Any] = nn.BatchNorm

  @nn.compact
  def __call__(self, x):
    """ResNet call with stem of 7x7 filters followed by max pool.

    Args:
      x: Input features of shape [N, H, W, C].

    Returns:
      output: A dictionary of level features or an array of output logits,
        depending on whether return_fatures are true or False. The feature
        shapes are [N, H, W, C] and logit shapes are [N, C].
    """
    block_group_features = {}
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype,
                             kernel_init=self.variable_init())

    if self.mode == base.ExecutionMode.TRAIN and self.batch_norm_group_size:
      axis_index_groups = model_utils.get_device_groups(
          self.batch_norm_group_size, x.shape[0])
    else:
      axis_index_groups = None

    norm = functools.partial(
        self.norm_layer,
        use_running_average=self.mode != base.ExecutionMode.TRAIN,
        momentum=self.batch_norm_momentum,
        epsilon=self.batch_norm_epsilon,
        dtype=self.dtype,
        axis_name='batch' if self.batch_norm_group_size else None,
        axis_index_groups=axis_index_groups,
    )

    x = conv(
        self.num_filters * self.width_scale, (7, 7), (2, 2),
        padding=[(3, 3), (3, 3)],
        name='init_conv')(x)
    x = norm(name='init_bn')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i * self.width_scale,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
            name=f'stage_{i}_block_{j}')(x)
      # Read out the features after each block group.
      if self.return_features:
        block_group_features[2 + i] = x

    if self.return_features:
      return block_group_features

    assert self.num_classes is not None and self.num_classes > 0, (
        f'Number of classes ({self.num_classes}) invalid!')
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype,
                 kernel_init=self.variable_init())(x)
    return x.astype(self.dtype)


ResNet9 = functools.partial(
    ResNet, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock)
ResNet18 = functools.partial(
    ResNet, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock)
ResNet34 = functools.partial(
    ResNet, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock)
ResNet50 = functools.partial(
    ResNet, stage_sizes=(3, 4, 6, 3), block_cls=BottleneckResNetBlock)
ResNet101 = functools.partial(
    ResNet, stage_sizes=(3, 4, 23, 3), block_cls=BottleneckResNetBlock)
ResNet152 = functools.partial(
    ResNet, stage_sizes=(3, 8, 36, 3), block_cls=BottleneckResNetBlock)
ResNet200 = functools.partial(
    ResNet, stage_sizes=(3, 24, 36, 3), block_cls=BottleneckResNetBlock)
