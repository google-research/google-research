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

"""Implementation of ResNet V1 in Flax.

"Deep Residual Learning for Image Recognition"
He et al., 2015, [https://arxiv.org/abs/1512.03385]
"""

import functools

from typing import Any, Tuple, Type, List, Optional, Callable, Sequence
import flax.linen as nn
import jax.numpy as jnp


Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1), use_bias=False)
Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3), use_bias=False)


class ResNetBlock(nn.Module):
  """ResNet block without bottleneck used in ResNet-18 and ResNet-34."""

  filters: int
  norm: Any
  kernel_dilation: Tuple[int, int] = (1, 1)
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x

    x = Conv3x3(
        self.filters,
        strides=self.strides,
        kernel_dilation=self.kernel_dilation,
        name="conv1")(x)
    x = self.norm(name="bn1")(x)
    x = nn.relu(x)
    x = Conv3x3(self.filters, name="conv2")(x)
    # Initializing the scale to 0 has been common practice since "Fixup
    # Initialization: Residual Learning Without Normalization" Tengyu et al,
    # 2019, [https://openreview.net/forum?id=H1gsz30cKX].
    x = self.norm(scale_init=nn.initializers.zeros, name="bn2")(x)

    if residual.shape != x.shape:
      residual = Conv1x1(
          self.filters, strides=self.strides, name="proj_conv")(
              residual)
      residual = self.norm(name="proj_bn")(residual)

    x = nn.relu(residual + x)
    return x


class BottleneckResNetBlock(ResNetBlock):
  """Bottleneck ResNet block used in ResNet-50 and larger."""

  @nn.compact
  def __call__(self, x):
    residual = x

    x = Conv1x1(self.filters, name="conv1")(x)
    x = self.norm(name="bn1")(x)
    x = nn.relu(x)
    x = Conv3x3(
        self.filters,
        strides=self.strides,
        kernel_dilation=self.kernel_dilation,
        name="conv2")(x)
    x = self.norm(name="bn2")(x)
    x = nn.relu(x)
    x = Conv1x1(4 * self.filters, name="conv3")(x)
    # Initializing the scale to 0 has been common practice since "Fixup
    # Initialization: Residual Learning Without Normalization" Tengyu et al,
    # 2019, [https://openreview.net/forum?id=H1gsz30cKX].
    x = self.norm(name="bn3")(x)

    if residual.shape != x.shape:
      residual = Conv1x1(
          4 * self.filters, strides=self.strides, name="proj_conv")(
              residual)
      residual = self.norm(name="proj_bn")(residual)

    x = nn.relu(residual + x)
    return x


class ResNetStage(nn.Module):
  """ResNet stage consistent of multiple ResNet blocks."""

  stage_size: int
  filters: int
  block_cls: Type[ResNetBlock]
  norm: Any
  first_block_strides: Tuple[int, int]

  @nn.compact
  def __call__(self, x):
    for i in range(self.stage_size):
      x = self.block_cls(
          filters=self.filters,
          norm=self.norm,
          strides=self.first_block_strides if i == 0 else (1, 1),
          name=f"block{i + 1}")(
              x)
    return x


class ResNet(nn.Module):
  """Construct ResNet V1 with `num_classes` outputs.

  Attributes:
    num_classes: Number of nodes in the final layer.
    block_cls: Class for the blocks. ResNet-50 and larger use
      `BottleneckResNetBlock` (convolutions: 1x1, 3x3, 1x1), ResNet-18 and
        ResNet-34 use `ResNetBlock` without bottleneck (two 3x3 convolutions).
    stage_sizes: List with the number of ResNet blocks in each stage. Number of
      stages can be varied.
    norm_type: Which type of normalization layer to apply. Options are:
      "batch": BatchNorm, "group": GroupNorm, "layer": LayerNorm. Defaults to
      BatchNorm.
    width_factor: Factor applied to the number of filters. The 64 * width_factor
      is the number of filters in the first stage, every consecutive stage
      doubles the number of filters.
    small_inputs: Bool, if True, ignore strides and skip max pooling in the root
      block and use smaller filter size.
    stage_strides: Stride per stage. This overrides all other arguments.
    include_top: Whether to include the fully-connected layer at the top
      of the network.
    axis_name: Axis name over which to aggregate batchnorm statistics.
  """
  num_classes: int
  block_cls: Type[ResNetBlock]
  stage_sizes: List[int]
  norm_type: str = "batch"
  width_factor: int = 1
  small_inputs: bool = False
  stage_strides: Optional[List[Tuple[int, int]]] = None
  include_top: bool = False
  axis_name: Optional[str] = None
  output_initializer: Callable[[Any, Sequence[int], Any], Any] = (
      nn.initializers.zeros)

  @nn.compact
  def __call__(self, x, *, train):
    """Apply the ResNet to the inputs `x`.

    Args:
      x: Inputs.
      train: Whether to use BatchNorm in training or inference mode.

    Returns:
      The output head with `num_classes` entries.
    """
    width = 64 * self.width_factor

    if self.norm_type == "batch":
      norm = functools.partial(
          nn.BatchNorm, use_running_average=not train, momentum=0.9,
          axis_name=self.axis_name)
    elif self.norm_type == "layer":
      norm = nn.LayerNorm
    elif self.norm_type == "group":
      norm = nn.GroupNorm
    else:
      raise ValueError(f"Invalid norm_type: {self.norm_type}")

    # Root block.
    x = nn.Conv(
        features=width,
        kernel_size=(7, 7) if not self.small_inputs else (3, 3),
        strides=(2, 2) if not self.small_inputs else (1, 1),
        use_bias=False,
        name="init_conv")(
            x)
    x = norm(name="init_bn")(x)

    if not self.small_inputs:
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

    # Stages.
    for i, stage_size in enumerate(self.stage_sizes):
      if i == 0:
        first_block_strides = (
            1, 1) if self.stage_strides is None else self.stage_strides[i]
      else:
        first_block_strides = (
            2, 2) if self.stage_strides is None else self.stage_strides[i]

      x = ResNetStage(
          stage_size,
          filters=width * 2**i,
          block_cls=self.block_cls,
          norm=norm,
          first_block_strides=first_block_strides,
          name=f"stage{i + 1}")(x)

    # Head.
    if self.include_top:
      x = jnp.mean(x, axis=(1, 2))
      x = nn.Dense(
          self.num_classes, kernel_init=self.output_initializer, name="head")(x)
    return x


ResNetWithBasicBlk = functools.partial(ResNet, block_cls=ResNetBlock)
ResNetWithBottleneckBlk = functools.partial(ResNet,
                                            block_cls=BottleneckResNetBlock)

ResNet18 = functools.partial(ResNetWithBasicBlk, stage_sizes=[2, 2, 2, 2])
ResNet34 = functools.partial(ResNetWithBasicBlk, stage_sizes=[3, 4, 6, 3])
ResNet50 = functools.partial(ResNetWithBottleneckBlk, stage_sizes=[3, 4, 6, 3])
ResNet101 = functools.partial(ResNetWithBottleneckBlk,
                              stage_sizes=[3, 4, 23, 3])
ResNet152 = functools.partial(ResNetWithBottleneckBlk,
                              stage_sizes=[3, 8, 36, 3])
ResNet200 = functools.partial(ResNetWithBottleneckBlk,
                              stage_sizes=[3, 24, 36, 3])
