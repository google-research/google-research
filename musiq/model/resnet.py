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

"""ResNet V1 with GroupNorm. https://arxiv.org/abs/1512.03385."""

from flax import nn
import jax.numpy as jnp


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  def apply(self, x):
    return x


def weight_standardize(w, axis, eps):
  w = w - jnp.mean(w, axis=axis)
  w = w / (jnp.std(w, axis=axis) + eps)
  return w


class StdConv(nn.Conv):

  def param(self, name, shape, initializer):
    param = super().param(name, shape, initializer)
    if name == "kernel":
      param = weight_standardize(param, axis=[0, 1, 2], eps=1e-5)
    return param


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self, x, nout, strides=(1, 1), bottleneck=True):
    features = nout
    nout = nout * 4 if bottleneck else nout
    needs_projection = x.shape[-1] != nout or strides != (1, 1)
    residual = x
    if needs_projection:
      residual = StdConv(
          residual, nout, (1, 1), strides, bias=False, name="conv_proj")
      residual = nn.GroupNorm(residual, epsilon=1e-4, name="gn_proj")

    if bottleneck:
      x = StdConv(x, features, (1, 1), bias=False, name="conv1")
      x = nn.GroupNorm(x, epsilon=1e-4, name="gn1")
      x = nn.relu(x)

    x = StdConv(x, features, (3, 3), strides, bias=False, name="conv2")
    x = nn.GroupNorm(x, epsilon=1e-4, name="gn2")
    x = nn.relu(x)

    last_kernel = (1, 1) if bottleneck else (3, 3)
    x = StdConv(x, nout, last_kernel, bias=False, name="conv3")
    x = nn.GroupNorm(
        x, epsilon=1e-4, name="gn3", scale_init=nn.initializers.zeros)
    x = nn.relu(residual + x)

    return x


class ResNetStage(nn.Module):

  def apply(self, x, block_size, nout, first_stride, bottleneck=True):
    x = ResidualUnit(
        x, nout, strides=first_stride, bottleneck=bottleneck, name="unit1")
    for i in range(1, block_size):
      x = ResidualUnit(
          x, nout, strides=(1, 1), bottleneck=bottleneck, name=f"unit{i + 1}")
    return x


class Model(nn.Module):
  """ResNetV1."""

  def apply(self,
            x,
            num_classes=1000,
            train=False,
            width_factor=1,
            num_layers=50):
    del train
    blocks, bottleneck = get_block_desc(num_layers)
    width = int(64 * width_factor)

    # Root block
    x = StdConv(x, width, (7, 7), (2, 2), bias=False, name="conv_root")
    x = nn.GroupNorm(x, name="gn_root")
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

    # Stages
    x = ResNetStage(
        x,
        blocks[0],
        width,
        first_stride=(1, 1),
        bottleneck=bottleneck,
        name="block1")
    for i, block_size in enumerate(blocks[1:], 1):
      x = ResNetStage(
          x,
          block_size,
          width * 2**i,
          first_stride=(2, 2),
          bottleneck=bottleneck,
          name=f"block{i + 1}")

    # Head
    x = jnp.mean(x, axis=(1, 2))
    x = IdentityLayer(x, name="pre_logits")
    x = nn.Dense(x, num_classes, kernel_init=nn.initializers.zeros, name="head")
    return x


# A dictionary mapping the number of layers in a resnet to the number of
# blocks in each stage of the model. The second argument indicates whether we
# use bottleneck layers or not.
def get_block_desc(num_layers):
  if isinstance(num_layers, list):  # Be robust to silly mistakes.
    num_layers = tuple(num_layers)
  return {
      5: ([1], True),  # Only strided blocks. Total stride 4.
      8: ([1, 1], True),  # Only strided blocks. Total stride 8.
      11: ([1, 1, 1], True),  # Only strided blocks. Total stride 16.
      14: ([1, 1, 1, 1], True),  # Only strided blocks. Total stride 32.
      9: ([1, 1, 1, 1], False),  # Only strided blocks. Total stride 32.
      18: ([2, 2, 2, 2], False),
      26: ([2, 2, 2, 2], True),
      34: ([3, 4, 6, 3], False),
      50: ([3, 4, 6, 3], True),
      101: ([3, 4, 23, 3], True),
      152: ([3, 8, 36, 3], True),
      200: ([3, 24, 36, 3], True)
  }.get(num_layers, (num_layers, True))
