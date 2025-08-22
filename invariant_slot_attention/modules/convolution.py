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

"""Convolutional module library."""

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from flax import linen as nn
import jax

Shape = Tuple[int]

DType = Any
Array = Any  # jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class SimpleCNN(nn.Module):
  """Simple CNN encoder with multiple Conv+ReLU layers."""

  features: Sequence[int]
  kernel_size: Sequence[Tuple[int, int]]
  strides: Sequence[Tuple[int, int]]
  transpose: bool = False
  use_batch_norm: bool = False
  axis_name: Optional[str] = None  # Over which axis to aggregate batch stats.
  padding: Union[str, Iterable[Tuple[int, int]]] = "SAME"
  resize_output: Optional[Iterable[int]] = None

  @nn.compact
  def __call__(self, inputs, train = False):
    num_layers = len(self.features)
    assert len(self.kernel_size) == num_layers, (
        "len(kernel_size) and len(features) must match.")
    assert len(self.strides) == num_layers, (
        "len(strides) and len(features) must match.")
    assert num_layers >= 1, "Need to have at least one layer."

    if self.transpose:
      conv_module = nn.ConvTranspose
    else:
      conv_module = nn.Conv

    x = conv_module(
        name="conv_simple_0",
        features=self.features[0],
        kernel_size=self.kernel_size[0],
        strides=self.strides[0],
        use_bias=False if self.use_batch_norm else True,
        padding=self.padding)(inputs)

    for i in range(1, num_layers):
      if self.use_batch_norm:
        x = nn.BatchNorm(
            momentum=0.9, use_running_average=not train,
            axis_name=self.axis_name, name=f"bn_simple_{i-1}")(x)

      x = nn.relu(x)
      x = conv_module(
          name=f"conv_simple_{i}",
          features=self.features[i],
          kernel_size=self.kernel_size[i],
          strides=self.strides[i],
          use_bias=False if (
              self.use_batch_norm and i < (num_layers-1)) else True,
          padding=self.padding)(x)

    if self.resize_output:
      x = jax.image.resize(
          x, list(x.shape[:-3]) + list(self.resize_output) + [x.shape[-1]],
          method=jax.image.ResizeMethod.LINEAR)
    return x


class CNN(nn.Module):
  """Flexible CNN model with Conv/Normalization/Pooling layers."""

  features: Sequence[int]
  kernel_size: Sequence[Tuple[int, int]]
  strides: Sequence[Tuple[int, int]]
  max_pool_strides: Sequence[Tuple[int, int]]
  layer_transpose: Sequence[bool]
  activation_fn: Callable[[Array], Array] = nn.relu
  norm_type: Optional[str] = None
  axis_name: Optional[str] = None  # Over which axis to aggregate batch stats.
  output_size: Optional[int] = None

  @nn.compact
  def __call__(self, inputs, train = False):
    num_layers = len(self.features)

    assert num_layers >= 1, "Need to have at least one layer."
    assert len(self.kernel_size) == num_layers, (
        "len(kernel_size) and len(features) must match.")
    assert len(self.strides) == num_layers, (
        "len(strides) and len(features) must match.")
    assert len(self.max_pool_strides) == num_layers, (
        "len(max_pool_strides) and len(features) must match.")
    assert len(self.layer_transpose) == num_layers, (
        "len(layer_transpose) and len(features) must match.")

    if self.norm_type:
      assert self.norm_type in {"batch", "group", "instance", "layer"}, (
          f"{self.norm_type} is unrecognizaed normalization")

    # Whether transpose conv or regular conv.
    conv_module = {False: nn.Conv, True: nn.ConvTranspose}

    if self.norm_type == "batch":
      norm_module = functools.partial(
          nn.BatchNorm, momentum=0.9, use_running_average=not train,
          axis_name=self.axis_name)
    elif self.norm_type == "group":
      norm_module = functools.partial(
          nn.GroupNorm, num_groups=32)
    elif self.norm_type == "layer":
      norm_module = nn.LayerNorm

    x = inputs
    for i in range(num_layers):
      x = conv_module[self.layer_transpose[i]](
          name=f"conv_{i}",
          features=self.features[i],
          kernel_size=self.kernel_size[i],
          strides=self.strides[i],
          use_bias=False if self.norm_type else True)(x)

      # Normalization layer.
      if self.norm_type:
        if self.norm_type == "instance":
          x = nn.GroupNorm(
              num_groups=self.features[i],
              name=f"{self.norm_type}_norm_{i}")(x)
        else:
          norm_module(name=f"{self.norm_type}_norm_{i}")(x)

      # Activation layer.
      x = self.activation_fn(x)

      # Max pooling layer.
      x = x if self.max_pool_strides[i] == (1, 1) else nn.max_pool(
          x, self.max_pool_strides[i], strides=self.max_pool_strides[i],
          padding="SAME")

    # Final dense layer.
    if self.output_size:
      x = nn.Dense(self.output_size, name="output_layer", use_bias=True)(x)
    return x
