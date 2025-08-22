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

"""Methods for stacking multiple same-config layers."""

import dataclasses
from typing import Any

import flax.linen as nn

from imp.max.core import constants
from imp.max.core import transforms
from imp.max.utils import typing


@dataclasses.dataclass
class SequentialStackCall(object):
  """Calls a stack of configured layers sequentially."""

  layers: list[nn.Module]

  def __call__(self, inputs, *args, **kwargs):
    for layer in self.layers:
      inputs = layer(inputs, *args, **kwargs)

    return inputs


class ReplicatedStack(nn.Module):
  """Replicates a module with same config and calls the stack."""

  module: nn.Module
  length: int
  prefix: str = 'layer_'
  config: dict[str, Any] | None = None

  def setup(self):
    config = self.config or {}
    self.stack = SequentialStackCall([
        self.module(name=f'{self.prefix}{n}', **config)
        for n in range(self.length)
    ])

  def __call__(self, *args, **kwargs):
    return self.stack(*args, **kwargs)


class RematScannedStack(nn.Module):
  """Remats and Scans a module with same config and calls the stack."""

  module: nn.Module
  length: int
  config: dict[str, Any] | None = None
  scan_axis: int = 0
  in_axes: typing.TransformAxes = nn.broadcast
  out_axes: typing.TransformAxes = 0
  sharding_axis: str | None = None
  stack_name: str = constants.TransformAxisName.STACK
  remat: str | None = None
  static_argnums: tuple[int, Ellipsis] = ()
  rng_keys: tuple[str, Ellipsis] = ()

  def setup(self):
    config = self.config or {}

    module = transforms.remat(  # pylint: disable=invalid-name
        module=self.module,
        level=self.remat,
        scanned=True,
        static_argnums=self.static_argnums,
    )

    self.stack = transforms.scan(
        module=module,
        length=self.length,
        scan_axis=self.scan_axis,
        in_axes=self.in_axes,
        out_axes=self.out_axes,
        sharding_axis=self.sharding_axis,
        rng_keys=self.rng_keys,
    )(name=self.stack_name, **config)

  def __call__(self, inputs, *args):
    outputs, _ = self.stack(inputs, *args)

    return outputs
