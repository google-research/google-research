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

"""Common types used by submodules."""

from typing import Any, Callable, Sequence

import aqt.jax.v2.config as aqt_config
import flax.linen as nn
from flax.training import train_state as flax_train_state
import jax
from jax import lax
import numpy as np

JaxNpArray = jax.Array | np.ndarray
# TODO(hassanak): Find a generic way to represent Flax Module typing
FlaxModule = Any
Precision = (
    str | lax.Precision | tuple[str, str] | tuple[lax.Precision, lax.Precision]
) | None
DotGeneral = Callable[Ellipsis, jax.Array]
ConvGeneralDilated = Callable[Ellipsis, jax.Array]
Shape = Sequence[int]
Axes = int | Shape
ClassificationHead = int | Sequence[tuple[str, int]] | None
ShardingAxis = (
    str | Sequence[str] | type(jax.sharding.PartitionSpec.UNCONSTRAINED) | None
)
ShardingAxes = Sequence[ShardingAxis] | None
FlaxMetaShardingAxes = tuple[str | None, Ellipsis]
PaddingFn = Callable[[jax.Array, Shape, Shape], jax.Array]
PaddingLike = str | int | Sequence[int | tuple[int, int]] | PaddingFn
PaddingShape = tuple[tuple[int, int], Ellipsis]
LaxPadding = str | Sequence[tuple[int, int]]
TransformAxis = int | type(nn.broadcast)
TransformAxes = TransformAxis | tuple[TransformAxis, Ellipsis]
Data = dict[str, Any]
ObjectiveModalityTokenWeight = tuple[tuple[str, str], float]
ObjectiveModalityPairWeight = tuple[tuple[str, str], float]
FlatConfigDict = dict[str | tuple[str, Ellipsis], Any]
# TODO(hassanak): Find a way to make this arbitrary depth
NestedDict = dict[str, Any] | dict[str, 'NestedDict']
FlaxParams = NestedDict
FlaxTrainState = flax_train_state.TrainState
GenericCallable = Callable[Ellipsis, Any]
AqtDotGeneralConfig = aqt_config.DotGeneral | None
DataLoaderCollection = tuple[dict[str, Any], Ellipsis]
MetricFn = (Callable[[JaxNpArray, JaxNpArray], NestedDict] |
            Callable[[JaxNpArray, JaxNpArray, JaxNpArray | None], NestedDict])


# New sharding API
ArrayShardingAxis = (
    str | Sequence[str] | None | type(jax.sharding.PartitionSpec.UNCONSTRAINED)
)
ShardingAxes = Sequence[ArrayShardingAxis] | None


# TODO(hassanak): completely remove upon migrating to new API
TpuMesh = tuple[int, int, int, int]  # (x, y, z, num_cores).
OtherMesh = tuple[int, int]
HardwareMesh = TpuMesh | OtherMesh
AxisRules = Sequence[tuple[str, jax.sharding.PartitionSpec]]
LogicalAxisRules = Sequence[tuple[str, str | None]]
