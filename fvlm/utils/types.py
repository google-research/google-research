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

"""Type annotation library."""

import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from clu import metrics
from flax import linen as nn
import jax.numpy as jnp
import tensorflow as tf


Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = Array
Shape = Iterable[int]

# Types for data preprocessors.
FeatureName = Tuple[str, Ellipsis]
Tensor = tf.Tensor
TFDType = tf.dtypes.DType
DictTensor = Dict[Union[int, str], Tensor]
NestedDictTensor = Dict[Union[int, str], Union[Tensor, DictTensor]]

# Generic dictionary of JAX arrays.
TextDict = Dict[str, Any]
PairArray = Tuple[Array, Optional[Array]]
TripleArray = Tuple[Array, Optional[Array], Optional[Array]]
TripleListArray = Tuple[Array, Optional[Array], Optional[List[Array]]]
DictArray = Dict[Union[int, str], Array]
DictPairArray = Dict[Union[int, str], PairArray]
MultilevelFeature = Union[Array, DictArray]
NestedDictArray = Dict[Union[int, str], Union[Array, DictArray]]
NestedTextDictArray = Dict[str, Union[Array, DictArray]]

# Generic loss function.
LossFn = Callable[[NestedDictArray, NestedDictArray], DictArray]
# Callable to instantiate a linen module.
ModelFn = Callable[Ellipsis, nn.Module]
OptModelFn = Optional[ModelFn]
# Data mapper function.
DataMapFn = Callable[[TextDict], TextDict]
# Common parser functions
TextParserFn = Callable[[Dict[str, Any], str], Dict[str, Any]]
# Initializer function
InitFn = Callable[[PRNGKey, Shape, DType], Array]
# Multitask metrics.
MetricDict = Dict[str, metrics.Metric]
Metric = metrics.Metric
# Fusion models.
FuseFeatureFn = Callable[[MultilevelFeature, Array, Array, int], Array]
MultilevelPairFeature = Union[Array, DictArray, PairArray, DictPairArray]
# Projection models.
ProjectFeatureFn = Callable[[MultilevelFeature, Optional[int]],
                            MultilevelFeature]

# Targets used in text_metrics.
Targets = Union[List[str], List[List[str]]]

# Loader types.
LoaderFn = Callable[[], tf.data.Dataset]
LoaderFns = List[LoaderFn]


@dataclasses.dataclass
class DatasetReturnObject:
  dataset: Union[Iterable[Any], tf.data.Dataset]
  weights: Optional[tf.Variable]


# General vocabulary for language model.
Vocab = Optional[Any]
