# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Types definition for the project."""

from typing import Callable, Dict, Tuple, TypeVar, Any

import jax.numpy as jnp

PRNGKey = jnp.array
Shape = Tuple[int, Ellipsis]
Initializer = Callable[[PRNGKey, Shape, jnp.dtype], jnp.ndarray]
AuxOutput = Dict[str, jnp.ndarray]
LayerInput = TypeVar("LayerInput")
LossOrMetric = Callable[[jnp.ndarray, jnp.ndarray, Any], Any]
