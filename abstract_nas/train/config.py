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

"""Config for training."""

import dataclasses
from typing import Any, Dict, Optional, Tuple, Union

import ml_collections as mlc

from abstract_nas.model.concrete import Graph
from abstract_nas.model.subgraph import SubgraphSpec

Tensor = Any


@dataclasses.dataclass
class Config:
  """A config for training."""
  config_dict: mlc.ConfigDict

  graph: Union[Graph, Tuple[Graph, Dict[str, Tensor]]]
  output_dir: Optional[str] = None

  subgraph: Optional[SubgraphSpec] = None

  init_dir: Optional[str] = None
  inherit_weights: bool = False
  freeze_inherited: bool = False
  train_subg_outputs: bool = False

  checkpoint_steps: int = 1000
