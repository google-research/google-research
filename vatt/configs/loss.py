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

# Lint as: python3
"""Config definition for different types of losses."""

import dataclasses
from typing import Tuple

from vatt.configs import base_config


@dataclasses.dataclass
class BaseLossConfig(base_config.Config):
  """Base configs for any type of losses.

  Attributes:
    name: .
    loss_weight: .
  """
  name: str = ""
  loss_weight: float = 1.0


@dataclasses.dataclass
class SymmetricNCE(BaseLossConfig):
  """Parameters for symmetrical nce / mil-nce loss."""

  name: str = "symmetric_nce"
  temperature: float = 0.07
  vid_txt_weight: float = 1.
  vid_aud_weight: float = 1.
  aud_txt_weight: float = 0.


@dataclasses.dataclass
class AsymmetricNCE(SymmetricNCE):
  """Parameters for asymmetrical nce / mil-nce loss."""

  name: str = "asymmetric_nce"


@dataclasses.dataclass
class LossStack(base_config.Config):
  """Common BatchNorm configs for all models."""

  bridge: Tuple[BaseLossConfig, Ellipsis] = ()
