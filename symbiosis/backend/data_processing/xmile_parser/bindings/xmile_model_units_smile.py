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

"""Definitions of units used in this model. SMILE standard version."""

import dataclasses
from typing import List
import xmile_globals
import xmile_units

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass
class XmileModelUnitsSmile:
  """Definitions of units used in this model. SMILE standard version."""

  class Meta:
    name = "model_units"
    namespace = xmile_globals.SMILE_NAMESPACE

  unit: List[xmile_units.XmileUnits] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "min_occurs": 1,
      },
  )
