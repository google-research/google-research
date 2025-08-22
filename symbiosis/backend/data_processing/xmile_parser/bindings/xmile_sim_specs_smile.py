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

"""Default simulation specifications for this model."""

import dataclasses
from typing import Optional
import xmile_globals

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileSimSpecsSmile:

  class Meta:
    name = "sim_specs"
    namespace = xmile_globals.SMILE_NAMESPACE

  method: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  time_units: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  pause_after_rates: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  instantaneous_flows: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  start: int = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
  stop: int = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
  dt: float = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
