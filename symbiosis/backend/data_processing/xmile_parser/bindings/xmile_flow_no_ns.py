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

"""Core building block of a model, also called a rate or a derivative."""

import dataclasses
import xmile_enums
import xmile_pts_no_ns


@dataclasses.dataclass
class XmileFlowNoNs:
  """Core building block of a model, also called a rate or a derivative.

  Flows represent the rate of change of a stock. Each timestep the
  flow multiplied by the timestep is added to the stock.
  """

  class Meta:
    name = "flow"
    namespace = ""

  name: str = dataclasses.field(
      metadata={
          "type": "Attribute",
      },
  )
  x: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  y: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  pts: xmile_pts_no_ns.XmilePtsNoNs = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
