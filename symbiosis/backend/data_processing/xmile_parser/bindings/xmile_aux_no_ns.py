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

"""Auxiliaries class."""

import dataclasses
from typing import List, Optional, Type

import xmile_alias_no_ns
import xmile_connector_no_ns
import xmile_dimensions
import xmile_enums
import xmile_event_poster
import xmile_flow_no_ns
import xmile_format
import xmile_gf
import xmile_group_no_ns
import xmile_range
import xmile_scale
import xmile_stock_no_ns


@dataclasses.dataclass(kw_only=True)
class XmileAuxNoNs:
  """Core building block of a model.

  Auxiliaries allow the isolation of any algebraic function that is
  used. They can both clarify a model and factor out important or
  repeated calculations. They can be defined using any algebraic
  expression (including a constant value), optionally in conjunction
  with a graphical function.
  """

  class Meta:
    name = "aux"
    namespace = ""

  name: str = dataclasses.field(
      metadata={
          "type": "Attribute",
      },
  )
  x: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  y: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  content: List[object] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Wildcard",
          "namespace": "##any",
          "mixed": True,
          "choices": (
              {
                  "name": "connector",
                  "type": xmile_connector_no_ns.XmileConnectorNoNs,
              },
              {
                  "name": "stock",
                  "type": xmile_stock_no_ns.XmileStockNoNs,
              },
              {
                  "name": "flow",
                  "type": xmile_flow_no_ns.XmileFlowNoNs,
              },
              {
                  "name": "aux",
                  "type": type["XmileAuxNoNs"],
              },
              {
                  "name": "alias",
                  "type": xmile_alias_no_ns.XmileAliasNoNs,
              },
              {
                  "name": "group",
                  "type": xmile_group_no_ns.XmileGroupNoNs,
              },
          ),
      },
  )
