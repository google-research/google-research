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

import dataclasses
from typing import List, Optional, Union
import xmile_globals
import xmile_graph
import xmile_group_input
import xmile_table

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileStackedContainer:

  class Meta:
    name = "stacked_container"
    namespace = xmile_globals.XMILE_NAMESPACE

  z_index: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  height: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  width: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  appearance: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  y: Union[float, int] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  x: Union[float, int] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  uid: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  visible_index: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  locked: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  graph: List[xmile_graph.XmileGraph] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  minimized: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  table: List[xmile_table.XmileTable] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  group_input: List[xmile_group_input.XmileGroupInput] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
