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
from typing import List
import xmile_alias_no_ns
import xmile_aux_no_ns
import xmile_connector_no_ns
import xmile_flow_no_ns
import xmile_group_isee
import xmile_stock_no_ns


@dataclasses.dataclass(kw_only=True)
class XmileViewNoNs:

  class Meta:
    name = "view"
    namespace = ""

  content: List[object] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Wildcard",
          "namespace": "##any",
          "mixed": True,
          "choices": (
              {
                  "name": "group",
                  "type": xmile_group_isee.XmileGroupIsee,
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
                  "name": "connector",
                  "type": xmile_connector_no_ns.XmileConnectorNoNs,
              },
              {
                  "name": "aux",
                  "type": xmile_aux_no_ns.XmileAuxNoNs,
              },
              {
                  "name": "alias",
                  "type": xmile_alias_no_ns.XmileAliasNoNs,
              },
          ),
      },
  )
