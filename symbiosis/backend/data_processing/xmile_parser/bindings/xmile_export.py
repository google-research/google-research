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

"""Exog."""

import dataclasses
from typing import Optional
import xmile_globals
import xmile_table

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileExport:
  """Export."""

  class Meta:
    name = "export"
    namespace = xmile_globals.XMILE_NAMESPACE

  enabled: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  frequency: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  orientation: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  resource: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  type_value: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "type",
          "type": "Attribute",
      },
  )
  worksheet: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  interval: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  precomputed: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      }
  )
  format: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
          "required": True,
      }
  )
  all: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  list_value: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "list",
          "type": "Element",
      },
  )
  table: Optional[xmile_table.XmileTable] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
