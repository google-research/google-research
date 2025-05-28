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

"""PayoffElement - ISEE Standard."""

import dataclasses
from typing import List, Optional
import xmile_globals
import xmile_time_range


__NAMESPACE__ = "http://iseesystems.com/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmilePayoffElement:
  """PayoffElement - ISEE Standard."""

  class Meta:
    name = "payoff_element"
    namespace = xmile_globals.ISEE_NAMESPACE

  name: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      },
  )
  weight: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  compare_name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  compare_run: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  tolerance: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  type_value: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "type",
          "type": "Attribute",
      },
  )
  time_range: Optional[xmile_time_range.XmileTimeRange] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
