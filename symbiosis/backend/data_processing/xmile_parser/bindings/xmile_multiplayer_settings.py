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
from typing import List, Optional
import xmile_globals
import xmile_role

__NAMESPACE__ = "http://iseesystems.com/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileMultiplayerSettings:

  class Meta:
    name = "multiplayer_settings"
    namespace = xmile_globals.ISEE_NAMESPACE

  include_chat: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  allow_observers: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  advance_time_increment: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  max_advance_time_in_ms: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  role_dimension: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  start_page_template: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  setup_widget_top: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  setup_widget_left: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  setup_widget_bottom: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  setup_widget_right: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  model_stops_when_players_drop: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  observer_start_page: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  enabled: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  role: List[xmile_role.XmileRole] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
