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
from typing import Optional, Union
import xmile_event
import xmile_globals

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileThreshold:

  class Meta:
    name = "threshold"
    namespace = xmile_globals.XMILE_NAMESPACE

  direction: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  value: Union[int, float] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  no_stop: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  repeat: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  event: xmile_event.XmileEvent = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
