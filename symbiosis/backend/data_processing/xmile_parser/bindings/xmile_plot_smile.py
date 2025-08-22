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
import xmile_entity_smile
import xmile_globals

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmilePlotSmile:

  class Meta:
    name = "plot"
    namespace = xmile_globals.SMILE_NAMESPACE

  index: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  pen_width: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  color: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  precision: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  show_y_axis: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  entity: xmile_entity_smile.XmileEntitySmile = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
