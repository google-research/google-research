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

"""TextBox class."""

import dataclasses
import xmile_globals

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileTextBoxSmile:
  """TextBox class."""

  class Meta:
    name = "text_box"
    namespace = xmile_globals.SMILE_NAMESPACE

  color: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  background: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  font_style: str = dataclasses.field(
      metadata={
          "name": "font-style",
          "type": "Attribute",
          "required": True,
      }
  )
  font_weight: str = dataclasses.field(
      metadata={
          "name": "font-weight",
          "type": "Attribute",
          "required": True,
      }
  )
  text_decoration: str = dataclasses.field(
      metadata={
          "name": "text-decoration",
          "type": "Attribute",
          "required": True,
      }
  )
  text_align: str = dataclasses.field(
      metadata={
          "name": "text-align",
          "type": "Attribute",
          "required": True,
      }
  )
  vertical_text_align: str = dataclasses.field(
      metadata={
          "name": "vertical-text-align",
          "type": "Attribute",
          "required": True,
      }
  )
  text_background: str = dataclasses.field(
      metadata={
          "name": "text-background",
          "type": "Attribute",
          "required": True,
      }
  )
  font_color: str = dataclasses.field(
      metadata={
          "name": "font-color",
          "type": "Attribute",
          "required": True,
      }
  )
  font_family: str = dataclasses.field(
      metadata={
          "name": "font-family",
          "type": "Attribute",
          "required": True,
      }
  )
  font_size: str = dataclasses.field(
      metadata={
          "name": "font-size",
          "type": "Attribute",
          "required": True,
      }
  )
  padding: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  border_color: str = dataclasses.field(
      metadata={
          "name": "border-color",
          "type": "Attribute",
          "required": True,
      }
  )
  border_width: str = dataclasses.field(
      metadata={
          "name": "border-width",
          "type": "Attribute",
          "required": True,
      }
  )
  border_style: str = dataclasses.field(
      metadata={
          "name": "border-style",
          "type": "Attribute",
          "required": True,
      }
  )
