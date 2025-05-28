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

import xmile_globals

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileTableSmile:

  class Meta:
    name = "table"
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
  orientation: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  wrap_text: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  auto_fit: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  use_alternate_row_colors: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  horizontal_scrolling_enabled: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  blank_column_width: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  column_width: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  hide_detail: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  interval: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  report_month_names: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
          "required": True,
      }
  )
  header_font_style: str = dataclasses.field(
      metadata={
          "name": "header-font-style",
          "type": "Attribute",
          "required": True,
      }
  )
  header_font_weight: str = dataclasses.field(
      metadata={
          "name": "header-font-weight",
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_decoration: str = dataclasses.field(
      metadata={
          "name": "header-text-decoration",
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_align: str = dataclasses.field(
      metadata={
          "name": "header-text-align",
          "type": "Attribute",
          "required": True,
      }
  )
  header_vertical_text_align: str = dataclasses.field(
      metadata={
          "name": "header-vertical-text-align",
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_background: str = dataclasses.field(
      metadata={
          "name": "header-text-background",
          "type": "Attribute",
          "required": True,
      }
  )
  header_font_color: str = dataclasses.field(
      metadata={
          "name": "header-font-color",
          "type": "Attribute",
          "required": True,
      }
  )
  header_font_family: str = dataclasses.field(
      metadata={
          "name": "header-font-family",
          "type": "Attribute",
          "required": True,
      }
  )
  header_font_size: str = dataclasses.field(
      metadata={
          "name": "header-font-size",
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_padding: int = dataclasses.field(
      metadata={
          "name": "header-text-padding",
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_border_color: str = dataclasses.field(
      metadata={
          "name": "header-text-border-color",
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_border_width: str = dataclasses.field(
      metadata={
          "name": "header-text-border-width",
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_border_style: str = dataclasses.field(
      metadata={
          "name": "header-text-border-style",
          "type": "Attribute",
          "required": True,
      }
  )
