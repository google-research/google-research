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

"""XMILE types."""

import dataclasses
from typing import Optional, Union

import xmile_enums


@dataclasses.dataclass
class XmileStyleType:
  """Style type."""

  class Meta:
    name = "style_type"

  border_width: Optional[Union[xmile_enums.StandardBorderWidthType, float]] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
  )
  border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
  )
  border_style: xmile_enums.StyleTypeBorderStyle = dataclasses.field(
      default=xmile_enums.StyleTypeBorderStyle.NONE,
      metadata={
          "type": "Attribute",
      },
  )
  font_family: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  font_style: xmile_enums.StyleTypeFontStyle = dataclasses.field(
      default=xmile_enums.StyleTypeFontStyle.NORMAL,
      metadata={
          "type": "Attribute",
      },
  )
  font_weight: xmile_enums.StyleTypeFontWeight = dataclasses.field(
      default=xmile_enums.StyleTypeFontWeight.NORMAL,
      metadata={
          "type": "Attribute",
      },
  )
  text_decoration: xmile_enums.StyleTypeTextDecoration = dataclasses.field(
      default=xmile_enums.StyleTypeTextDecoration.NORMAL,
      metadata={
          "type": "Attribute",
      },
  )
  text_align: xmile_enums.TextAlign = dataclasses.field(
      default=xmile_enums.TextAlign.LEFT,
      metadata={
          "type": "Attribute",
      },
  )
  vertical_text_align: xmile_enums.VerticalTextAlign = dataclasses.field(
      default=xmile_enums.VerticalTextAlign.CENTER,
      metadata={
          "type": "Attribute",
      },
  )
  font_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
  )
  text_background: Optional[Union[xmile_enums.StandardColorType, str]] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
  )
  font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  padding: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  color: Optional[Union[xmile_enums.StandardColorType, str]] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
  )
  background: Optional[Union[xmile_enums.StandardColorType, str]] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
  )
  z_index: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  height: Optional[float] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  width: Optional[float] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  label_side: Optional[xmile_enums.StyleTypeLabelSide] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  label_angle: Optional[float] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
