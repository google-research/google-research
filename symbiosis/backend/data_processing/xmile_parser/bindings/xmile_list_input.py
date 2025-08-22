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

"""ListInput class."""

import dataclasses
from typing import List, Optional, Union
import xmile_enums
import xmile_globals


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileListInput:
  """XmileListInput class."""

  class Meta:
    name = "list_input"
    namespace = xmile_globals.XMILE_NAMESPACE

  name: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      },
  )
  column_width: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      },
  )
  font_style: xmile_enums.ListInputFontStyle = dataclasses.field(
      default=xmile_enums.ListInputFontStyle.NORMAL,
      metadata={
          "type": "Attribute",
      },
  )
  font_weight: xmile_enums.ListInputFontWeight = dataclasses.field(
      default=xmile_enums.ListInputFontWeight.NORMAL,
      metadata={
          "type": "Attribute",
      },
  )
  text_decoration: xmile_enums.ListInputTextDecoration = dataclasses.field(
      default=xmile_enums.ListInputTextDecoration.NORMAL,
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
  numeric_input: List["XmileListInput.NumericInput"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "min_occurs": 1,
      },
  )
  index: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
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
  border_style: Optional[xmile_enums.ListInputBorderStyle] = dataclasses.field(
      default=xmile_enums.ListInputBorderStyle.NONE,
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
  x: Optional[float] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  y: Optional[float] = dataclasses.field(
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
  label_side: Optional[xmile_enums.ListInputLabelSide] = dataclasses.field(
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

  @dataclasses.dataclass(kw_only=True)
  class NumericInput:
    """NumericInput class."""

    entity: Optional["XmileListInput.NumericInput.Entity"] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
        },
    )
    min: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    max: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    precision: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )

    @dataclasses.dataclass(kw_only=True)
    class Entity:
      name: str = dataclasses.field(
          metadata={
              "type": "Attribute",
              "required": True,
          },
      )
      index: Optional[str] = dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
