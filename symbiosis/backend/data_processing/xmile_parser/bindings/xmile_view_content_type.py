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
from typing import List, Optional, Type, Union

import xmile_entity
import xmile_enums
import xmile_globals
import xmile_graphical_input
import xmile_graphics_frame
import xmile_image
import xmile_knob
import xmile_link
import xmile_list_input
import xmile_numeric_display
import xmile_plot
import xmile_popup
import xmile_shape
import xmile_slider
import xmile_sound
import xmile_stacked_container
import xmile_style
import xmile_switch
import xmile_switch_action
import xmile_text_box
import xmile_video


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileViewContentType:
  """Content of a view or container."""

  class Meta:
    name = "view_content_type"

  style: List[xmile_style.XmileStyle] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  stock: List["XmileViewContentType.Stock"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  flow: List["XmileViewContentType.Flow"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  aux: List["XmileViewContentType.Aux"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  module: List["XmileViewContentType.Module"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  group: List["XmileViewContentType.Group"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  connector: List["XmileViewContentType.Connector"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  alias: List["XmileViewContentType.Alias"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  text_box: List[xmile_text_box.XmileTextBox] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  graphics_frame: List[xmile_graphics_frame.XmileGraphicsFrame] = (
      dataclasses.field(
          default_factory=list,
          metadata={
              "type": "Element",
              "namespace": xmile_globals.XMILE_NAMESPACE,
          },
      )
  )
  graph: List["XmileViewContentType.Graph"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  table: List["XmileViewContentType.Table"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  button: List["XmileViewContentType.Button"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  slider: List[xmile_slider.XmileSlider] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  knob: List[xmile_knob.XmileKnob] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  list_input: List[xmile_list_input.XmileListInput] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  numeric_display: List[xmile_numeric_display.XmileNumericDisplay] = (
      dataclasses.field(
          default_factory=list,
          metadata={
              "type": "Element",
              "namespace": xmile_globals.XMILE_NAMESPACE,
          },
      )
  )
  switch: List[xmile_switch.XmileSwitch] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  graphical_input: List[xmile_graphical_input.XmileGraphicalInput] = (
      dataclasses.field(
          default_factory=list,
          metadata={
              "type": "Element",
              "namespace": xmile_globals.XMILE_NAMESPACE,
          },
      )
  )
  stacked_container: List[xmile_stacked_container.XmileStackedContainer] = (
      dataclasses.field(
          default_factory=list,
          metadata={
              "type": "Element",
              "namespace": xmile_globals.XMILE_NAMESPACE,
          },
      )
  )
  options: List["XmileViewContentType.Options"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  width: Optional[float] = dataclasses.field(
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

  @dataclasses.dataclass(kw_only=True)
  class Stock:
    shape: Optional[xmile_shape.XmileShape] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    name: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.StockBorderStyle] = dataclasses.field(
        default=xmile_enums.StockBorderStyle.NONE,
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
    font_style: xmile_enums.StockFontStyle = dataclasses.field(
        default=xmile_enums.StockFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.StockFontWeight = dataclasses.field(
        default=xmile_enums.StockFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.StockTextDecoration = dataclasses.field(
        default=xmile_enums.StockTextDecoration.NORMAL,
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
    label_side: Optional[xmile_enums.StockLabelSide] = dataclasses.field(
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
  class Flow:
    pts: "XmileViewContentType.Flow.Pts" = dataclasses.field(
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
            "required": True,
        }
    )
    shape: Optional[xmile_shape.XmileShape] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    name: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.FlowBorderStyle] = dataclasses.field(
        default=xmile_enums.FlowBorderStyle.NONE,
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
    font_style: xmile_enums.FlowFontStyle = dataclasses.field(
        default=xmile_enums.FlowFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.FlowFontWeight = dataclasses.field(
        default=xmile_enums.FlowFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.FlowTextDecoration = dataclasses.field(
        default=xmile_enums.FlowTextDecoration.NORMAL,
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
    label_side: Optional[xmile_enums.FlowLabelSide] = dataclasses.field(
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
    class Pts:
      pt: List["XmileViewContentType.Flow.Pts.Pt"] = dataclasses.field(
          default_factory=list,
          metadata={
              "type": "Element",
              "namespace": xmile_globals.XMILE_NAMESPACE,
              "min_occurs": 1,
          },
      )

      @dataclasses.dataclass(kw_only=True)
      class Pt:
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

  @dataclasses.dataclass(kw_only=True)
  class Aux:
    shape: Optional[xmile_shape.XmileShape] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    name: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.AuxBorderStyle] = dataclasses.field(
        default=xmile_enums.AuxBorderStyle.NONE,
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
    font_style: xmile_enums.AuxFontStyle = dataclasses.field(
        default=xmile_enums.AuxFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.AuxFontWeight = dataclasses.field(
        default=xmile_enums.AuxFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.AuxTextDecoration = dataclasses.field(
        default=xmile_enums.AuxTextDecoration.NORMAL,
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
    label_side: Optional[xmile_enums.AuxLabelSide] = dataclasses.field(
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
  class Module:
    shape: Optional[xmile_shape.XmileShape] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    name: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.ModuleBorderStyle] = dataclasses.field(
        default=xmile_enums.ModuleBorderStyle.NONE,
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
    font_style: xmile_enums.ModuleFontStyle = dataclasses.field(
        default=xmile_enums.ModuleFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.ModuleFontWeight = dataclasses.field(
        default=xmile_enums.ModuleFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.ModuleTextDecoration = dataclasses.field(
        default=xmile_enums.ModuleTextDecoration.NORMAL,
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
    label_side: Optional[xmile_enums.ModuleLabelSide] = dataclasses.field(
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
  class Group:
    item: List["XmileViewContentType.Group.Item"] = dataclasses.field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    name: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    locked: Optional[bool] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.GroupBorderStyle] = dataclasses.field(
        default=xmile_enums.GroupBorderStyle.NONE,
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
    font_style: xmile_enums.GroupFontStyle = dataclasses.field(
        default=xmile_enums.GroupFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.GroupFontWeight = dataclasses.field(
        default=xmile_enums.GroupFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.GroupTextDecoration = dataclasses.field(
        default=xmile_enums.GroupTextDecoration.NORMAL,
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
    label_side: Optional[xmile_enums.GroupLabelSide] = dataclasses.field(
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
    class Item:
      uid: Optional[int] = dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )

  @dataclasses.dataclass(kw_only=True)
  class Connector:
    from_value: "XmileViewContentType.Connector.From" = dataclasses.field(
        metadata={
            "name": "from",
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
            "required": True,
        }
    )
    to: str = dataclasses.field(
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
            "required": True,
        }
    )
    uid: Optional[int] = dataclasses.field(
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
    angle: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    delay_mark: bool = dataclasses.field(
        default=False,
        metadata={
            "type": "Attribute",
        },
    )
    line_style: xmile_enums.ConnectorLineStyle = dataclasses.field(
        default=xmile_enums.ConnectorLineStyle.SOLID,
        metadata={
            "type": "Attribute",
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.ConnectorBorderStyle] = dataclasses.field(
        default=xmile_enums.ConnectorBorderStyle.NONE,
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
    font_style: xmile_enums.ConnectorFontStyle = dataclasses.field(
        default=xmile_enums.ConnectorFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.ConnectorFontWeight = dataclasses.field(
        default=xmile_enums.ConnectorFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.ConnectorTextDecoration = dataclasses.field(
        default=xmile_enums.ConnectorTextDecoration.NORMAL,
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

    @dataclasses.dataclass(kw_only=True)
    class From:
      content: List[object] = dataclasses.field(
          default_factory=list,
          metadata={
              "type": "Wildcard",
              "namespace": "##any",
              "mixed": True,
              "choices": (
                  {
                      "name": "alias",
                      "type": Type["XmileViewContentType.Connector.From.Alias"],
                      "namespace": xmile_globals.XMILE_NAMESPACE,
                  },
              ),
          },
      )

      @dataclasses.dataclass(kw_only=True)
      class Alias:
        uid: Optional[int] = dataclasses.field(
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

  @dataclasses.dataclass(kw_only=True)
  class Alias:
    shape: Optional[xmile_shape.XmileShape] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    of: str = dataclasses.field(
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
            "required": True,
        }
    )
    uid: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.AliasBorderStyle] = dataclasses.field(
        default=xmile_enums.AliasBorderStyle.NONE,
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
    font_style: xmile_enums.AliasFontStyle = dataclasses.field(
        default=xmile_enums.AliasFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.AliasFontWeight = dataclasses.field(
        default=xmile_enums.AliasFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.AliasTextDecoration = dataclasses.field(
        default=xmile_enums.AliasTextDecoration.NORMAL,
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
    label_side: Optional[xmile_enums.AliasLabelSide] = dataclasses.field(
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
  class Graph:
    plot: List[xmile_plot.XmilePlot] = dataclasses.field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
            "min_occurs": 1,
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: xmile_enums.GraphBorderStyle = dataclasses.field(
        default=xmile_enums.GraphBorderStyle.NONE,
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
    font_style: xmile_enums.GraphFontStyle = dataclasses.field(
        default=xmile_enums.GraphFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.GraphFontWeight = dataclasses.field(
        default=xmile_enums.GraphFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.GraphTextDecoration = dataclasses.field(
        default=xmile_enums.GraphTextDecoration.NORMAL,
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
    type_value: xmile_enums.GraphType = dataclasses.field(
        default=xmile_enums.GraphType.TIME_SERIES,
        metadata={
            "name": "type",
            "type": "Attribute",
        },
    )
    title: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    doc: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    show_grid: bool = dataclasses.field(
        default=True,
        metadata={
            "type": "Attribute",
        },
    )
    num_x_grid_lines: int = dataclasses.field(
        default=0,
        metadata={
            "type": "Attribute",
        },
    )
    num_y_grid_lines: int = dataclasses.field(
        default=0,
        metadata={
            "type": "Attribute",
        },
    )
    num_y_labels: int = dataclasses.field(
        default=0,
        metadata={
            "type": "Attribute",
        },
    )
    x_axis_title: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    right_axis_title: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    right_axis_auto_scale: bool = dataclasses.field(
        default=True,
        metadata={
            "type": "Attribute",
        },
    )
    right_axis_multi_scale: bool = dataclasses.field(
        default=True,
        metadata={
            "type": "Attribute",
        },
    )
    left_axis_title: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    left_axis_auto_scale: bool = dataclasses.field(
        default=True,
        metadata={
            "type": "Attribute",
        },
    )
    left_axis_multi_scale: bool = dataclasses.field(
        default=True,
        metadata={
            "type": "Attribute",
        },
    )
    plot_numbers: bool = dataclasses.field(
        default=False,
        metadata={
            "type": "Attribute",
        },
    )
    comparative: bool = dataclasses.field(
        default=False,
        metadata={
            "type": "Attribute",
        },
    )
    from_value: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "name": "from",
            "type": "Attribute",
        },
    )
    to: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )

  @dataclasses.dataclass(kw_only=True)
  class Table:
    column_width: int = dataclasses.field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    interval: Union[float, xmile_enums.ReportIntervalValue] = dataclasses.field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    item: List["XmileViewContentType.Table.Item"] = dataclasses.field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
            "min_occurs": 1,
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.TableBorderStyle] = dataclasses.field(
        default=xmile_enums.TableBorderStyle.NONE,
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
    font_style: xmile_enums.TableFontStyle = dataclasses.field(
        default=xmile_enums.TableFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.TableFontWeight = dataclasses.field(
        default=xmile_enums.TableFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.TableTextDecoration = dataclasses.field(
        default=xmile_enums.TableTextDecoration.NORMAL,
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
    title: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    doc: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    orientation: xmile_enums.Orientation = dataclasses.field(
        default=xmile_enums.Orientation.VERTICAL,
        metadata={
            "type": "Attribute",
        },
    )
    blank_column_width: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    report_balances: xmile_enums.ReportBalances = dataclasses.field(
        default=xmile_enums.ReportBalances.BEGINNING,
        metadata={
            "type": "Attribute",
        },
    )
    report_flows: xmile_enums.ReportFlows = dataclasses.field(
        default=xmile_enums.ReportFlows.INSTANTANEOUS,
        metadata={
            "type": "Attribute",
        },
    )
    comparative: bool = dataclasses.field(
        default=False,
        metadata={
            "type": "Attribute",
        },
    )
    wrap_text: bool = dataclasses.field(
        default=False,
        metadata={
            "type": "Attribute",
        },
    )
    header_border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    header_border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    header_border_style: xmile_enums.TableHeaderBorderStyle = dataclasses.field(
        default=xmile_enums.TableHeaderBorderStyle.NONE,
        metadata={
            "type": "Attribute",
        },
    )
    header_font_family: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    header_font_style: xmile_enums.TableHeaderFontStyle = dataclasses.field(
        default=xmile_enums.TableHeaderFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    header_font_weight: xmile_enums.TableHeaderFontWeight = dataclasses.field(
        default=xmile_enums.TableHeaderFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    header_text_decoration: xmile_enums.TableHeaderTextDecoration = (
        dataclasses.field(
            default=xmile_enums.TableHeaderTextDecoration.NORMAL,
            metadata={
                "type": "Attribute",
            },
        )
    )
    header_text_align: xmile_enums.TextAlign = dataclasses.field(
        default=xmile_enums.TextAlign.LEFT,
        metadata={
            "type": "Attribute",
        },
    )
    header_vertical_text_align: xmile_enums.VerticalTextAlign = (
        dataclasses.field(
            default=xmile_enums.VerticalTextAlign.CENTER,
            metadata={
                "type": "Attribute",
            },
        )
    )
    header_font_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    header_text_background: Optional[
        Union[xmile_enums.StandardColorType, str]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    header_font_size: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    header_padding: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    header_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    header_background: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    header_z_index: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )

    @dataclasses.dataclass(kw_only=True)
    class Item:
      entity: Optional["XmileViewContentType.Table.Item.Entity"] = (
          dataclasses.field(
              default=None,
              metadata={
                  "type": "Element",
                  "namespace": xmile_globals.XMILE_NAMESPACE,
              },
          )
      )
      border_width: Optional[
          Union[xmile_enums.StandardBorderWidthType, float]
      ] = dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
      border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
          dataclasses.field(
              default=None,
              metadata={
                  "type": "Attribute",
              },
          )
      )
      border_style: Optional[xmile_enums.ItemBorderStyle] = dataclasses.field(
          default=xmile_enums.ItemBorderStyle.NONE,
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
      font_style: xmile_enums.ItemFontStyle = dataclasses.field(
          default=xmile_enums.ItemFontStyle.NORMAL,
          metadata={
              "type": "Attribute",
          },
      )
      font_weight: xmile_enums.ItemFontWeight = dataclasses.field(
          default=xmile_enums.ItemFontWeight.NORMAL,
          metadata={
              "type": "Attribute",
          },
      )
      text_decoration: xmile_enums.ItemTextDecoration = dataclasses.field(
          default=xmile_enums.ItemTextDecoration.NORMAL,
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
      precision: Optional[float] = dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
      scale_by: int = dataclasses.field(
          default=1,
          metadata={
              "type": "Attribute",
          },
      )
      delimit_000s: bool = dataclasses.field(
          default=False,
          metadata={
              "type": "Attribute",
          },
      )
      display_as: xmile_enums.ItemDisplayAs = dataclasses.field(
          default=xmile_enums.ItemDisplayAs.NUMBER,
          metadata={
              "type": "Attribute",
          },
      )
      type_value: xmile_enums.TableItemType = dataclasses.field(
          default=xmile_enums.TableItemType.VARIABLE,
          metadata={
              "name": "type",
              "type": "Attribute",
          },
      )
      index: Optional[int] = dataclasses.field(
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
            }
        )
        index: Optional[str] = dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

  @dataclasses.dataclass(kw_only=True)
  class Button:
    image: Optional[xmile_image.XmileImage] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    video: Optional[xmile_video.XmileVideo] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    sound: Optional[xmile_sound.XmileSound] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    menu_action: Optional["XmileViewContentType.Button.MenuAction"] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Element",
                "namespace": xmile_globals.XMILE_NAMESPACE,
            },
        )
    )
    switch_action: Optional[xmile_switch_action.XmileSwitchAction] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Element",
                "namespace": xmile_globals.XMILE_NAMESPACE,
            },
        )
    )
    popup: Optional[xmile_popup.XmilePopup] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    link: Optional[xmile_link.XmileLink] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.ButtonBorderStyle] = dataclasses.field(
        default=xmile_enums.ButtonBorderStyle.NONE,
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
    font_style: xmile_enums.ButtonFontStyle = dataclasses.field(
        default=xmile_enums.ButtonFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.ButtonFontWeight = dataclasses.field(
        default=xmile_enums.ButtonFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.ButtonTextDecoration = dataclasses.field(
        default=xmile_enums.ButtonTextDecoration.NORMAL,
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
    label_side: Optional[xmile_enums.ButtonLabelSide] = dataclasses.field(
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
    transparency: xmile_enums.TransparencyType = dataclasses.field(
        default=xmile_enums.TransparencyType.OPAQUE,
        metadata={
            "type": "Attribute",
        },
    )
    style: xmile_enums.ButtonStyle = dataclasses.field(
        default=xmile_enums.ButtonStyle.SQUARE,
        metadata={
            "type": "Attribute",
        },
    )
    label: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    clicking_sound: bool = dataclasses.field(
        default=True,
        metadata={
            "type": "Attribute",
        },
    )
    show_name: bool = dataclasses.field(
        default=True,
        metadata={
            "type": "Attribute",
        },
    )
    show_number: bool = dataclasses.field(
        default=True,
        metadata={
            "type": "Attribute",
        },
    )
    show_min_max: bool = dataclasses.field(
        default=True,
        metadata={
            "type": "Attribute",
        },
    )
    name: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    uid: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )

    @dataclasses.dataclass(kw_only=True)
    class MenuAction:
      value: xmile_enums.MenuActionChoices = dataclasses.field(
          metadata={
              "required": True,
          }
      )
      resource: Optional[str] = dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
      worksheet: Optional[str] = dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )
      all: bool = dataclasses.field(
          default=False,
          metadata={
              "type": "Attribute",
          },
      )
      run_name: Optional[str] = dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
          },
      )

  @dataclasses.dataclass(kw_only=True)
  class Options:
    entity: List[xmile_entity.XmileEntity] = dataclasses.field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": xmile_globals.XMILE_NAMESPACE,
            "min_occurs": 1,
        },
    )
    border_width: Optional[
        Union[xmile_enums.StandardBorderWidthType, float]
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    border_color: Optional[Union[xmile_enums.StandardColorType, str]] = (
        dataclasses.field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )
    )
    border_style: Optional[xmile_enums.OptionsBorderStyle] = dataclasses.field(
        default=xmile_enums.OptionsBorderStyle.NONE,
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
    font_style: xmile_enums.OptionsFontStyle = dataclasses.field(
        default=xmile_enums.OptionsFontStyle.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    font_weight: xmile_enums.OptionsFontWeight = dataclasses.field(
        default=xmile_enums.OptionsFontWeight.NORMAL,
        metadata={
            "type": "Attribute",
        },
    )
    text_decoration: xmile_enums.OptionsTextDecoration = dataclasses.field(
        default=xmile_enums.OptionsTextDecoration.NORMAL,
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
    label_side: Optional[xmile_enums.OptionsLabelSide] = dataclasses.field(
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
    show_name: bool = dataclasses.field(
        default=False,
        metadata={
            "type": "Attribute",
        },
    )
    clicking_sound: bool = dataclasses.field(
        default=False,
        metadata={
            "type": "Attribute",
        },
    )
    layout: xmile_enums.OptionsLayout = dataclasses.field(
        default=xmile_enums.OptionsLayout.VERTICAL,
        metadata={
            "type": "Attribute",
        },
    )
    vertical_spacing: int = dataclasses.field(
        default=2,
        metadata={
            "type": "Attribute",
        },
    )
    horizontal_spacing: int = dataclasses.field(
        default=2,
        metadata={
            "type": "Attribute",
        },
    )
    uid: int = dataclasses.field(
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
