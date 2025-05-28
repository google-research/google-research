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
from typing import List, Optional, Union
from xsdata.models.datatype import XmlDateTime
import xmile_globals
import xmile_plot_smile

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileGraphSmile:

  class Meta:
    name = "graph"
    namespace = xmile_globals.SMILE_NAMESPACE

  color: Optional[str] = dataclasses.field(
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
  background: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  font_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-style",
          "type": "Attribute",
      },
  )
  font_weight: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-weight",
          "type": "Attribute",
      },
  )
  text_decoration: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "text-decoration",
          "type": "Attribute",
      },
  )
  text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "text-align",
          "type": "Attribute",
      },
  )
  vertical_text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "vertical-text-align",
          "type": "Attribute",
      },
  )
  text_background: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "text-background",
          "type": "Attribute",
      },
  )
  font_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-color",
          "type": "Attribute",
      },
  )
  font_family: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-family",
          "type": "Attribute",
      },
  )
  font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-size",
          "type": "Attribute",
      },
  )
  padding: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  border_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "border-color",
          "type": "Attribute",
      },
  )
  border_width: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "border-width",
          "type": "Attribute",
      },
  )
  border_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "border-style",
          "type": "Attribute",
      },
  )
  axis_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-color",
          "type": "Attribute",
      },
  )
  grid_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "grid-color",
          "type": "Attribute",
      },
  )
  legend_position: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "legend-position",
          "type": "Attribute",
      },
  )
  axis_title_font_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-font-style",
          "type": "Attribute",
      },
  )
  axis_title_font_weight: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-font-weight",
          "type": "Attribute",
      },
  )
  axis_title_text_decoration: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-text-decoration",
          "type": "Attribute",
      },
  )
  axis_title_text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-text-align",
          "type": "Attribute",
      },
  )
  axis_title_vertical_text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-vertical-text-align",
          "type": "Attribute",
      },
  )
  axis_title_text_background: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-text-background",
          "type": "Attribute",
      },
  )
  axis_title_font_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-font-color",
          "type": "Attribute",
      },
  )
  axis_title_font_family: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-font-family",
          "type": "Attribute",
      },
  )
  axis_title_font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-font-size",
          "type": "Attribute",
      },
  )
  axis_title_text_padding: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-text-padding",
          "type": "Attribute",
      },
  )
  axis_title_text_border_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-text-border-color",
          "type": "Attribute",
      },
  )
  axis_title_text_border_width: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-text-border-width",
          "type": "Attribute",
      },
  )
  axis_title_text_border_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-title-text-border-style",
          "type": "Attribute",
      },
  )
  axis_label_font_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-font-style",
          "type": "Attribute",
      },
  )
  axis_label_font_weight: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-font-weight",
          "type": "Attribute",
      },
  )
  axis_label_text_decoration: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-text-decoration",
          "type": "Attribute",
      },
  )
  axis_label_text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-text-align",
          "type": "Attribute",
      },
  )
  axis_label_vertical_text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-vertical-text-align",
          "type": "Attribute",
      },
  )
  axis_label_text_background: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-text-background",
          "type": "Attribute",
      },
  )
  axis_label_font_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-font-color",
          "type": "Attribute",
      },
  )
  axis_label_font_family: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-font-family",
          "type": "Attribute",
      },
  )
  axis_label_font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-font-size",
          "type": "Attribute",
      },
  )
  axis_label_text_padding: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-text-padding",
          "type": "Attribute",
      },
  )
  axis_label_text_border_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-text-border-color",
          "type": "Attribute",
      },
  )
  axis_label_text_border_width: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-text-border-width",
          "type": "Attribute",
      },
  )
  axis_label_text_border_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "axis-label-text-border-style",
          "type": "Attribute",
      },
  )
  show_grid: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  plot_numbers: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  use_five_segments: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  date_time: Optional[XmlDateTime] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  time_precision: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  from_value: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "name": "from",
          "type": "Attribute",
      },
  )
  to: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  title: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  plot: List[xmile_plot_smile.XmilePlotSmile] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
