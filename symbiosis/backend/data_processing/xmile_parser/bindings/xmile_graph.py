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
import xmile_globals
import xmile_plot
import xmile_series_styles

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileGraph:

  class Meta:
    name = "graph"
    namespace = xmile_globals.XMILE_NAMESPACE

  color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  background: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  font_color: Optional[str] = dataclasses.field(
      default=None,
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
  font_weight: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  text_decoration: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  grid_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  graph_area_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  legend_position: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  transparent: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  hide_border: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  axis_title_font_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_font_weight: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_text_decoration: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_vertical_text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_font_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_font_family: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_text_padding: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_text_border_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_text_border_width: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_title_text_border_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_font_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_font_weight: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_text_decoration: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_vertical_text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_font_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_font_family: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_text_padding: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_text_border_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_text_border_width: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  axis_label_text_border_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  series_styles: Optional[xmile_series_styles.XmileSeriesStyles] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
              "namespace": xmile_globals.ISEE_NAMESPACE,
          },
      )
  )
  x: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  y: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  width: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  height: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  locked: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  comparative: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  comparative_override_thickness: Optional[Union[float, int]] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Attribute",
              "namespace": xmile_globals.ISEE_NAMESPACE,
          },
      )
  )
  stash: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  type_value: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "type",
          "type": "Attribute",
      },
  )
  sum_stacked_legend: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  stacked_show_hover_values: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  time_format_id: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  x_axis_auto_scale: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  connect_dots: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  single_point: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  points_only: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  time_precision: Optional[Union[int, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  time_scale_by: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  stacked_bar: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  show_grid: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  tick_type: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  include_units_in_legend: Optional[bool] = dataclasses.field(
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
  label_pie_slices: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  num_x_grid_lines: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  num_y_grid_lines: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  num_x_labels: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  page_title: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  num_y_labels: Optional[int] = dataclasses.field(
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
  fill_intensity: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  to: Optional[int] = dataclasses.field(
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
  stacked_area_alpha: Optional[float] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  x_axis_title: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  allow_blowup: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  allow_zero_axis: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  left_axis_title: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  left_axis_multi_scale: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  left_axis_auto_scale: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  left_include_units: Optional[bool] = dataclasses.field(
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
  stacked_area_y_max: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  right_axis_multi_scale: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  right_axis_auto_scale: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  right_include_units: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  plot: List[xmile_plot.XmilePlot] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
