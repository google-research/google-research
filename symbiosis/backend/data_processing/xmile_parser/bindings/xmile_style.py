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

"""Style class."""

import dataclasses
from typing import List, Optional

import xmile_animation_object_smile
import xmile_annotation
import xmile_aux
import xmile_button
import xmile_connector
import xmile_dual_slider
import xmile_financial_table
import xmile_flow
import xmile_gauge
import xmile_globals
import xmile_graph
import xmile_graphical_input
import xmile_group
import xmile_group_input
import xmile_iframe
import xmile_knob
import xmile_lamp
import xmile_loop_indicator
import xmile_module
import xmile_navigation_widget
import xmile_numeric_display
import xmile_numeric_input
import xmile_options
import xmile_pie_input
import xmile_placeholder
import xmile_selector
import xmile_shape_smile
import xmile_sim_speed_slider
import xmile_slider
import xmile_spatial_map
import xmile_stacked_container
import xmile_stock
import xmile_switch
import xmile_table
import xmile_text_box
import xmile_time_slider


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileStyle:

  class Meta:
    name = "style"
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
  font_style: Optional[str] = dataclasses.field(
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
  text_decoration: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  vertical_text_align: Optional[str] = dataclasses.field(
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
  font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
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
          "type": "Attribute",
      },
  )
  border_width: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  border_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  content: List[object] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Wildcard",
          "namespace": "##any",
          "mixed": True,
          "choices": (
              {
                  "name": "stock",
                  "type": xmile_stock.XmileStock,
              },
              {
                  "name": "flow",
                  "type": xmile_flow.XmileFlow,
              },
              {
                  "name": "placeholder",
                  "type": xmile_placeholder.XmilePlaceholder,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "module",
                  "type": xmile_module.XmileModule,
              },
              {
                  "name": "aux",
                  "type": xmile_aux.XmileAux,
              },
              {
                  "name": "group",
                  "type": xmile_group.XmileGroup,
              },
              {
                  "name": "connector",
                  "type": xmile_connector.XmileConnector,
              },
              {
                  "name": "text_box",
                  "type": xmile_text_box.XmileTextBox,
              },
              {
                  "name": "loop_indicator",
                  "type": xmile_loop_indicator.XmileLoopIndicator,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "numeric_display",
                  "type": xmile_numeric_display.XmileNumericDisplay,
              },
              {
                  "name": "graph",
                  "type": xmile_graph.XmileGraph,
              },
              {
                  "name": "table",
                  "type": xmile_table.XmileTable,
              },
              {
                  "name": "button",
                  "type": xmile_button.XmileButton,
              },
              {
                  "name": "annotation",
                  "type": xmile_annotation.XmileAnnotation,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "slider",
                  "type": xmile_slider.XmileSlider,
              },
              {
                  "name": "dual_slider",
                  "type": xmile_dual_slider.XmileDualSlider,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "sim_speed_slider",
                  "type": xmile_sim_speed_slider.XmileSimSpeedSlider,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "time_slider",
                  "type": xmile_time_slider.XmileTimeSlider,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "pie_input",
                  "type": xmile_pie_input.XmilePieInput,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "knob",
                  "type": xmile_knob.XmileKnob,
              },
              {
                  "name": "numeric_input",
                  "type": xmile_numeric_input.XmileNumericInput,
              },
              {
                  "name": "switch",
                  "type": xmile_switch.XmileSwitch,
              },
              {
                  "name": "options",
                  "type": xmile_options.XmileOptions,
              },
              {
                  "name": "graphical_input",
                  "type": xmile_graphical_input.XmileGraphicalInput,
              },
              {
                  "name": "group_input",
                  "type": xmile_group_input.XmileGroupInput,
              },
              {
                  "name": "lamp",
                  "type": xmile_lamp.XmileLamp,
              },
              {
                  "name": "gauge",
                  "type": xmile_gauge.XmileGauge,
              },
              {
                  "name": "spatial_map",
                  "type": xmile_spatial_map.XmileSpatialMap,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "animation_object",
                  "type": (
                      xmile_animation_object_smile.XmileAnimationObjectSmile
                  ),
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "navigation_widget",
                  "type": xmile_navigation_widget.XmileNavigationWidget,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "shape",
                  "type": xmile_shape_smile.XmileShapeSmile,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "selector",
                  "type": xmile_selector.XmileSelector,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "iframe",
                  "type": xmile_iframe.XmileIframe,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "financial_table",
                  "type": xmile_financial_table.XmileFinancialTable,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "stacked_container",
                  "type": xmile_stacked_container.XmileStackedContainer,
              },
          ),
      },
  )
