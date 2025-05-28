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
from typing import List, Optional, Type, Union
import xmile_alias
import xmile_annotation
import xmile_aux
import xmile_button
import xmile_chapter
import xmile_connector
import xmile_extra_collection_variables
import xmile_flow
import xmile_gauge
import xmile_globals
import xmile_graphical_input
import xmile_graphics_frame
import xmile_group
import xmile_highlight_list
import xmile_knob
import xmile_loop_indicator
import xmile_module
import xmile_navigation_widget
import xmile_numeric_display
import xmile_numeric_input
import xmile_options
import xmile_pie_input
import xmile_selector
import xmile_shape_smile
import xmile_slider
import xmile_stacked_container
import xmile_stock
import xmile_style
import xmile_switch
import xmile_text_box
import xmile_variables
import xmile_views_no_ns
import xmile_visible_list


@dataclasses.dataclass(kw_only=True)
class XmileTemplates:

  class Meta:
    name = "templates"
    namespace = xmile_globals.ISEE_NAMESPACE

  view: List["XmileView"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
          "min_occurs": 1,
      },
  )


@dataclasses.dataclass(kw_only=True)
class XmileModelView:

  class Meta:
    name = "model_view"
    namespace = xmile_globals.ISEE_NAMESPACE

  uid: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  x: Union[int, float] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  y: Union[int, float] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  width: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  height: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  locked: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  zoom: Union[float, int] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  center_x: Union[int, float] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  center_y: Union[float, int] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  locked_view: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  transparent: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      }
  )
  use_visible_list: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  show_ltm_animation: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  show_ltm_table: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  include_equation: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  include_units: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  include_documentation: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  model: "XmileModel" = dataclasses.field(
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
          "required": True,
      }
  )
  visible_list: Optional[xmile_visible_list.XmileVisibleList] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
          },
      )
  )
  highlight_list: xmile_highlight_list.XmileHighlightList = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )


@dataclasses.dataclass(kw_only=True)
class XmileView:

  class Meta:
    name = "view"
    namespace = xmile_globals.XMILE_NAMESPACE

  show_pages: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  template_view: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  background: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  use_lettered_polarity: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  converter_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  name_only_modules: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  page_order: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  page_width: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  page_height: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  page_cols: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  page_rows: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  scroll_x: Optional[Union[int, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  scroll_y: Optional[Union[int, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  zoom: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  popup_graphs_are_comparative: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  enable_non_negative_highlights: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  home_view: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  disable_non_negative_highlights: Optional[bool] = dataclasses.field(
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
  content: List[object] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Wildcard",
          "namespace": "##any",
          "mixed": True,
          "choices": (
              {
                  "name": "style",
                  "type": xmile_style.XmileStyle,
              },
              {
                  "name": "graphics_frame",
                  "type": xmile_graphics_frame.XmileGraphicsFrame,
              },
              {
                  "name": "group",
                  "type": xmile_group.XmileGroup,
              },
              {
                  "name": "aux",
                  "type": xmile_aux.XmileAux,
              },
              {
                  "name": "connector",
                  "type": xmile_connector.XmileConnector,
              },
              {
                  "name": "model_view",
                  "type": XmileModelView,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "graphical_input",
                  "type": xmile_graphical_input.XmileGraphicalInput,
              },
              {
                  "name": "knob",
                  "type": xmile_knob.XmileKnob,
              },
              {
                  "name": "gauge",
                  "type": xmile_gauge.XmileGauge,
              },
              {
                  "name": "slider",
                  "type": xmile_slider.XmileSlider,
              },
              {
                  "name": "stacked_container",
                  "type": xmile_stacked_container.XmileStackedContainer,
              },
              {
                  "name": "flow",
                  "type": xmile_flow.XmileFlow,
              },
              {
                  "name": "stock",
                  "type": xmile_stock.XmileStock,
              },
              {
                  "name": "numeric_display",
                  "type": xmile_numeric_display.XmileNumericDisplay,
              },
              {
                  "name": "module",
                  "type": xmile_module.XmileModule,
              },
              {
                  "name": "loop_indicator",
                  "type": xmile_loop_indicator.XmileLoopIndicator,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "numeric_input",
                  "type": xmile_numeric_input.XmileNumericInput,
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
                  "name": "navigation_widget",
                  "type": xmile_navigation_widget.XmileNavigationWidget,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "options",
                  "type": xmile_options.XmileOptions,
              },
              {
                  "name": "switch",
                  "type": xmile_switch.XmileSwitch,
              },
              {
                  "name": "shape",
                  "type": xmile_shape_smile.XmileShapeSmile,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "text_box",
                  "type": xmile_text_box.XmileTextBox,
              },
              {
                  "name": "alias",
                  "type": xmile_alias.XmileAlias,
              },
              {
                  "name": "pie_input",
                  "type": xmile_pie_input.XmilePieInput,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "view",
                  "type": Type["XmileView"],
              },
              {
                  "name": "templates",
                  "type": XmileTemplates,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "selector",
                  "type": xmile_selector.XmileSelector,
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
              {
                  "name": "simulation_delay",
                  "type": Union[float, str],
                  "namespace": xmile_globals.ISEE_NAMESPACE,
              },
          ),
      },
  )


@dataclasses.dataclass(kw_only=True)
class XmileStory:

  class Meta:
    name = "story"
    namespace = xmile_globals.ISEE_NAMESPACE

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
  view: List[XmileView] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  chapter: Optional[xmile_chapter.XmileChapter] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )


@dataclasses.dataclass(kw_only=True)
class XmileStories:

  class Meta:
    name = "stories"
    namespace = xmile_globals.ISEE_NAMESPACE

  story: List[XmileStory] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "min_occurs": 1,
      },
  )


@dataclasses.dataclass(kw_only=True)
class XmileViews:

  class Meta:
    name = "views"
    namespace = xmile_globals.XMILE_NAMESPACE

  style: Optional[xmile_style.XmileStyle] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  view: List[XmileView] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "min_occurs": 1,
      },
  )
  templates: Optional[XmileTemplates] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  stories: Optional[XmileStories] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )


@dataclasses.dataclass(kw_only=True)
class XmileModel:

  class Meta:
    name = "model"
    namespace = xmile_globals.XMILE_NAMESPACE

  name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  encryption_scheme: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "encryption-scheme",
          "type": "Attribute",
      },
  )
  encryption_key_type: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "encryption-key-type",
          "type": "Attribute",
      },
  )
  iv: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  hmac: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  implicit_lock: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  value: str = dataclasses.field(default="")
  collect_data: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  require_server_engine: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  collect_page_tracking_data: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  extra_collection_variables: Optional[
      xmile_extra_collection_variables.XmileExtraCollectionVariables
  ] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  run: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  variables: Optional[xmile_variables.XmileVariables] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  docs_oasis_open_org_xmile_ns_xmile_v1_0_views: Optional[XmileViews] = (
      dataclasses.field(
          default=None,
          metadata={
              "name": "views",
              "type": "Element",
          },
      )
  )
  views: Optional[xmile_views_no_ns.XmileViewsNoNs] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": "",
      },
  )
