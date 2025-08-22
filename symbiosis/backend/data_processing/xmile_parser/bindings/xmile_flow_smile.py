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

"""Core building block of a model, also called a rate or a derivative."""

import dataclasses
from typing import List, Optional, Union

import xmile_alias_smile
import xmile_connector_smile
import xmile_gf_smile
import xmile_globals
import xmile_graph_smile
import xmile_group_smile
import xmile_interface_smile
import xmile_module_smile
import xmile_numeric_display_smile
import xmile_pad_icon_smile
import xmile_pts_smile
import xmile_stacked_container_smile
import xmile_table_smile
import xmile_text_box_smile


__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileFlowSmile:
  """Core building block of a model, also called a rate or a derivative.

  Flows represent the rate of change of a stock. Each timestep the
  flow multiplied by the timestep is added to the stock.

  SMILE standard version.
  """

  class Meta:
    name = "flow"
    namespace = xmile_globals.SMILE_NAMESPACE

  name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  doc: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  eqn: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  non_negative: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  units: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  display: Optional["XmileDisplaySmile"] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  label_side: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  label_angle: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )


@dataclasses.dataclass(kw_only=True)
class XmileStockSmile:

  class Meta:
    name = "stock"
    namespace = xmile_globals.SMILE_NAMESPACE

  name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  eqn: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  inflow: List[str] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  outflow: List[str] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  non_negative: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  units: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  display: Optional["XmileDisplaySmile"] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  label_side: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  label_angle: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )


@dataclasses.dataclass(kw_only=True)
class XmileModelSmile:

  class Meta:
    name = "model"
    namespace = xmile_globals.SMILE_NAMESPACE

  stock: List[XmileStockSmile] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  flow: List[XmileFlowSmile] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  aux: List["XmileAuxSmile"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  display: Optional["XmileDisplaySmile"] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  interface: Optional[xmile_interface_smile.XmileInterfaceSmile] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
          },
      )
  )
  stories: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  style: Optional["XmileStyleSmile"] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  font_name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-name",
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
  font_size: Optional[Union[str, int]] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-size",
          "type": "Attribute",
      },
  )


@dataclasses.dataclass(kw_only=True)
class XmileStyleSmile:

  class Meta:
    name = "style"
    namespace = xmile_globals.SMILE_NAMESPACE

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
  stock: Optional[XmileStockSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  flow: Optional[XmileFlowSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  module: Optional[xmile_module_smile.XmileModuleSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  aux: Optional["XmileAuxSmile"] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  group: Optional[xmile_group_smile.XmileGroupSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  connector: Optional[xmile_connector_smile.XmileConnectorSmile] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
          },
      )
  )
  text_box: Optional[xmile_text_box_smile.XmileTextBoxSmile] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
          },
      )
  )
  numeric_display: Optional[
      xmile_numeric_display_smile.XmileNumericDisplaySmile
  ] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  graph: Optional[xmile_graph_smile.XmileGraphSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  table: Optional[xmile_table_smile.XmileTableSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  model: Optional[XmileModelSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  interface: Optional[xmile_interface_smile.XmileInterfaceSmile] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
          },
      )
  )


@dataclasses.dataclass(kw_only=True)
class XmileDisplaySmile:

  class Meta:
    name = "display"
    namespace = xmile_globals.SMILE_NAMESPACE

  show_pages: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
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
  page_rows: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  converter_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  animate_flows: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  animate_stocks: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  animate_converters: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  report_balances: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  report_flows: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  alias: Optional[xmile_alias_smile.XmileAliasSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  page_cols: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
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
  connector: List[xmile_connector_smile.XmileConnectorSmile] = (
      dataclasses.field(
          default_factory=list,
          metadata={
              "type": "Element",
          },
      )
  )
  stacked_container: Optional[
      xmile_stacked_container_smile.XmileStackedContainerSmile
  ] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  pad_icon: Optional[xmile_pad_icon_smile.XmilePadIconSmile] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
          },
      )
  )
  style: Optional[XmileStyleSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  label_side: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  label_angle: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  x: Optional[Union[int, float]] = dataclasses.field(
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
  width: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  height: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  pts: Optional[xmile_pts_smile.XmilePtsSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )


@dataclasses.dataclass(kw_only=True)
class XmileAuxSmile:

  class Meta:
    name = "aux"
    namespace = xmile_globals.SMILE_NAMESPACE

  name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  doc: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  flow_concept: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  eqn: Optional[Union[str, float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  gf: Optional[xmile_gf_smile.XmileGfSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  units: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  display: Optional[XmileDisplaySmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  label_side: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  label_angle: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
