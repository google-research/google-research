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

"""Auxiliaries class."""

import dataclasses
from decimal import Decimal
from typing import List, Optional, Union
import xmile_alias
import xmile_connector
import xmile_dimensions
import xmile_element
import xmile_enums
import xmile_event_poster
import xmile_flow
import xmile_format
import xmile_gf
import xmile_globals
import xmile_group
import xmile_range
import xmile_scale
import xmile_shape
import xmile_stock

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileAux:
  """Core building block of a model.

  Auxiliaries allow the isolation of any algebraic function that is
  used. They can both clarify a model and factor out important or
  repeated calculations. They can be defined using any algebraic
  expression (including a constant value), optionally in conjunction
  with a graphical function.
  """

  class Meta:
    name = "aux"
    namespace = xmile_globals.XMILE_NAMESPACE

  color: Optional[str] = dataclasses.field(
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
  background: Optional[str] = dataclasses.field(
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
  text_decoration: Optional[str] = dataclasses.field(
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
  label_angle: Optional[Union[int, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
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
  width: Optional[Union[int, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  height: Optional[Union[int, float]] = dataclasses.field(
      default=None,
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
  shape: Optional[xmile_shape.XmileShape] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  doc: Optional[Union[str, int, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  access: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  autocreated: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  flow_concept: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  eqn: Optional[Union[str, int, float, Decimal]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  dimensions: Optional[xmile_dimensions.XmileDimensions] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  element: List[xmile_element.XmileElement] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  format: Optional[xmile_format.XmileFormat] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  scale: Optional[xmile_scale.XmileScale] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  range: Optional[xmile_range.XmileRange] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  delay_aux: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  gf: Optional[xmile_gf.XmileGf] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  init_eqn: Optional[Union[str, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  summing: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  loopscore: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  pathscore: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  units: Optional[Union[str, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  aux: List["XmileAux"] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  connector: List[xmile_connector.XmileConnector] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  stock: List[xmile_stock.XmileStock] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  flow: List[xmile_flow.XmileFlow] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  alias: List[xmile_alias.XmileAlias] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  group: List[xmile_group.XmileGroup] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "sequence": 1,
      },
  )
  event_poster: Optional[xmile_event_poster.XmileEventPoster] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
          },
      )
  )
