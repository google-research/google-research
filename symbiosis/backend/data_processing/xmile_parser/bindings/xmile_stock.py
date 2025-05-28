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

"""Stock."""

import dataclasses
from decimal import Decimal
from typing import List, Optional, Union
import xmile_conveyor
import xmile_dimensions
import xmile_element
import xmile_event_poster
import xmile_format
import xmile_globals
import xmile_oven
import xmile_range
import xmile_scale
import xmile_shape


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileStock:
  """Core building block of a model, also called level or state.

  Stocks accumulate. Their value at the start of the simulation must
  be set as either a constant or with an initial equation. The initial
  equation is evaluated only once, at the beginning of the simulation.
  """

  class Meta:
    name = "stock"
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
  background: Optional[str] = dataclasses.field(
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
  font_family: Optional[str] = dataclasses.field(
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
  name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
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
  eqn: Optional[Union[int, str, float, Decimal]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
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
  doc: Optional[Union[str, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  element: List[xmile_element.XmileElement] = dataclasses.field(
      default_factory=list,
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
  dimensions: Optional[xmile_dimensions.XmileDimensions] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  conveyor: Optional[xmile_conveyor.XmileConveyor] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  oven: Optional[xmile_oven.XmileOven] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  queue: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  units: Optional[Union[str, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  shape: Optional[xmile_shape.XmileShape] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
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
