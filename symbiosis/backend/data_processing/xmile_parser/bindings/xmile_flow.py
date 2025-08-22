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
import xmile_dimensions
import xmile_element
import xmile_format
import xmile_gf
import xmile_pts
import xmile_range
import xmile_globals


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileFlow:
  """Core building block of a model, also called a rate or a derivative.

  Flows represent the rate of change of a stock. Each timestep the
  flow multiplied by the timestep is added to the stock.
  """

  class Meta:
    name = "flow"
    namespace = xmile_globals.XMILE_NAMESPACE

  color: Optional[str] = dataclasses.field(
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
  segment_with_valve: Optional[int] = dataclasses.field(
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
  font_size: Optional[str] = dataclasses.field(
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
  name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  pts: Optional[xmile_pts.XmilePts] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  spreadflow: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  doc: Optional[str] = dataclasses.field(
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
  dimensions: Optional[xmile_dimensions.XmileDimensions] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  timestamped: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  access: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  eqn: Optional[Union[str, int, float]] = dataclasses.field(
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
  leak: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  init_eqn: Optional[Union[int, str]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  gf: Optional[xmile_gf.XmileGf] = dataclasses.field(
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
  range: Optional[xmile_range.XmileRange] = dataclasses.field(
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
