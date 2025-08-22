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
import decimal
from typing import Optional, Union

import xmile_from
import xmile_globals
import xmile_pts


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileConnector:

  class Meta:
    name = "connector"
    namespace = xmile_globals.XMILE_NAMESPACE

  uid: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
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
  font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  thickness: Optional[Union[int, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  polarity: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  polarity_placement: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  delay_mark: Optional[bool] = dataclasses.field(
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
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  angle: Optional[Union[float, int, decimal.Decimal]] = dataclasses.field(
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
  pts: Optional[xmile_pts.XmilePts] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  from_value: Optional[Union[str, xmile_from.XmileFrom]] = dataclasses.field(
      default=None,
      metadata={
          "name": "from",
          "type": "Element",
      },
  )
  to: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
