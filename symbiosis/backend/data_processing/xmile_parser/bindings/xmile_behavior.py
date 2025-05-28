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

"""Simulation behavior definitions that are inherited/cascaded through all models defined in this XMILE document."""

import dataclasses
from typing import Optional
import xmile_boolean_or_empty_type
import xmile_globals

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileBehavior:
  """Simulation behavior definitions that are inherited/cascaded through all models defined in this XMILE document."""

  class Meta:
    name = "behavior"
    namespace = xmile_globals.XMILE_NAMESPACE

  non_negative: Optional[
      xmile_boolean_or_empty_type.XmileBooleanOrEmptyType
  ] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  stock: Optional["XmileBehavior.Stock"] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  flow: Optional["XmileBehavior.Flow"] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "required": True,
      },
  )

  @dataclasses.dataclass(kw_only=True)
  class Stock:
    non_negative: Optional[
        xmile_boolean_or_empty_type.XmileBooleanOrEmptyType
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
        },
    )

  @dataclasses.dataclass(kw_only=True)
  class Flow:
    non_negative: Optional[
        xmile_boolean_or_empty_type.XmileBooleanOrEmptyType
    ] = dataclasses.field(
        default=None,
        metadata={
            "type": "Element",
        },
    )
