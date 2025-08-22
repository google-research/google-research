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
from typing import Optional, Union
import xmile_globals


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileAction:

  class Meta:
    name = "action"
    namespace = xmile_globals.XMILE_NAMESPACE

  immediately_update_on_user_input: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  conditional_entity: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  target: Optional[str] = dataclasses.field(
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
  model: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  x: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  y: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  reset_after: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  reset_value: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  controller_value: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  iseesystems_com_xmile_entity: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "entity",
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  view: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  story: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  entity: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  conditional_attribute: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  conditional_attribute_value: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  time: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  url: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  value: str = dataclasses.field(default="")
  type_value: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "type",
          "type": "Attribute",
      },
  )
  sound: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
