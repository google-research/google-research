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

"""Item class."""

import dataclasses
from typing import Optional, Union
import xmile_data_smile
import xmile_entity
import xmile_format
import xmile_globals

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileItem:
  """Item class."""

  class Meta:
    name = "item"
    namespace = xmile_globals.XMILE_NAMESPACE

  type_value: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "type",
          "type": "Attribute",
      },
  )
  format: Optional[xmile_format.XmileFormat] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  entity: Optional[xmile_entity.XmileEntity] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  data: Optional[xmile_data_smile.XmileDataSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  uid: Optional[Union[int, str]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
