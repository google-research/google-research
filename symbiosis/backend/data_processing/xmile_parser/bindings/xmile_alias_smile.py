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

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileAliasSmile:

  class Meta:
    name = "alias"
    namespace = xmile_globals.SMILE_NAMESPACE

  font_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-style",
          "type": "Attribute",
      },
  )
  uid: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
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
  label_side: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  of: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
