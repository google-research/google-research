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

"""The root element for the XMILE - SMILE version file."""

import dataclasses
from typing import Optional

import xmile_globals


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileSmile:
  """The root element for the XMILE - SMILE version file.

  Note that certain child elements (i.e. model) are required for the
  primary file but not for included files.
  """

  class Meta:
    name = "smile"
    namespace = xmile_globals.XMILE_NAMESPACE

  version: float = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  namespace: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  uses_arrays: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  uses_conveyor: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  uses_submodels: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  uses_queue: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
