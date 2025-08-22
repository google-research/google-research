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

"""Event."""

import dataclasses
from typing import Optional
import xmile_globals
import xmile_link


__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileEvent:
  """Event."""

  class Meta:
    name = "event"
    namespace = xmile_globals.XMILE_NAMESPACE

  for_role: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  sim_action: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  kind: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
          "required": True,
      }
  )
  link: Optional[xmile_link.XmileLink] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  text_box: str = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
