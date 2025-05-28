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

"""Control - ISEE Standard."""

import dataclasses
from typing import Optional, Union
import xmile_control
import xmile_exog
import xmile_gf
import xmile_globals

__NAMESPACE__ = "http://iseesystems.com/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileControl:
  """Control - ISEE Standard."""

  class Meta:
    name = "control"
    namespace = xmile_globals.ISEE_NAMESPACE

  priority: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  name: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  type_value: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "type",
          "type": "Attribute",
      },
  )
  exog: Optional[xmile_exog.XmileExog] = dataclasses.field(
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
          "namespace": xmile_globals.XMILE_NAMESPACE,
      },
  )
  value: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
