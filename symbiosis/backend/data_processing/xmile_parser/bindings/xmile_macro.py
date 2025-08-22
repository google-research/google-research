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

"""Defines macros that provide custom functions or building blocks that can be

used in model entity equations.
"""

import dataclasses
from typing import List, Optional, Union
import xmile_enums
import xmile_globals
import xmile_model_view
import xmile_parm
import xmile_variables

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileMacro:
  """Defines macros that provide custom functions or building blocks that can be

  used in model entity equations.
  """

  class Meta:
    name = "macro"
    namespace = xmile_globals.XMILE_NAMESPACE

  name: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      },
  )
  eqn: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  parm: List[Union[str, xmile_parm.XmileParm]] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "min_occurs": 1,
      },
  )
  variables: xmile_variables.XmileVariables = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      },
  )
  views: Optional[xmile_model_view.XmileViews] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  view: Optional[xmile_model_view.XmileView] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
