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

"""Graphical function, alternately called lookup functions or table functions. SMILE Standard Version."""

import dataclasses
from typing import Optional
import xmile_globals
import xmile_xscale_smile
import xmile_yscale_smile

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileGfSmile:
  """Graphical function, alternately called lookup functions or table functions.

  They represent a (potentially) non-linear relationship. Graphical
  functions may occur as part of other variables or (with a name) as a
  stand alone variable.

  SMILE Standard Version.
  """

  class Meta:
    name = "gf"
    namespace = xmile_globals.SMILE_NAMESPACE

  xscale: Optional[xmile_xscale_smile.XmileXscaleSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  yscale: xmile_yscale_smile.XmileYscaleSmile = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      },
  )
  xpts: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  ypts: Optional[str] = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      },
  )
  type_value: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "type",
          "type": "Attribute",
      },
  )
