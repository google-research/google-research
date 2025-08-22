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

"""Grouping of variables, also called a sector or view."""

import dataclasses
from typing import List, Optional

import xmile_globals

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileGroupSmile:
  """A grouping of variables, also called a sector or view.

  SMILE Standard version.
  """

  class Meta:
    name = "group"
    namespace = xmile_globals.SMILE_NAMESPACE

  color: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  font_color: str = dataclasses.field(
      metadata={
          "name": "font-color",
          "type": "Attribute",
          "required": True,
      }
  )
