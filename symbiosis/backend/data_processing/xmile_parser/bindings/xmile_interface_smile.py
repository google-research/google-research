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
import xmile_button_smile
import xmile_globals

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileInterfaceSmile:

  class Meta:
    name = "interface"
    namespace = xmile_globals.SMILE_NAMESPACE

  show_pages: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  page_width: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  page_height: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  simulation_delay: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  pages: Optional[object] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  button: Optional[xmile_button_smile.XmileButtonSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  font_name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-name",
          "type": "Attribute",
      },
  )
  font_family: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-family",
          "type": "Attribute",
      },
  )
  font_size: Optional[Union[str, int]] = dataclasses.field(
      default=None,
      metadata={
          "name": "font-size",
          "type": "Attribute",
      },
  )
