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

"""Graphics Frame."""

import dataclasses
from typing import Optional, Union
import xmile_enums
import xmile_globals
import xmile_image

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileGraphicsFrame:
  """Graphics Frame."""

  class Meta:
    name = "graphics_frame"
    namespace = xmile_globals.XMILE_NAMESPACE

  border_width: Optional[Union[str, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  border_style: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  z_index: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  border_color: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  fill: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  uid: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  x: Union[int, float] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  y: Union[int, float] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  width: Union[int, float] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  height: Union[int, float] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  locked: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  image: Optional[xmile_image.XmileImage] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  video: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
