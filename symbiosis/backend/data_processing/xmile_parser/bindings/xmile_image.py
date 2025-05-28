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

"""Picture of the model in JPG, GIF, TIF, or PNG format."""

import dataclasses
import xmile_globals
from typing import Optional

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileImage:
  """Picture of the model in JPG, GIF, TIF, or PNG format.

  The resource attribute is optional and may specify a relative file
  path, an absolute file path, or a URL.  The picture data may also be
  embedded inside the tag in Data URI format, using base64 encoding.
  """

  class Meta:
    name = "image"
    namespace = xmile_globals.XMILE_NAMESPACE

  value: str = dataclasses.field(
      default="",
  )
  size_to_parent: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
      },
  )
  width: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      },
  )
  height: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      },
  )
  fixed_aspect_ratio: Optional[bool] = dataclasses.field(
      default=None,
      metadata={"type": "Attribute", "namespace": xmile_globals.ISEE_NAMESPACE},
  )
