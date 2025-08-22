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

"""Scale class."""

import dataclasses
from typing import Optional, Union

import xmile_globals

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileScale:
  """Default scale for a variable output."""

  class Meta:
    name = "scale"
    namespace = xmile_globals.XMILE_NAMESPACE

  min: Optional[Union[int, str, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  max: Optional[Union[int, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
