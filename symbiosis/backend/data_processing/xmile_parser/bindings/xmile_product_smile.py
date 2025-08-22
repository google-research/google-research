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
import xmile_globals

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileProductSmile:

  class Meta:
    name = "product"
    namespace = xmile_globals.SMILE_NAMESPACE

  version: Optional[Union[str, float]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  saved_by_v1: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  compatibility: Optional[float] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  lang: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  value: str = dataclasses.field(default="")
