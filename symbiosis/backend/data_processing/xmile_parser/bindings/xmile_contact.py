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

"""Contact class."""

import dataclasses
from typing import List

import xmile_globals

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileContact:
  """Contact class."""

  class Meta:
    name = "contact"
    namespace = xmile_globals.XMILE_NAMESPACE

  address: List[str] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  phone: List[str] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  fax: List[str] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  email: List[str] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  website: List[str] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
