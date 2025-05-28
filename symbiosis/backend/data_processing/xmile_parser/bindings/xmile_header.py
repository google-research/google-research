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

"""Provides information about the origin of the model and required capabilities."""

import dataclasses
from typing import Optional

import xmile_globals
import xmile_options
import xmile_product
import xmile_smile

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileHeader:
  """Provides information about the origin of the model and required capabilities.

  Note that certain child elements (i.e. product, options, vendor) are
  required for the primary file but not for included files.
  """

  class Meta:
    name = "header"
    namespace = xmile_globals.XMILE_NAMESPACE

  smile: Optional[xmile_smile.XmileSmile] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  name: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  uuid: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  options: Optional[xmile_options.XmileOptions] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  vendor: str = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
  product: xmile_product.XmileProduct = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
