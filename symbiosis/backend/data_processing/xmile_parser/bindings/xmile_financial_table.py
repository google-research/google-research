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

"""Financial Table."""

import dataclasses
from typing import Optional
import xmile_globals

__NAMESPACE__ = "http://iseesystems.com/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileFinancialTable:

  class Meta:
    name = "financial_table"
    namespace = xmile_globals.ISEE_NAMESPACE

  color: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  background: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  text_align: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  font_family: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  font_size: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  hide_border: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  auto_fit: bool = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  first_column_width: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  other_column_width: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_font_style: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_font_weight: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_decoration: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_align: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_vertical_text_align: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_font_color: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_font_family: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_font_size: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_padding: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_border_color: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_border_width: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header_text_border_style: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
