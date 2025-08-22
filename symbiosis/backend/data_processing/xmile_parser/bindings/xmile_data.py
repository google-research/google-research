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

"""Data class."""

import dataclasses
from typing import List
import xmile_export
import xmile_globals
import xmile_import_mod

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileData:
  """Data class."""

  class Meta:
    name = "data"
    namespace = xmile_globals.XMILE_NAMESPACE

  export: List[xmile_export.XmileExport] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  import_value: List[xmile_import_mod.XmileImport] = dataclasses.field(
      default_factory=list,
      metadata={"name": "import", "type": "Element"},
  )
