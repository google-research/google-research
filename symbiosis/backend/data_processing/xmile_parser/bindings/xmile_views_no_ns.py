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

"""List of views or diagrams pertaining to the model."""

import dataclasses
from typing import List, Optional
import xmile_globals
import xmile_view_no_ns


@dataclasses.dataclass(kw_only=True)
class XmileViewsNoNs:
  """List of views or diagrams pertaining to the model."""

  class Meta:
    name = "views"
    namespace = ""

  view: xmile_view_no_ns.XmileViewNoNs = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      },
  )
