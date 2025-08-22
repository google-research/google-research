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

"""PayoffSpecsList - ISEE standard."""

import dataclasses
from typing import List
import xmile_globals
import xmile_payoff_specs


__NAMESPACE__ = "http://iseesystems.com/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmilePayoffSpecsList:
  """PayoffSpecsList - ISEE standard."""

  class Meta:
    name = "payoff_specs_list"
    namespace = xmile_globals.ISEE_NAMESPACE

  payoff_specs: List[xmile_payoff_specs.XmilePayoffSpecs] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "required": True,
      },
  )
