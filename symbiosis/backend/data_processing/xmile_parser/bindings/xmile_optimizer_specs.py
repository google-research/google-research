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

"""OptimizerSpecs - ISEE Standard."""

import dataclasses
from typing import List, Optional
import xmile_globals
import xmile_parameter
import xmile_payoff


__NAMESPACE__ = "http://iseesystems.com/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileOptimizerSpecs:
  """OptimizerSpecs - ISEE Standard."""

  class Meta:
    name = "optimizer_specs"
    namespace = xmile_globals.ISEE_NAMESPACE

  name: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      },
  )
  method: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  additional_starts: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      },
  )
  report_interval: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  confidence_range: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  init_step: Optional[float] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  parameter: List[xmile_parameter.XmileParameter] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
  payoff: Optional[xmile_payoff.XmilePayoff] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
