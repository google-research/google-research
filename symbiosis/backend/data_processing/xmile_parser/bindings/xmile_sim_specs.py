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

"""Default simulation specifications for this model."""

import dataclasses
import decimal
from typing import Optional, Union
import xmile_dt
import xmile_enums
import xmile_globals
import xmile_run

__NAMESPACE__ = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"


@dataclasses.dataclass(kw_only=True)
class XmileSimSpecs:
  """Default simulation specifications for this model."""

  class Meta:
    name = "sim_specs"
    namespace = xmile_globals.XMILE_NAMESPACE

  sim_duration: Optional[Union[float, int, str]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  run_prefix: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  simulation_delay: Optional[Union[float, int]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  restore_on_start: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  save_interval: Optional[Union[int, float, decimal.Decimal]] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  method: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  time_units: Union[str, int] = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  pause_interval: Optional[float] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
      },
  )
  pause_after_rates: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  instantaneous_flows: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  ignore_module_errors: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  loop_scores: Optional[bool] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  loop_exhaustive_allowed: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  interaction_mode: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Attribute",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  start: Union[int, decimal.Decimal] = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
  stop: Union[int, str, decimal.Decimal] = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
  dt: Union[xmile_dt.XmileDt, float, int, decimal.Decimal] = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
  run: Optional[xmile_run.XmileRun] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
