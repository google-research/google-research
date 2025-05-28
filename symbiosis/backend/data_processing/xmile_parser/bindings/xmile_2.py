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
from typing import List

import xmile_data
import xmile_dimensions
import xmile_dimensions_smile
import xmile_equation_prefs
import xmile_flow_smile
import xmile_globals
import xmile_header
import xmile_header_smile
import xmile_model_units
import xmile_model_units_smile
import xmile_model_view
import xmile_prefs
import xmile_sensi_specs
import xmile_sim_specs
import xmile_sim_specs_smile


__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class Xmile2:

  class Meta:
    name = "xmile"
    namespace = xmile_globals.SMILE_NAMESPACE

  version: float = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  level: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  header: xmile_header.XmileHeader = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
          "namespace": xmile_globals.XMILE_NAMESPACE,
      }
  )
  header_smile: xmile_header_smile.XmileHeaderSmile = dataclasses.field(
      metadata={
          "name": "header",
          "type": "Element",
          "required": True,
      }
  )
  sim_specs: xmile_sim_specs.XmileSimSpecs = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
          "namespace": xmile_globals.XMILE_NAMESPACE,
      }
  )
  sim_specs_smile: xmile_sim_specs_smile.XmileSimSpecsSmile = dataclasses.field(
      metadata={
          "name": "sim_specs",
          "type": "Element",
          "required": True,
      }
  )
  sensi_specs: xmile_sensi_specs.XmileSensiSpecs = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
          "namespace": xmile_globals.ISEE_NAMESPACE,
      }
  )
  dimensions: xmile_dimensions.XmileDimensions = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
          "namespace": xmile_globals.XMILE_NAMESPACE,
      }
  )
  dimensions_smile: xmile_dimensions_smile.XmileDimensionsSmile = (
      dataclasses.field(
          metadata={
              "name": "dimensions",
              "type": "Element",
              "required": True,
          }
      )
  )
  prefs: List[xmile_prefs.XmilePrefs] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
          "min_occurs": 1,
      },
  )
  equation_prefs: xmile_equation_prefs.XmileEquationPrefs = dataclasses.field(
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
          "required": True,
      }
  )
  style: xmile_flow_smile.XmileStyleSmile = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      }
  )
  model_units: xmile_model_units.XmileModelUnits = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
          "namespace": xmile_globals.XMILE_NAMESPACE,
      }
  )
  model_units_smile: xmile_model_units_smile.XmileModelUnitsSmile = dataclasses.field(
      metadata={
          "name": "model_units",
          "type": "Element",
          "required": True,
      }
  )
  model: xmile_model_view.XmileModel = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
          "namespace": xmile_globals.XMILE_NAMESPACE,
      }
  )
  model_smile: xmile_flow_smile.XmileModelSmile = dataclasses.field(
      metadata={
          "name": "model",
          "type": "Element",
          "required": True,
      }
  )
  data: xmile_data.XmileData = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
          "namespace": xmile_globals.XMILE_NAMESPACE,
      }
  )
