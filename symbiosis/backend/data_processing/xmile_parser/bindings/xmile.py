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

"""The root element for the XMILE file."""

import dataclasses
from typing import List, Optional

import xmile_controls
import xmile_data
import xmile_default_format
import xmile_dimensions
import xmile_globals
import xmile_header
import xmile_macro
import xmile_model_units
import xmile_model_view
import xmile_multiplayer_settings
import xmile_optimizer_specs_list
import xmile_payoff_specs_list
import xmile_prefs
import xmile_sensi_specs
import xmile_sensi_specs_list
import xmile_sim_specs
import xmile_time_formats


@dataclasses.dataclass(kw_only=True)
class Xmile:
  """The root element for the XMILE file.

  Note that certain child elements (i.e. model) are required for the
  primary file but not for included files.
  """

  class Meta:
    name = "xmile"
    namespace = xmile_globals.XMILE_NAMESPACE

  version: float = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      },
  )
  header: xmile_header.XmileHeader = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      },
  )
  sim_specs: xmile_sim_specs.XmileSimSpecs = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      },
  )
  dimensions: Optional[xmile_dimensions.XmileDimensions] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "required": True,
      },
  )
  prefs: xmile_prefs.XmilePrefs = dataclasses.field(
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
          "required": True,
      },
  )
  multiplayer_settings: Optional[
      xmile_multiplayer_settings.XmileMultiplayerSettings
  ] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  time_formats: Optional[xmile_time_formats.XmileTimeFormats] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
              "namespace": xmile_globals.ISEE_NAMESPACE,
          },
      )
  )
  default_format: Optional[xmile_default_format.XmileDefaultFormat] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
          },
      )
  )
  data: Optional[xmile_data.XmileData] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
      },
  )
  sensi_specs_list: Optional[xmile_sensi_specs_list.XmileSensiSpecsList] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
              "namespace": xmile_globals.ISEE_NAMESPACE,
          },
      )
  )
  optimizer_specs_list: Optional[
      xmile_optimizer_specs_list.XmileOptimizerSpecsList
  ] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  payoff_specs_list: Optional[xmile_payoff_specs_list.XmilePayoffSpecsList] = (
      dataclasses.field(
          default=None,
          metadata={
              "type": "Element",
              "namespace": xmile_globals.ISEE_NAMESPACE,
          },
      )
  )
  data_manager: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  sensi_specs: Optional[xmile_sensi_specs.XmileSensiSpecs] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  model_units: xmile_model_units.XmileModelUnits = dataclasses.field(
      metadata={
          "type": "Element",
          "required": True,
      },
  )
  model: List[xmile_model_view.XmileModel] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
          "min_occurs": 1,
      },
  )
  controls: Optional[xmile_controls.XmileControls] = dataclasses.field(
      default=None,
      metadata={
          "type": "Element",
          "namespace": xmile_globals.ISEE_NAMESPACE,
      },
  )
  macro: List[xmile_macro.XmileMacro] = dataclasses.field(
      default_factory=list,
      metadata={
          "type": "Element",
      },
  )
