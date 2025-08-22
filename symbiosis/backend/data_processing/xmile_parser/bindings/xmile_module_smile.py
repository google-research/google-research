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

"""Module class."""

import dataclasses

import xmile_globals

__NAMESPACE__ = "http://www.systemdynamics.org/XMILE"


@dataclasses.dataclass(kw_only=True)
class XmileModuleSmile:
  """Modules are placeholders in the variables section for submodels.

  If present, this must appear in every model that references that
  submodel.
  """

  class Meta:
    name = "module"
    namespace = xmile_globals.SMILE_NAMESPACE

  label_side: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  label_angle: int = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
  size: str = dataclasses.field(
      metadata={
          "type": "Attribute",
          "required": True,
      }
  )
