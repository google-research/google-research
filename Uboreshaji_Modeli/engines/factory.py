# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Factory for model engines."""

from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.engines import base
from Uboreshaji_Modeli.engines import owl


def get_engine(model_flavor):
  """Returns the engine for the given model flavor."""
  if model_flavor == config.ModelFlavor.OWL_V2_TORCH:
    return owl.Owlv2Engine()


  raise ValueError(f"Unsupported model flavor: {model_flavor}")
