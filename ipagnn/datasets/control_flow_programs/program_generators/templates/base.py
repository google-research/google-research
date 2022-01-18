# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""The classes shared by all template definitions."""

import dataclasses
from typing import Any, Dict, List

from ipagnn.datasets.control_flow_programs.program_generators import top_down_refinement


@dataclasses.dataclass
class TemplateData:
  weighted_templates: List[top_down_refinement.WeightedTemplate]
  root_object: top_down_refinement.ThingWithHoles
  hole_type_weights: Dict[Any, int]
  start_with_initialization: bool = False


class ConfigurableTemplate(top_down_refinement.HoleFillerTemplate):
  """A hole filler template that accepts a config object."""
  precedence = 1

  def __init__(self, config):
    self.config = config
    super(ConfigurableTemplate, self).__init__()
