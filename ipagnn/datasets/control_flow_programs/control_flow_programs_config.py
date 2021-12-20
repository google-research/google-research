# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Config definition for the control_flow_programs dataset."""

import tensorflow_datasets as tfds

from ipagnn.datasets.control_flow_programs import control_flow_programs_version

DEFAULT_FEATURE_SETS = ["human_readable", "code", "trace", "output", "cfg"]


class ControlFlowProgramsConfig(tfds.core.BuilderConfig):
  """BuilderConfig for ControlFlowPrograms."""

  def __init__(self,
               program_generator_config,
               max_examples=None,
               feature_sets=None,
               partial_program=False,
               **kwargs):
    """BuilderConfig for ControlFlowPrograms dataset.

    Args:
      program_generator_config: The program generator config for the programs to
        generator.
      max_examples: If set, the number of examples to include in the dataset.
      feature_sets: The list of feature sets to include in each example.
      partial_program: If True, poke a hole in the programs.
      **kwargs: Additional kwargs for the BuilderConfig.
    """
    description = "ControlFlowPrograms dataset."
    super(ControlFlowProgramsConfig, self).__init__(
        version=control_flow_programs_version.VERSION,
        description=description,
        **kwargs)

    self.program_generator_config = program_generator_config
    self.max_examples = max_examples
    self.feature_sets = feature_sets or DEFAULT_FEATURE_SETS
    self.partial_program = partial_program
