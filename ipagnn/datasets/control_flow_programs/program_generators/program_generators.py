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

"""This module routes calls to generate programs to the right program generator.

The Learned Interpreters project supports multiple program generators, each with
different configs. This module sends requests to generate a program to the
appropriate program generator module.
"""

from ipagnn.datasets.control_flow_programs.program_generators import arithmetic_if_repeats
from ipagnn.datasets.control_flow_programs.program_generators import arithmetic_repeats
from ipagnn.datasets.control_flow_programs.program_generators import multivar_arithmetic
from ipagnn.datasets.control_flow_programs.program_generators import template_programs
from ipagnn.datasets.control_flow_programs.program_generators import toy_programs


def generate_python_source_and_partial_python_source(
    length, program_generator_config):
  """Generates Python code and partial code according to the config."""
  config_type = type(program_generator_config).__name__
  if config_type == 'ArithmeticRepeatsConfig':
    return arithmetic_repeats.generate_python_source_and_partial_python_source(
        length, program_generator_config)
  elif config_type == 'TemplatesConfig':
    return template_programs.generate_python_source_and_partial_python_source(
        length, program_generator_config)
  elif config_type == 'ToyProgramsConfig':
    return toy_programs.generate_python_source_and_partial_python_source(
        length, program_generator_config)
  else:
    raise ValueError('Unexpected program generator config.',
                     program_generator_config)


def generate_python_source(length, program_generator_config):
  """Generates Python code according to the config."""
  config_type = type(program_generator_config).__name__
  if config_type == 'ArithmeticRepeatsConfig':
    return arithmetic_repeats.generate_python_source(
        length, program_generator_config)
  elif config_type == 'ArithmeticIfRepeatsConfig':
    return arithmetic_if_repeats.generate_python_source(
        length, program_generator_config)
  elif config_type == 'TemplatesConfig':
    return template_programs.generate_python_source(
        length, program_generator_config)
  elif config_type == 'ToyProgramsConfig':
    return toy_programs.generate_python_source(length, program_generator_config)
  elif config_type == 'MultivarArithmeticConfig':
    return multivar_arithmetic.generate_python_source(
        length, program_generator_config)
  else:
    raise ValueError('Unexpected program generator config.',
                     program_generator_config)
