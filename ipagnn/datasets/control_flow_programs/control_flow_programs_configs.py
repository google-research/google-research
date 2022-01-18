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

"""Configs for the control_flow_programs dataset."""

import dataclasses
import itertools

from typing import Any, Optional, Text
from absl import logging  # pylint: disable=unused-import

from ipagnn.datasets.control_flow_programs import control_flow_programs_config
from ipagnn.datasets.control_flow_programs.program_generators import arithmetic_if_repeats_config
from ipagnn.datasets.control_flow_programs.program_generators import arithmetic_repeats_config
from ipagnn.datasets.control_flow_programs.program_generators import multivar_arithmetic
from ipagnn.datasets.control_flow_programs.program_generators import template_programs
from ipagnn.datasets.control_flow_programs.program_generators import toy_programs
from ipagnn.datasets.control_flow_programs.program_generators.templates import arithmetic as arithmetic_templates
from ipagnn.datasets.control_flow_programs.program_generators.templates import runtime_error as runtime_error_templates


ControlFlowProgramsConfig = control_flow_programs_config.ControlFlowProgramsConfig
ArithmeticIfRepeatsConfig = arithmetic_if_repeats_config.ArithmeticIfRepeatsConfig
ArithmeticRepeatsConfig = arithmetic_repeats_config.ArithmeticRepeatsConfig
MultivarArithmeticConfig = multivar_arithmetic.MultivarArithmeticConfig
TemplatesConfig = template_programs.TemplatesConfig
ToyProgramsConfig = toy_programs.ToyProgramsConfig

TINY_DATASET_SIZE = 2500
SMALL_DATASET_SIZE = 25000
MEDIUM_DATASET_SIZE = 50000
LARGE_DATASET_SIZE = 5000000

ALL_BASES = [10]
ALL_LENGTHS = [2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
               200, 400, 1000]


def _base_label(base):
  if base == 2:
    return "binary"
  elif base == 3:
    return "ternary"
  elif base == 10:
    return "decimal"
  else:
    return f"b{base}"


def _dataset_size(length):
  if length >= 200:
    return TINY_DATASET_SIZE
  elif length >= 50:
    return SMALL_DATASET_SIZE
  elif length >= 20:
    return MEDIUM_DATASET_SIZE
  else:
    return LARGE_DATASET_SIZE


@dataclasses.dataclass
class ConfigSettings:
  """Settings for generating a ControlFlowProgramsConfig.

  One of generator_config or generator_config_fn should be set.
  If generator_config is set dataclasses.replace will be used with replacements.
  If generator_config_fn is set, replacements form the fn args.
  """
  generator_config: Optional[Any] = None
  generator_config_fn: Optional[Any] = None
  name_format: Text = "unnamed"
  include_partial: bool = False
  max_examples: Optional[int] = None


DEFAULT_BASE = 10
DEFAULT_LENGTH = 10
DEFAULT_ARITHMETIC_REPEATS_SETTINGS = ConfigSettings(
    generator_config=ArithmeticRepeatsConfig(
        base=DEFAULT_BASE,
        length=DEFAULT_LENGTH,
        num_digits=1,
        max_repeat_statements=20,
        max_repetitions=9,
        max_repeat_block_size=5,
        repeat_probability=0.25,
    ),
    name_format="{}-L{}",
    include_partial=True,
)
STRAIGHTLINE_ARITHMETIC_SETTINGS = ConfigSettings(
    generator_config=ArithmeticRepeatsConfig(
        base=DEFAULT_BASE,
        length=DEFAULT_LENGTH,
        num_digits=1,
        max_repeat_statements=0,
        start_with_initialization=True,
    ),
    name_format="{}-straightline-L{}",
    include_partial=True,
)
NESTED_ARITHMETIC_REPEATS_SETTINGS = ConfigSettings(
    generator_config=ArithmeticRepeatsConfig(
        base=DEFAULT_BASE,
        length=DEFAULT_LENGTH,
        num_digits=1,
        max_repeat_statements=20,
        max_repetitions=9,
        max_repeat_block_size=9,
        repeat_probability=0.25,
        permit_nested_repeats=True,
    ),
    name_format="{}-nested-L{}",
    include_partial=True,
)


def make_templates_program_config(base, length):
  length_distribution = template_programs.get_train_length_distribution(length)
  return TemplatesConfig(
      base=base, length=length,
      length_distribution=length_distribution,
      template_data_fn=arithmetic_templates.get_template_data)
TEMPLATES_PROGRAMS_SETTINGS = ConfigSettings(
    generator_config_fn=make_templates_program_config,
    name_format="{}-templates-L{}",
    include_partial=True,
)


def make_large_state_templates_program_config(base, length):
  del base  # Unused.
  length_distribution = template_programs.get_train_length_distribution(length)
  return TemplatesConfig(
      base=1000, length=length,
      max_value=9,
      mod=1000,
      output_mod=1000,
      length_distribution=length_distribution,
      template_data_fn=arithmetic_templates.get_template_data)
LARGE_STATE_TEMPLATES_PROGRAMS_SETTINGS = ConfigSettings(
    generator_config_fn=make_large_state_templates_program_config,
    name_format="{}-large-state-L{}",
    include_partial=True,
)


def make_large_state_train_templates_programs_config(base, length):
  del base  # Unused.
  length_distribution = template_programs.get_train_length_distribution(length)
  return TemplatesConfig(
      base=1000, length=length,
      max_value=9,
      mod=1000,
      output_mod=1000,
      length_distribution=length_distribution,
      template_data_fn=arithmetic_templates.get_template_data)
LARGE_STATE_TRAIN_TEMPLATES_PROGRAMS_SETTINGS = ConfigSettings(
    generator_config_fn=make_large_state_train_templates_programs_config,
    name_format="{}-large-state-train-L{}",
    include_partial=True,
)


def make_large_state_test_templates_programs_config(base, length):
  del base  # Unused.
  length_distribution = template_programs.get_test_length_distribution(length)
  return TemplatesConfig(
      base=1000, length=length,
      max_value=9,
      mod=1000,
      output_mod=1000,
      length_distribution=length_distribution,
      template_data_fn=arithmetic_templates.get_template_data)
LARGE_STATE_TEST_TEMPLATES_PROGRAMS_SETTINGS = ConfigSettings(
    generator_config_fn=make_large_state_test_templates_programs_config,
    name_format="{}-large-state-test-L{}",
    include_partial=True,
)


def make_runtime_error_programs_config(base, length):
  del base  # Unused.
  length_distribution = template_programs.get_train_length_distribution(
      length, min_train_length=3)
  return TemplatesConfig(
      base=1000, length=length,
      max_value=9,
      mod=1000,
      output_mod=1000,
      exact_length=True,
      length_distribution=length_distribution,
      encoder_name="text",
      template_data_fn=runtime_error_templates.get_template_data)
RUNTIME_ERROR_PROGRAMS_SETTINGS = ConfigSettings(
    generator_config_fn=make_runtime_error_programs_config,
    name_format="{}-multivar-templates-train-L{}",
    include_partial=True,
)


SMALL_NESTED_ARITHMETIC_REPEATS_SETTINGS = ConfigSettings(
    generator_config=ArithmeticRepeatsConfig(
        base=DEFAULT_BASE,
        length=DEFAULT_LENGTH,
        num_digits=1,
        max_repeat_statements=20,
        max_repetitions=9,
        max_repeat_block_size=9,
        repeat_probability=0.25,
        permit_nested_repeats=True,
    ),
    name_format="{}-nested-L{}-small",
    include_partial=True,
    max_examples=128,
)

MULTIDIGIT_ARITHMETIC_REPEATS_SETTINGS = ConfigSettings(
    generator_config=ArithmeticRepeatsConfig(
        base=DEFAULT_BASE,
        length=DEFAULT_LENGTH,
        num_digits=4,
        max_repeat_statements=10,
        max_repetitions=9,
        max_repeat_block_size=5,
    ),
    name_format="multidigit-{}-L{}",
    include_partial=True,
)

MULTIVAR_ARITHMETIC_SETTINGS = ConfigSettings(
    generator_config=MultivarArithmeticConfig(
        base=DEFAULT_BASE,
        length=DEFAULT_LENGTH,
        variables=5,
        constant_probability=0.75,
    ),
    name_format="multivar-{}-L{}",
    include_partial=False,
)


def make_builder_configs(settings, replacements, max_examples=None):
  """Makes the builder configs with the settings specified."""
  builder_configs = []
  for s in settings:
    for replacement_values in itertools.product(*replacements.values()):
      replacement_dict = dict(zip(replacements.keys(), replacement_values))
      if s.generator_config is not None:
        config = dataclasses.replace(s.generator_config, **replacement_dict)
      elif s.generator_config_fn is not None:
        config = s.generator_config_fn(**replacement_dict)
      else:
        raise ValueError("No generator_config or generator_config_fn found.")

      base = replacement_dict["base"]
      base_label = _base_label(base)
      max_examples = (max_examples or s.max_examples
                      or _dataset_size(config.length))
      name = s.name_format.format(base_label, config.length)
      builder_config = ControlFlowProgramsConfig(
          name=name,
          program_generator_config=config,
          max_examples=max_examples,
      )
      builder_configs.append(builder_config)
      if s.include_partial:
        builder_config = ControlFlowProgramsConfig(
            name=name + "-partial",
            program_generator_config=config,
            max_examples=max_examples,
            partial_program=True,
        )
        builder_configs.append(builder_config)
  return builder_configs


def get_builder_configs():
  """Constructs the list to use for BUILDER_CONFIGS."""
  # The builder configs to publish.
  builder_configs = (
      [
          ControlFlowProgramsConfig(  # 1 dataset
              name="toy-programs",
              program_generator_config=ToyProgramsConfig(base=10),
              max_examples=32,
          ),
      ]
      + make_builder_configs(
          settings=[
              RUNTIME_ERROR_PROGRAMS_SETTINGS,  # 5 datasets
          ],
          replacements={"base": [10], "length": [5, 10, 20, 50, 100]},
      )
      + make_builder_configs(
          settings=[
              LARGE_STATE_TEMPLATES_PROGRAMS_SETTINGS,  # 14 datasets
              STRAIGHTLINE_ARITHMETIC_SETTINGS,  # 12 datasets
          ],
          replacements={"base": [10], "length": [20, 30, 40, 60, 70, 80, 90]},
          max_examples=10000,
      )
  )

  # A superset of builder configs we care about.
  more_builder_configs = (
      make_builder_configs(
          settings=[
              DEFAULT_ARITHMETIC_REPEATS_SETTINGS,
              MULTIVAR_ARITHMETIC_SETTINGS,
              MULTIDIGIT_ARITHMETIC_REPEATS_SETTINGS,
              NESTED_ARITHMETIC_REPEATS_SETTINGS,
              SMALL_NESTED_ARITHMETIC_REPEATS_SETTINGS,
              TEMPLATES_PROGRAMS_SETTINGS,
              LARGE_STATE_TEMPLATES_PROGRAMS_SETTINGS,
          ],
          replacements={"base": ALL_BASES, "length": ALL_LENGTHS},
      ) +
      make_builder_configs(
          settings=[
              RUNTIME_ERROR_PROGRAMS_SETTINGS,
          ],
          replacements={"base": ALL_BASES,
                        "length": [n for n in ALL_LENGTHS if n > 3]},
      ) +
      [
          ControlFlowProgramsConfig(
              name="toy-programs",
              program_generator_config=ToyProgramsConfig(base=10),
              max_examples=32,
          ),
          ControlFlowProgramsConfig(
              name="addition-decimal-L10",
              program_generator_config=ArithmeticRepeatsConfig(
                  base=10,
                  length=10,
                  ops=["+="],
                  max_repeat_statements=0,
              ),
              max_examples=1024,
          ),
          ControlFlowProgramsConfig(
              name="addition-b5-L10",
              program_generator_config=ArithmeticRepeatsConfig(
                  base=5,
                  length=10,
                  ops=["+="],
                  max_repeat_statements=0,
              ),
              max_examples=1024,
          ),
          ControlFlowProgramsConfig(
              name="addition-binary-L10",
              program_generator_config=ArithmeticRepeatsConfig(
                  base=2,
                  length=10,
                  ops=["+="],
                  max_repeat_statements=0,
              ),
              max_examples=1024,
          ),
          ControlFlowProgramsConfig(
              name="decimal-L10-small",
              program_generator_config=ArithmeticRepeatsConfig(
                  base=10,
                  length=10,
                  max_repeat_statements=2,
              ),
              max_examples=32,
          ),
          ControlFlowProgramsConfig(
              name="multivar-arithmetic-L10",
              program_generator_config=MultivarArithmeticConfig(
                  base=10, length=10),
              max_examples=32,
          ),
          ControlFlowProgramsConfig(
              name="multivar-arithmetic-L10-V4",
              program_generator_config=MultivarArithmeticConfig(
                  base=10, length=10, variables=4),
              max_examples=32,
          ),
          ControlFlowProgramsConfig(
              name="multivar-arithmetic-L50-V4",
              program_generator_config=MultivarArithmeticConfig(
                  base=10, length=50, variables=4),
              max_examples=32,
          ),
          ControlFlowProgramsConfig(
              name="tmp",
              program_generator_config=ArithmeticRepeatsConfig(
                  base=10,
                  length=10,
                  max_repeat_statements=2,
              ),
              max_examples=32,
              feature_sets=["human_readable", "output", "cfg"],
          ),
      ]
  )

  # Include all configs, but only once each.
  config_names = set(config.name for config in builder_configs)
  for config in more_builder_configs:
    if config.name not in config_names:
      builder_configs.append(config)
      config_names.add(config.name)

  return builder_configs
