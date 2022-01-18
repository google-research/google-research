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

"""Top-down refinement program generator."""

import dataclasses
from typing import Any, Callable, Dict, Optional, Text, Tuple

from absl import logging  # pylint: disable=unused-import
import numpy as np

from ipagnn.datasets.control_flow_programs import control_flow_programs_version
from ipagnn.datasets.control_flow_programs.program_generators import constants
from ipagnn.datasets.control_flow_programs.program_generators import top_down_refinement
from ipagnn.datasets.control_flow_programs.program_generators.templates import base as base_templates

DEFAULT_OPS = ("+=", "-=", "*=")

Hole = top_down_refinement.Hole
Program = top_down_refinement.ThingWithHoles
Template = top_down_refinement.HoleFillerTemplate


@dataclasses.dataclass
class TemplatesConfig:
  """The Config object for Template Programs."""
  base: int
  length: int
  template_data_fn: Callable[[Any], base_templates.TemplateData]
  max_value: Optional[int] = None
  num_digits: int = 1
  max_repetitions: int = 9
  exact_length: bool = False
  ops: Tuple[Text, Ellipsis] = DEFAULT_OPS
  encoder_name: Text = "simple"
  mod: Optional[int] = 10
  output_mod: Optional[int] = None
  length_distribution: Optional[Dict[int, float]] = None

  _template_data: Optional[base_templates.TemplateData] = None

  @property
  def template_data(self):
    if self._template_data is None:
      self._template_data = self.template_data_fn(self)
    return self._template_data


def generate_python_source(length, config, rng=None):
  """Generates Python code according to the config."""
  statements = _generate_statements(length, config, rng)
  return "\n".join(statements)


def generate_python_source_and_partial_python_source(length, config):
  """Generates Python code according to the config."""

  # Keep generating programs until we find one that we can poke a hole in.
  found_program = False
  while not found_program:
    statements = _generate_statements(length, config)
    hole_statement_indexes = []
    for index, statement in enumerate(statements):
      # TODO(dbieber): defer to specific TemplateData for whether hole
      # is acceptable here.
      if ("while" not in statement and "if" not in statement
          and "v0" in statement
          and index != 0):
        hole_statement_indexes.append(index)
    if hole_statement_indexes:
      found_program = True

  hole_statement_index = np.random.choice(hole_statement_indexes)
  full_source = "\n".join(statements)

  statement = statements[hole_statement_index]
  indent = (len(statement) - len(statement.lstrip())) / constants.INDENT_SPACES
  indent_str = constants.INDENT_STRING * int(indent)

  partial_statements = statements.copy()
  partial_statements[hole_statement_index] = f"{indent_str}_ = 0"
  partial_source = "\n".join(partial_statements)
  return full_source, partial_source


def get_train_length_distribution(max_train_length, min_train_length=2):
  """Create a distribution over program lengths for training.

  The max_train_length is given probability 90%.
  Each smaller length is given half the remaining probability, up to the
  min_train_length.

  Args:
    max_train_length: The maximum length to allocate any probability.
    min_train_length: The minimum length to allocate any probability.
  Returns:
    A dict mapping length to the probability allocated to programs of that
    length.
  """
  assert max_train_length >= min_train_length
  ps = {max_train_length: 0.9}
  remaining_probability = 1.0 - ps[max_train_length]
  length = max_train_length - 1
  while length > min_train_length:
    ps[length] = remaining_probability / 2
    remaining_probability /= 2
    length -= 1

  ps[min_train_length] = ps.get(min_train_length, 0) + remaining_probability
  return ps


def get_test_length_distribution(max_test_length):
  lengths = range(10, max_test_length + 1)
  unnormalized_probabilities = [1/length for length in lengths]
  total = sum(unnormalized_probabilities)
  probabilities = [p / total for p in unnormalized_probabilities]
  return dict(zip(lengths, probabilities))


def _generate_statements(length, config, rng=None):
  """Generates `length` statements representing a control flow program.

  Before 0.0.44:
    Mostly (90% of the time) generates statements at the requested length.
    A small fraction of the time generates smaller programs.
    The smaller the program length, the less likely it is.
  Version 0.0.44:
    Uses config.length_distribution to determine the length program to
    generate.

  Args:
    length: The target program length. If config.exact_length, this length will
      be used exactly. Otherwise, smaller programs are permitted.
    config: The dataset config.
    rng: (optional) A numpy RandomState.
  Returns:
    The list of statements in the generated program.
  """
  if rng is None:
    rng = np.random.RandomState()

  if config.exact_length:
    return _generate_statements_with_length(length, config, rng)

  if control_flow_programs_version.at_least("0.0.44"):
    assert config.length_distribution is not None
    lengths, probabilities = zip(*config.length_distribution.items())
    length = rng.choice(lengths, p=probabilities)
    return _generate_statements_with_length(length, config, rng)

  if not config.exact_length:
    if rng.random() < 0.10:  # 90% of the time, generate the full length.
      while length > 2 and rng.random() > .5:
        # Of the remaining 10% of programs half are length - 1.
        # Of the still remaining 5%, half are length - 2, etc.
        length -= 1
  return _generate_statements_with_length(length, config, rng)


def _generate_statements_with_length(length, config, rng):
  """Generates `length` statements representing a control flow program."""
  found = False
  while not found:
    statements = _generate_statements_with_cost(
        target_cost=length, config=config, rng=rng)
    if len(statements) == length:
      found = True
  return statements


def _generate_statements_with_cost(target_cost, config, rng):
  """Generates a list of statements representing a control flow program.

  Args:
    target_cost: Approximately the number of statements to generate.
    config: The TemplatesConfig specifying the properties of the program
      to generate.
    rng: A numpy RandomState.
  Returns:
    A list of statements, each statement being a string.
  """
  template_data = config.template_data
  start_with_initialization = template_data.start_with_initialization
  weighted_templates = template_data.weighted_templates
  root_object = template_data.root_object

  if start_with_initialization:
    target_cost = target_cost - 1

  distribution = top_down_refinement.RefinementDistribution(
      hole_selection_weights=template_data.hole_type_weights,
      weighted_templates=weighted_templates,
  )
  found = False
  while not found:
    try:
      statements = top_down_refinement.top_down_construct(
          root_object=root_object,
          target_cost=target_cost,
          refinement_distribution=distribution,
          rng=rng)
      found = True
    except ValueError as e:
      print("Retrying", e)

  init_statements = []
  if start_with_initialization:
    start_value = np.random.randint(config.base)
    init_statements = [f"v0 = {start_value}"]
  return init_statements + statements
