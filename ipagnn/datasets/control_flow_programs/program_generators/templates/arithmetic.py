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

"""These templates define a simple distribution over programs with control flow.

The programs consist of a primary variable, v0, arithmetic statements, and
potentially control flow nested in complex arrangements. If, if-else, and while
loops with counter variables are included as control flow.
"""

import dataclasses
import enum
from typing import FrozenSet

import numpy as np

from ipagnn.datasets.control_flow_programs import control_flow_programs_version
from ipagnn.datasets.control_flow_programs.program_generators import constants
from ipagnn.datasets.control_flow_programs.program_generators import top_down_refinement
from ipagnn.datasets.control_flow_programs.program_generators.templates import base as base_templates

ConfigurableTemplate = base_templates.ConfigurableTemplate
TemplateData = base_templates.TemplateData

Hole = top_down_refinement.Hole
Program = top_down_refinement.ThingWithHoles


class HoleType(enum.Enum):
  """A type of hole for this task."""
  STMT = "STMT"  # A single statement or control flow block
  STMTS = "STMTS"  # Possibly empty list of statements (no jumps)
  STMTS_NONEMPTY = "STMTS_NONEMPTY"  # Nonempty list of statements (no jumps)
  BLOCK = "BLOCK"  # Nonempty list of statements, which might end in a jump


_STMT_COST = 1
_BLOCK_COST = 1
_STMTS_COST = 0
_STMTS_NONEMPTY_COST = 1


# Assign weights so that we avoid being forced into suboptimal choices later.
# For instance, it's always fine to stop generating statements, so we give
# adding more statements a low weight, and choose it less often. But it's
# annoying to be forced to insert "pass" everywhere due to lack of space, so we
# give partially-expanded single statements much more weight.
HOLE_TYPE_WEIGHTS = {
    HoleType.STMT: 100,
    HoleType.BLOCK: 10,
    HoleType.STMTS: 1,
    HoleType.STMTS_NONEMPTY: 100,
}


@dataclasses.dataclass(frozen=True)
class HoleMetadata:
  indent: int = 0
  inside_loop: bool = False
  used_variables: FrozenSet[str] = frozenset()

  @property
  def indent_str(self):
    return constants.INDENT_STRING * self.indent


def _max_value(config):
  if config.max_value is not None:
    return config.max_value
  return config.base ** config.num_digits - 1


class PassTemplate(ConfigurableTemplate):
  """No-op pass statement."""
  precedence = 0
  fills_type = HoleType.STMT
  weight = 1
  required_cost = 1

  def fill(self, hole, rng):
    return Program(1, [], lambda: [f"{hole.metadata.indent_str}pass"])


class ArithmeticTemplate(ConfigurableTemplate):
  """Arithmetic statement."""
  precedence = 1
  fills_type = HoleType.STMT
  weight = 15
  required_cost = 1

  def fill(self, hole, rng):
    op = np.random.choice(self.config.ops)
    max_value = _max_value(self.config)
    operand = np.random.randint(1, max_value + 1)
    return Program(1, [],
                   lambda: [f"{hole.metadata.indent_str}v0 {op} {operand}"])


class StatementsTemplate(ConfigurableTemplate):
  """A list of statements with at least one statement."""
  precedence = 1
  fills_type = HoleType.STMTS_NONEMPTY
  weight = 1
  required_cost = _STMT_COST + _STMTS_COST

  def fill(self, hole, rng):

    def build(stmt, rest):
      return stmt + rest

    stmt_hole = Hole(HoleType.STMT, hole.metadata)
    rest_hole = Hole(HoleType.STMTS, hole.metadata)
    return Program(0, [stmt_hole, rest_hole], build)


class SomeStatementsTemplate(ConfigurableTemplate):
  """Insert some statements.

  STMTS -> STMTS_NONEMPTY
  """
  precedence = 1
  fills_type = HoleType.STMTS
  weight = 8
  required_cost = _STMTS_NONEMPTY_COST

  def fill(self, hole, rng):
    stmts_hole = Hole(HoleType.STMTS_NONEMPTY, hole.metadata)
    return Program(0, [stmts_hole], lambda stmts: stmts)


class OneStatementTemplate(ConfigurableTemplate):
  """Insert one statement.

  STMTS -> STMT
  """
  precedence = 1
  fills_type = HoleType.STMTS
  weight = 1
  required_cost = _STMTS_NONEMPTY_COST

  def fill(self, hole, rng):
    stmt_hole = Hole(HoleType.STMT, hole.metadata)
    return Program(0, [stmt_hole], lambda stmts: stmts)


class NoMoreStatementsTemplate(ConfigurableTemplate):
  """Don't insert any statements (as a last resort)."""
  precedence = 0
  fills_type = HoleType.STMTS
  weight = 1
  required_cost = 0

  def fill(self, hole, rng):
    return Program(0, [], lambda: [])


def _select_counter_variable(used_variables):
  num_variables = 10  # TODO(dbieber): num_variables is hardcoded.
  max_variable = num_variables - 1
  allowed_variables = (
      set(range(1, max_variable + 1)) - set(used_variables))
  return np.random.choice(list(allowed_variables))


class RepeatTemplate(ConfigurableTemplate):
  """Construct an if block."""
  precedence = 1
  fills_type = HoleType.STMT
  weight = 5
  required_cost = 3 + _BLOCK_COST

  def can_fill(self, hole, space, templates):
    return len(hole.metadata.used_variables) < 9

  def fill(self, hole, rng):
    counter_var = _select_counter_variable(hole.metadata.used_variables)
    num_repeats = np.random.randint(2, self.config.max_repetitions + 1)

    def build(body):
      return [
          f"{hole.metadata.indent_str}v{counter_var} = {num_repeats}",
          f"{hole.metadata.indent_str}while v{counter_var} > 0:",
          f"{hole.metadata.indent_str}{constants.INDENT_STRING}v{counter_var} -= 1",
      ] + body

    block_hole = Hole(
        HoleType.BLOCK,
        dataclasses.replace(
            hole.metadata,
            indent=hole.metadata.indent + 1,
            used_variables=hole.metadata.used_variables | {counter_var},
            inside_loop=True))
    return Program(3, [block_hole], build)


class IfStatementTemplate(ConfigurableTemplate):
  """Construct an if block."""
  precedence = 1
  fills_type = HoleType.STMT
  weight = 5
  required_cost = 1 + _BLOCK_COST

  def fill(self, hole, rng):
    max_value = _max_value(self.config)
    operand = np.random.randint(max_value + 1)
    cond_op = np.random.choice(["<", "<=", ">", ">="])
    use_mod_conds = control_flow_programs_version.at_least("0.0.42")
    if use_mod_conds:
      test = f"v0 % 10 {cond_op} {operand}"
    else:
      test = f"v0 {cond_op} {operand}"

    def build(body):
      return [f"{hole.metadata.indent_str}if {test}:"] + body

    block_hole = Hole(HoleType.BLOCK,
                      dataclasses.replace(
                          hole.metadata, indent=hole.metadata.indent + 1))
    return Program(1, [block_hole], build)


class IfElseBlockTemplate(ConfigurableTemplate):
  """Construct an if/else block."""
  precedence = 1
  fills_type = HoleType.STMT
  weight = 5
  required_cost = 2 + 2 * _BLOCK_COST

  def fill(self, hole, rng):
    max_value = _max_value(self.config)
    operand = np.random.randint(max_value + 1)
    cond_op = np.random.choice(["<", "<=", ">", ">="])
    use_mod_conds = control_flow_programs_version.at_least("0.0.42")
    if use_mod_conds:
      test = f"v0 % 10 {cond_op} {operand}"
    else:
      test = f"v0 {cond_op} {operand}"

    def build(body, orelse):
      return (
          [f"{hole.metadata.indent_str}if {test}:"]
          + body
          + [f"{hole.metadata.indent_str}else:"]
          + orelse
      )

    block_hole = Hole(HoleType.BLOCK,
                      dataclasses.replace(
                          hole.metadata, indent=hole.metadata.indent + 1))
    return Program(2, [block_hole, block_hole], build)

# Block kinds:


class BreakTemplate(ConfigurableTemplate):
  """Block that ends by breaking out of the containing loop."""
  precedence = 1
  fills_type = HoleType.BLOCK
  weight = 5
  required_cost = 1 + _STMTS_COST

  def can_fill(self, hole, space, templates):
    return hole.metadata.inside_loop

  def fill(self, hole, rng):
    stmts_hole = Hole(HoleType.STMTS, hole.metadata)
    return Program(
        1, [stmts_hole],
        lambda stmts: stmts + [f"{hole.metadata.indent_str}break"])


class ContinueTemplate(ConfigurableTemplate):
  """Block that ends by continuing to the next iteration of the loop."""
  precedence = 1
  fills_type = HoleType.BLOCK
  weight = 5
  required_cost = 1 + _STMTS_COST

  def can_fill(self, hole, space, templates):
    return hole.metadata.inside_loop

  def fill(self, hole, rng):
    stmts_hole = Hole(HoleType.STMTS, hole.metadata)
    return Program(
        1, [stmts_hole],
        lambda stmts: stmts + [f"{hole.metadata.indent_str}continue"])


class FallthroughTemplate(ConfigurableTemplate):
  """Block that ends by falling through to the outer block.

  Note that every block has to contain at least one statement.
  """
  precedence = 1
  fills_type = HoleType.BLOCK
  weight = 30
  required_cost = _STMTS_NONEMPTY_COST

  def fill(self, hole, rng):
    stmts_hole = Hole(HoleType.STMTS_NONEMPTY, hole.metadata)
    return Program(0, [stmts_hole], lambda stmts: stmts)


TEMPLATE_CLASSES = [
    PassTemplate,
    ArithmeticTemplate,
    StatementsTemplate,
    SomeStatementsTemplate,
    OneStatementTemplate,
    NoMoreStatementsTemplate,
    RepeatTemplate,
    IfStatementTemplate,
    IfElseBlockTemplate,
    BreakTemplate,
    ContinueTemplate,
    FallthroughTemplate,
]


def get_template_data(config):
  weighted_templates = []
  for template_cls in TEMPLATE_CLASSES:
    weighted_templates.append(top_down_refinement.WeightedTemplate(
        template_cls(config),
        weight=template_cls.weight))

  return TemplateData(
      weighted_templates=weighted_templates,
      root_object=Program(
          0, [Hole(HoleType.STMTS_NONEMPTY, HoleMetadata())], lambda x: x),
      hole_type_weights=HOLE_TYPE_WEIGHTS,
      start_with_initialization=True,
  )
