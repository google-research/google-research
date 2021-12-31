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

"""These templates define a program distribution that includes GCD, LCM, etc.

Valid functions under this distribution include GCD, LCM, is-prime, fibonacci
numbers, totient

The following constructions make up the grammar defined by these templates:

# iN {=,+=,-=,*=,//=} constant
# iN {=,+=,-=,*=,//=} iN
# i0 = iN {%,+,-,*, //} iN
# bN {=, |=, &=, ^=} constant
# bN = iN {==, >, !=} 0
# while bN
# for iN in range(constant)
# if bN
# continue / break / pass
"""

import dataclasses
import enum
from typing import FrozenSet

from ipagnn.datasets.control_flow_programs.program_generators import constants
from ipagnn.datasets.control_flow_programs.program_generators import top_down_refinement
from ipagnn.datasets.control_flow_programs.program_generators.templates import base as base_templates

ConfigurableTemplate = base_templates.ConfigurableTemplate
TemplateData = base_templates.TemplateData

Program = top_down_refinement.ThingWithHoles


class Hole(top_down_refinement.Hole):

  @property
  def indent(self):
    return self.metadata.indent_str


class HoleType(enum.Enum):
  """A type of hole for this task."""
  PROGRAM = "PROGRAM"
  INIT = "INIT"
  STMT = "STMT"
  SIMPLE_STMT = "SIMPLE_STMT"
  COMPOUND_STMT = "COMPOUND_STMT"
  LOOP_STMT = "LOOP_STMT"
  IF_STMT = "IF_STMT"
  BLOCK = "BLOCK"
  BLOCK_END = "BLOCK_END"
  CF_STMT = "CF_STMT"
  STMTS = "STMTS"
  STMTS_NONEMPTY = "STMTS_NONEMPTY"
  STMTS_OPTIONAL = "STMTS_OPTIONAL"
  BVAR = "BVAR"
  BAUG_OP = "BAUG_OP"
  BCONSTANT = "BCONSTANT"
  IVAR = "IVAR"
  IAUG_OP = "IAUG_OP"
  IBIN_OP = "IBIN_OP"
  ICONSTANT = "ICONSTANT"


HOLE_TYPE_WEIGHTS = {
    HoleType.PROGRAM: 10,
    HoleType.INIT: 10,
    HoleType.STMT: 100,
    HoleType.SIMPLE_STMT: 100,
    HoleType.COMPOUND_STMT: 100,
    HoleType.LOOP_STMT: 100,
    HoleType.IF_STMT: 100,
    HoleType.BLOCK: 100,
    HoleType.BLOCK_END: 10,
    HoleType.CF_STMT: 10,
    HoleType.STMTS: 5,
    HoleType.STMTS_NONEMPTY: 50,
    HoleType.STMTS_OPTIONAL: 5,
    HoleType.BVAR: 10,
    HoleType.BAUG_OP: 10,
    HoleType.BCONSTANT: 10,
    HoleType.IVAR: 10,
    HoleType.IAUG_OP: 10,
    HoleType.IBIN_OP: 10,
    HoleType.ICONSTANT: 10,
}


ALL_COSTS = {
    HoleType.PROGRAM: 7,
    HoleType.INIT: 6,
    HoleType.STMT: 1,
    HoleType.SIMPLE_STMT: 1,
    HoleType.COMPOUND_STMT: 2,
    HoleType.LOOP_STMT: 1,
    HoleType.IF_STMT: 1,
    HoleType.BLOCK: 1,
    HoleType.BLOCK_END: 0,
    HoleType.CF_STMT: 1,
    HoleType.STMTS: 0,
    HoleType.STMTS_NONEMPTY: 1,
    HoleType.STMTS_OPTIONAL: 0,
    HoleType.BVAR: 0,
    HoleType.BAUG_OP: 0,
    HoleType.BCONSTANT: 0,
    HoleType.IVAR: 0,
    HoleType.IAUG_OP: 0,
    HoleType.IBIN_OP: 0,
    HoleType.ICONSTANT: 0,
}


@dataclasses.dataclass(frozen=True)
class HoleMetadata:
  indent: int = 0
  inside_loop: bool = False
  used_variables: FrozenSet[str] = frozenset()

  @property
  def indent_str(self):
    return constants.INDENT_STRING * self.indent


def const(x):
  return lambda *unused: x


def concat(*lines):
  return sum(lines, [])


def fmt(format_string):
  return lambda *args: [format_string.format(*args)]


TEMPLATE_CLASSES = []


def register_template(template_cls):
  TEMPLATE_CLASSES.append(template_cls)


@register_template
class ProgramTemplate(ConfigurableTemplate):
  fills_type = HoleType.PROGRAM
  required_cost = ALL_COSTS[HoleType.INIT] + 1
  weight = 10

  def fill(self, hole, rng):
    init = Hole(HoleType.INIT, hole.metadata)
    body = Hole(HoleType.STMTS_NONEMPTY, hole.metadata)
    return Program(0, [init, body], concat)


def _max_value(config):
  if config.max_value is not None:
    return config.max_value
  return config.base ** config.num_digits - 1


# pylint: disable=missing-class-docstring
@register_template
class InitTemplate(ConfigurableTemplate):
  fills_type = HoleType.INIT
  required_cost = ALL_COSTS[HoleType.INIT]
  weight = 10

  def fill(self, hole, rng):
    max_value = _max_value(self.config)
    i0 = rng.randint(max_value + 1)
    i1 = rng.randint(max_value + 1)
    i2 = rng.randint(max_value + 1)
    i3 = rng.randint(max_value + 1)
    b0 = rng.choice([True, False])
    b1 = rng.choice([True, False])
    lines = [f"v0 = {i0}",
             f"v1 = {i1}",
             f"v2 = {i2}",
             f"v3 = {i3}",
             f"b0 = {b0}",
             f"b1 = {b1}"]
    return Program(len(lines), [], const(lines))


@register_template
class IntVarAugOpConstant(ConfigurableTemplate):
  fills_type = HoleType.SIMPLE_STMT
  required_cost = 1
  weight = 10

  def fill(self, hole, rng):
    ivar = Hole(HoleType.IVAR, hole.metadata)
    iaug_op = Hole(HoleType.IAUG_OP, hole.metadata)
    iconstant = Hole(HoleType.ICONSTANT, hole.metadata)
    return Program(1, [ivar, iaug_op, iconstant],
                   fmt(f"{hole.indent}{{0}} {{1}} {{2}}"))


@register_template
class IntVarAugOpIntVar(ConfigurableTemplate):
  fills_type = HoleType.SIMPLE_STMT
  required_cost = 1
  weight = 10

  def fill(self, hole, rng):
    ivar = Hole(HoleType.IVAR, hole.metadata)
    iaug_op = Hole(HoleType.IAUG_OP, hole.metadata)
    return Program(1, [ivar, iaug_op, ivar],
                   fmt(f"{hole.indent}{{0}} {{1}} {{2}}"))


@register_template
class IntRegisterIntVarOpIntVar(ConfigurableTemplate):
  fills_type = HoleType.SIMPLE_STMT
  required_cost = 1
  weight = 10

  def fill(self, hole, rng):
    ivar = Hole(HoleType.IVAR, hole.metadata)
    ibin_op = Hole(HoleType.IBIN_OP, hole.metadata)
    return Program(1, [ivar, ibin_op, ivar],
                   fmt(f"{hole.indent}i0 = {{0}} {{1}} {{2}}"))


@register_template
class BVarAugOpConstant(ConfigurableTemplate):
  fills_type = HoleType.SIMPLE_STMT
  required_cost = 1
  weight = 5

  def fill(self, hole, rng):
    bvar = Hole(HoleType.BVAR, hole.metadata)
    baug_op = Hole(HoleType.BAUG_OP, hole.metadata)
    bconstant = Hole(HoleType.BCONSTANT, hole.metadata)
    return Program(1, [bvar, baug_op, bconstant],
                   fmt(f"{hole.indent}{{0}} {{1}} {{2}}"))


@register_template
class BVarBinOpIntVarZero(ConfigurableTemplate):
  fills_type = HoleType.SIMPLE_STMT
  required_cost = 1
  weight = 5

  def fill(self, hole, rng):
    bvar = Hole(HoleType.BVAR, hole.metadata)
    ivar = Hole(HoleType.IVAR, hole.metadata)
    # TODO(dbieber): Should be comparator ops
    ibin_op = Hole(HoleType.IBIN_OP, hole.metadata)
    return Program(1, [bvar, ivar, ibin_op],
                   fmt(f"{hole.indent}{{0}} = {{1}} {{2}} 0"))


@register_template
class BVar(ConfigurableTemplate):
  fills_type = HoleType.BVAR
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    index = rng.randint(2)
    return Program(0, [], const(f"b{index}"))


@register_template
class BAugOp(ConfigurableTemplate):
  fills_type = HoleType.BAUG_OP
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    op = rng.choice(("=", "|=", "&=", "^="))
    return Program(0, [], const(op))


@register_template
class BConstant(ConfigurableTemplate):
  fills_type = HoleType.BCONSTANT
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    value = rng.choice((True, False))
    return Program(0, [], const(value))


@register_template
class IVar(ConfigurableTemplate):
  fills_type = HoleType.IVAR
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    index = rng.randint(4)
    return Program(0, [], const(f"v{index}"))


@register_template
class IAugOp(ConfigurableTemplate):
  fills_type = HoleType.IAUG_OP
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    op = rng.choice(("=", "+=", "-=", "*=", "//="))
    return Program(0, [], const(op))


@register_template
class IBinOp(ConfigurableTemplate):
  fills_type = HoleType.IBIN_OP
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    op = rng.choice(("%", "+", "-", "*", "//"))
    return Program(0, [], const(op))


@register_template
class IConstant(ConfigurableTemplate):
  fills_type = HoleType.ICONSTANT
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    max_value = _max_value(self.config)
    value = rng.randint(1, max_value + 1)
    return Program(0, [], const(value))


@register_template
class LoopCompound(ConfigurableTemplate):
  fills_type = HoleType.COMPOUND_STMT
  required_cost = 2
  weight = 25

  def fill(self, hole, rng):
    cf_stmt = Hole(HoleType.LOOP_STMT, hole.metadata)
    block_metadata = dataclasses.replace(
        hole.metadata,
        indent=hole.metadata.indent + 1,
        inside_loop=True)
    block = Hole(HoleType.BLOCK, block_metadata)
    return Program(0, [cf_stmt, block], concat)


@register_template
class IfCompound(ConfigurableTemplate):
  fills_type = HoleType.COMPOUND_STMT
  required_cost = 2
  weight = 15

  def fill(self, hole, rng):
    cf_stmt = Hole(HoleType.IF_STMT, hole.metadata)
    block_metadata = dataclasses.replace(
        hole.metadata,
        indent=hole.metadata.indent + 1)
    block = Hole(HoleType.BLOCK, block_metadata)
    return Program(0, [cf_stmt, block], concat)


@register_template
class While(ConfigurableTemplate):
  fills_type = HoleType.LOOP_STMT
  required_cost = 1
  weight = 15

  def fill(self, hole, rng):
    bvar = Hole(HoleType.BVAR, hole.metadata)
    return Program(1, [bvar], fmt(f"{hole.indent}while {{0}}:"))


# @register_template
class Repeat(ConfigurableTemplate):
  """Construct a repeat block."""
  fills_type = HoleType.LOOP_STMT
  required_cost = 3
  weight = 15

  def fill(self, hole, rng):

    def build(ivar, iconstant):
      return [
          f"{hole.indent}{ivar} = {iconstant}",
          f"{hole.indent}while {ivar} > 0:",
          f"{hole.indent}{constants.INDENT_STRING}{ivar} -= 1",
      ]

    ivar = Hole(HoleType.IVAR, hole.metadata)
    iconstant = Hole(HoleType.ICONSTANT, hole.metadata)
    return Program(3, [ivar, iconstant], build)


@register_template
class If(ConfigurableTemplate):
  fills_type = HoleType.IF_STMT
  required_cost = 1
  weight = 10

  def fill(self, hole, rng):
    bvar = Hole(HoleType.BVAR, hole.metadata)
    return Program(1, [bvar],
                   fmt(f"{hole.indent}if {{0}}:"))


@register_template
class IfElse(ConfigurableTemplate):
  fills_type = HoleType.COMPOUND_STMT
  required_cost = 4
  weight = 15

  def fill(self, hole, rng):
    def build(bvar, block1, block2):
      return (
          [f"{hole.indent}if {bvar}:"]
          + block1
          + [f"{hole.indent}else:"]
          + block2
      )
    bvar = Hole(HoleType.BVAR, hole.metadata)
    block_metadata = dataclasses.replace(
        hole.metadata,
        indent=hole.metadata.indent + 1)
    block = Hole(HoleType.BLOCK, block_metadata)
    return Program(2, [bvar, block, block], build)


@register_template
class Block(ConfigurableTemplate):
  fills_type = HoleType.BLOCK
  required_cost = 1
  weight = 10

  def fill(self, hole, rng):
    stmts = Hole(HoleType.STMTS_NONEMPTY, hole.metadata)
    block_end = Hole(HoleType.BLOCK_END, hole.metadata)
    return Program(0, [stmts, block_end], concat)


@register_template
class StatementsNonEmpty(ConfigurableTemplate):
  fills_type = HoleType.STMTS_NONEMPTY
  required_cost = 1
  weight = 10

  def fill(self, hole, rng):
    stmt = Hole(HoleType.STMT, hole.metadata)
    stmts = Hole(HoleType.STMTS_OPTIONAL, hole.metadata)
    return Program(0, [stmt, stmts], concat)


@register_template
class StatementsOptionalNonEmpty(ConfigurableTemplate):
  fills_type = HoleType.STMTS_OPTIONAL
  required_cost = 1
  weight = 10

  def fill(self, hole, rng):
    stmts = Hole(HoleType.STMTS_NONEMPTY, hole.metadata)
    return Program(0, [stmts], concat)


@register_template
class StatementsOptionalEmpty(ConfigurableTemplate):
  fills_type = HoleType.STMTS_OPTIONAL
  required_cost = 0
  precedence = 0
  weight = 10

  def fill(self, hole, rng):
    return Program(0, [], concat)


@register_template
class Continue(ConfigurableTemplate):
  fills_type = HoleType.BLOCK_END
  required_cost = 1
  weight = 15

  def can_fill(self, hole, space, templates):
    return hole.metadata.inside_loop

  def fill(self, hole, rng):
    return Program(1, [], const([f"{hole.indent}continue"]))


@register_template
class Break(ConfigurableTemplate):
  fills_type = HoleType.BLOCK_END
  required_cost = 1
  weight = 15

  def can_fill(self, hole, space, templates):
    return hole.metadata.inside_loop

  def fill(self, hole, rng):
    return Program(1, [], const([f"{hole.indent}break"]))


@register_template
class Fallthrough(ConfigurableTemplate):
  fills_type = HoleType.BLOCK_END
  required_cost = 0
  weight = 15

  def fill(self, hole, rng):
    return Program(0, [], const([]))


@register_template
class SimpleStatement(ConfigurableTemplate):
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 30

  def fill(self, hole, rng):
    stmt = Hole(HoleType.SIMPLE_STMT, hole.metadata)
    return Program(0, [stmt], concat)


@register_template
class CompoundStatement(ConfigurableTemplate):
  fills_type = HoleType.STMT
  required_cost = 2
  weight = 10

  def fill(self, hole, rng):
    stmt = Hole(HoleType.COMPOUND_STMT, hole.metadata)
    return Program(0, [stmt], concat)


def get_template_data(config):
  weighted_templates = []
  for template_cls in TEMPLATE_CLASSES:
    weighted_templates.append(top_down_refinement.WeightedTemplate(
        template_cls(config),
        weight=template_cls.weight))

  return TemplateData(
      weighted_templates=weighted_templates,
      root_object=Program(
          0, [Hole(HoleType.PROGRAM, HoleMetadata())], lambda x: x),
      hole_type_weights=HOLE_TYPE_WEIGHTS,
      start_with_initialization=False,
  )
