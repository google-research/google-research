# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Generation templates for Python programs using numbers and control flow.

For this task, we assume that every variable holds a number.

Random sampling uses numpy for consistency with top_down_refinement, so that we
can control the whole random sampling using a single seed.
"""

import enum
from typing import Optional, Tuple

import dataclasses
import gast
import numpy as np

from gfsa.datasets.random_python import top_down_refinement

# Convenient type aliases
Hole = top_down_refinement.Hole
ASTWithHoles = top_down_refinement.ThingWithHoles
ASTNodeTemplate = top_down_refinement.HoleFillerTemplate


class ASTHoleType(enum.Enum):
  """A type of hole for this task."""
  NUMBER = "NUMBER"  # An integer or float expression
  BOOL = "BOOL"  # A boolean expression
  STMT = "STMT"  # A single statement or control flow block
  STMTS = "STMTS"  # Possibly empty list of statements (no jumps)
  STMTS_NONEMPTY = "STMTS_NONEMPTY"  # Nonempty list of statements (no jumps)
  BLOCK = "BLOCK"  # Nonempty list of statements, which might end in a jump


@dataclasses.dataclass(frozen=True)
class ASTHoleMetadata:
  """Context for what is valid inside this hole.

  Attributes:
    names_in_scope: Collection of names currently in scope. Stored as a tuple to
      ensure determinism given a seed, since iteration order for sets depends on
      a per-process Python randomization seed (see
      https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED)
    inside_function: Whether this hole is inside a function (so that returning
      is allowed).
    inside_loop: Whether this hole is inside a loop (so that break/continue is
      allowed)
    op_depth: Depth of this hole in a nested expression, used to limit the
      complexity of expressions.
  """
  names_in_scope: Tuple[str, Ellipsis]
  inside_function: bool
  inside_loop: bool
  op_depth: int


_NUMBER_COST = 1
_BOOL_COST = 1
_STMT_COST = 1
_BLOCK_COST = 1
_STMTS_COST = 0
_STMTS_NONEMPTY_COST = 1

ALL_COSTS = {
    ASTHoleType.NUMBER: _NUMBER_COST,
    ASTHoleType.BOOL: _BOOL_COST,
    ASTHoleType.STMT: _STMT_COST,
    ASTHoleType.BLOCK: _BLOCK_COST,
    ASTHoleType.STMTS: _STMTS_COST,
    ASTHoleType.STMTS_NONEMPTY: _STMTS_NONEMPTY_COST,
}


def make_name(identifier):
  """Returns a gast.Name for the given string identifier.

  Convenience function to avoid having to specify all the fields we don't
  care about. NotImplemented is used as a sentinel value, since gast usually
  populates that according to context, but we don't bother.

  Args:
    identifier: Identifier to use.
  """
  return gast.Name(
      id=identifier, ctx=NotImplemented, annotation=None, type_comment=None)


##########################
# Numbers
##########################


class NameReferenceTemplate(ASTNodeTemplate):
  """Reference an existing name."""
  fills_type = ASTHoleType.NUMBER
  required_cost = 1

  def can_fill(self, hole):
    return bool(hole.metadata.names_in_scope)

  def fill(self, hole, rng):
    name = rng.choice(hole.metadata.names_in_scope)
    return ASTWithHoles(1, [], lambda: make_name(name))


class ConstIntTemplate(ASTNodeTemplate):
  """Use a literal integer between 0 and 100."""
  fills_type = ASTHoleType.NUMBER
  required_cost = 1

  def fill(self, hole, rng):
    i = rng.randint(0, 100)
    return ASTWithHoles(1, [], lambda: gast.Constant(value=i, kind=None))


class BinOpTemplate(ASTNodeTemplate):
  """Mathematical operation on two numbers."""
  fills_type = ASTHoleType.NUMBER
  required_cost = 2 + 2 * _NUMBER_COST

  def __init__(self, max_depth=None):
    self.max_depth = max_depth

  def can_fill(self, hole):
    return self.max_depth is None or hole.metadata.op_depth < self.max_depth

  def fill(self, hole, rng):
    op = rng.choice([gast.Add, gast.Sub, gast.Mult, gast.Div])()

    def build(left, right):
      return gast.BinOp(left=left, op=op, right=right)

    sub_hole = Hole(
        ASTHoleType.NUMBER,
        dataclasses.replace(hole.metadata, op_depth=hole.metadata.op_depth + 1))
    return ASTWithHoles(2, [sub_hole, sub_hole], build)


class FunctionCallTemplate(ASTNodeTemplate):
  """Applies a function to some number of arguments."""
  fills_type = ASTHoleType.NUMBER

  def __init__(self, num_args, names, max_depth=None):
    self.num_args = num_args
    self.max_depth = max_depth
    self.names = names

  @property
  def required_cost(self):
    return 2 + self.num_args * _NUMBER_COST

  def can_fill(self, hole):
    return self.max_depth is None or hole.metadata.op_depth < self.max_depth

  def fill(self, hole, rng):
    name = rng.choice(self.names)

    def build(*args):
      return gast.Call(func=make_name(name), args=list(args), keywords=[])

    sub_hole = Hole(
        ASTHoleType.NUMBER,
        dataclasses.replace(hole.metadata, op_depth=hole.metadata.op_depth + 1))
    return ASTWithHoles(2, [sub_hole] * self.num_args, build)


##########################
# Booleans
##########################


class CompareTemplate(ASTNodeTemplate):
  """Compare two numbers."""
  fills_type = ASTHoleType.BOOL
  required_cost = 2 + 2 * _NUMBER_COST

  def fill(self, hole, rng):
    op = rng.choice([gast.Eq, gast.NotEq, gast.Lt, gast.LtE, gast.Gt,
                     gast.GtE])()

    def build(left, right):
      return gast.Compare(left=left, ops=[op], comparators=[right])

    number_hole = Hole(ASTHoleType.NUMBER, hole.metadata)
    return ASTWithHoles(2, [number_hole, number_hole], build)


class BoolOpTemplate(ASTNodeTemplate):
  """And/or between two booleans."""
  fills_type = ASTHoleType.BOOL
  required_cost = 2 + 2 * _BOOL_COST

  def __init__(self, max_depth=None):
    self.max_depth = max_depth

  def can_fill(self, hole):
    return self.max_depth is None or hole.metadata.op_depth < self.max_depth

  def fill(self, hole, rng):
    op = rng.choice([gast.And, gast.Or])()

    def build(left, right):
      return gast.BoolOp(op=op, values=[left, right])

    bool_hole = Hole(
        ASTHoleType.BOOL,
        dataclasses.replace(hole.metadata, op_depth=hole.metadata.op_depth + 1))
    return ASTWithHoles(2, [bool_hole, bool_hole], build)


class ConstBoolTemplate(ASTNodeTemplate):
  """Literal true or false."""
  fills_type = ASTHoleType.BOOL
  required_cost = 1

  def fill(self, hole, rng):
    value = rng.choice([True, False])
    return ASTWithHoles(1, [], lambda: gast.Constant(value=value, kind=None))


##########################
# Atomic statements
##########################


class AssignExistingTemplate(ASTNodeTemplate):
  """Assign to an existing variable."""
  fills_type = ASTHoleType.STMT
  required_cost = 2 + _NUMBER_COST

  def can_fill(self, hole):
    return bool(hole.metadata.names_in_scope)

  def fill(self, hole, rng):
    name = rng.choice(hole.metadata.names_in_scope)

    def build(v):
      return gast.Assign(targets=[make_name(name)], value=v)

    number_hole = Hole(ASTHoleType.NUMBER, hole.metadata)
    return ASTWithHoles(2, [number_hole], build)


class PassTemplate(ASTNodeTemplate):
  """No-op."""
  fills_type = ASTHoleType.STMT
  required_cost = 1

  def fill(self, hole, rng):
    return ASTWithHoles(1, [], gast.Pass)


class PrintNumberTemplate(ASTNodeTemplate):
  """Print out a number."""
  fills_type = ASTHoleType.STMT
  required_cost = 3 + _NUMBER_COST

  def fill(self, hole, rng):

    def build(v):
      return gast.Expr(
          value=gast.Call(func=make_name("print"), args=[v], keywords=[]))

    number_hole = Hole(ASTHoleType.NUMBER, hole.metadata)
    return ASTWithHoles(3, [number_hole], build)


##########################
# Composite statements
##########################


class IfBlockTemplate(ASTNodeTemplate):
  """Construct an if block."""
  fills_type = ASTHoleType.STMT
  required_cost = 1 + _BOOL_COST + _BLOCK_COST

  def fill(self, hole, rng):

    def build(test, body):
      return gast.If(test=test, body=body, orelse=[])

    test_hole = Hole(ASTHoleType.BOOL, hole.metadata)
    block_hole = Hole(ASTHoleType.BLOCK, hole.metadata)
    return ASTWithHoles(1, [test_hole, block_hole], build)


class IfElseBlockTemplate(ASTNodeTemplate):
  """Construct an if/else block."""
  fills_type = ASTHoleType.STMT
  required_cost = 1 + _BOOL_COST + 2 * _BLOCK_COST

  def fill(self, hole, rng):

    def build(test, body, orelse):
      return gast.If(test=test, body=body, orelse=orelse)

    test_hole = Hole(ASTHoleType.BOOL, hole.metadata)
    block_hole = Hole(ASTHoleType.BLOCK, hole.metadata)
    return ASTWithHoles(1, [test_hole, block_hole, block_hole], build)


class ForRangeBlockTemplate(ASTNodeTemplate):
  """Construct a for loop with a fresh variable over a range."""
  fills_type = ASTHoleType.STMT
  required_cost = 6 + _NUMBER_COST + _BLOCK_COST

  def fill(self, hole, rng):
    fresh_name = f"v{len(hole.metadata.names_in_scope)}"
    assert fresh_name not in hole.metadata.names_in_scope

    def build(maxval, body):
      return gast.For(
          target=make_name(fresh_name),
          iter=gast.Call(
              func=make_name("range"),
              args=[
                  gast.Call(func=make_name("int"), args=[maxval], keywords=[])
              ],
              keywords=[]),
          body=body,
          orelse=[],
          type_comment=None)

    number_hole = Hole(ASTHoleType.NUMBER, hole.metadata)
    body_hole = Hole(
        ASTHoleType.BLOCK,
        dataclasses.replace(
            hole.metadata,
            inside_loop=True,
            names_in_scope=hole.metadata.names_in_scope + (fresh_name,)))
    return ASTWithHoles(6, [number_hole, body_hole], build)


class WhileBlockTemplate(ASTNodeTemplate):
  """Construct a while loop."""
  fills_type = ASTHoleType.STMT
  required_cost = 1 + _BOOL_COST + _BLOCK_COST

  def fill(self, hole, rng):

    def build(test, body):
      return gast.While(test=test, body=body, orelse=[])

    test_hole = Hole(ASTHoleType.BOOL, hole.metadata)
    body_hole = Hole(ASTHoleType.BLOCK,
                     dataclasses.replace(hole.metadata, inside_loop=True))
    return ASTWithHoles(1, [test_hole, body_hole], build)


##########################
# Blocks
##########################

# A block represents a contigouous sequence of statements that might end with
# a return, break, or continue.


class ReturnNothingTemplate(ASTNodeTemplate):
  """Block that ends with a bare return."""
  fills_type = ASTHoleType.BLOCK
  required_cost = 1 + _STMTS_COST

  def can_fill(self, hole):
    return hole.metadata.inside_function

  def fill(self, hole, rng):
    stmts_hole = Hole(ASTHoleType.STMTS, hole.metadata)
    return ASTWithHoles(1, [stmts_hole],
                        lambda stmts: stmts + [gast.Return(value=None)])


class ReturnNumberTemplate(ASTNodeTemplate):
  """Block that ends by returning a number."""
  fills_type = ASTHoleType.BLOCK
  required_cost = 1 + _NUMBER_COST + _STMTS_COST

  def can_fill(self, hole):
    return hole.metadata.inside_function

  def fill(self, hole, rng):
    stmts_hole = Hole(ASTHoleType.STMTS, hole.metadata)
    number_hole = Hole(ASTHoleType.NUMBER, hole.metadata)
    return ASTWithHoles(1, [stmts_hole, number_hole],
                        lambda stmts, v: stmts + [gast.Return(value=v)])


class BreakTemplate(ASTNodeTemplate):
  """Block that ends by breaking out of the containing loop."""
  fills_type = ASTHoleType.BLOCK
  required_cost = 1 + _STMTS_COST

  def can_fill(self, hole):
    return hole.metadata.inside_loop

  def fill(self, hole, rng):
    stmts_hole = Hole(ASTHoleType.STMTS, hole.metadata)
    return ASTWithHoles(1, [stmts_hole], lambda stmts: stmts + [gast.Break()])


class ContinueTemplate(ASTNodeTemplate):
  """Block that ends by coninuing to the next iteration of the loop."""
  fills_type = ASTHoleType.BLOCK
  required_cost = 1 + _STMTS_COST

  def can_fill(self, hole):
    return hole.metadata.inside_loop

  def fill(self, hole, rng):
    stmts_hole = Hole(ASTHoleType.STMTS, hole.metadata)
    return ASTWithHoles(1, [stmts_hole],
                        lambda stmts: stmts + [gast.Continue()])


class FallthroughTemplate(ASTNodeTemplate):
  """Block that ends by falling through to the outer block.

  Note that every block has to contain at least one statement.
  """
  fills_type = ASTHoleType.BLOCK
  required_cost = _STMTS_NONEMPTY_COST

  def fill(self, hole, rng):
    stmts_hole = Hole(ASTHoleType.STMTS_NONEMPTY, hole.metadata)
    return ASTWithHoles(0, [stmts_hole], lambda stmts: stmts)


##########################
# Nonempty statements
##########################

# We handle fresh variables as a special case here, because they then are
# available to the following statements in the block.


class NewAssignTemplate(ASTNodeTemplate):
  """Assign to a new variable, and make it possible to use it later."""
  fills_type = ASTHoleType.STMTS_NONEMPTY
  required_cost = 2 + _NUMBER_COST + _STMTS_COST

  def fill(self, hole, rng):
    fresh_name = f"v{len(hole.metadata.names_in_scope)}"
    assert fresh_name not in hole.metadata.names_in_scope

    def build(v, rest):
      return [gast.Assign(targets=[make_name(fresh_name)], value=v)] + rest

    number_hole = Hole(ASTHoleType.NUMBER, hole.metadata)
    rest_hole = Hole(
        ASTHoleType.STMTS,
        dataclasses.replace(
            hole.metadata,
            names_in_scope=hole.metadata.names_in_scope + (fresh_name,)))
    return ASTWithHoles(2, [number_hole, rest_hole], build)


class NormalStatementTemplate(ASTNodeTemplate):
  """Add a normal statement."""
  fills_type = ASTHoleType.STMTS_NONEMPTY
  required_cost = _STMT_COST + _STMTS_COST

  def fill(self, hole, rng):

    def build(stmt, rest):
      return [stmt] + rest

    stmt_hole = Hole(ASTHoleType.STMT, hole.metadata)
    rest_hole = Hole(ASTHoleType.STMTS, hole.metadata)
    return ASTWithHoles(0, [stmt_hole, rest_hole], build)


##########################
# Possibly empty lists
##########################


class SomeStatementsTemplate(ASTNodeTemplate):
  """Insert some statements."""
  fills_type = ASTHoleType.STMTS
  required_cost = _STMTS_NONEMPTY_COST

  def fill(self, hole, rng):
    stmts_hole = Hole(ASTHoleType.STMTS_NONEMPTY, hole.metadata)
    return ASTWithHoles(0, [stmts_hole], lambda stmts: stmts)


class NoMoreStatementsTemplate(ASTNodeTemplate):
  """Don't insert any statements (as a last resort)."""
  fills_type = ASTHoleType.STMTS
  required_cost = 0

  def fill(self, hole, rng):
    return ASTWithHoles(0, [], lambda: [])


##########################
#  Sampling distributions
##########################

# Note regarding hole selection weights:
# We assign weights so that we avoid being forced into suboptimal choices later.
# For instance, it's always fine to stop generating statements, so we give
# adding more statements a low weight, and choose it less often. But it's
# annoying to be forced to insert "pass" everywhere due to lack of space, so we
# give partially-expanded single statements much more weight.

#  This distribution tends to create complex, nested control flow.
CFG_DISTRIBUTION = top_down_refinement.RefinementDistribution(
    hole_selection_weights={
        ASTHoleType.NUMBER: 3,
        ASTHoleType.BOOL: 10,
        ASTHoleType.STMT: 100,
        ASTHoleType.BLOCK: 10,
        ASTHoleType.STMTS: 1,
        ASTHoleType.STMTS_NONEMPTY: 100,
    },
    weighted_templates=[
        # Numbers
        top_down_refinement.WeightedTemplate(
            NameReferenceTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(ConstIntTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(BinOpTemplate(), weight=10),
        # Bools
        top_down_refinement.WeightedTemplate(CompareTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(BoolOpTemplate(), weight=3),
        top_down_refinement.WeightedTemplate(ConstBoolTemplate(), weight=2),
        # Statements
        top_down_refinement.WeightedTemplate(
            AssignExistingTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(PassTemplate(), weight=1),
        top_down_refinement.WeightedTemplate(PrintNumberTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(IfBlockTemplate(), weight=5),
        top_down_refinement.WeightedTemplate(IfElseBlockTemplate(), weight=5),
        top_down_refinement.WeightedTemplate(ForRangeBlockTemplate(), weight=5),
        top_down_refinement.WeightedTemplate(WhileBlockTemplate(), weight=3),
        # Blocks
        top_down_refinement.WeightedTemplate(ReturnNothingTemplate(), weight=5),
        top_down_refinement.WeightedTemplate(ReturnNumberTemplate(), weight=5),
        top_down_refinement.WeightedTemplate(BreakTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(ContinueTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(FallthroughTemplate(), weight=30),
        # Nonempty statement sequences
        top_down_refinement.WeightedTemplate(NewAssignTemplate(), weight=5),
        top_down_refinement.WeightedTemplate(
            NormalStatementTemplate(), weight=15),
        # Possibly empty statement sequences
        top_down_refinement.WeightedTemplate(
            SomeStatementsTemplate(), weight=1),
        top_down_refinement.WeightedTemplate(
            NoMoreStatementsTemplate(), weight=1, precedence=0),
    ])

# This distribution tends to create complex data flow.
DATAFLOW_DISTRIBUTION = top_down_refinement.RefinementDistribution(
    hole_selection_weights={
        ASTHoleType.NUMBER: 3,
        ASTHoleType.BOOL: 10,
        ASTHoleType.STMT: 100,
        ASTHoleType.BLOCK: 10,
        ASTHoleType.STMTS: 1,
        ASTHoleType.STMTS_NONEMPTY: 100,
    },
    weighted_templates=[
        # Numbers
        top_down_refinement.WeightedTemplate(
            NameReferenceTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(ConstIntTemplate(), weight=2),
        top_down_refinement.WeightedTemplate(
            BinOpTemplate(max_depth=3), weight=7),
        # Bools
        top_down_refinement.WeightedTemplate(CompareTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(
            BoolOpTemplate(max_depth=3), weight=3),
        top_down_refinement.WeightedTemplate(ConstBoolTemplate(), weight=2),
        # Statements
        top_down_refinement.WeightedTemplate(
            AssignExistingTemplate(), weight=20),
        top_down_refinement.WeightedTemplate(PassTemplate(), weight=1),
        top_down_refinement.WeightedTemplate(PrintNumberTemplate(), weight=5),
        top_down_refinement.WeightedTemplate(IfBlockTemplate(), weight=2),
        top_down_refinement.WeightedTemplate(IfElseBlockTemplate(), weight=2),
        top_down_refinement.WeightedTemplate(ForRangeBlockTemplate(), weight=2),
        top_down_refinement.WeightedTemplate(WhileBlockTemplate(), weight=2),
        # Blocks
        top_down_refinement.WeightedTemplate(ReturnNothingTemplate(), weight=3),
        top_down_refinement.WeightedTemplate(ReturnNumberTemplate(), weight=3),
        top_down_refinement.WeightedTemplate(BreakTemplate(), weight=10),
        top_down_refinement.WeightedTemplate(ContinueTemplate(), weight=6),
        top_down_refinement.WeightedTemplate(FallthroughTemplate(), weight=40),
        # Nonempty statement sequences
        top_down_refinement.WeightedTemplate(NewAssignTemplate(), weight=5),
        top_down_refinement.WeightedTemplate(
            NormalStatementTemplate(), weight=15),
        # Possibly empty statement sequences
        top_down_refinement.WeightedTemplate(
            SomeStatementsTemplate(), weight=1),
        top_down_refinement.WeightedTemplate(
            NoMoreStatementsTemplate(), weight=1, precedence=0),
    ])


# Meta-distribution for perturbed examples
def make_dataflow_fns_distribution(
    rng,
    weights_temperature = 0,
    max_depth_expected = 3,
    max_depth_maximum = 3):
  """Randomly sample a refinement distribution.

  Args:
    rng: Random number generator to use.
    weights_temperature: Dirichlet temperature to use when adjusting weights.
    max_depth_expected: Expected value of maximum expression nesting depth.
    max_depth_maximum: Maximum value of maximum expression nesting depth.

  Returns:
    A refinement distribution for examples.
  """
  if rng:
    max_depth = rng.binomial(max_depth_maximum,
                             max_depth_expected / max_depth_maximum)
  else:
    assert weights_temperature == 0
    assert max_depth_expected == max_depth_maximum
    max_depth = max_depth_maximum

  groups = [
      [  # Numbers
          top_down_refinement.WeightedTemplate(
              NameReferenceTemplate(), weight=10),
          top_down_refinement.WeightedTemplate(ConstIntTemplate(), weight=2),
          top_down_refinement.WeightedTemplate(
              BinOpTemplate(max_depth=max_depth), weight=5),
          top_down_refinement.WeightedTemplate(
              FunctionCallTemplate(
                  num_args=1, names=["foo_1", "bar_1"], max_depth=max_depth),
              weight=3),
          top_down_refinement.WeightedTemplate(
              FunctionCallTemplate(
                  num_args=2, names=["foo_2", "bar_2"], max_depth=max_depth),
              weight=2),
          top_down_refinement.WeightedTemplate(
              FunctionCallTemplate(
                  num_args=4, names=["foo_4", "bar_4"], max_depth=max_depth),
              weight=1),
      ],
      [  # Bools
          top_down_refinement.WeightedTemplate(CompareTemplate(), weight=10),
          top_down_refinement.WeightedTemplate(
              BoolOpTemplate(max_depth=max_depth), weight=3),
          top_down_refinement.WeightedTemplate(ConstBoolTemplate(), weight=2),
      ],
      [  # Statements
          top_down_refinement.WeightedTemplate(
              AssignExistingTemplate(), weight=20),
          top_down_refinement.WeightedTemplate(PassTemplate(), weight=1),
          top_down_refinement.WeightedTemplate(PrintNumberTemplate(), weight=5),
          top_down_refinement.WeightedTemplate(IfBlockTemplate(), weight=2),
          top_down_refinement.WeightedTemplate(IfElseBlockTemplate(), weight=2),
          top_down_refinement.WeightedTemplate(
              ForRangeBlockTemplate(), weight=2),
          top_down_refinement.WeightedTemplate(WhileBlockTemplate(), weight=2),
      ],
      [  # Blocks
          top_down_refinement.WeightedTemplate(
              ReturnNothingTemplate(), weight=3),
          top_down_refinement.WeightedTemplate(
              ReturnNumberTemplate(), weight=3),
          top_down_refinement.WeightedTemplate(BreakTemplate(), weight=10),
          top_down_refinement.WeightedTemplate(ContinueTemplate(), weight=6),
          top_down_refinement.WeightedTemplate(
              FallthroughTemplate(), weight=40),
      ],
      [  # Nonempty statement sequences
          top_down_refinement.WeightedTemplate(NewAssignTemplate(), weight=5),
          top_down_refinement.WeightedTemplate(
              NormalStatementTemplate(), weight=15),
      ]
  ]
  weighted_templates = [
      # Possibly empty statement sequences
      top_down_refinement.WeightedTemplate(SomeStatementsTemplate(), weight=1),
      top_down_refinement.WeightedTemplate(
          NoMoreStatementsTemplate(), weight=1, precedence=0),
  ]
  for group in groups:
    weights = np.array([template.weight for template in group])
    weights = weights / np.sum(weights)
    if rng and weights_temperature > 0:
      weights = np.random.dirichlet(weights / weights_temperature)
    weighted_templates.extend(
        dataclasses.replace(template, weight=weight)
        for template, weight in zip(group, weights))

  return top_down_refinement.RefinementDistribution(
      hole_selection_weights={
          ASTHoleType.NUMBER: 3,
          ASTHoleType.BOOL: 10,
          ASTHoleType.STMT: 100,
          ASTHoleType.BLOCK: 10,
          ASTHoleType.STMTS: 1,
          ASTHoleType.STMTS_NONEMPTY: 100,
      },
      weighted_templates=weighted_templates,
  )


# Dataflow distribution with function calls.
DATAFLOW_FNS_DISTRIBUTION = make_dataflow_fns_distribution(rng=None)
