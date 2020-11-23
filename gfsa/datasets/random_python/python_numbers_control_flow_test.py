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
"""Tests for gfsa.datasets.google.random_python.python_numbers_control_flow."""

from absl.testing import absltest
from absl.testing import parameterized
import gast
import numpy as np
from gfsa.datasets.random_python import python_numbers_control_flow
from gfsa.datasets.random_python import top_down_refinement


class PythonNumbersControlFlowTest(parameterized.TestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters([{
      "testcase_name": type(wt.template).__name__,
      "template": wt.template
  } for wt in python_numbers_control_flow.CFG_DISTRIBUTION.weighted_templates])
  # pylint: enable=g-complex-comprehension
  def test_buildable(self, template):
    """Test that each template can be built when given acceptable arguments."""
    rng = np.random.RandomState(1234)

    # Construct a hole that this template can always fill.
    hole = top_down_refinement.Hole(
        template.fills_type,
        python_numbers_control_flow.ASTHoleMetadata(
            names_in_scope=("a",),
            inside_function=True,
            inside_loop=True,
            op_depth=0))
    self.assertTrue(template.can_fill(hole))

    # Make sure we can build this object with no errors.
    filler = template.fill(hole, rng)
    dummy_values = {
        python_numbers_control_flow.ASTHoleType.NUMBER:
            (lambda: gast.Constant(value=1, kind=None)),
        python_numbers_control_flow.ASTHoleType.BOOL:
            (lambda: gast.Constant(value=True, kind=None)),
        python_numbers_control_flow.ASTHoleType.STMT: gast.Pass,
        python_numbers_control_flow.ASTHoleType.STMTS: (lambda: []),
        python_numbers_control_flow.ASTHoleType.STMTS_NONEMPTY:
            (lambda: [gast.Pass()]),
        python_numbers_control_flow.ASTHoleType.BLOCK: (lambda: [gast.Pass()]),
    }
    hole_values = [dummy_values[h.hole_type]() for h in filler.holes]
    value = filler.build(*hole_values)

    # Check the type of the value that was built.
    if template.fills_type in (
        python_numbers_control_flow.ASTHoleType.STMTS_NONEMPTY,
        python_numbers_control_flow.ASTHoleType.BLOCK):
      self.assertTrue(value)
      for item in value:
        self.assertIsInstance(item, gast.stmt)
    elif template.fills_type == python_numbers_control_flow.ASTHoleType.STMTS:
      for item in value:
        self.assertIsInstance(item, gast.stmt)
    elif template.fills_type == python_numbers_control_flow.ASTHoleType.STMT:
      self.assertIsInstance(value, gast.stmt)
    elif template.fills_type in (python_numbers_control_flow.ASTHoleType.NUMBER,
                                 python_numbers_control_flow.ASTHoleType.BOOL):
      self.assertIsInstance(value, gast.expr)
    else:
      raise NotImplementedError(f"Unexpected fill type {template.fills_type}; "
                                "please update this test.")

    # Check that cost reflects number of AST nodes.
    total_cost = 0
    if isinstance(value, gast.AST):
      for _ in gast.walk(value):
        total_cost += 1
    else:
      for item in value:
        for _ in gast.walk(item):
          total_cost += 1

    self.assertEqual(template.required_cost, total_cost)

    cost_without_holes = total_cost - sum(
        python_numbers_control_flow.ALL_COSTS[h.hole_type]
        for h in filler.holes)

    self.assertEqual(filler.cost, cost_without_holes)

    # Check determinism
    for _ in range(20):
      rng = np.random.RandomState(1234)
      redo_value = template.fill(hole, rng).build(*hole_values)
      if isinstance(value, list):
        self.assertEqual([gast.dump(v) for v in value],
                         [gast.dump(v) for v in redo_value])
      else:
        self.assertEqual(gast.dump(value), gast.dump(redo_value))


if __name__ == "__main__":
  absltest.main()
