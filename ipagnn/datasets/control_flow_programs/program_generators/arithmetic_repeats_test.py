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

"""Tests for the control_flow_programs.program_generator module."""

import random

from absl.testing import absltest
from ipagnn.datasets.control_flow_programs.program_generators import arithmetic_repeats
from ipagnn.datasets.control_flow_programs.program_generators import arithmetic_repeats_config


class ProgramGeneratorTest(absltest.TestCase):

  def test_generate_statements(self):
    config = arithmetic_repeats_config.ArithmeticRepeatsConfig(
        base=10,
        length=10,
        max_repeat_statements=5,
        max_repetitions=2,
    )
    statements, unused_hole_statement_index = (
        arithmetic_repeats._generate_statements(config.length, config))
    self.assertLen(statements, 10)

  def test_to_python_source(self):
    statements = [
        ['+=', 0, 5],  # v0 += 5
        ['*=', 0, 6],  # v0 *= 6
        [arithmetic_repeats.REPEAT_OP, 5, 2],  # repeat 5x 2 statements:
        ['-=', 0, 3],  # v0 -= 3
        ['+=', 0, 1],  # v0 += 1
        ['+=', 0, 1],  # v0 += 1
    ]
    code = arithmetic_repeats._to_python_source(statements, num_variables=2)
    self.assertEqual(code, """
v0 += 5
v0 *= 6
v1 = 5
while v1 > 0:
  v1 -= 1
  v0 -= 3
  v0 += 1
v0 += 1
""".strip())

  def test_to_python_source_nested_repeats(self):
    random.seed(0)
    statements = [
        ['+=', 0, 5],  # v0 += 5
        ['*=', 0, 6],  # v0 *= 6
        [arithmetic_repeats.REPEAT_OP, 5, 5],  # repeat 5x 5 statements:
        ['-=', 0, 3],  # v0 *= 6
        [arithmetic_repeats.REPEAT_OP, 2, 2],  # repeat 2x 2 statements:
        ['-=', 0, 3],  # v0 -= 3
        ['+=', 0, 1],  # v0 += 1
        ['+=', 0, 2],  # v0 += 2
        ['+=', 0, 1],  # v0 += 1
    ]
    code = arithmetic_repeats._to_python_source(statements, num_variables=10)
    self.assertEqual(code, """
v0 += 5
v0 *= 6
v7 = 5
while v7 > 0:
  v7 -= 1
  v0 -= 3
  v8 = 2
  while v8 > 0:
    v8 -= 1
    v0 -= 3
    v0 += 1
  v0 += 2
v0 += 1
""".strip())

if __name__ == '__main__':
  absltest.main()
