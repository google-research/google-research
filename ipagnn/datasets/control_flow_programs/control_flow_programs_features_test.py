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

"""Tests for control_flow_programs_features.py."""

from absl.testing import absltest
from ipagnn.datasets.control_flow_programs import control_flow_programs_features
from ipagnn.datasets.control_flow_programs import python_interpreter


class ControlFlowProgramsFeaturesTest(absltest.TestCase):

  def setUp(self):
    super(ControlFlowProgramsFeaturesTest, self).setUp()
    self.executor = python_interpreter.ExecExecutor()

  def test_generate_example_from_python_object(self):
    python_source = """
v0 *= 2
v1 = 2
while v1 > 0:
  v1 -= 1
  v0 += 1
v0 -= 3
    """.strip()
    example = (
        control_flow_programs_features.generate_example_from_python_object(
            executor=self.executor,
            base=10,
            python_object=python_source,
            tokens_per_statement=4,
            target_output_length=1,
            mod=10,
            output_mod=None)
    )
    self.assertEqual(example['code_statements'], python_source)
    self.assertEqual(example['trace_statements'], """
v0 *= 2
v1 = 2
(v1 > 0)
v1 -= 1
v0 += 1
(v1 > 0)
v1 -= 1
v0 += 1
(v1 > 0)
v0 -= 3
    """.strip())
    self.assertEqual(example['target_output'], [1])


if __name__ == '__main__':
  absltest.main()
