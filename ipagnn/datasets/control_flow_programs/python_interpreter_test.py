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

"""Tests for python_interpreter.py."""

from absl.testing import absltest
from ipagnn.datasets.control_flow_programs import python_interpreter
from ipagnn.datasets.control_flow_programs import python_interpreter_trace
from ipagnn.datasets.control_flow_programs import python_programs
from ipagnn.datasets.control_flow_programs.program_generators import arithmetic_repeats_config
from ipagnn.datasets.control_flow_programs.program_generators import program_generators


class PythonInterpreterTest(absltest.TestCase):

  def setUp(self):
    super(PythonInterpreterTest, self).setUp()
    self.executor = python_interpreter.ExecExecutor()

  def test_evaluate_cfg_on_random_program(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    config = arithmetic_repeats_config.ArithmeticRepeatsConfig(
        base=10,
        length=10,
        max_repeat_statements=10,
        max_repetitions=9,
        max_repeat_block_size=5,
        repeat_probability=0.2,
        permit_nested_repeats=True,
    )
    python_source = program_generators.generate_python_source(
        config.length, config)
    cfg = python_programs.to_cfg(python_source)
    values = python_interpreter.evaluate_cfg(
        self.executor, cfg, initial_values=initial_values)
    self.assertIn('v0', values)

  def test_evaluate_cfg(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    python_source = """
v1 = 2
while v1 > 0:
  v1 -= 1
  v0 += 2
  v0 -= 1
  v0 *= 4
    """.strip()
    cfg = python_programs.to_cfg(python_source)
    values = python_interpreter.evaluate_cfg(
        self.executor, cfg, initial_values=initial_values)
    self.assertEqual(values['v0'], 36)

  def test_evaluate_cfg_break(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    python_source = """
v1 = 2
while v1 > 0:
  v1 -= 1
  v0 += 2
  break
    """.strip()
    cfg = python_programs.to_cfg(python_source)
    values = python_interpreter.evaluate_cfg(
        self.executor, cfg, initial_values=initial_values)
    self.assertEqual(values['v0'], 3)

  def test_evaluate_cfg_straightline(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    python_source = """
v0 += 5
v0 *= 6
v0 -= 3
    """.strip()
    cfg = python_programs.to_cfg(python_source)
    final_values = python_interpreter.evaluate_cfg(
        self.executor, cfg, initial_values=initial_values)
    final_value = final_values['v0']
    self.assertEqual(33, final_value)

  def test_evaluate_cfg_mod_n(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    python_source = """
v0 += 5
v0 *= 6
v0 -= 3
    """.strip()
    cfg = python_programs.to_cfg(python_source)
    final_values = python_interpreter.evaluate_cfg(
        self.executor, cfg, mod=5, initial_values=initial_values)
    final_value = final_values['v0']
    self.assertEqual(3, final_value)

  def test_evaluate_cfg_while(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    python_source = """
v0 += 5
v0 *= 6
v1 = 5
while v1 > 0:
  v1 -= 1
  v0 -= 3
    """.strip()
    cfg = python_programs.to_cfg(python_source)
    final_values = python_interpreter.evaluate_cfg(
        self.executor, cfg, initial_values=initial_values)
    final_value = final_values['v0']
    self.assertEqual(21, final_value)

  def test_evaluate_cfg_multi_statement_while(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    python_source = """
v0 += 5
v0 *= 6
v1 = 5
while v1 > 0:
  v1 -= 1
  v0 -= 3
  v0 += 1
    """.strip()
    cfg = python_programs.to_cfg(python_source)
    final_values = python_interpreter.evaluate_cfg(
        self.executor, cfg, initial_values=initial_values)
    final_value = final_values['v0']
    self.assertEqual(26, final_value)

  def test_evaluate_cfg_nested_whiles(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    python_source = """
v0 += 5
v0 *= 6
v1 = 3
while v1 > 0:
  v1 -= 1
  v2 = 3
  while v2 > 0:
    v2 -= 1
    v0 -= 1
    v0 -= 1
    v0 -= 1
    """.strip()
    cfg = python_programs.to_cfg(python_source)
    final_values = python_interpreter.evaluate_cfg(
        self.executor, cfg, initial_values=initial_values)
    final_value = final_values['v0']
    self.assertEqual(9, final_value)

  def test_evaluate_cfg_trace(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    python_source = """
v0 += 5
v0 *= 6
v1 = 2
while v1 > 0:
  v1 -= 1
  v0 -= 3
  v0 += 1
    """.strip()
    cfg = python_programs.to_cfg(python_source)
    trace_fn = python_interpreter_trace.make_trace_fn(python_source, cfg)
    final_values = python_interpreter.evaluate_cfg(
        self.executor, cfg, initial_values=initial_values, trace_fn=trace_fn)
    final_value = final_values['v0']
    expected_trace_cfg_node_indexes = [0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 3]
    expected_trace_line_indexes = expected_trace_cfg_node_indexes.copy()
    expected_trace_values = [
        [{'v0': 6}],
        [{'v0': 36}],
        [{'v0': 36, 'v1': 2}],
        [{'v0': 36, 'v1': 2, 'vBranch': True}],
        [{'v0': 36, 'v1': 1, 'vBranch': True}],
        [{'v0': 33, 'v1': 1, 'vBranch': True}],
        [{'v0': 34, 'v1': 1, 'vBranch': True}],
        [{'v0': 34, 'v1': 1, 'vBranch': True}],
        [{'v0': 34, 'v1': 0, 'vBranch': True}],
        [{'v0': 31, 'v1': 0, 'vBranch': True}],
        [{'v0': 32, 'v1': 0, 'vBranch': True}],
        [{'v0': 32, 'v1': 0, 'vBranch': False}],
    ]
    expected_cfg_node_index_values = [
        [{'v0': 6}],  # v0 += 5
        [{'v0': 36}],  # v0 *= 6
        [{'v0': 36, 'v1': 2}],  # v1 = 2
        [{'v0': 36, 'v1': 2, 'vBranch': True},  # vBranch = (v1 > 0)
         {'v0': 34, 'v1': 1, 'vBranch': True},
         {'v0': 32, 'v1': 0, 'vBranch': False}],
        [{'v0': 36, 'v1': 1, 'vBranch': True},  # v1 -= 1
         {'v0': 34, 'v1': 0, 'vBranch': True}],
        [{'v0': 33, 'v1': 1, 'vBranch': True},  # v0 -= 3
         {'v0': 31, 'v1': 0, 'vBranch': True}],
        [{'v0': 34, 'v1': 1, 'vBranch': True},  # v0 += 1
         {'v0': 32, 'v1': 0, 'vBranch': True}],
    ]
    expected_line_index_values = expected_cfg_node_index_values
    self.assertEqual(final_value, 32)
    self.assertEqual(trace_fn.trace.trace_values, expected_trace_values)
    self.assertEqual(trace_fn.trace.trace_cfg_node_indexes,
                     expected_trace_cfg_node_indexes)
    self.assertEqual(trace_fn.trace.trace_line_indexes,
                     expected_trace_line_indexes)
    self.assertEqual(trace_fn.trace.cfg_node_index_values,
                     expected_cfg_node_index_values)
    self.assertEqual(trace_fn.trace.line_index_values,
                     expected_line_index_values)

  def test_evaluate_cfg_nested_while_trace(self):
    initial_value = 1  # v0 = 1
    initial_values = {'v0': initial_value}
    python_source = """
v0 += 49
v1 = 2
while v1 > 0:
  v1 -= 1
  v2 = 2
  while v2 > 0:
    v2 -= 1
    v0 -= 3
    v0 += 1
    """.strip()
    cfg = python_programs.to_cfg(python_source)
    trace_fn = python_interpreter_trace.make_trace_fn(python_source, cfg)
    final_values = python_interpreter.evaluate_cfg(
        self.executor, cfg, initial_values=initial_values, trace_fn=trace_fn)
    final_value = final_values['v0']
    expected_trace_cfg_node_indexes = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5,
        2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 2]
    expected_trace_values = [
        [{'v0': 50}],
        [{'v0': 50, 'v1': 2}],
        [{'v0': 50, 'v1': 2, 'vBranch': True}],
        [{'v0': 50, 'v1': 1, 'vBranch': True}],
        [{'v0': 50, 'v1': 1, 'v2': 2, 'vBranch': True}],
        [{'v0': 50, 'v1': 1, 'v2': 2, 'vBranch': True}],
        [{'v0': 50, 'v1': 1, 'v2': 1, 'vBranch': True}],
        [{'v0': 47, 'v1': 1, 'v2': 1, 'vBranch': True}],
        [{'v0': 48, 'v1': 1, 'v2': 1, 'vBranch': True}],
        [{'v0': 48, 'v1': 1, 'v2': 1, 'vBranch': True}],
        [{'v0': 48, 'v1': 1, 'v2': 0, 'vBranch': True}],
        [{'v0': 45, 'v1': 1, 'v2': 0, 'vBranch': True}],
        [{'v0': 46, 'v1': 1, 'v2': 0, 'vBranch': True}],
        [{'v0': 46, 'v1': 1, 'v2': 0, 'vBranch': False}],
        [{'v0': 46, 'v1': 1, 'v2': 0, 'vBranch': True}],
        [{'v0': 46, 'v1': 0, 'v2': 0, 'vBranch': True}],
        [{'v0': 46, 'v1': 0, 'v2': 2, 'vBranch': True}],
        [{'v0': 46, 'v1': 0, 'v2': 2, 'vBranch': True}],
        [{'v0': 46, 'v1': 0, 'v2': 1, 'vBranch': True}],
        [{'v0': 43, 'v1': 0, 'v2': 1, 'vBranch': True}],
        [{'v0': 44, 'v1': 0, 'v2': 1, 'vBranch': True}],
        [{'v0': 44, 'v1': 0, 'v2': 1, 'vBranch': True}],
        [{'v0': 44, 'v1': 0, 'v2': 0, 'vBranch': True}],
        [{'v0': 41, 'v1': 0, 'v2': 0, 'vBranch': True}],
        [{'v0': 42, 'v1': 0, 'v2': 0, 'vBranch': True}],
        [{'v0': 42, 'v1': 0, 'v2': 0, 'vBranch': False}],
        [{'v0': 42, 'v1': 0, 'v2': 0, 'vBranch': False}],
    ]
    expected_cfg_node_index_values = [
        [{'v0': 50}],  # v0 += 49
        [{'v0': 50, 'v1': 2}],  # v1 = 2
        [{'v0': 50, 'v1': 2, 'vBranch': True},  # vBranch = (v1 > 0)
         {'v0': 46, 'v1': 1, 'v2': 0, 'vBranch': True},
         {'v0': 42, 'v1': 0, 'v2': 0, 'vBranch': False}],
        [{'v0': 50, 'v1': 1, 'vBranch': True},
         {'v0': 46, 'v1': 0, 'v2': 0, 'vBranch': True}],  # v1 -= 1
        [{'v0': 50, 'v1': 1, 'v2': 2, 'vBranch': True},
         {'v0': 46, 'v1': 0, 'v2': 2, 'vBranch': True}],  # v2 = 2
        [{'v0': 50, 'v1': 1, 'v2': 2, 'vBranch': True},
         {'v0': 48, 'v1': 1, 'v2': 1, 'vBranch': True},
         {'v0': 46, 'v1': 1, 'v2': 0, 'vBranch': False},
         {'v0': 46, 'v1': 0, 'v2': 2, 'vBranch': True},
         {'v0': 44, 'v1': 0, 'v2': 1, 'vBranch': True},
         {'v0': 42, 'v1': 0, 'v2': 0, 'vBranch': False}],  # vBranch = (v2 > 0)
        [{'v0': 50, 'v1': 1, 'v2': 1, 'vBranch': True},
         {'v0': 48, 'v1': 1, 'v2': 0, 'vBranch': True},
         {'v0': 46, 'v1': 0, 'v2': 1, 'vBranch': True},
         {'v0': 44, 'v1': 0, 'v2': 0, 'vBranch': True}],  # v2 -= 1
        [{'v0': 47, 'v1': 1, 'v2': 1, 'vBranch': True},
         {'v0': 45, 'v1': 1, 'v2': 0, 'vBranch': True},
         {'v0': 43, 'v1': 0, 'v2': 1, 'vBranch': True},
         {'v0': 41, 'v1': 0, 'v2': 0, 'vBranch': True}],  # v0 -= 3
        [{'v0': 48, 'v1': 1, 'v2': 1, 'vBranch': True},
         {'v0': 46, 'v1': 1, 'v2': 0, 'vBranch': True},
         {'v0': 44, 'v1': 0, 'v2': 1, 'vBranch': True},
         {'v0': 42, 'v1': 0, 'v2': 0, 'vBranch': True}],  # v0 += 1
    ]
    self.assertEqual(trace_fn.trace.trace_cfg_node_indexes,
                     expected_trace_cfg_node_indexes)
    self.assertEqual(trace_fn.trace.trace_values, expected_trace_values)
    self.assertEqual(trace_fn.trace.cfg_node_index_values,
                     expected_cfg_node_index_values)
    self.assertEqual(final_value, 42)


if __name__ == '__main__':
  absltest.main()
