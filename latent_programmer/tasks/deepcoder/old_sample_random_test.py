# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for old_sample_random."""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

from latent_programmer.tasks.deepcoder import deepcoder_dsl
from latent_programmer.tasks.deepcoder import experiment as exp_module
from latent_programmer.tasks.deepcoder import old_sample_random

FLAGS = flags.FLAGS


class DatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._saved_flags = flagsaver.save_flag_values()

  def tearDown(self):
    flagsaver.restore_flag_values(self._saved_flags)
    super().tearDown()

  @parameterized.product(
      max_length=[5, 20],
      max_int=[50, 256],
      num_inputs=[1, 2],
      num_examples=[1, 3, 5],
  )
  def test_random_inputs(self, max_length, max_int, num_inputs, num_examples):
    with flagsaver.flagsaver(deepcoder_max_list_length=max_length,
                             deepcoder_max_int=max_int):
      example_inputs = old_sample_random.random_inputs(num_examples, num_inputs)
      # Shape is (num_examples, num_inputs).
      self.assertLen(example_inputs, num_examples)
      self.assertTrue(all(len(inputs) == num_inputs
                          for inputs in example_inputs))
      # Every individual input is ok.
      for inputs in example_inputs:
        self.assertTrue(all(type(x) in [int, list] for x in inputs))
        self.assertTrue(all(deepcoder_dsl.validate_result(x) for x in inputs))
      # Input types are consistent across examples.
      for i in range(num_inputs):
        self.assertTrue(all(type(inputs[i]) == type(example_inputs[0][i])  # pylint: disable=unidiomatic-typecheck
                            for inputs in example_inputs))

  def test_random_new_variable(self):
    x4 = deepcoder_dsl.variable_token(4)
    existing_variables = list(deepcoder_dsl.ALL_VARIABLES)
    existing_variables.remove(x4)
    self.assertEqual(old_sample_random.random_new_variable(existing_variables,
                                                           ordered=False), x4)

    existing_variables = ['x0', 'x1', 'x2']
    self.assertEqual(old_sample_random.random_new_variable(existing_variables,
                                                           ordered=True), 'x3')

  def test_random_new_variable_raises(self):
    for ordered in [True, False]:
      with self.assertRaises(ValueError):
        old_sample_random.random_new_variable(deepcoder_dsl.ALL_VARIABLES,
                                              ordered=ordered)

  @parameterized.parameters(
      ('Head', {int: [], list: [1]}, True),
      ('Head', {int: [1, 2], list: []}, False),
      ('Take', {int: [], list: [1, 2]}, False),
      ('Drop', {int: [1, 2], list: [3, 4]}, True),
      ('Map', {int: [], list: [1, 2]}, True),
      ('Map', {int: [1], list: []}, False),
      ('ZipWith', {int: [], list: [1]}, True),
      ('ZipWith', {int: [], list: [1, 2]}, True),
      ('ZipWith', {int: [1], list: []}, False),
  )
  def test_is_valid_operation(self, operation, var_dict, expected):
    op = deepcoder_dsl.TOKEN_TO_OPERATION[operation]
    self.assertEqual(old_sample_random.is_valid_operation(op, var_dict),
                     expected)

  @parameterized.parameters(
      (['x3 = 2 | x4 = [ 6 7 ] | x5 = [ 6 7 ]',
        'x3 = 1 | x4 = [ 5 6 ] | x5 = [ 5 6 ]'], True),
      (['x3 = 2 | x4 = [ 6 7 ] | x5 = [ 6 7 ]',
        'x3 = [ 5 7 ] | x4 = [ 5 7 ] | x5 = [ 5 7 ]'], True),
      (['x3 = 2 | x4 = [ 6 7 ] | x5 = [ 6 7 ]',
        'x3 = [ 5 6 ] | x4 = 1 | x5 = [ 5 6 ]'], False),
      (['x3 = 2 | x4 = [ 6 7 ] | x5 = [ 6 7 ]',
        'x3 = 1 | x4 = [ 5 6 ] | x5 = [ 6 7 ]'], False),
  )
  def test_is_redundant(self, states, expected):
    program_states = [deepcoder_dsl.ProgramState.from_str(state)
                      for state in states]
    self.assertEqual(old_sample_random.is_redundant(program_states), expected)

  @parameterized.named_parameters(
      ('no_dead_code',
       ('x3 = INPUT | x1 = INPUT | x2 = Sum x3 | x5 = Head x1 | '
        'x7 = Take x5 x2 | x6 = Sort x7'),
       False),
      ('dead_input',
       ('x3 = INPUT | x1 = INPUT | x2 = Sum x3 | x5 = Head x3 | '
        'x7 = Take x5 x2 | x6 = Sort x7'),
       False),
      ('dead_x5',
       ('x3 = INPUT | x1 = INPUT | x2 = Sum x3 | x5 = Head x3 | '
        'x7 = Take x1 x2 | x6 = Sort x7'),
       True),
  )
  def test_has_dead_code(self, program_str, expected):
    program = deepcoder_dsl.Program.from_str(program_str)
    self.assertEqual(old_sample_random.has_dead_code(program), expected)

  @parameterized.named_parameters(
      ('sort_min_has_duplicate_2',
       'x3 = INPUT | x4 = Sort x3 | x6 = Minimum x4', True),
      ('sort_max_all_different',
       'x3 = INPUT | x4 = Sort x3 | x6 = Maximum x4', False),
  )
  def test_has_duplicate_output(self, program_str, expected):
    program = deepcoder_dsl.Program.from_str(program_str)
    example_inputs = [[[4, 2, 3]], [[6, 8, 5]], [[2, 7]]]
    self.assertEqual(old_sample_random.has_duplicate_output(program,
                                                            example_inputs),
                     expected)

  @parameterized.named_parameters(
      ('sort_min_always_2',
       'x3 = INPUT | x4 = Sort x3 | x6 = Minimum x4', True),
      ('sort_max_different',
       'x3 = INPUT | x4 = Sort x3 | x6 = Maximum x4', False),
      ('sum_always_9',
       'x3 = INPUT | x8 = Sum x3', True),
      ('head_different',
       'x3 = INPUT | x8 = Head x3', False),
      ('always_all_zeros',
       ('x3 = INPUT | x8 = ZipWith (-) x3 x3 | x5 = Minimum x3 '
        '| x4 = Take x5 x8'),
       True),
  )
  def test_has_constant_output(self, program_str, expected):
    program = deepcoder_dsl.Program.from_str(program_str)
    example_inputs = [[[4, 2, 3]], [[2, 7]]]
    self.assertEqual(old_sample_random.has_constant_output(program,
                                                           example_inputs),
                     expected)

  @parameterized.product(
      num_inputs=[1, 2],
      num_statements=[2, 4],  # All generalization tasks should support these.
      experiment=list(exp_module.Experiment),
      is_train=[True, False],
      canonical_variable_order=[True, False],
      max_length=[5, 20],
      max_int=[50, 256],
  )
  def test_random_task(self, num_inputs, num_statements, experiment, is_train,
                       canonical_variable_order, max_length, max_int):
    for _ in range(10):
      with flagsaver.flagsaver(deepcoder_max_list_length=max_length,
                               deepcoder_max_int=max_int):
        task = old_sample_random.random_task(
            num_examples=5,
            num_inputs=num_inputs,
            num_statements=num_statements,
            is_train=is_train,
            canonical_variable_order=canonical_variable_order,
            experiment=experiment)
        program = task.program
        self.assertLen(program, num_statements)
        self.assertLen(task.examples, 5)
        for example in task.examples:
          self.assertEqual(program.run(example.inputs).get_output(),
                           example.output)
        if canonical_variable_order:
          # Test programs should have variable names in order.
          self.assertStartsWith(str(task.program), 'x0 = INPUT | x1 = ')


if __name__ == '__main__':
  absltest.main()
