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

"""Tests for sample_random."""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

from latent_programmer.tasks.deepcoder import deepcoder_dsl
from latent_programmer.tasks.deepcoder import experiment as exp_module
from latent_programmer.tasks.deepcoder import sample_random

FLAGS = flags.FLAGS


class DatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._saved_flags = flagsaver.save_flag_values()
    FLAGS.deepcoder_mod = 0  # Tests don't use mod unless otherwise specified.

  def tearDown(self):
    flagsaver.restore_flag_values(self._saved_flags)
    super().tearDown()

  def test_random_int(self):
    for _ in range(100):
      random_int = sample_random.random_int()
      self.assertBetween(
          random_int, deepcoder_dsl.MIN_INT, deepcoder_dsl.MAX_INT)

  @parameterized.named_parameters(
      ('10', 10),
      ('20', 20),
  )
  def test_random_int_with_mod(self, mod):
    with flagsaver.flagsaver(deepcoder_mod=mod):
      for _ in range(100):
        random_int = sample_random.random_int()
        self.assertBetween(random_int, 0, mod - 1)

  def test_random_list(self):
    for _ in range(100):
      random_list = sample_random.random_list()
      self.assertLessEqual(len(random_list), deepcoder_dsl.MAX_LIST_LENGTH)
      for elem in random_list:
        self.assertBetween(elem, deepcoder_dsl.MIN_INT, deepcoder_dsl.MAX_INT)

  @parameterized.named_parameters(
      ('1', 1),
      ('2', 2),
      ('3', 3),
  )
  def test_random_inputs(self, num_inputs):
    for _ in range(10):
      inputs = sample_random.random_inputs(num_inputs)
      self.assertLen(inputs, num_inputs)
      self.assertTrue(all(type(x) in [int, list] for x in inputs))

  @parameterized.named_parameters(
      ('single_int', [[1], [5]]),
      ('single_list', [[[6, 2, 5]]]),
      ('int_and_list', [[4, [8, 4, 3, 5]], [1, [6, 3]], [9, [5]]]),
  )
  def test_random_inputs_like(self, existing_inputs):
    first = existing_inputs[0]
    for _ in range(10):
      new_inputs = sample_random.random_inputs_like(existing_inputs)
      total_inputs = existing_inputs + [new_inputs]
      self.assertLen(new_inputs, len(first))
      self.assertTrue(all(type(x) == type(y)  # pylint: disable=unidiomatic-typecheck
                          for x, y in zip(new_inputs, first)))
      self.assertLen(set(str(inputs) for inputs in total_inputs),
                     len(total_inputs))

  def test_random_inputs_like_no_repeats(self):
    with flagsaver.flagsaver(deepcoder_mod=5):
      existing_inputs = [[4], [2], [0], [1]]
      new_inputs = sample_random.random_inputs_like(existing_inputs)
      self.assertEqual(new_inputs, [3])  # Only non-duplicate choice.

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
    self.assertEqual(sample_random.is_valid_operation(op, var_dict), expected)

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
    self.assertEqual(sample_random.is_redundant(program_states), expected)

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
    self.assertEqual(sample_random.has_dead_code(program), expected)

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
       'x3 = INPUT | x8 = ZipWith (-) x3 x3 | x5 = Minimum x3 | x4 = Take x5 x8',
       True),
  )
  def test_has_constant_output(self, program_str, expected):
    program = deepcoder_dsl.Program.from_str(program_str)
    example_inputs = [[[4, 2, 3]], [[2, 7]]]
    self.assertEqual(sample_random.has_constant_output(program, example_inputs),
                     expected)

  @parameterized.parameters(
      ([[[1, 2, 3], 2], [[4, 5, 6], 2]], 1, exp_module.Experiment.NONE),
      ([[10, [1, 10, 100]]], 2, exp_module.Experiment.SWITCH_CONCEPT_ORDER),
      ([[[1, 2, 3], [1, 10]], [[1, 2], [3, 4]]], 1,
       exp_module.Experiment.EXTEND_OP_FUNCTIONALITY),
      ([[[1, 2, 3], 2], [[4, 5, 6], 2]], 5, exp_module.Experiment.NONE),
      ([[10, [1, 10, 100]]], 5, exp_module.Experiment.SWITCH_CONCEPT_ORDER),
      ([[[1, 2, 3], [1, 10]], [[1, 2], [3, 4]]], 5,
       exp_module.Experiment.EXTEND_OP_FUNCTIONALITY),
  )
  def test_random_program(self, example_inputs, num_statements, experiment):
    for _ in range(10):
      for is_train in [True, False]:
        random_program = sample_random.random_program(
            example_inputs, num_statements, is_train, experiment)
        self.assertLen(random_program, num_statements)
        for inputs in example_inputs:
          self.assertIsNotNone(random_program.run(inputs))


if __name__ == '__main__':
  absltest.main()
