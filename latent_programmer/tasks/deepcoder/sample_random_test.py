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

"""Tests for deepcoder_dsl."""

from absl import flags
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
