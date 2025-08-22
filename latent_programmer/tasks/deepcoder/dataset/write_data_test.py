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

"""Tests for sample_random."""

import random

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf

from latent_programmer.tasks.deepcoder import deepcoder_dsl
from latent_programmer.tasks.deepcoder import experiment as exp_module
from latent_programmer.tasks.deepcoder.dataset import write_data

FLAGS = flags.FLAGS


class WriteDataTest(parameterized.TestCase):

  def test_serialize_decomposition_examples(self):
    # Checks that task is serialized correctly using a hard-coded example.
    program = deepcoder_dsl.Program.from_str(
        'x5 = INPUT | x1 = INPUT | x2 = Map (+1) x5 | x7 = Take x1 x2')
    examples = [deepcoder_dsl.Example([[4, 1, 3], 2], [5, 2]),
                deepcoder_dsl.Example([[2, 5, 8, 7], 3], [3, 6, 9])]
    task = deepcoder_dsl.ProgramTask(program, examples)

    results = write_data.serialize_decomposition_examples(task)
    result_0_example = tf.train.Example.FromString(results[0])
    result_1_example = tf.train.Example.FromString(results[1])

    expected_result_0 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'inputs':
                    write_data._bytes_feature(['x5 = [ 4 1 3 ] | x1 = 2',
                                               'x5 = [ 2 5 8 7 ] | x1 = 3']),
                'outputs':
                    write_data._bytes_feature(['[ 5 2 ]', '[ 3 6 9 ]']),
                'next_part':
                    write_data._bytes_feature(['[ 5 2 4 ]', '[ 3 6 9 8 ]']),
                'corrupted_next_part':
                    # This feature is generated randomly. Copy it over from the
                    # actual result, to effectively ignore it.
                    write_data._bytes_feature(
                        result_0_example.features.feature['corrupted_next_part']
                        .bytes_list.value),
                'program_part':
                    write_data._bytes_feature(['x2 = Map (+1) x5']),
            }))
    expected_result_1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'inputs':
                    write_data._bytes_feature(
                        ['x5 = [ 4 1 3 ] | x1 = 2 | x2 = [ 5 2 4 ]',
                         'x5 = [ 2 5 8 7 ] | x1 = 3 | x2 = [ 3 6 9 8 ]']),
                'outputs':
                    write_data._bytes_feature(['[ 5 2 ]', '[ 3 6 9 ]']),
                'next_part':
                    write_data._bytes_feature(['[ 5 2 ]', '[ 3 6 9 ]']),
                'corrupted_next_part':
                    # This feature is generated randomly.
                    write_data._bytes_feature(
                        result_1_example.features.feature['corrupted_next_part']
                        .bytes_list.value),
                'program_part':
                    write_data._bytes_feature(['x7 = Take x1 x2']),
            }))
    self.assertLen(results, 2)
    self.assertEqual(result_0_example, expected_result_0)
    self.assertEqual(result_1_example, expected_result_1)

  def test_serialize_entire_program_example(self):
    # Checks that task is serialized correctly using a hard-coded example.
    program = deepcoder_dsl.Program.from_str(
        'x5 = INPUT | x1 = INPUT | x2 = Map (+1) x5 | x7 = Take x1 x2')
    examples = [deepcoder_dsl.Example([[4, 1, 3], 2], [5, 2]),
                deepcoder_dsl.Example([[2, 5, 8, 7], 3], [3, 6, 9])]
    task = deepcoder_dsl.ProgramTask(program, examples)

    result = write_data.serialize_entire_program_example(task)
    expected_result = tf.train.Example(
        features=tf.train.Features(
            feature={
                'inputs':
                    write_data._bytes_feature(['x5 = [ 4 1 3 ] | x1 = 2',
                                               'x5 = [ 2 5 8 7 ] | x1 = 3']),
                'outputs':
                    write_data._bytes_feature(['[ 5 2 ]', '[ 3 6 9 ]']),
                'program':
                    write_data._bytes_feature([str(program)]),
            }))
    self.assertEqual(tf.train.Example.FromString(result), expected_result)

  @parameterized.parameters(
      # These cases are chosen because they run quickly.
      (exp_module.Experiment.COMPOSE_DIFFERENT_CONCEPTS, True),
      (exp_module.Experiment.SWITCH_CONCEPT_ORDER, False),
  )
  def test_generate_tasks_for_experiment(self, experiment, is_train):
    with flagsaver.flagsaver(num_examples=3,
                             deepcoder_max_list_length=5,
                             deepcoder_max_int=50):
      random.seed(0)
      tasks = write_data.generate_tasks_for_experiment(
          experiment.name, is_train, canonical_variable_order=True,
          num_programs=10)
      for task in tasks:
        for example in task.examples:
          self.assertEqual(task.program.run(example.inputs).get_output(),
                           example.output)


if __name__ == '__main__':
  absltest.main()
