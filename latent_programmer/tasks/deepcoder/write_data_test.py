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

import tensorflow as tf

from latent_programmer.tasks.deepcoder import deepcoder_dsl
from latent_programmer.tasks.deepcoder.dataset import write_data

FLAGS = flags.FLAGS


class WriteDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._saved_flags = flagsaver.save_flag_values()
    FLAGS.deepcoder_mod = 0  # Tests don't use mod unless otherwise specified.

  def tearDown(self):
    flagsaver.restore_flag_values(self._saved_flags)
    super().tearDown()

  def test_serialize_decomposition_examples(self):
    # Checks that task is serialized correctly using a hard-coded example.
    program = deepcoder_dsl.Program.from_str(
        'x0 = INPUT | x1 = INPUT | x2 = Map +1 x0 | x3 = Take x1 x2')
    examples = [deepcoder_dsl.Example([[4, 1, 3], 2], [5, 2])]
    task = deepcoder_dsl.ProgramTask(program, examples)

    results = write_data.serialize_decomposition_examples(task)
    expected_result_0 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'inputs':
                    write_data._bytes_feature(['x0 = [ 4 , 1 , 3 ] | x1 = 2']),
                'outputs':
                    write_data._bytes_feature(['[ 5 , 2 ]']),
                'next_part':
                    write_data._bytes_feature(['[ 5 , 2 , 4 ]']),
                'program_part':
                    write_data._bytes_feature(['x2 = Map +1 x0']),
            }))
    expected_result_1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'inputs':
                    write_data._bytes_feature(
                        ['x0 = [ 4 , 1 , 3 ] | x1 = 2 | x2 = [ 5 , 2 , 4 ]']),
                'outputs':
                    write_data._bytes_feature(['[ 5 , 2 ]']),
                'next_part':
                    write_data._bytes_feature(['[ 5 , 2 ]']),
                'program_part':
                    write_data._bytes_feature(['x3 = Take x1 x2']),
            }))
    self.assertLen(results, 2)
    self.assertEqual(tf.train.Example.FromString(results[0]), expected_result_0)
    self.assertEqual(tf.train.Example.FromString(results[1]), expected_result_1)

  def test_serialize_entire_program_example(self):
    # Checks that task is serialized correctly using a hard-coded example.
    program = deepcoder_dsl.Program.from_str(
        'x0 = INPUT | x1 = INPUT | x2 = Map +1 x0 | x3 = Take x1 x2')
    examples = [deepcoder_dsl.Example([[4, 1, 3], 2], [5, 2])]
    task = deepcoder_dsl.ProgramTask(program, examples)

    result = write_data.serialize_entire_program_example(task)
    expected_result = tf.train.Example(
        features=tf.train.Features(
            feature={
                'inputs':
                    write_data._bytes_feature(['x0 = [ 4 , 1 , 3 ] | x1 = 2']),
                'outputs':
                    write_data._bytes_feature(['[ 5 , 2 ]']),
                'program':
                    write_data._bytes_feature([str(program)]),
            }))
    self.assertEqual(tf.train.Example.FromString(result), expected_result)

  # TODO(kshi,jxihong): add a test for generate_task_for_experiment


if __name__ == '__main__':
  absltest.main()
