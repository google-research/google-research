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

"""Tests for utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
import utils


class TestTensorUtils(parameterized.TestCase):

  @parameterized.parameters(
      dict(input_iterable=[], batch_size=1, expected=[]),
      dict(input_iterable=[], batch_size=2, expected=[]),
      dict(input_iterable=[1], batch_size=1, expected=[[1]]),
      dict(input_iterable=[1], batch_size=2, expected=[[1]]),
      dict(input_iterable=[1, 2], batch_size=1, expected=[[1], [2]]),
      dict(input_iterable=[1, 2], batch_size=2, expected=[[1, 2]]),
      dict(input_iterable=[1, 2, 3], batch_size=2, expected=[[1, 2], [3]]),
      dict(
          input_iterable=[1, 2, 3, 4], batch_size=2, expected=[[1, 2], [3, 4]]),
  )
  def testBatchIterable(self, input_iterable, batch_size, expected):
    actual = list(utils.batch_iterable(input_iterable, batch_size))

    self.assertEqual(actual, expected)

  def testSparseToOneHot(self):
    seq = 'AY'
    expected_output = [[
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0.
    ],
                       [
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 1.
                       ]]
    self.assertListEqual(expected_output,
                         utils.residues_to_one_hot(seq).tolist())

  @parameterized.named_parameters(
      dict(
          testcase_name='pad by nothing',
          input_one_hot=np.array([[
              1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0.
          ]]),
          pad_length=1,
          expected=np.array([[
              1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0.
          ]])),
      dict(
          testcase_name='pad with one element',
          input_one_hot=np.array([[
              1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0.
          ]]),
          pad_length=2,
          expected=np.array([[
              1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0.
          ],
                             [
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0.
                             ]])),
  )
  def testPadOneHotSameLength(self, input_one_hot, pad_length, expected):
    actual = utils.pad_one_hot(input_one_hot, pad_length)

    self.assertTrue(
        np.allclose(actual, expected),
        msg='Actual: ' + str(actual) + '\nExpected: ' + str(expected))

  def test_absolute_paths_of_files_in_dir(self):
    test_dir = self.create_tempdir().full_path
    file_to_create = os.path.join(test_dir, 'a_file.txt')
    with open(file_to_create, 'w'):
      pass

    expected = [file_to_create]

    actual = utils.absolute_paths_of_files_in_dir(test_dir)

    self.assertCountEqual(actual, expected)


if __name__ == '__main__':
  tf.test.main()
