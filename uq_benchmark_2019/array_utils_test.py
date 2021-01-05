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

# Lint as: python2, python3
"""Tests for array_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from six.moves import range
import tensorflow.compat.v2 as tf
from uq_benchmark_2019 import array_utils

STRUCT = {
    'k1': [{'a': 1, 'b': 2, 'c': 3}, {'a': 2, 'b': 3, 'c': 4}],
    'k2': [{'a': 5, 'b': 6, 'c': 7}, {'a': 8, 'b': 9, 'c': 0}],
    'k3': [{'a': 5, 'b': 6}, {'a': 8, 'b': 9}],
}


class ArrayUtilsTest(absltest.TestCase):

  def test_threaded_map_structure(self):
    fn = lambda x: x+1
    actual = array_utils.threaded_map_structure(fn, STRUCT)
    expected = tf.nest.map_structure(fn, STRUCT)
    self.assertEqual(actual, expected)

  def test_threaded_map_structure_2args(self):
    fn = lambda x, y: x+y
    actual = array_utils.threaded_map_structure(fn, STRUCT, STRUCT)
    expected = tf.nest.map_structure(fn, STRUCT, STRUCT)
    self.assertEqual(actual, expected)

  def test_threaded_map_structure_fn_with_error(self):
    fn = lambda x: x / 0
    result = array_utils.threaded_map_structure(fn, STRUCT)
    tf.nest.assert_same_structure(result, STRUCT)
    expected_flat = [None] * len(tf.nest.flatten(STRUCT))
    self.assertEqual(tf.nest.flatten(result), expected_flat)

  def test_simple_slices(self):
    self.assertEqual(1, array_utils.slice_structure(STRUCT, ['k1', 0, 'a']))
    self.assertEqual(8, array_utils.slice_structure(STRUCT, ['k2', 1, 'a']))
    self.assertEqual(4, array_utils.slice_structure(STRUCT, ['k1', -1, 'c']))

  def test_all_slice_dict(self):
    actual = array_utils.slice_structure(
        STRUCT, [array_utils.SLICE.ALL, 0, 'a'])
    expected = {'k1': 1, 'k2': 5, 'k3': 5}
    self.assertEqual(expected, actual)

  def test_all_slice_list(self):
    actual = array_utils.slice_structure(STRUCT, ['k1', array_utils.SLICE.ALL])
    self.assertEqual(actual, STRUCT['k1'])

  def test_all_slice_set(self):
    struct = set(range(10))
    with self.assertRaises(NotImplementedError):
      array_utils.slice_structure(struct, [array_utils.SLICE.ALL])

  def test_list_slice_from_list(self):
    actual = array_utils.slice_structure(STRUCT, ['k1', [0, 1, 1], 'a'])
    self.assertEqual(actual, [1, 2, 2])

  def test_list_slice_from_dict(self):
    actual = array_utils.slice_structure(STRUCT, [['k1', 'k2', 'k1'], 0, 'b'])
    self.assertEqual(actual, [2, 6, 2])

  def test_invalid_key(self):
    with self.assertRaises(KeyError):
      array_utils.slice_structure(STRUCT, ['k3', [0], 'c'])
    with self.assertRaises(IndexError):
      array_utils.slice_structure(STRUCT, ['k3', [3], 'c'])

  def test_base_case(self):
    self.assertEqual(STRUCT, array_utils.slice_structure(STRUCT, []))

if __name__ == '__main__':
  absltest.main()
