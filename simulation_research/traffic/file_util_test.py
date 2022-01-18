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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os
import sys
import tempfile

from absl.testing import absltest
import numpy as np
from six.moves import cPickle

from simulation_research.traffic import file_util


class UtilTest(absltest.TestCase):

  def setUp(self):
    super(UtilTest, self).setUp()
    self._output_dir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())

  def test_append_line_to_file(self):
    r"""Tests the output file.

    The output file contains the following.
    hello world
    (hello) "world"
    (hello) !!!!!!!!!!! @~#$%^&*()_+"world"
    aaaaaaaa
    bbbbbbbbbb
    backslash\ backslash
    backslash\ backslash
    backslash\\ backslash
    backslash\\\ backslash
    backslash\\ backslash
    """
    input_lines = ['hello world',
                   '(hello) "world"',
                   '(hello) !!!!!!!!!!! @~#$%^&*()_+"world"',
                   'aaaaaaaa\nbbbbbbbbbb',
                   r'backslash\ backslash',
                   'backslash\\ backslash',
                   r'backslash\\ backslash',
                   r'backslash\\\ backslash',
                   'backslash\\\\ backslash']
    file_path = os.path.join(self._output_dir, 'test_append_line_to_file.txt')
    for line in input_lines:
      file_util.append_line_to_file(file_path, line)
    self.assertTrue(file_util.f_exists(file_path))
    # Note that the linebreak in the input_lines[3].
    target_lines = ['hello world',
                    '(hello) "world"',
                    '(hello) !!!!!!!!!!! @~#$%^&*()_+"world"',
                    'aaaaaaaa',
                    'bbbbbbbbbb',
                    r'backslash\ backslash',
                    'backslash\\ backslash',
                    r'backslash\\ backslash',
                    r'backslash\\\ backslash',
                    'backslash\\\\ backslash']
    with file_util.f_open(file_path, 'r') as actual_file:
      line_counter = 0
      read_lines = actual_file.readlines()
      for line in read_lines:
        # Linebreak is appended to the target string.
        self.assertEqual(line, target_lines[line_counter] + '\n')
        line_counter += 1
    target_line_number = len(target_lines)
    self.assertEqual(target_line_number, line_counter)

  def test_save_load_variable(self):
    file_path = os.path.join(self._output_dir, 'test_output_data.pkl')

    # Case 1: Nested dictionary.
    data = {'zz': 1, 'b': 234, 123: 'asdfa', 'dict': {'a': 123, 't': 123}}
    file_util.save_variable(file_path, data)
    actual_variable = file_util.load_variable(file_path)
    self.assertEqual(data, actual_variable)
    self.assertIsInstance(actual_variable, dict)

    # Case 2: 2-level nested dictionary.
    data = collections.defaultdict(
        lambda: collections.defaultdict(list))
    data['first']['A'] = [1, 2, 3]
    data['first']['B'] = [1, 2, 3]
    data['second']['B'] = [1, 2, 3]
    data['second']['C'] = [1, 2, 3]
    data['third']['C'] = [1, 2, 3]
    data['third']['D'] = [1, 2, 3]
    data['path'] = 'asdfas/asdf/asdfasdf/'
    file_util.save_variable(file_path, data)
    actual_variable = file_util.load_variable(file_path)
    self.assertEqual(data, actual_variable)
    self.assertIsInstance(actual_variable, dict)

    # Case 3: Large array. If the size is too large, the test will timeout.
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] * 10000
    file_util.save_variable(file_path, data)
    actual_variable = file_util.load_variable(file_path)
    self.assertListEqual(data, actual_variable)
    self.assertIsInstance(actual_variable, list)

    # Case 4: numpy array.
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] * 10
    data = np.array(data)
    file_util.save_variable(file_path, data)
    actual_variable = file_util.load_variable(file_path)
    np.testing.assert_array_equal(data, actual_variable)
    self.assertIsInstance(actual_variable, np.ndarray)

    # Case 5: A list of tuples.
    x = [1, 2, 3]
    y = ['a', 'b', 'c']
    data = zip(x, y)
    # Saving zip variable does not affect the iterative variable.
    file_util.save_variable(file_path, data)
    actual_variable = file_util.load_variable(file_path)
    # python2 treats `actual_variable` as a list, however, python3 treats it as
    # an iterative object.
    self.assertListEqual(list(actual_variable), list(data))

    # Case 6: In python2, the itertools.tee cannot be saved by cPickle. However,
    # in python3, it can be saved.
    x = [1, 2, 3]
    y = ['a', 'b', 'c']
    data = zip(x, y)
    data_tee, _ = itertools.tee(data)
    python_version = sys.version_info[0]
    try:
      file_util.save_variable(file_path, data_tee)
      pickle_save_correctly = True
    except cPickle.PicklingError:
      pickle_save_correctly = False
    self.assertTrue((pickle_save_correctly and python_version == 3) or
                    (not pickle_save_correctly and python_version == 2))


if __name__ == '__main__':
  absltest.main()
