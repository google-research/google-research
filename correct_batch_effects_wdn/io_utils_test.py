# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for io_utils library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import pandas
import tensorflow as tf

from correct_batch_effects_wdn import io_utils

FLAGS = flags.FLAGS

FOLD_DATA = {
    'sequence': ['AATT', 'AGTC'],
    'count1': [0, 12],
    'count2': [1, 2],
}


class IOTest(tf.test.TestCase):

  def setUp(self):
    super(IOTest, self).setUp()
    self.base_path = os.path.join(FLAGS.test_tmpdir, 'test.h5')
    df = pandas.DataFrame(FOLD_DATA)
    io_utils.write_dataframe_to_hdf5(df, self.base_path)

  def testReadHDF5(self):
    df = pandas.DataFrame(FOLD_DATA)
    actual = io_utils.read_dataframe_from_hdf5(self.base_path)
    self.assertTrue(actual.equals(df))

  def testWriteHDF5Raises(self):
    self.assertRaises(TypeError, io_utils.write_dataframe_to_hdf5,
                      'type should be DataFrame', 'path/to/file.h5')


if __name__ == '__main__':
  tf.test.main()
