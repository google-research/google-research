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

"""Tests for Distance library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pandas.util.testing as pandas_testing
import tensorflow.compat.v1 as tf

from correct_batch_effects_wdn import distance


class DistanceTest(tf.test.TestCase):

  def setUp(self):
    super(DistanceTest, self).setUp()
    self.m = pd.DataFrame({
        'v1': [1.0, 3.0],
        'v2': [2.0, 2.0]}, index=['a', 'b'])
    self.v = pd.DataFrame({
        'v1': [1.0, 3.0],
        'v2': [2.0, 1.0]}, index=['a', 'b'])
    # cosine distance between m[0,:] and m[1,:]
    self.dcos = 1.0 - 7.0/np.sqrt(5.0 * 13.0)
    # euclidean distance between m[0,:] and m[1,:]
    self.deuc = np.sqrt((-2.0)**2.0 + 0.0**2.0)

  def testCombine(self):
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    result = distance._combine(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_almost_equal(expected, result)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = distance._combine(np.array([1.0, 2.0]),
                               np.array([[3.0, 4.0],
                                         [5.0, 6.0]]))
    np.testing.assert_almost_equal(expected, result)

  def testSplit(self):
    expected = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    result = distance._split(np.array([1.0, 2.0, 3.0, 4.0]), 2, 2)
    np.testing.assert_almost_equal(expected[0], result[0])
    np.testing.assert_almost_equal(expected[1], result[1])
    expected = (np.array([1.0, 2.0]), np.array([[3.0, 4.0],
                                                [5.0, 6.0]]))
    result = distance._split(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 2, 2)
    np.testing.assert_almost_equal(expected[0], result[0])
    np.testing.assert_almost_equal(expected[1], result[1])

  def testMatrix(self):
    expected = pd.DataFrame(
        {'a': [0.0, self.dcos],
         'b': [self.dcos, 0.0]}, index=['a', 'b'])
    pandas_testing.assert_frame_equal(
        expected,
        distance.matrix('cosine', self.m))

    expected = pd.DataFrame(
        {'a': [0.0, self.deuc],
         'b': [self.deuc, 0.0]}, index=['a', 'b'])
    pandas_testing.assert_frame_equal(
        expected,
        distance.matrix('euclidean', self.m))

    euc = lambda v1, v2: np.sqrt((v2 - v1).dot(v2 - v1))
    pandas_testing.assert_frame_equal(
        expected,
        distance.matrix(euc, self.m))


if __name__ == '__main__':
  tf.test.main()
