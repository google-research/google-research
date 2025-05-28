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

"""Tests for protenn.test_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from protenn import test_util


class TestUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='empty df',
          df1=pd.DataFrame(),
          df2=pd.DataFrame(),
      ),
      dict(
          testcase_name='one column, ints',
          df1=pd.DataFrame({'col1': [1, 2, 3]}),
          df2=pd.DataFrame({'col1': [1, 2, 3]}),
      ),
      dict(
          testcase_name='one column, floats',
          df1=pd.DataFrame({'col1': [1., 2.]}),
          df2=pd.DataFrame({'col1': [1., 2.]}),
      ),
      dict(
          testcase_name='one column, sets',
          df1=pd.DataFrame({'col1': [
              {1., 2.},
              {3., 4.},
          ]}),
          df2=pd.DataFrame({'col1': [
              {1., 2.},
              {3., 4.},
          ]}),
      ),
      dict(
          testcase_name='one column, np.arrays',
          df1=pd.DataFrame({'col1': [
              np.array([1., 2.]),
              np.array([3., 4.]),
          ]}),
          df2=pd.DataFrame({'col1': [
              np.array([1., 2.]),
              np.array([3., 4.]),
          ]}),
      ),
      dict(
          testcase_name='two columns, ints and floats',
          df1=pd.DataFrame({
              'col1': ['a', 'b'],
              'col2': [1., 2.],
          }),
          df2=pd.DataFrame({
              'col1': ['a', 'b'],
              'col2': [1., 2.],
          }),
      ),
      dict(
          testcase_name='two columns, strings and floats, reordered',
          df1=pd.DataFrame({
              'col1': ['a', 'b'],
              'col2': [1., 2.],
          }),
          df2=pd.DataFrame({
              'col1': ['b', 'a'],
              'col2': [2., 1.],
          }),
          order_by_column='col1',
      ),
      dict(
          testcase_name='two columns, strings and np.arrays, reordered',
          df1=pd.DataFrame({
              'col1': ['a', 'b'],
              'col2': [
                  np.array([1., 2.]),
                  np.array([3., 4.]),
              ],
          }),
          df2=pd.DataFrame({
              'col1': ['b', 'a'],
              'col2': [
                  np.array([3., 4.]),
                  np.array([1., 2.]),
              ],
          }),
          order_by_column='col1',
      ),
  )
  def test_assert_dataframes_equal_no_error(self,
                                            df1,
                                            df2,
                                            order_by_column=None):
    test_util.assert_dataframes_equal(self, df1, df2, order_by_column)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty df and nonempty df',
          df1=pd.DataFrame(),
          df2=pd.DataFrame({'col1': [1., 2., 3.]}),
      ),
      dict(
          testcase_name='one column, different lengths',
          df1=pd.DataFrame({'col1': [1, 2]}),
          df2=pd.DataFrame({'col1': [1, 2, 3]}),
      ),
      dict(
          testcase_name='one column, sets',
          df1=pd.DataFrame({'col1': [
              {1., 2.},
              {3., 4.},
          ]}),
          df2=pd.DataFrame({'col1': [
              {1., 2.},
              set([]),
          ]}),
      ),
      dict(
          testcase_name='one column, np.arrays',
          df1=pd.DataFrame({'col1': [
              np.array([1., 2.]),
              np.array([3., 4.]),
          ]}),
          df2=pd.DataFrame({'col1': [
              np.array([1., 2.]),
              np.array([]),
          ]}),
      ),
      dict(
          testcase_name='two columns, ints and floats',
          df1=pd.DataFrame({
              'col1': ['a', 'b'],
              'col2': [1., 2.],
          }),
          df2=pd.DataFrame({
              'col1': ['a', 'b'],
              'col2': [1., 9999999999999.],
          }),
      ),
      dict(
          testcase_name='two columns, strings and floats, reordered',
          df1=pd.DataFrame({
              'col1': ['a', 'b'],
              'col2': [1., 2.],
          }),
          df2=pd.DataFrame({
              'col1': ['b', 'a'],
              'col2': [9999999999999., 1.],
          }),
          order_by_column='col1',
      ),
  )
  def test_assert_dataframes_equal_error(self, df1, df2, order_by_column=None):
    with self.assertRaises(AssertionError):
      test_util.assert_dataframes_equal(self, df1, df2, order_by_column)

  def test_assert_dataframes_equal_nan_equal_nan(self):
    df1 = pd.DataFrame({'col1': [float('nan'),]})

    test_util.assert_dataframes_equal(self, df1, df1, nan_equals_nan=True)

  def test_assert_dataframes_equal_nan_raises(self):
    df1 = pd.DataFrame({'col1': [float('nan'),]})

    with self.assertRaisesRegex(AssertionError, 'nan'):
      test_util.assert_dataframes_equal(self, df1, df1, nan_equals_nan=False)


if __name__ == '__main__':
  absltest.main()
