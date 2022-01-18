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

"""Tests for pandas_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized

import pandas as pd

from aqt.utils import pandas_utils


class PandasUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.df1 = pd.DataFrame({
        'name': ['aa', 'ab', 'cc', 'dd'],
        'run': [1, 1, 2, 3],
        'user': ['sam', 'max', 'max', 'jo']
    })

    self.df2 = pd.DataFrame({
        'name': ['aa', 'ab', 'cc'],
        'run': [1, 1, 2],
        'test1': ['x', 'y', 'z'],
        'test2': [10, 20, 30]
    })

    self.df3 = pd.DataFrame({
        'name': ['a', 'a', 'a', 'b', 'b', 'b'],
        'len': [2, 4, 6, 9, 10, 11],
        'dis': [1, 1, 1, 5, 6, 7],
    })

  @parameterized.named_parameters(
      dict(
          testcase_name='run_is_1',
          column_name='run',
          values=1,
          exp=pd.DataFrame(
              {
                  'name': ['aa', 'ab'],
                  'run': [1, 1],
                  'user': ['sam', 'max']
              },
              index=[0, 1])),
      dict(
          testcase_name='run_is_1_or_2',
          column_name='run',
          values=[1, 2],
          exp=pd.DataFrame(
              {
                  'name': ['aa', 'ab', 'cc'],
                  'run': [1, 1, 2],
                  'user': ['sam', 'max', 'max']
              },
              index=[0, 1, 2])),
  )
  def test_select_rows_by_column_values(self, column_name, values, exp):
    res = pandas_utils.select_rows_by_column_values(
        self.df1, column_name=column_name, values=values)
    pd.testing.assert_frame_equal(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='name_starts_with_a',
          column_name='name',
          regex_str='a.*',
          exp=pd.DataFrame({
              'name': ['aa', 'ab'],
              'run': [1, 1],
              'user': ['sam', 'max']
          })),
      dict(
          testcase_name='user_is_sam',
          column_name='user',
          regex_str='sam',
          exp=pd.DataFrame({
              'name': ['aa'],
              'run': [1],
              'user': ['sam']
          })),
  )
  def test_select_rows_by_regex(self, column_name, regex_str, exp):
    res = pandas_utils.select_rows_by_regex(
        self.df1, column_name=column_name, regex_str=regex_str)
    pd.testing.assert_frame_equal(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='run_is_not_1',
          column_name='run',
          values=[1],
          exp=pd.DataFrame(
              {
                  'name': ['cc', 'dd'],
                  'run': [2, 3],
                  'user': ['max', 'jo']
              },
              index=[2, 3])),
      dict(
          testcase_name='run_is_not_1_or_2',
          column_name='run',
          values=[1, 2],
          exp=pd.DataFrame({
              'name': ['dd'],
              'run': [3],
              'user': ['jo']
          },
                           index=[3])),
  )
  def test_drop_rows_by_column_values(self, column_name, values, exp):
    res = pandas_utils.drop_rows_by_column_values(
        self.df1, column_name=column_name, values=values)
    pd.testing.assert_frame_equal(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='name_does_not_start_with_a',
          column_name='name',
          regex_str='a.*',
          exp=pd.DataFrame(
              {
                  'name': ['cc', 'dd'],
                  'run': [2, 3],
                  'user': ['max', 'jo']
              },
              index=[2, 3])),
      dict(
          testcase_name='user_is_not_sam',
          column_name='user',
          regex_str='sam',
          exp=pd.DataFrame(
              {
                  'name': ['ab', 'cc', 'dd'],
                  'run': [1, 2, 3],
                  'user': ['max', 'max', 'jo']
              },
              index=[1, 2, 3])),
  )
  def test_drop_rows_by_regex(self, column_name, regex_str, exp):
    res = pandas_utils.drop_rows_by_regex(
        self.df1, column_name=column_name, regex_str=regex_str)
    pd.testing.assert_frame_equal(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='keep_name_column',
          columns_to_keep=['name'],
          exp=pd.DataFrame({
              'name': ['aa', 'ab', 'cc'],
          })),
      dict(
          testcase_name='keep_name_and_test2_columns',
          columns_to_keep=['run', 'test2'],
          exp=pd.DataFrame({
              'run': [1, 1, 2],
              'test2': [10, 20, 30]
          })),
  )
  def test_filter_columns(self, columns_to_keep, exp):
    res = pandas_utils.filter_columns(self.df2, columns_to_keep=columns_to_keep)
    pd.testing.assert_frame_equal(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='keep_columns_with_test_in_name',
          column_regex='test.*',
          exp=pd.DataFrame({
              'test1': ['x', 'y', 'z'],
              'test2': [10, 20, 30]
          })),
      dict(
          testcase_name='keep_columns_with_run_in_name',
          column_regex='run',
          exp=pd.DataFrame({
              'run': [1, 1, 2],
          })),
  )
  def test_filter_columns_by_regex(self, column_regex, exp):
    res = pandas_utils.filter_columns_by_regex(
        self.df2, column_regex=column_regex)
    pd.testing.assert_frame_equal(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='drop_columns_with_test_in_name',
          columns_to_drop=['run'],
          exp=pd.DataFrame({
              'name': ['aa', 'ab', 'cc'],
              'test1': ['x', 'y', 'z'],
              'test2': [10, 20, 30]
          })),
      dict(
          testcase_name='drop_columns_with_run_in_name',
          columns_to_drop=['run', 'test2'],
          exp=pd.DataFrame({
              'name': ['aa', 'ab', 'cc'],
              'test1': ['x', 'y', 'z'],
          })),
  )
  def test_drop_columns(self, columns_to_drop, exp):
    res = pandas_utils.drop_columns(self.df2, columns_to_drop=columns_to_drop)
    pd.testing.assert_frame_equal(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='drop_columns_with_test_in_name',
          column_regex='test.*',
          exp=pd.DataFrame({
              'name': ['aa', 'ab', 'cc'],
              'run': [1, 1, 2],
          })),
      dict(
          testcase_name='drop_columns_with_run_in_name',
          column_regex='run',
          exp=pd.DataFrame({
              'name': ['aa', 'ab', 'cc'],
              'test1': ['x', 'y', 'z'],
              'test2': [10, 20, 30]
          })),
  )
  def test_drop_columns_by_regex(self, column_regex, exp):
    res = pandas_utils.drop_columns_by_regex(
        self.df2, column_regex=column_regex)
    pd.testing.assert_frame_equal(res, exp)

  def test_group_by_with_aggregation(self):
    res = pandas_utils.group_by_with_aggregation(
        self.df3, by='name', agg_column_names=['len', 'dis'])
    self.assertEqual(res.shape, (2, 5))
    self.assertSequenceEqual(list(res.len['mean']), [4, 10])
    self.assertSequenceEqual(list(res.len['std']), [2, 1])
    self.assertSequenceEqual(list(res.dis['mean']), [1, 6])
    self.assertSequenceEqual(list(res.dis['std']), [0, 1])

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_string',
          column_name='name',
          pattern='a',
          repl='d',
          exp=pd.DataFrame({
              'name': ['dd', 'db', 'cc', 'dd'],
              'run': [1, 1, 2, 3],
              'user': ['sam', 'max', 'max', 'jo']
          })),
      dict(
          testcase_name='regex_with_groups',
          column_name='user',
          pattern=r'(sa)',
          repl=r'\g<1>ttt',
          exp=pd.DataFrame({
              'name': ['aa', 'ab', 'cc', 'dd'],
              'run': [1, 1, 2, 3],
              'user': ['satttm', 'max', 'max', 'jo']
          })),
      )
  def test_rename_values_in_column(self, column_name, pattern, repl, exp):
    res = pandas_utils.rename_values_in_column(
        self.df1, column_name=column_name, pattern=pattern, repl=repl)
    pd.testing.assert_frame_equal(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_string',
          pattern='user',
          repl='owner',
          exp=pd.DataFrame({
              'name': ['aa', 'ab', 'cc', 'dd'],
              'run': [1, 1, 2, 3],
              'owner': ['sam', 'max', 'max', 'jo']
          })),
      dict(
          testcase_name='regex_with_groups',
          pattern=r'(a)m',
          repl=r'x\g<1>',
          exp=pd.DataFrame({
              'nxae': ['aa', 'ab', 'cc', 'dd'],
              'run': [1, 1, 2, 3],
              'user': ['sam', 'max', 'max', 'jo']
          })),
  )
  def test_rename_column_headers(self, pattern, repl, exp):
    res = pandas_utils.rename_column_headers(
        self.df1, pattern=pattern, repl=repl)
    pd.testing.assert_frame_equal(res, exp)

  def test_apply_filter_drop_rename_operations(self):
    res = pandas_utils.apply_filter_drop_rename_operations(
        self.df1,
        row_filter_args=[('run', [1, 2])],
        row_regex_filter_args=[('user', r'ma.*')],
        rename_row_value_args=[('name', r'a(.*)', r'\g<1>')],
        drop_columns_by_regex_args=['user'],
        rename_column_name_args=[(r'(a)m', r'x\g<1>')],
        sort_by_args=([('run', False)]),
    )
    exp = pd.DataFrame({
        'nxae': ['cc', 'b'],
        'run': [2, 1],
    })
    pd.testing.assert_frame_equal(res, exp)


if __name__ == '__main__':
  absltest.main()
