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

"""Tests for helper."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from covid_vhh_design import helper


class HelperTest(parameterized.TestCase):

  def test_max_impute_inf_with_df(self):
    df = pd.DataFrame(dict(
        a=[0, 1, 2, 3],
        b=[0.1, np.inf, 0.2, np.inf],
        c=[np.inf] * 4,
    ))
    expected = pd.DataFrame(dict(
        a=[0, 1, 2, 3],
        b=[0.1, 0.2, 0.2, 0.2],
        c=[np.nan] * 4,
    ))
    actual = helper.max_impute_inf(df)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

  @parameterized.parameters(
      ('a', [0, 1, 2]),
      ('b', [0, 2]),
  )
  def test_drop_inf_df(self, col_name, expected_idx):
    df = pd.DataFrame({
        'a': [1, 2, np.nan],
        'b': [None, np.inf, 3],
    })
    actual_remaining_idx = helper.drop_inf_df(df, column=col_name).index
    self.assertSequenceEqual(list(actual_remaining_idx), expected_idx)

  @parameterized.parameters(
      (('s1', 's2', 's3'), {'s1': 'x1', 's3': 'x3'}, ('x1', 's2', 'x3')),
      (('s1', 's2', 's3'), (('s2', 'x2'), ('s4', 'x4')), ('s1', 'x2', 's3')),
  )
  def test_map_values(self, values, mapping, expected):
    actual = helper.map_values(pd.Series(values), mapping)
    self.assertSequenceEqual(list(actual), expected)

  def test_map_columns(self):
    df = pd.DataFrame(dict(
        a=[1, 2, 3],
        b=['x', 'y', 'z'],
    ))
    column_mappings = dict(
        a={
            1: 10,
            3: 30
        },
        b={
            'x': 'xx',
            'y': 'yy',
            'z': 'zz'
        },
        z={'x'},
    )
    expected_df = pd.DataFrame(dict(
        a=[10, 2, 30],
        b=['xx', 'yy', 'zz'],
    ))
    actual_df = helper.map_columns(df, ignore_missing=True, **column_mappings)
    pd.testing.assert_frame_equal(actual_df, expected_df)

  def test_safe_merge_passed(self):
    df1 = pd.DataFrame(dict(
        key=[1, 2, 3],
        value=[0.1, 0.2, 0.3]))
    df2 = pd.DataFrame(dict(
        key=[1, 2, 3, 4, 5],
        value2=[-3, -2, -1, 0, 1]))
    expected_df = pd.DataFrame(dict(
        key=[1, 2, 3],
        value=[0.1, 0.2, 0.3],
        value2=[-3, -2, -1]))
    actual_df = helper.safe_merge(df1, df2)
    pd.testing.assert_frame_equal(actual_df, expected_df)

  def test_safe_merge_fails(self):
    df1 = pd.DataFrame(dict(
        key=[1, 2, 3],
        value=[0.1, 0.2, 0.3]))
    df2 = pd.DataFrame(dict(
        key=[1, 3, 4, 5],
        value2=[-3, -1, 0, 1]))
    with self.assertRaisesRegex(RuntimeError, 'Merge changed the length'):
      _ = helper.safe_merge(df1, df2)


if __name__ == '__main__':
  absltest.main()
