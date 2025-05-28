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

"""Tests utilities for Proteinfer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
import numpy as np


def model_testdata_path():
  return os.path.join(
      absltest.get_default_test_srcdir(),
      os.path.dirname(__file__),
  )


def savedmodel_path():
  return os.path.join(model_testdata_path(), 'testdata/per_residue_saved_model')


def assert_dataframes_equal(abseil_testcase_instance,
                            actual,
                            expected,
                            sort_by_column=None,
                            nan_equals_nan=False):
  """Assert dataframes equal up to reordering of columns and rows.

  Supports non-indexable datatypes in fields, like `set` and np.ndarray.

  Args:
    abseil_testcase_instance: absltest.TestCase (or parameterized.TestCase).
      E.g. pass 'self' from within an absltest.TestCase.
    actual: pd.DataFrame.
    expected: pd.DataFrame.
    sort_by_column: optional string name of a column. This column must be
      sortable (e.g. an int, not an np.array).
    nan_equals_nan: bool. If true, then allow nan == nan.
  """
  abseil_testcase_instance.assertEqual(
      len(actual), len(expected),
      'Lengths were not equal: {}\nand\n{}'.format(actual, expected))

  potential_error_message = 'actual:\n{}\nexpected:\n{}'.format(
      actual, expected)

  abseil_testcase_instance.assertSetEqual(
      set(actual.columns), set(expected.columns), potential_error_message)

  if len(set(actual.columns)) == 0:  # pylint: disable=g-explicit-length-test
    # Both dataframes are empty.
    return

  # Sort rows of DFs in same way, based on just one of the columns.
  if sort_by_column:
    actual = actual.sort_values(by=sort_by_column)
    expected = expected.sort_values(by=sort_by_column)

  actual_records = actual.to_dict('records')
  expected_records = expected.to_dict('records')

  for actual_record, expected_record in zip(actual_records, expected_records):
    abseil_testcase_instance.assertCountEqual(actual_record.keys(),
                                              expected_record.keys(),
                                              potential_error_message)
    for col_name in actual_record.keys():
      actual_value = actual_record[col_name]
      expected_value = expected_record[col_name]
      if isinstance(actual_value, np.ndarray):
        np.testing.assert_allclose(
            actual_value, expected_value, err_msg=potential_error_message)
      elif isinstance(actual_value, float) and np.isnan(actual_value):
        if nan_equals_nan and np.isnan(actual_value) and np.isnan(
            expected_value):
          continue
        else:
          raise AssertionError(
              actual_value, expected_value,
              'Actual value is nan, and nan is not equal to anything. '
              '{} != {}. {}'.format(actual_value, expected_value,
                                    potential_error_message))
      else:
        abseil_testcase_instance.assertEqual(actual_value, expected_value,
                                             potential_error_message)
