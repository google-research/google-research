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

"""Pandas dataframe wrapper functions for frequently used operations."""
import functools
import re
from typing import Any, List, Optional, Sequence, Tuple, Union

import pandas as pd

Regex = str
NewStr = str
ColumnName = str
ColumnValue = Any
OrderAscending = bool
SingleColumnFilter = Tuple[ColumnName, List[ColumnValue]]
SingleColumnRegex = Tuple[ColumnName, Regex]
SingleColumnRegexReplace = Tuple[ColumnName, Regex, Regex]
ColumnNameRegexReplace = Tuple[Regex, Regex]
SortBy = Sequence[Tuple[ColumnName, OrderAscending]]


def select_rows_by_column_values(df, column_name,
                                 values):
  """Selects rows where column values are in specified list.

  Args:
    df: A pandas dataframe.
    column_name: Name of the column to filter the rows by.
    values: Single value or list of values to filter by.

  Returns:
    Filtered dataframe.

  """
  if not isinstance(values, List):
    values = [values]
  return df[df[column_name].isin(values)]


def select_rows_by_regex(df, column_name,
                         regex_str):
  """Returns pandas dataframe rows matching regex in a specific column.

  Returns a dataframe with a subset of rows where the value in the column
  contains the regex.

  Args:
    df: A pandas dataframe.
    column_name: Name of the column to select the rows by. Has to contain string
      values only.
    regex_str: a string or regex string to match column values against.

  Returns:
    Filtered dataframe.

  """
  return df[df[column_name].str.contains(regex_str)]


def drop_rows_by_column_values(df, column_name,
                               values):
  """Drop rows where their column values are specified list.

  Args:
    df: A pandas dataframe.
    column_name: Name of the column to filter the rows by.
    values: List of values to filter by.

  Returns:
    Filtered dataframe.

  """
  return df[~df[column_name].isin(values)]


def drop_rows_by_regex(df, column_name,
                       regex_str):
  """Returns pandas dataframe rows matching regex in column dropped.

  Args:
    df: A pandas dataframe.
    column_name: Name of the column to filter the rows by. Has to contain string
      values only.
    regex_str: a string or regex string to match column values against.

  Returns:
    Filtered dataframe.

  """
  return df[~df[column_name].str.contains(regex_str)]


def filter_columns(df, columns_to_keep=List[str]):
  """Returns dataframe with subset of columns."""
  columns_to_keep = [c for c in columns_to_keep if c in list(df.columns)]
  return df[columns_to_keep]


def filter_columns_by_regex(df,
                            column_regex):
  """Returns dataframe with subset of columns matching regex."""
  return df.filter(regex=column_regex)


def drop_columns(df, columns_to_drop=List[str]):
  """Returns dataframe with specified columns dropped."""
  columns_to_drop = [c for c in columns_to_drop if c in list(df.columns)]
  return df.drop(columns=columns_to_drop)


def drop_columns_by_regex(df, column_regex):
  """Returns dataframe with columns matching the regex dropped."""
  return df[df.columns.drop(list(df.filter(regex=column_regex)))]


def group_by_with_aggregation(
    df,
    by,
    agg_column_names,
    aggregators = ('mean', 'std')
):
  """Group by with aggregation.

  Args:
    df: Dataframe to perform group by on.
    by: Which column to group by.
    agg_column_names: Which columns to aggregate over. Have to be columns
      containing numeric values.
    aggregators: Which aggregation functions to apply.

  Returns:
    Multi-level dataframe where each row corresponds to a group and contains
    aggregated values for specified agg_column_names and aggregators.

  """
  agg_dict = {}
  for col in agg_column_names:
    agg_dict[col] = aggregators

  return df.groupby(by, as_index=False).agg(agg_dict)


def rename_values_in_column(df, column_name, pattern,
                            repl):
  """Renames row values of dataframe by regex replacement.

  Args:
    df: A pandas dataframe.
    column_name: In which column to rename values. Has to be string column.
    pattern: Regex pattern to replace.
    repl: Replacement string to insert where old string was.

  Returns:
    Dataframe with renamed columns.
  """
  df[column_name] = df[column_name].apply(
      lambda x: re.sub(pattern=pattern, repl=repl, string=x))
  return df


def rename_column_headers(df, pattern,
                          repl):
  """Renames column headers of dataframe by regex replacement.

  Args:
    df: A pandas dataframe.
    pattern: Regex pattern to replace.
    repl: Replacement string to insert where old string was.

  Returns:
    Dataframe with renamed columns.
  """

  def _regex_replace_fn(x, pattern, repl):
    return re.sub(pattern=pattern, repl=repl, string=x)

  return df.rename(
      columns=functools.partial(_regex_replace_fn, pattern=pattern, repl=repl))


def apply_filter_drop_rename_operations(
    df,
    row_filter_args = None,
    row_regex_filter_args = None,
    rename_row_value_args = None,
    drop_columns_by_regex_args = None,
    rename_column_name_args = None,
    sort_by_args = None):
  r"""Function to apply series of filter, drop and rename operations.

  Tthe user can specify how to filter the dataframe rows, which
  columns to drop, and how to rename columns. For more customized pandas
  operations, please use report_utils.load_all_reports() to load all reports
  into a dataframe.

  Usage example:
    df = pandas_utils.apply_filters_and_renaming(
    df,
    row_filter_args=
      [('xid', [18925472, 18925219, 18925394])],
    row_regex_filter_args=
      [('model_dir', '.*4bit.*')],
    rename_row_value_args=
      [('experiment_name', r'leaderboard_full_model_(.*)_wanglisa-.*',
      r'\g<1>')],
    drop_columns_by_regex_args=['.*unsmoothed.*'],
    rename_column_name_args=[('_translate--de-en:test', '')],
    sort_by_args=([('eval/loss', True), ('eval/bleu', False)]),
  )

  Args:
    df: Dataframe to run filter and rename operations on.
    row_filter_args: List of row filters to be applied sequentially. Each row
      filter is a tuple of (column_name, filter_list). Each row filter is
      equivalent to a SQL statement of the form SELECT * WHERE column_name IN
      filter_list.
    row_regex_filter_args: Each row filter is a tuple of (column_name, regex).
      Will select the rows where column values match the regex.
    rename_row_value_args: Arguments for replacing values in columns. List of
      tuples (column_name, old_string, new_string).
    drop_columns_by_regex_args: Drop columns matching the provided regex.
    rename_column_name_args: Arguments to rename column headers. List of tuples
      (old_string (can be regex), new_string).
    sort_by_args: Arguments for df.sort_values(), List of tuples of (by,
      ascending), where `by` is a column sort by, and `ascending` is a bool
      indicating whether the column should be sorted in ascending order. First
      tuple will be used as primary sorting axis, etc. See documentation
      pandas.DataFrame.sort_values() for more details.

  Returns:
    A pandas dataframe.

  """
  for column_name, value in row_filter_args or []:
    df = select_rows_by_column_values(df, column_name=column_name, values=value)

  for column_name, regex in row_regex_filter_args or []:
    df = select_rows_by_regex(df, column_name=column_name, regex_str=regex)

  for column_name, pattern, repl in rename_row_value_args or []:
    df = rename_values_in_column(
        df, column_name=column_name, pattern=pattern, repl=repl)

  for regex in drop_columns_by_regex_args or []:
    df = drop_columns_by_regex(df, column_regex=regex)

  for pattern, repl in rename_column_name_args or []:
    df = rename_column_headers(df, pattern=pattern, repl=repl)

  if sort_by_args is not None:
    by = [tup[0] for tup in sort_by_args]
    ascending = [tup[1] for tup in sort_by_args]
    df = df.sort_values(by=by, ascending=ascending)

  df = df.reset_index(drop=True)
  return df


def boxplot_with_group_by_sorted_by_median(df,
                                           column_to_plot,
                                           by):
  """Returns a box plot for dataframe column, after group by operation.

  Args:
    df: Dataframe to plot.
    column_to_plot: Which column to plot values of.
    by: Which column(s) to use to group by. Can be any valid input to
      pandas.DataFrame.groupby().

  Returns:
    A matplotlib boxplot of type matplotlib.axes._subplots.AxesSubplot.

  """
  grouped_df = pd.DataFrame(
      {group: values[column_to_plot] for group, values in df.groupby(by)})
  medians = grouped_df.median().sort_values()
  return grouped_df[medians.index].boxplot(rot=0, return_type='axes')
