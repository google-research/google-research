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

"""Provides helper functions."""

import collections
from collections.abc import Mapping, Sequence
import os
from typing import Any

import numpy as np
import pandas as pd
from tensorflow.io import gfile

DataFrameOrSeries = pd.DataFrame | pd.Series


class Bunch(collections.OrderedDict):
  """An `OrderedDict` that allows accessed an item x by `bunch.x`.

  Example:
    bunch = Bunch(a=1, b=20)
    assert bunch.a == bunch['a']
    bunch.c = 2
    assert 'c' in bunch
  """

  def __setattr__(self, key, value):
    self[key] = value

  def __dir__(self):
    return self.keys()

  def __getattr__(self, key):
    try:
      return self[key]
    except KeyError as e:
      raise AttributeError(key) from e

  def __setstate__(self, state):
    pass


def to_list(values: Any, keep_none: bool = False) -> list[Any]:
  """Converts `values` of any type to a `list`."""
  if (
      isinstance(values, collections.abc.Iterable)
      and not isinstance(values, str)
      and not isinstance(values, dict)
  ):
    return list(values)
  elif values is None and not keep_none:
    return []
  else:
    return [values]


def ordered_intersection(
    ordered_values: Sequence[Any], values: Sequence[Any]
) -> list[Any]:
  """Returns the values in ordered_values that exist in values preserving order.

  Similar to set(ordered_values) & set(values), but preserves the order of
  values in `ordered_values`.

  Example:
    ordered_values = [3, 1, 2]
    values = {4, 2, 1}
    ordered_intersection([3, 2, 1], {1, 4, 2}) == [2, 1])

  Args:
    ordered_values: A sequence of values.
    values: A sequence of values.

  Returns:
    Values in `ordered_values` that exist in `values`.
  """
  return [value for value in ordered_values if value in set(values)]


def tprint(value: Any, *args, **kwargs) -> None:
  """Calls `print` with `flush=True`."""
  if args:
    value %= args
  print(value, flush=True, **kwargs)


def unique(values: Sequence[Any], sort: bool = False) -> Any:
  """Returns a list of unique values.

  Args:
    values: An iterable of values.
    sort: Whether to sort the resulting values. If `None`, values will be sorted
      only if they do not contain NaNs.

  Returns:
    A list of (sorted) unique values.
  """
  values = list(set(values))
  if sort is None:
    sort = not pd.isna(values).any()
  return sorted(values) if sort else values


def get_unique_value(values: Sequence[Any]) -> Any:
  """Returns the unique value of `values`.

  Expects `values` to be an iterable of a single unique values, which it
  returns. E.g. returns 'a' if `values == ['a', 'a', 'a']`.

  Args:
    values: An iterable.

  Raises:
    ValueError: If the number of unique values of `values` is not one.

  Returns:
    A scalar value.
  """
  unique_values = unique(values)
  if len(unique_values) != 1:
    raise ValueError('More than one value!: %s' % unique_values[:50])
  return unique_values[0]


def infer_compression(path: str) -> str | None:
  """Infers the Pandas compression given a file path (see docu pd.to_csv)."""
  ext = os.path.splitext(path)[1][1:]
  if ext == 'gz':
    return 'gzip'
  elif ext in ['bz2', 'zip', 'xy']:
    return ext
  else:
    return None


def read_csv(
    path: str, compression: str | None = 'infer', **kwargs
) -> pd.DataFrame:
  """Reads a `pd.DataFrame` from a (compressed) CSV file.

  Args:
    path: The file path (gfile supported).
    compression: The compression level. If 'infer', will be derived from `path`.
      E.g. 'foo.csv.gz' will be mapped to 'gzip'. See the pandas docstring of
      `pd.read_csv` for more details.
    **kwargs: Named arguments passed to `pd.read_csv`.

  Returns:
    A `pd.DataFrame`.
  """
  if compression == 'infer':
    compression = infer_compression(path)

  mode = 'rb' if compression else 'r'
  with gfile.GFile(path, mode) as f:
    return pd.read_csv(f, compression=compression, **kwargs)


def impute_inf(
    df_or_series: DataFrameOrSeries,
    imputation_fn: ...,
    fill_value: float = np.nan,
) -> DataFrameOrSeries:
  """Replaces Inf values of a Series or DataFrame.

  Args:
    df_or_series: A Series or DataFrame.
    imputation_fn: A function such as `np.max` or `np.mean` that is applied to
      non-Inf values and returns a single value by which Inf values are
      replaced.
    fill_value: The value that will be used to impute Infs if all values of a
      Series are Inf.

  Returns:
    df_or_series without Inf values.
  """
  if isinstance(df_or_series, pd.DataFrame):
    return df_or_series.apply(
        impute_inf, imputation_fn=imputation_fn, fill_value=fill_value
    )
  else:
    series = df_or_series.copy()
    idx = np.isinf(series)
    series.loc[idx] = (
        fill_value if idx.all() else imputation_fn(series.loc[~idx])
    )
    return series


def max_impute_inf(df_or_series: DataFrameOrSeries) -> DataFrameOrSeries:
  """Max imputes Inf values of a Series or DataFrame."""
  return impute_inf(df_or_series, np.max)


def drop_inf_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
  """Drop rows with `inf` values in column."""
  return df.loc[~np.isinf(df.loc[:, column])]


def map_values(
    df_or_series: DataFrameOrSeries,
    mapping: Mapping[Any, Any],
    ignore_unmapped: bool = True,
    **kwargs,
) -> DataFrameOrSeries:
  """Maps `values` according the `mapping` dict or tuple."""
  mapping = dict(mapping)
  if ignore_unmapped:
    return df_or_series.map(lambda x: mapping.get(x, x), **kwargs)
  else:
    return df_or_series.map(mapping, **kwargs)


def map_columns(
    df: pd.DataFrame, ignore_missing: bool = False, **column_mappings
) -> pd.DataFrame:
  """Replaces values of one or more columns.

  Args:
    df: A DataFrame.
    ignore_missing: Whether to ignore column in `column_mappings` that do not
      exist in `df`.
    **column_mappings: A dict mapping columns to a dict mapping values to
      replacement values.

  Returns:
    A copy of `df` with replaced column values.
  """
  df = df.copy()
  for column, mapping in column_mappings.items():
    if column in df.columns:
      df[column] = map_values(df[column], mapping)
    elif not ignore_missing:
      raise ValueError(
          'Column "%s" does not exist! Existing columns: %s'
          % ((column, ' '.join(sorted(df.columns))))
      )
  return df


def safe_merge(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
  """Returns `df.merge(*args, **kargs)` and fails if the length changes."""
  merged = df.merge(*args, **kwargs)
  if len(merged) != len(df):
    raise RuntimeError(
        'Merge changed the length from %d to %d!' % (len(df), len(merged))
    )
  return merged
