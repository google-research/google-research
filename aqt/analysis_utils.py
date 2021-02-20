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

"""Collection of commonly used convenience functions for experiment analysis."""
from typing import Any, Callable, Dict, List

import dataclasses
import pandas as pd
import tree

from aqt import report_utils


def flatten_with_joined_string_paths(
    dictionary):
  """Flattens nested dict to single level dict with joined paths as keys."""
  flattened = tree.flatten_with_path(structure=dictionary)
  flattened_dict = {}
  # join path tuples to single string
  for path_tuple, val in flattened:
    # convert all path elements to strings
    path = [str(s) for s in path_tuple]
    path = '/'.join(path)
    flattened_dict[path] = val
  return flattened_dict


def convert_report_to_flat_dict_default(
    report):
  """Selects subset of report and flattens it to a single level dict.

  This function selects all information except what's stored under the fields
  `report_query_args` and `metadata_corp`.

  This function serves as an example for how to parse an ExperimentReport
  into a dataframe row by flattening it to a row_dict, with keys corresponding
  to dataframe columns.

  The ExperimentReport dataclass likely contains more information then you need
  for your analysis, so you can write your own function to pick and choose the
  information you want. You can refer to report_utils.ExperimentReport for
  documentation of all available fields.

  You can pass your custom function into convert_reports_to_dataframe().

  Args:
    report: An instance of ExperimentReport.

  Returns:
    A flattened dict representing a dataframe row.
  """

  row_dict = {}

  # Add smoothed metrics if present
  if report.metrics is not None:
    flattened_metrics = dict(flatten_with_joined_string_paths(report.metrics))
    # merge dicts
    row_dict = {**row_dict, **flattened_metrics}

  # Add unsmoothed metrics if present
  if report.unsmoothed_metrics is not None:
    flattened_unsmoothed_metrics = dict(
        flatten_with_joined_string_paths(report.unsmoothed_metrics))
    flattened_unsmoothed_metrics = {
        f'unsmoothed/{k}': v for k, v in flattened_unsmoothed_metrics.items()
    }
    # merge dicts
    row_dict = {**row_dict, **flattened_unsmoothed_metrics}


  # Ignore following fields because they have already been added, or we chose
  # not to include them.
  report_fields_to_ignore = {
      'metrics',
      'unsmoothed_metrics',
  }
  # Add other report fields.
  for field in dataclasses.fields(report):
    if field.name not in report_fields_to_ignore:
      row_dict[field.name] = getattr(report, field.name)

  return row_dict


def convert_reports_to_dataframe(
    reports,
    convert_report_to_flat_dict_fn = convert_report_to_flat_dict_default
):
  """Converts a list of ExperimentReport instances to a pandas dataframe.

  Args:
    reports: List of ExperimentReport instances. Each instance will correspond
      to a row in the dataframe.
    convert_report_to_flat_dict_fn: Function to use for converting an
      ExperimentReport to a flat dict, which will then be read in as a pandas
      dataframe row. The keys in the flat dict are interpreted as column names,
      the values as entries for that row. Please refer to
      `convert_report_to_flat_dict_default()` as an example.

  Returns:
    A pandas dataframe populated with information extracted from the reports.
  """

  rows = [convert_report_to_flat_dict_fn(rep) for rep in reports]
  return pd.DataFrame(rows)


def clickable_link(link, display_str = 'link'):
  """Converts a link string into a clickable link with html tag.

  WARNING: This function is not safe to use for untrusted inputs since the
  generated HTML is not sanitized.

  Usage:
    df.style.format(clickable_link, subset=['col_name'])

  Args:
    link: A link string without formatting.
    display_str: What text the link should display.

  Returns:
    HTML-formatted link.

  """
  return f'<a href="{link}">{display_str}</a>'


