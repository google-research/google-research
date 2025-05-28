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

"""Common utilities for interfacing with BigQuery."""

import dataclasses
import enum
import re
from typing import Mapping, Optional, Sequence


@dataclasses.dataclass
class BQTablePathParts:
  """Data object with the parts of a full BigQuery table.

  Attributes:
    project_id: The project id for the project that the BigQuery table is in.
    bq_dataset_name: The name of the dataset where the table resides.
    bq_table_name: The table's name.
  """

  project_id: str
  bq_dataset_name: str
  bq_table_name: str

  @property
  def full_table_id(self):
    """The full path 'project.dataset.table' of the BigQuery table."""
    return f"{self.project_id}.{self.bq_dataset_name}.{self.bq_table_name}"

  @property
  def escaped_table_id(self):
    """The full path of the BigQuery table escaped with `."""
    return f"`{self.full_table_id}`"

  @classmethod
  def from_full_path(cls, full_big_query_path):
    """Parses a full BigQuery path into components (project, dataset, table).

    Args:
      full_big_query_path: The full path (e.g.
        project_id.dataset_name.table_name) to a BigQuery table.

    Returns:
      A BQTablePathParts object.
    """
    project_prefix_pattern = re.compile(
        r"`?(?P<project_id>[^.`]+)\."
        r"(?P<bq_dataset_name>[^.`]+)"
        r"(?:$|\.(?P<bq_table_name>[^.`]+)`?$)",
        re.VERBOSE,
    )

    path_match = project_prefix_pattern.match(full_big_query_path)
    if not path_match:
      raise ValueError(
          f"The BigQuery table path {full_big_query_path} did not have the "
          "expected format of project_id.dataset_name.table_name"
      )

    output_dict = path_match.groupdict()
    for name, match in output_dict.items():
      if not match:
        raise ValueError(
            f"The {name} was not found in the BQ path {full_big_query_path}"
        )

    # Note that the regex match names must match the attribute names.
    return cls(
        project_id=output_dict["project_id"],
        bq_dataset_name=output_dict["bq_dataset_name"],
        bq_table_name=output_dict["bq_table_name"],
    )


def where_statement_from_clauses(
    where_clauses, conjunction = "AND"
):
  """Constructs an optional where statement from list of conditions.

  Args:
    where_clauses: A sequence of strings each of which is a valid clause for the
      table being queried.
    conjunction: The conjunction used to join the clauses. Does not need to
      include spaces as they will be added later.

  Returns:
    Either an empty string if there were no clauses or a where clause with
    whitespace on both sides.
  """
  if not where_clauses:
    return ""

  join_str = f") {conjunction} ("
  combined_clauses = f"({join_str.join(where_clauses)})"
  return f"\nWHERE {combined_clauses} "


@enum.unique
class SplitColumnValues(enum.Enum):
  TRAIN = enum.auto()
  VALIDATE = enum.auto()
  TEST = enum.auto()


def create_split_column_conditions(
    split_column,
):
  return {
      mode: (f"{split_column} = '{mode.name}'",) if split_column else ()
      for mode in SplitColumnValues
  }
