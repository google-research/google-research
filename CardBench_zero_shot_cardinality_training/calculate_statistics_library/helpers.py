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

"""Helper functions for statistics generation."""

from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector


run_query = database_connector.run_query
DBType = database_connector.DBType

TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE


def remove_anything_after_first_parenthesis(column_type):
  return column_type.split("(")[0]


def printif(flag, msg):
  if flag:
    print(msg)


table_info_cache = {}


def copy_table(
    projectname,
    datasetname,
    tablename,
    target_row_number,
    partition_pred,
    sampled_tables_dataset_name,
    dbs,
):
  """Copy a sql table and apply a limit."""
  copy_query = f"""
    CREATE OR REPLACE TABLE `{sampled_tables_dataset_name}.{projectname}_{datasetname}_{tablename}` AS(
      SELECT *
      FROM `{projectname}.{datasetname}.{tablename}`
      WHERE {partition_pred}
      LIMIT {target_row_number}
    )
  """
  try:
    _, _ = run_query(dbs["data_dbtype"], copy_query, dbs["data_dbclient"])
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(e)
    print(">>>>>>>>>>>> ERROR IN QUERY :\n" + copy_query)

  return


def get_partitioning_info(
    projectname,
    datasetname,
    tablename,
    dbclient,
    dbtype,
):
  """Get partitioning infor stored in BQ tables_info_table."""
  requested_key = f"{projectname}.{datasetname}.{tablename}"
  if requested_key not in table_info_cache:
    query = (
        "SELECT project_name, dataset_name, table_name, row_count, "
        "is_partitioned, partition_column, partition_column_type "
        f"FROM `{TABLES_INFO_TABLE}` where dataset_name = '{datasetname}'"
    )
    queryjob, _ = run_query(dbtype, query, dbclient)
    for rowres in queryjob:
      key = f"{rowres['project_name']}.{rowres['dataset_name']}.{rowres['table_name']}"
      table_info_cache[key] = {
          "project_name": projectname,
          "dataset_name": datasetname,
          "table_name": tablename,
          "row_count": rowres["row_count"],
          "is_partitioned": rowres["is_partitioned"],
          "partition_column": rowres["partition_column"],
          "partition_column_type": rowres["partition_column_type"],
      }
  return (
      table_info_cache[requested_key]["is_partitioned"],
      table_info_cache[requested_key]["partition_column"],
      table_info_cache[requested_key]["partition_column_type"],
      table_info_cache[requested_key]["row_count"],
  )


def build_partitioned_predicate(
    is_partitioned,
    table,
    part_col,
    part_col_type,
    use_table_name = False,
):
  """Build predicate to bypass BQ partitioning restriction."""
  # When a table is partitioned BQ does allow querying without a predicate on
  # the partition column. This function builds a predicate that is always true.
  if not is_partitioned:
    return "1 = 1"
  elif part_col_type == "DATE":
    if use_table_name:
      pred = f"{table}.{part_col} > '1000-01-01' "
    else:
      pred = f"{part_col} > '1000-01-01' "
  elif part_col_type == "TIMESTAMP":
    if use_table_name:
      pred = f"{table}.{part_col} > '1000-01-01 00:00:00' "
    else:
      pred = f"{part_col} > '1000-01-01 00:00:00' "
  elif part_col_type == "DATETIME":
    if use_table_name:
      pred = f"{table}.{part_col} > '1000-01-01 00:00:00' "
    else:
      pred = f"{part_col} > '1000-01-01 00:00:00' "
  elif part_col_type in ["INT32", "INT64", "UINT32", "UINT64"]:
    if use_table_name:
      pred = f"{table}.{part_col}  <= cast('+inf' as float64) "
    else:
      pred = f"{part_col}  <= cast('+inf' as float64) "
  else:
    raise ValueError(
        "wrong coltype " + str(part_col) + " " + str(part_col_type)
    )

  return pred
