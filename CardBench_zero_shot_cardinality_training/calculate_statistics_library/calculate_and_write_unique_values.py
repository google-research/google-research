# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Collect column unique values."""

from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers

INTERNAL = False

build_partitioned_predicate = helpers.build_partitioned_predicate
get_partitioning_info = helpers.get_partitioning_info
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
DBType = database_connector.DBType
TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_INFO_TABLE = configuration.COLUMNS_INFO_TABLE
COLUMNS_STATS_TABLE = configuration.COLUMNS_STATS_TABLE
BQ_INFO_SCHEMA_COLUMNS = configuration.BQ_INFO_SCHEMA_COLUMNS
TYPES_TO_TABLES = configuration.TYPES_TO_TABLES


def calculate_and_write_unique_values_internal(
    projectname,
    column_list,
    extra_stats_table,
    dbs,
):
  """Get the unique values for each column in the dataset."""
  count_total = 0
  for datasetname, tablename, columnname in column_list:
    print(
        " calculating unique values for: ", datasetname, tablename, columnname
    )
    is_partitioned, partition_column, partition_column_type, _ = (
        get_partitioning_info(
            projectname,
            datasetname,
            tablename,
            dbs["metadata_dbclient"],
            dbs["metadata_dbtype"],
        )
    )
    partitioning_predicate = build_partitioned_predicate(
        is_partitioned, partition_column, partition_column_type
    )
    queryjob = []
    query = (
        f"SELECT distinct `{columnname}` as val FROM"
        f" `{projectname}.{datasetname}.{tablename}` WHERE"
        f" {partitioning_predicate}"
    )
    try:
      queryjob = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)
    unique_vals = []
    for row in queryjob:
      unique_vals.append(row["val"])

    query = (
        f"UPDATE `{extra_stats_table}` SET uniq_vals = (( SELECT"
        f" ARRAY_AGG(distinct  `{columnname}`) as val  FROM"
        f" `{projectname}.{datasetname}.{tablename}` WHERE"
        f" {partitioning_predicate}))"
    )

    query = (
        f"{query} WHERE project_name='{projectname}' AND"
        f" dataset_name='{datasetname}' AND table_name='{tablename}' AND"
        f" column_name='{columnname}'"
    )
    count_total += 1
    try:
      _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)
      count_total -= 1
  return count_total


def calculate_and_write_unique_values(
    projectname,
    datasetname,
    dbs,
    categorical_threshold = 10000,
):
  """Calculate and writes columnn unique values to configuration.COLUMNS_STATS_TABLE."""
  columns_per_type = {}
  for sqltype in ["TIME", "TIMESTAMP", "DATE", "DATETIME", "STRING"]:
    type_table = TYPES_TO_TABLES[sqltype]
    columns_per_type[sqltype] = {
        "table": type_table,
        "column_list": [],
    }
    query = (
        "SELECT  ci.dataset_name, ci.table_name, ci.column_name,"
        f" ci.column_type FROM `{COLUMNS_STATS_TABLE}` as ci  WHERE"
        f" ci.dataset_name = '{datasetname}' AND ci.project_name ="
        f" '{projectname}' AND ci.column_type = '{sqltype}' AND"
        f" ci.num_unique	 <= {str(categorical_threshold)} and ci.num_unique >"
        f" 0 and ci.null_frac <> 1 AND EXISTS (select * from `{type_table}`"
        " as est where est.column_name = ci.column_name and  est.table_name"
        " = ci.table_name and est.dataset_name = ci.dataset_name AND"
        " est.project_name = ci.project_name and array_length(uniq_vals)"
        " = 0)"
    )

    queryjob = run_query(
        dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
    )
    for row in queryjob:
      columns_per_type[sqltype]["column_list"].append([
          row["dataset_name"],
          row["table_name"],
          row["column_name"],
      ])

  rows_written_to_extra_stats_table = 0
  if not INTERNAL:
    for column_type in columns_per_type:
      rows_written_to_extra_stats_table += (
          calculate_and_write_unique_values_internal(
              projectname=projectname,
              column_list=columns_per_type[column_type]["column_list"],
              extra_stats_table=columns_per_type[column_type]["table"],
              dbs=dbs,
          )
      )

  print("Unique Values inserted:", rows_written_to_extra_stats_table)
