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

"""Collect percet extra statistics."""

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


def calculate_and_store_percentiles_bq(
    projectname,
    datasetname,
    tablename,
    columnname,
    extra_stats_table,
    dbs,
):
  """Collects columnn percentiles to build histograms, Big Query version."""
  query = (
      f"UPDATE `{extra_stats_table}` SET percentiles = (select"
      f" approx_quantiles(`{columnname}`, 9) FROM"
      f" `{projectname}.{datasetname}.{tablename}` ) WHERE"
      f" project_name='{projectname}' AND dataset_name='{datasetname}' AND"
      f" table_name='{tablename}' AND column_name='{columnname}'"
  )

  success = True
  try:
    _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)
    success = False

  if success:
    return 1
  return 0


def calculate_and_write_percentiles_internal(
    projectname,
    datasetname,
    tablename,
    columnname,
    columntype,
    extra_stats_table,
    dbs,
):
  """Routes to the database specific function."""
  print(
      "   Calculate percentiles for ",
      datasetname,
      " ",
      tablename,
      " ",
      columnname,
      " ",
      columntype,
  )
  assert columntype in ["INT64", "FLOAT64", "NUMERIC"]

  if dbs["data_dbtype"] == DBType.BIGQUERY:
    return calculate_and_store_percentiles_bq(
        projectname,
        datasetname,
        tablename,
        columnname,
        extra_stats_table,
        dbs,
    )
  else:
    raise ValueError("dbtype not supported yet: " + str(dbs["data_dbtype"]))


def calculate_and_write_percentiles(
    projectname, datasetname, dbs
):
  """Collects columnn percentiles to build histograms."""
  tasks = []
  for column_type in ["INT64", "FLOAT64", "NUMERIC"]:
    type_table = TYPES_TO_TABLES[column_type]
    query = (
        "select ci.dataset_name, ci.table_name, ci.column_name,"
        f" ci.column_type FROM `{COLUMNS_STATS_TABLE}` as ci  WHERE"
        f" ci.dataset_name = '{datasetname}' AND ci.project_name ="
        f" '{projectname}' AND ci.column_type LIKE '{column_type}%' and"
        " ci.num_unique > 0 and ci.null_frac <> 1 AND EXISTS (select * from"
        f" `{type_table}` as est where est.column_name = ci.column_name and"
        " est.table_name = ci.table_name and est.dataset_name ="
        " ci.dataset_name AND est.project_name = ci.project_name and"
        " array_length(est.percentiles) = 0)"
    )
    queryjob = run_query(
        dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
    )
    for row in queryjob:
      tasks.append([
          projectname,
          row["dataset_name"],
          row["table_name"],
          row["column_name"],
          column_type,
          type_table,
      ])

  rows_written_to_database = 0
  if not INTERNAL:
    for x in tasks:
      rows_written_to_database += calculate_and_write_percentiles_internal(
          projectname=x[0],
          datasetname=x[1],
          tablename=x[2],
          columnname=x[3],
          columntype=x[4],
          extra_stats_table=x[5],
          dbs=dbs,
      )
  print("rows_written_to_database percentiles:", rows_written_to_database)
