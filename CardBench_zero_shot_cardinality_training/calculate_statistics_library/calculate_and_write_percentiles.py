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
get_sql_table_string = database_connector.get_sql_table_string


def calculate_and_store_percentiles_bq(
    projectname,
    datasetname,
    tablename,
    columnname,
    extra_stats_table,
    columntype,
    dbs,
):
  """Collects columnn percentiles to build histograms, Big Query version."""
  table_sql_string = get_sql_table_string(
      dbs["data_dbtype"], f"{projectname}.{datasetname}.{tablename}"
  )
  query = (
      f" select approx_quantiles(`{columnname}`, 9) as quantiles FROM"
      f" {table_sql_string}"
  )
  quantiles = None
  try:
    queryjob, _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
    quantiles = get_query_result_first_row(dbs["data_dbtype"], queryjob)[
        "quantiles"
    ]
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)

  quantiles = [str(x) for x in quantiles]
  if columntype in ["UINT32", "UINT64"]:
    quantiles = (
        "(SELECT ARRAY(select cast(_ as BIGNUMERIC) FROM UNNEST((select"
        f" {quantiles})) _))"
    )
  quantiles_str = ", ".join(quantiles)
  query = (
      f"UPDATE `{extra_stats_table}` SET percentiles = [{quantiles_str}] "
      f"WHERE project_name='{projectname}' AND dataset_name='{datasetname}' AND"
      f" table_name='{tablename}' AND column_name='{columnname}'"
  )

  success = True
  try:
    _, _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)
    success = False

  if success:
    return 1
  return 0


def types_to_collect_percentiles(columntype):
  return (
      columntype == "INT64"
      or columntype == "INT32"
      or columntype == "UINT32"
      or columntype == "UINT64"
      or columntype == "FLOAT64"
      or columntype == "DOUBLE"
      or columntype == "NUMERIC"
      or columntype == "DECIMAL"
  )


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
  assert types_to_collect_percentiles(columntype)

  if dbs["data_dbtype"] == DBType.BIGQUERY:
    return calculate_and_store_percentiles_bq(
        projectname,
        datasetname,
        tablename,
        columnname,
        extra_stats_table,
        columntype,
        dbs,
    )
  else:
    raise ValueError("dbtype not supported yet: " + str(dbs["data_dbtype"]))


def calculate_and_write_percentiles(
    projectname, datasetname, dbs
):
  """Collects columnn percentiles to build histograms."""
  tasks = []
  for column_type in TYPES_TO_TABLES.keys():
    if not types_to_collect_percentiles(column_type):
      continue
    type_table = TYPES_TO_TABLES[column_type]
    type_table_sql_string = get_sql_table_string(
        dbs["metadata_dbtype"], type_table
    )
    columns_stats_table_sql_string = get_sql_table_string(
        dbs["metadata_dbtype"], COLUMNS_STATS_TABLE
    )
    query = (
        "select ci.dataset_name, ci.table_name, ci.column_name, ci.column_type"
        f" FROM {columns_stats_table_sql_string} as ci  WHERE ci.dataset_name ="
        f" '{datasetname}' AND ci.project_name = '{projectname}' AND"
        f" ci.column_type LIKE '{column_type}%' and ci.num_unique > 0 and"
        " ci.null_frac <> 1 AND EXISTS (select * from"
        f" {type_table_sql_string} as est where est.column_name ="
        " ci.column_name and est.table_name = ci.table_name and"
        " est.dataset_name = ci.dataset_name AND est.project_name ="
        " ci.project_name and array_length(est.percentiles) = 0)"
    )
    queryjob, _ = run_query(
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
