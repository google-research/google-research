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

"""Collect column statistics."""

from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers


INTERNAL = False

build_partitioned_predicate = helpers.build_partitioned_predicate
get_partitioning_info = helpers.get_partitioning_info
is_type_we_dont_collect_stats = configuration.is_type_we_dont_collect_stats
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
DBType = database_connector.DBType
TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_INFO_TABLE = configuration.COLUMNS_INFO_TABLE
COLUMNS_STATS_TABLE = configuration.COLUMNS_STATS_TABLE
BQ_INFO_SCHEMA_COLUMNS = configuration.BQ_INFO_SCHEMA_COLUMNS
TYPES_TO_TABLES = configuration.TYPES_TO_TABLES
get_sql_table_string = database_connector.get_sql_table_string
printif = helpers.printif


def calculate_and_write_column_statistics_internal(
    projectname, datasetname, tablename, dbs
):
  """Run queries to calculate column statistics and insert them to the configuration.COLUMNS_STATS_TABLE, returns the number of rows written to the database."""

  count_total = 0
  is_partitioned, partition_column, partition_column_type, row_count = (
      get_partitioning_info(
          projectname,
          datasetname,
          tablename,
          dbs["metadata_dbclient"],
          dbs["metadata_dbtype"],
      )
  )
  partitioned_predicate = build_partitioned_predicate(
      is_partitioned, tablename, partition_column, partition_column_type
  )
  table_sql_string = get_sql_table_string(
      dbs["data_dbtype"], f"{projectname}.{datasetname}.{tablename}"
  )
  # Find columns we need to collect stats for. These are the columns of the
  # specified tables that are not alreay in the
  # configuration.COLUMNS_STATS_TABLE.
  queryjob_col = []
  columns_info_table_sql_string = get_sql_table_string(
      dbs["metadata_dbtype"], COLUMNS_INFO_TABLE
  )
  columns_stats_table_sql_string = get_sql_table_string(
      dbs["metadata_dbtype"], COLUMNS_STATS_TABLE
  )

  query = (
      "SELECT it.column_name, it.column_type FROM"
      f" {columns_info_table_sql_string} as it WHERE it.project_name ="
      f" '{projectname}' AND it.dataset_name = '{datasetname}' AND"
      f" it.table_name = '{tablename}' AND not exists (select * from"
      f" {columns_stats_table_sql_string} as ts where ts.project_name ="
      f" '{projectname}'  and ts.dataset_name = '{datasetname}'  and"
      f" ts.table_name = '{tablename}' and ts.column_name = it.column_name )"
  )
  try:
    queryjob_col, _ = run_query(
        dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)

  count = 0
  preamble = (
      f"INSERT INTO {columns_stats_table_sql_string} (`project_name`,"
      " `dataset_name`, `table_name`, `column_name`,`column_type`,"
      " `null_frac`, `num_unique`,`row_count`)VALUES "
  )
  newvals = ""
  count_null = -1
  num_unique = -1
  for row in queryjob_col:
    columname = row["column_name"]
    coltype = row["column_type"]
    dont_collect_for_column_type, _ = is_type_we_dont_collect_stats(coltype)
    if dont_collect_for_column_type:
      continue
    query = (
        f"SELECT count(distinct `{columname}`) as num_unique FROM"
        f" {table_sql_string} WHERE"
        f" {partitioned_predicate}"
    )
    try:
      queryjob, _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
      rowres = get_query_result_first_row(dbs["data_dbtype"], queryjob)
      num_unique = rowres["num_unique"]
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)

    query = (
        "SELECT  count(*) as count_null FROM"
        f" {table_sql_string} WHERE `{columname}` IS"
        f" NULL AND {partitioned_predicate}"
    )
    try:
      queryjob, _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
      rowres = get_query_result_first_row(dbs["data_dbtype"], queryjob)
      count_null = rowres["count_null"]
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
      print(">>>> " + str(e))
      print(">>>> " + query)

    nan_ratio = 0
    if row_count > 0:
      nan_ratio = count_null / row_count

    if newvals:
      newvals = newvals + ", "

    newvals = (
        f"{newvals} ('{projectname}', '{datasetname}', '{tablename}',"
        f" '{columname}', '{coltype}', {str(nan_ratio)}, {str(num_unique)},"
        f" {str(row_count)})"
    )
    count += 1
    if count >= 200:
      query = preamble + newvals
      count_total = count_total + count
      try:
        _, _ = run_query(
            dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
        )
        count = 0
        newvals = ""
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
        print(">>>> " + str(e))
        print(">>>> " + query)

  if newvals:
    query = preamble + newvals
    count_total = count_total + count
    try:
      _, _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
      print(">>>> " + str(e))
      print(">>>> " + query)
  return count_total


def calculate_and_write_column_statistics(
    projectname, datasetname, dbs
):
  """Calculate basic columnn statistics and store them in configuration.COLUMNS_STATS_TABLE."""

  # Find tables of datasets and store them in a list of lists
  num_rows_written = 0
  dataset_table_pairs = []
  tables_info_table_sql_string = get_sql_table_string(
      dbs["metadata_dbtype"], TABLES_INFO_TABLE
  )
  query = (
      f"SELECT table_name FROM {tables_info_table_sql_string} WHERE"
      f" dataset_name = '{datasetname}' AND project_name = '{projectname}'"
  )
  print(query)
  queryjob, _ = run_query(
      dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
  )
  for row in queryjob:
    dataset_table_pairs.append([datasetname, row["table_name"]])
    print(f"table considered for columns : {row['table_name']}")

  if not INTERNAL:
    for x in dataset_table_pairs:
      num_rows_written += calculate_and_write_column_statistics_internal(
          projectname=projectname,
          datasetname=datasetname,
          tablename=x[1],
          dbs=dbs,
      )
  print("Number of rows written to the column stats table:", num_rows_written)
