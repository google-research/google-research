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

"""Collect table information."""

from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers


build_partitioned_predicate = helpers.build_partitioned_predicate
get_partitioning_info = helpers.get_partitioning_info
is_type_we_dont_collect_stats = configuration.is_type_we_dont_collect_stats
run_query = database_connector.run_query
DBType = database_connector.DBType
get_query_result_first_row = database_connector.get_query_result_first_row
TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_INFO_TABLE = configuration.COLUMNS_INFO_TABLE
BQ_INFO_SCHEMA_COLUMNS = configuration.BQ_INFO_SCHEMA_COLUMNS
TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE


def collect_and_write_table_information_bq(
    projectname, datasetname, dbclient, dbtype
):
  """Get table info from BigQuery."""

  # returns a dict of table information
  table_comb_info = {}
  query = (
      "select * from (  SELECT table_name, max(is_partitioning_column) as"
      " ispart, max(clustering_ordinal_position) as isclust FROM"
      f" `{projectname}.{datasetname}{BQ_INFO_SCHEMA_COLUMNS}` group by"
      " table_name ) as tables where  NOT EXISTS (select * from"
      f" `{TABLES_INFO_TABLE}` as it where  it.project_name = '{projectname}'"
      f" AND it.dataset_name = '{datasetname}' AND it.table_name ="
      " tables.table_name)"
  )
  queryjob = ""
  try:
    queryjob, _ = run_query(dbtype, query, dbclient)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)
  for row in queryjob:
    ispart = False
    if row["ispart"] == "YES":
      ispart = True
    isclust = False
    if row["isclust"] != "null":
      isclust = True

    table_name = row["table_name"]
    query2 = (
        "SELECT count(*) as cnt FROM"
        f" `{projectname}.{datasetname}.{table_name}`"
    )
    queryjob2, _ = run_query(dbtype, query2, dbclient)
    rowrescnt = get_query_result_first_row(dbtype, queryjob2)

    part_col_name = "empty"
    part_col_type = "empty"
    if ispart:
      query3 = f"""
          SELECT
            (CASE ((
                  SELECT COUNT(*)
                  FROM `{projectname}.{datasetname}.INFORMATION_SCHEMA.COLUMNS`
                  WHERE table_name = '{table_name}' AND is_partitioning_column = 'YES'))
             WHEN 1 THEN (( SELECT column_name FROM `{projectname}.{datasetname}.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = '{table_name}' AND is_partitioning_column = 'YES'))
             ELSE NULL
             END) as part_col_name,
             (CASE ((
                  SELECT COUNT(*)
                  FROM `{projectname}.{datasetname}.INFORMATION_SCHEMA.COLUMNS`
                  WHERE table_name = '{table_name}' AND is_partitioning_column = 'YES'))
             WHEN 1 THEN (( SELECT data_type FROM `{projectname}.{datasetname}.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = '{table_name}' AND is_partitioning_column = 'YES'))
             ELSE NULL
             END) as part_col_type
          """
      queryjob3, _ = run_query(dbtype, query3, dbclient)
      rowres3 = get_query_result_first_row(dbtype, queryjob3)
      part_col_type = rowres3["part_col_type"]
      part_col_name = rowres3["part_col_name"]

    table_comb_info[row["table_name"]] = {
        "is_partitioned": ispart,  # bool
        "is_clustered": isclust,  # bool
        "row_count": rowrescnt["cnt"],  # int
        "data_size_gib": -1,  # int
        "file_count": -1,  # int
        "partition_column": part_col_name,  # string
        "partition_column_type": part_col_type,  # string
        "clustered_columns": [],  # list of strings TODO
    }
  return table_comb_info




def collect_and_write_table_information(
    projectname, datasetname, dbs
):
  """Collect table information and write to configuration.TABLES_INFO_TABLE."""

  # The following information is needed for each table:
  #   - table name (string)
  #   - row_count (int)
  #   - data_size_gib (int)
  #   - file_count (int), if file is stored in multiple files
  #   - is_partitioned (bool), if the table is partitioned
  #   - is_clustered (bool), if the table is clustered
  #   - partition_column, the column used for partitioning
  #   - clustered_columns, the columns used for clustering

  if dbs["data_dbtype"] == DBType.BIGQUERY:
    table_comb_info = collect_and_write_table_information_bq(
        projectname, datasetname, dbs["data_dbclient"], dbs["data_dbtype"]
    )
  else:
    raise ValueError("dbtype not supported yet: " + str(dbs["data_dbtype"]))

  total_rows_added = 0
  insert_query_preamble = (
      f"INSERT INTO `{TABLES_INFO_TABLE}` (`project_name`, `dataset_name`,"
      " `table_name`, `row_count`, `data_size_gib`, `file_count`,"
      " `is_partitioned`, `is_clustered`, `partition_column`,"
      " `clustered_columns`, `partition_column_type` )VALUES"
  )

  count = 0
  newvals = ""
  for ti in table_comb_info:
    if newvals:
      newvals = newvals + ", "

    newvals = (
        f"{newvals} ('{projectname}', '{datasetname}', '{ti}',"
        f" {table_comb_info[ti]['row_count']},"
        f" {table_comb_info[ti]['data_size_gib']},"
        f" {table_comb_info[ti]['file_count']},"
        f" {table_comb_info[ti]['is_partitioned']},"
        f" {table_comb_info[ti]['is_clustered']},"
        f" '{table_comb_info[ti]['partition_column']}',"
        f" {table_comb_info[ti]['clustered_columns']},"
        f" '{table_comb_info[ti]['partition_column_type']}')"
    )
    count += 1
    # limits inserted rows of a single query to 200 to avoid size limit
    if count >= 200:
      query = insert_query_preamble + newvals
      try:
        _, _ = run_query(
            dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
        print(">>>> " + str(e))
        print(">>>> " + query)
      total_rows_added = total_rows_added + count
      count = 0
      newvals = ""

  if newvals:
    total_rows_added = total_rows_added + count
    query = insert_query_preamble + newvals
    try:
      _, _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)
  print(
      "Table information total rows added to database: " + str(total_rows_added)
  )
