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

"""Collect column histograms."""

from typing import Any
from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers

INTERNAL = False

build_partitioned_predicate = helpers.build_partitioned_predicate
run_query = database_connector.run_query
DBType = database_connector.DBType
get_query_result_first_row = database_connector.get_query_result_first_row
tables_info_table = configuration.TABLES_INFO_TABLE
columns_info_table = configuration.COLUMNS_INFO_TABLE
info_schema_columns = configuration.BQ_INFO_SCHEMA_COLUMNS
columns_stats_table = configuration.COLUMNS_STATS_TABLE
columns_histogram_table = configuration.COLUMNS_HISTOGRAM_TABLE
columns_string_extra_stats_table = (
    configuration.COLUMNS_STRING_EXTRA_STATS_TABLE
)
types_to_collect_stats = configuration.TYPES_TO_COLLECT_STATS


def calculate_and_write_column_histograms_internal_bq(
    projectname,
    datasetname,
    tablename,
    columns,
    dbtype,
    dbclient,
):
  """Generate the query to get column histogram."""
  histograms = []
  # Get table level information from columns.
  if not columns:
    return
  is_partitioned = columns[0]["is_partitioned"]
  partition_column = columns[0]["partition_column"]
  partition_column_type = columns[0]["partition_column_type"]
  partitioned_predicate = build_partitioned_predicate(
      is_partitioned, partition_column, partition_column_type
  )

  # Generate approx quantitles for all columns.
  approx_quantiles_list = [
      f'APPROX_QUANTILES(`{column["column_name"]}`, {column["num_quantiles"]})'
      f' AS `{column["column_name"]}`'
      for column in columns
      if column["num_quantiles"] > 0
  ]

  if not approx_quantiles_list:
    return []

  print(
      f"Getting histogram for {len(columns)} columns of the table"
      f" `{projectname}.{datasetname}.{tablename}`"
  )
  generate_quantitles_query = (
      f'SELECT {", ".join(approx_quantiles_list)} FROM'
      f" `{projectname}.{datasetname}.{tablename}` WHERE"
      f" {partitioned_predicate}"
  )

  try:
    queryjob = run_query(dbtype, generate_quantitles_query, dbclient)
    row = get_query_result_first_row(dbtype, queryjob)
    for column in columns:
      if column["num_quantiles"] > 0:
        histograms.append({
            "column_name": column["column_name"],
            "column_type": column["column_type"],
            "quantiles": [str(item) for item in row[column["column_name"]]],
        })
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(e)
    print(">>>>>>>>>>>> ERROR IN QUERY :\n" + generate_quantitles_query)
    # Retry to query for each column to get around error like
    # "400 Cannot query rows larger than 100MB limit."
    if len(columns) > 1:
      for column in columns:
        h = calculate_and_write_column_histograms_internal_bq(
            projectname, datasetname, tablename, [column], dbtype, dbclient
        )
        histograms.extend(h)
  return histograms


def calculate_and_write_column_histograms_internal(
    projectname,
    datasetname,
    tablename,
    columns,
    dbs,
):
  """Calculate column histogram."""

  if dbs["data_dbtype"] == DBType.BIGQUERY:
    histograms = calculate_and_write_column_histograms_internal_bq(
        projectname,
        datasetname,
        tablename,
        columns,
        dbs["data_dbtype"],
        dbs["data_dbclient"],
    )
  else:
    raise ValueError("Unsupported database type: " + dbs["data_dbtype"])

  insert_row_list = [
      f"('{projectname}', '{datasetname}', '{tablename}',"
      f" '{histogram['column_name']}', '{histogram['column_type']}',"
      f" {histogram['quantiles']})"
      for histogram in histograms
  ]

  insert_query = (
      f"INSERT INTO `{columns_histogram_table}` (`project_name`,"
      " `dataset_name`, `table_name`, `column_name`,`column_type`,"
      f" `approx_quantiles_100`) VALUES {', '.join(insert_row_list)}"
  )

  try:
    _ = run_query(
        dbs["metadata_dbtype"], insert_query, dbs["metadata_dbclient"]
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(e)
    print(">>>>>>>>>>>>> ERROR IN QUERY :" + insert_query)
    return


def calculate_and_write_column_histograms(
    projectname, datasetname, dbs
):
  """Calculate histograms for column, the result is stored in the configuration.COLUMNS_HISTOGRAM_TABLE table."""

  selected_column_types = ", ".join(
      [f'"{type}"' for type in types_to_collect_stats.keys()]
  )
  # Get list of columns to collect histograms
  query = f"""WITH column_candidates AS (
      SELECT
        col.*,
      FROM `{columns_stats_table}` AS col
      WHERE TRUE
        AND col.project_name = \'{projectname}\'
        AND col.dataset_name = \'{datasetname}\'
        AND col.column_type IN ({selected_column_types})
    ), column_candidates_w_part_info AS (
      SELECT
        col.*,
        tab.is_partitioned,
        tab.partition_column,
        tab.partition_column_type,
      FROM column_candidates AS col
      INNER JOIN `{tables_info_table}` AS tab
        ON col.project_name = tab.project_name
          AND col.dataset_name = tab.dataset_name
          AND col.table_name = tab.table_name
    ), column_candidates_group_by_table AS (
      SELECT
        project_name,
        dataset_name,
        table_name,
        ARRAY_AGG(STRUCT(
          column_name,
          column_type,
          num_unique,
          100 as num_quantiles,
          is_partitioned,
          partition_column,
          partition_column_type
        )) AS columns_arr
      FROM column_candidates_w_part_info
      GROUP BY project_name, dataset_name, table_name, column_type
    )
    SELECT *
    FROM column_candidates_group_by_table
  """

  query_job = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])

  if not INTERNAL:
    for row in query_job:
      calculate_and_write_column_histograms_internal(
          projectname=row["project_name"],
          datasetname=row["dataset_name"],
          tablename=row["table_name"],
          columns=row["columns_arr"],
          dbs=dbs,
      )
