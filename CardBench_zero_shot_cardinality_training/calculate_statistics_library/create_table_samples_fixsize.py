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

"""Create table sample."""

from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers

INTERNAL = False

build_partitioned_predicate = helpers.build_partitioned_predicate
copy_table = helpers.copy_table
get_partitioning_info = helpers.get_partitioning_info
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
DBType = database_connector.DBType
tables_info_table = configuration.TABLES_INFO_TABLE
columns_info_table = configuration.COLUMNS_INFO_TABLE
columns_stats_table = configuration.COLUMNS_STATS_TABLE
info_schema_columns = configuration.BQ_INFO_SCHEMA_COLUMNS
types_to_tables = configuration.TYPES_TO_TABLES
get_sql_table_string = database_connector.get_sql_table_string
table_exists = database_connector.table_exists


def create_table_samples_fixsize_internal_bq(
    projectname,
    datasetname,
    tablename,
    target_row_number,
    row_count,
    is_partitioned,
    partition_column,
    partition_column_type,
    sampled_tables_dataset_name,
    sampled_table_path,
    dbs,
):
  """Calculates tabel sample, Big Query version."""
  try:
    partition_pred = build_partitioned_predicate(
        is_partitioned, tablename, partition_column, partition_column_type
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(e)
    print(
        ">>>>>>>>>>>> ERROR IN PARTITION PREDICATE for: ",
        projectname,
        datasetname,
        tablename,
    )
    return

  # Sample requested is larger than the table size, copy entire table
  if target_row_number >= row_count:
    copy_table(
        projectname,
        datasetname,
        tablename,
        target_row_number,
        partition_pred,
        sampled_tables_dataset_name,
        dbs,
    )
    return

  table_sql_string = get_sql_table_string(
      dbs["data_dbtype"], f"{projectname}.{datasetname}.{tablename}"
  )
  sampled_table_sql_string = get_sql_table_string(
      dbs["data_dbtype"], sampled_table_path
  )

  sampling_query = f"""
    CREATE OR REPLACE TABLE {sampled_table_sql_string} AS(
      SELECT *
      FROM {table_sql_string}
      WHERE {partition_pred}
      ORDER BY FARM_FINGERPRINT(GENERATE_UUID())
      LIMIT {target_row_number}
    )
  """
  try:
    _, _ = run_query(dbs["data_dbtype"], sampling_query, dbs["data_dbclient"])
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(e)
    print(">>>>>>>>>>>> ERROR IN QUERY :\n" + sampling_query)


def create_table_samples_fixsize_internal(
    projectname,
    datasetname,
    tablename,
    target_row_number,
    row_count,
    is_partitioned,
    partition_column,
    partition_column_type,
    sampled_tables_dataset_name,
    sampled_table_path,
    dbs,
):
  """Calculates tabel sample."""

  if dbs["data_dbtype"] == DBType.BIGQUERY:
    create_table_samples_fixsize_internal_bq(
        projectname,
        datasetname,
        tablename,
        target_row_number,
        row_count,
        is_partitioned,
        partition_column,
        partition_column_type,
        sampled_tables_dataset_name,
        sampled_table_path,
        dbs,
    )


def create_table_samples_fixsize(
    projectname,
    datasetname,
    sampled_tables_dataset_name,
    dbs,
    target_row_number,
):
  """Create tables with fixed size."""

  tables_info_table_sql_string = get_sql_table_string(
      dbs["metadata_dbtype"], tables_info_table
  )
  if dbs["data_dbtype"] == DBType.BIGQUERY:
    sampled_tables_dataset_name_sql_string = get_sql_table_string(
        dbs["metadata_dbtype"], f"{sampled_tables_dataset_name}.__TABLES__"
    )
    query = f"""
      SELECT
        tb.project_name,
        tb.dataset_name,
        tb.table_name,
        tb.row_count,
        tb.is_partitioned,
        tb.partition_column,
        tb.partition_column_type
      FROM {tables_info_table_sql_string} AS tb
      WHERE TRUE
        AND tb.table_name not like '_SEARCH_INDEX_%'
        AND tb.project_name = "{projectname}"
        AND tb.dataset_name = "{datasetname}"
        AND NOT EXISTS(
          SELECT 1
          FROM {sampled_tables_dataset_name_sql_string} AS sample
          WHERE TRUE
            AND sample.table_id = CONCAT(tb.project_name, "_", tb.dataset_name, "_", tb.table_name)
            # Conditions for correct sampled tables.
            AND (sample.row_count = {target_row_number} OR sample.row_count = tb.row_count)
        )
    """
  else:
    raise ValueError(f"Unsupported database type: {dbs['metadata_dbtype']}")
  query_job, _ = run_query(
      dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
  )

  if not INTERNAL:
    for row in query_job:
      sampled_table_path = f"{sampled_tables_dataset_name}.{row['project_name']}_{row['dataset_name']}_{row['table_name']}"
      create_table_samples_fixsize_internal(
          projectname=row["project_name"],
          datasetname=row["dataset_name"],
          tablename=row["table_name"],
          target_row_number=target_row_number,
          row_count=row["row_count"],
          is_partitioned=row["is_partitioned"],
          partition_column=row["partition_column"],
          partition_column_type=row["partition_column_type"],
          sampled_tables_dataset_name=sampled_tables_dataset_name,
          sampled_table_path=sampled_table_path,
          dbs=dbs,
      )

