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

"""Collect column information."""

from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers


run_query = database_connector.run_query
DBType = database_connector.DBType
TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_INFO_TABLE = configuration.COLUMNS_INFO_TABLE
BQ_INFO_SCHEMA_COLUMNS = configuration.BQ_INFO_SCHEMA_COLUMNS
remove_anything_after_first_parenthesis = (
    helpers.remove_anything_after_first_parenthesis
)


def get_col_info_from_bq(
    data_dbclient, data_dbtype, projectname, datasetname
):
  """Get column information from BigQuery."""
  column_info = []
  query = (
      f"SELECT * FROM `{projectname}.{datasetname}{BQ_INFO_SCHEMA_COLUMNS}` as"
      " isc WHERE isc.table_name IN (select table_name from"
      f" `{TABLES_INFO_TABLE}` where project_name = '{projectname}' AND"
      f" dataset_name = '{datasetname}') AND NOT EXISTS (SELECT * from"
      f" `{COLUMNS_INFO_TABLE}` as it where  it.project_name = '{projectname}'"
      f" AND it.dataset_name = '{datasetname}' AND it.table_name ="
      " isc.table_name AND it.column_name = isc.column_name)"
  )
  try:
    queryjob, _ = run_query(data_dbtype, query, data_dbclient)
    for row in queryjob:
      column_info.append({
          "table_name": row["table_name"],
          "column_name": row["column_name"],
          "column_type": row["data_type"],
      })
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)

  return column_info


def collect_and_write_column_information(
    projectname, datasetname, dbs
):
  """Get column information and write them to configuration.COLUMNS_INFO_TABLE."""

  # For each column the following information is needed:
  # - column name (string)
  # - column type (string)

  if dbs["data_dbtype"] == DBType.BIGQUERY:
    column_info = get_col_info_from_bq(
        dbs["data_dbclient"], dbs["data_dbtype"], projectname, datasetname
    )
  else:
    raise ValueError("dbtype not supported yet: " + str(dbs["data_dbtype"]))

  count = 0
  rows_added = 0
  newvals = ""
  insert_query_preamble = (
      f"INSERT INTO `{COLUMNS_INFO_TABLE}` (`project_name`, `dataset_name`,"
      " `table_name`, `column_name`, `column_type`) VALUES "
  )
  for col in column_info:
    if newvals:
      newvals = newvals + ", "
    sanitized_column_type = remove_anything_after_first_parenthesis(
        col["column_type"]
    )
    newvals = (
        f"{newvals} ('{projectname}', '{datasetname}',"
        f" '{col['table_name']}',  '{col['column_name']}',"
        f" '{sanitized_column_type}')"
    )
    count += 1
    if count >= 200:
      query = insert_query_preamble + newvals
      try:
        _, _ = run_query(
            dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
        )
        rows_added += count
        count = 0
        newvals = ""
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
        print(">>>> " + str(e))
        print(">>>> " + query)
  if newvals:
    query = insert_query_preamble + newvals
    rows_added += count
    try:
      _, _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)
  print("collect column information, rows added:", rows_added)
