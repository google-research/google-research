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

"""Helper functions for run_queries.py."""

import os
from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector


WORKLOAD_DEFINITION_TABLE = configuration.WORKLOAD_DEFINITION_TABLE
DIRECTORY_PATH_QUERY_FILES = configuration.DIRECTORY_PATH_QUERY_FILES
QUERY_RUN_INFORMATION_TABLE = configuration.QUERY_RUN_INFORMATION_TABLE
TEMP_QUERY_RUN_INFORMATION_TABLE_PREFIX = (
    configuration.TEMP_QUERY_RUN_INFORMATION_TABLE_PREFIX
)
DBType = database_connector.DBType
create_database_connection = database_connector.create_database_connection
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row


def get_workload_info(
    workload_id_to_run, metadata_dbclient, metadata_dbtype
):
  """Get the workload information from the metadata database."""
  id_of_num_queries = -1
  queries_file_path = ""
  num_queries = -1
  query = (
      "SELECT  queries_file_path, parameter_keys, parameter_values "
      f"FROM `{WORKLOAD_DEFINITION_TABLE}` "
      f"WHERE workload_id = {workload_id_to_run}"
  )
  try:
    queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
    for row in queryjob:
      for ki in enumerate(row["parameter_keys"]):
        if ki[1] == "num_queries_to_generate":
          id_of_num_queries = ki[0]
      num_queries = int(row["parameter_values"][id_of_num_queries])
      queries_file_path = row["queries_file_path"]
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + str(e))
    return

  queries_file_name = queries_file_path.split("/")[-1]
  queries_file_path = os.path.join(
      DIRECTORY_PATH_QUERY_FILES, queries_file_name
  )
  return queries_file_path, num_queries


def select_next_query_run_id(
    workload_id_to_run, metadata_dbclient, metadata_dbtype
):
  """Get the query run id from the metadata database."""
  query = (
      "SELECT  max(query_run_id) as max_query_run_id "
      f"FROM `{QUERY_RUN_INFORMATION_TABLE}`"
      f"WHERE workload_id = {workload_id_to_run}"
  )
  try:
    queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + str(e))
    return

  next_query_run_id = -1
  for row in queryjob:
    next_query_run_id = row["max_query_run_id"]
  # if table is empty
  if next_query_run_id is None:
    next_query_run_id = -1
  next_query_run_id = next_query_run_id + 1
  return next_query_run_id


def create_temp_query_run_information_table(
    workload_id,
    query_run_id,
    metadata_dbclient,
    metadata_dbtype,
):
  """Save query results in a temporary table instead of the QUERY_RUN_INFORMATION_TABLE."""

  temp_query_run_information_table = f"{TEMP_QUERY_RUN_INFORMATION_TABLE_PREFIX}{workload_id}_{query_run_id}_temp_table"

  create_query = (
      f"CREATE OR REPLACE TABLE `{temp_query_run_information_table}`"
      " (workload_id integer, query_run_id integer, query_string string,"
      " database_query_id string, cardinality integer)"
  )

  try:
    _, _ = run_query(metadata_dbtype, create_query, metadata_dbclient)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + create_query)
    print(e)
    return
  return temp_query_run_information_table


def get_preamble_insert_query_result(
    temp_query_run_information_table,
):
  return (
      f"INSERT INTO `{temp_query_run_information_table}` (`workload_id`,"
      " `query_run_id`, `query_string`, `database_query_id`,`cardinality`)"
      " VALUES "
  )


def result_cleanup(
    temp_query_run_information_table,
    metadata_dbclient,
    metadata_dbtype,
):
  """Copy the results from temp_query_run_information_table to the QUERY_RUN_INFORMATION_TABLE."""

  query = (
      f"INSERT INTO `{QUERY_RUN_INFORMATION_TABLE}` (workload_id, query_run_id,"
      " query_string, database_query_id, cardinality) SELECT workload_id,"
      " query_run_id, query_string, database_query_id, cardinality FROM "
      f" `{temp_query_run_information_table}`"
  )
  try:
    _ = run_query(metadata_dbtype, query, metadata_dbclient)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(e)
  else:
    query = f"drop table `{temp_query_run_information_table}`"
    try:
      _ = run_query(metadata_dbtype, query, metadata_dbclient)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + str(e))
