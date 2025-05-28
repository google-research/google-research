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

"""Run a workload with a given workload id."""

from collections.abc import Sequence
import os
import traceback
from typing import Any

from absl import app

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.run_queries_library import run_queries_helpers

WORKLOAD_DEFINITION_TABLE = configuration.WORKLOAD_DEFINITION_TABLE
QUERY_RUN_INFORMATION_TABLE = configuration.QUERY_RUN_INFORMATION_TABLE
TEMP_QUERY_RUN_INFORMATION_TABLE_PREFIX = (
    configuration.TEMP_QUERY_RUN_INFORMATION_TABLE_PREFIX
)
DBType = database_connector.DBType
create_database_connection = database_connector.create_database_connection
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
get_query_cardinality = database_connector.get_query_cardinality
get_workload_info = run_queries_helpers.get_workload_info
select_next_query_run_id = run_queries_helpers.select_next_query_run_id
create_temp_query_run_information_table = (
    run_queries_helpers.create_temp_query_run_information_table
)
get_preamble_insert_query_result = (
    run_queries_helpers.get_preamble_insert_query_result
)
result_cleanup = run_queries_helpers.result_cleanup

INTERNAL = False
open_file = open
file_exists = os.path.exists
remove_file = os.remove


def run_queries(argv):
  """Calculate statistics for a set of datasets and write them to the metadata database."""

  # The code is designed to work with BigQuery but to also be extensible to
  # other databases. Further the code uses two databases one that contains the
  # data we are calculating statistics for and one that stores the calculated
  # statistics.

  dbs = {
      # used to query the database
      "data_dbtype": configuration.DATA_DBTYPE,
      "data_dbclient": create_database_connection(configuration.DATA_DBTYPE),
      # used to stored the collected statistics
      "metadata_dbtype": configuration.METADATA_DBTYPE,
      "metadata_dbclient": create_database_connection(
          configuration.METADATA_DBTYPE
      ),
  }
  if len(argv) == 1:
    print(
        "No arguments were provided. Please re-run with the necessary args"
        "run_queries workload_id"
    )
    return

  workload_id_to_run = argv[1]
  ## find workload information
  queries_file_path, _ = get_workload_info(
      workload_id_to_run, dbs["metadata_dbclient"], dbs["metadata_dbtype"]
  )
  ## Find the query run id to use. Each time a the queries of a workload are
  ## run a new query run id is created.
  query_run_id = select_next_query_run_id(
      workload_id_to_run, dbs["metadata_dbclient"], dbs["metadata_dbtype"]
  )

  # Store the results in a temporary table and at the end move them to the
  # QUERY_RUN_INFORMATION_TABLE. This is to avoid errors when multiple writers
  # are trying to write to the same table at the same time.
  temp_query_run_information_table = create_temp_query_run_information_table(
      workload_id_to_run,
      query_run_id,
      dbs["metadata_dbclient"],
      dbs["metadata_dbtype"],
  )

  print("Create temp table: ", temp_query_run_information_table)

  insert_result_query_preamble = get_preamble_insert_query_result(
      temp_query_run_information_table
  )

  queries_to_run = []
  with open_file(queries_file_path, "rt") as queries_file:
    lines = queries_file.readlines()
    for line in lines:
      queries_to_run.append(line)

  if not INTERNAL:
    num_queries_run = run_queries_worker(
        insert_result_query_preamble,
        queries_to_run,
        dbs,
        workload_id_to_run,
        query_run_id,
    )

  print("Queries run in total :", num_queries_run)

  # cleanup the results
  result_cleanup(
      temp_query_run_information_table,
      dbs["metadata_dbclient"],
      dbs["metadata_dbtype"],
  )


def clean_query_string(query):
  return query.replace("'", "'").replace("\n", "").replace(";", "")


def run_queries_worker(
    insert_result_query_preamble,
    queries_to_run,
    dbs,
    workload_id,
    query_run_id,
):
  """Run a set of queries and store the cardinalities."""

  query_strings = []
  database_query_ids = []
  cardinalities = []

  for query in queries_to_run:
    print("-----\nExecuting: ", query)
    try:
      # run with 7 minutes timeout
      queryjob, job_id = run_query(
          dbs["data_dbtype"], query, dbs["data_dbclient"], 420
      )
      if "rwcnt" in query:
        cardinality = get_query_result_first_row(dbs["data_dbtype"], queryjob)[
            "rwcnt"
        ]
      else:
        cardinality = get_query_cardinality(dbs["data_dbtype"], queryjob)
      database_query_ids.append(job_id)
      cardinalities.append(cardinality)
      query_to_save = query
      if query.startswith("select count(*) as rwcnt from ("):
        query_to_save = query.split("select count(*) as rwcnt from (")[1]
        query_to_save = query_to_save.split(");")[0]
      query_strings.append(query_to_save)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> Error in query :", query, str(e))
      print(
          ">>>>>>>>>>>>>>>>>>>>>>>>>>>> Error in query :",
          traceback.format_exc(),
      )
      database_query_ids.append("error in query execution")
      cardinalities.append(-1)
      query_strings.append(query)

  values = ""
  for i in range(len(queries_to_run)):
    if values:
      values = values + ", "
    cleaned_query_string = clean_query_string(query_strings[i])
    values = (
        f'{values} ({workload_id}, {query_run_id}, "{cleaned_query_string}",'
        f" '{database_query_ids[i]}', {cardinalities[i]})"
    )
  query = insert_result_query_preamble + values
  if values:
    try:
      _, _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + str(e))
      print(traceback.print_exc())

  return len(query_strings)


if __name__ == "__main__":
  app.run(run_queries)
