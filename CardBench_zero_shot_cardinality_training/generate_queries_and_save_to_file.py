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

"""Generate sql queries and write them to a file."""

from collections.abc import Sequence
import datetime
import traceback
from typing import Any

from absl import app

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers
from CardBench_zero_shot_cardinality_training.generate_queries_library import query_generator


TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
run_query = database_connector.run_query
create_database_connection = database_connector.create_database_connection
get_query_result_first_row = database_connector.get_query_result_first_row
DBType = database_connector.DBType
build_partitioned_predicate = helpers.build_partitioned_predicate
DIRECTORY_PATH_QUERY_FILES = configuration.DIRECTORY_PATH_QUERY_FILES
DIRECTORY_PATH_JSON_FILES = configuration.DIRECTORY_PATH_JSON_FILES
Filter_Predicate_Operator = query_generator.Operator
Aggregation_Function = query_generator.Aggregator

generate_queries = query_generator.generate_queries

WOKRLOAD_DEFINITION_TABLE = configuration.WORKLOAD_DEFINITION_TABLE


def get_next_workload_id(dbs):
  """Returns the next available workload id."""
  next_workload_id = -1
  query = (
      f"SELECT  max(workload_id) as maxid FROM `{WOKRLOAD_DEFINITION_TABLE}`"
  )
  try:
    queryjob, _ = run_query(
        dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
    )
    next_workload_id = get_query_result_first_row(
        dbs["metadata_dbtype"], queryjob
    )["maxid"]
    if next_workload_id is None:
      next_workload_id = -1
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> error: " + str(e))

  next_workload_id = next_workload_id + 1

  return next_workload_id


def save_workload_information(
    projectname,
    datasetname,
    next_workload_id,
    output_file_name,
    parameters,
    dbs,
):
  """Saves the workload information to the database."""
  parameter_keys = []
  parameter_vals = []
  for k in parameters.keys():
    parameter_keys.append(k)
    if isinstance(parameters[k], str):
      parameter_vals.append(parameters[k])
    else:
      parameter_vals.append(str(parameters[k]))

  query = (
      f"INSERT INTO `{WOKRLOAD_DEFINITION_TABLE}` (`project_name`,"
      " `dataset_name`, `creation_date`, `workload_id`,`queries_file_path`,"
      f" `parameter_keys`, `parameter_values` ) VALUES ('{projectname}',"
      f" '{datasetname}',  '{str(datetime.datetime.now())}',"
      f" {next_workload_id}, '{output_file_name}', {parameter_keys},"
      f" {parameter_vals})"
  )
  try:
    _, _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> error: " + str(e))
    return


def get_table_partitioning_predicates(
    projectname, datasetname, dbs
):
  """Returns the partitioning predicates for each table."""
  partitioning_predidate_per_table = {}
  query = (
      "SELECT table_name, is_partitioned, partition_column,"
      f" partition_column_type FROM `{TABLES_INFO_TABLE}` WHERE project_name"
      f" = '{projectname}' AND dataset_name = '{datasetname}'"
  )

  try:
    queryjob, _ = run_query(
        dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> error: " + str(e))
    return

  for row in queryjob:
    partitioning_predidate_per_table[row["table_name"]] = (
        build_partitioned_predicate(
            row["is_partitioned"],
            row["table_name"],
            row["partition_column"],
            row["partition_column_type"],
            True,
        )
    )
  return partitioning_predidate_per_table


def generate_queries_and_save_to_file(_):
  """Generate and write queries to file."""

  projectname = configuration.PROJECT_NAME
  datasetnames = configuration.DATASET_NAMES

  dbs = {
      "data_dbtype": configuration.DATA_DBTYPE,
      "data_dbclient": create_database_connection(configuration.DATA_DBTYPE),
      "metadata_dbtype": configuration.METADATA_DBTYPE,
      "metadata_dbclient": create_database_connection(
          configuration.METADATA_DBTYPE
      ),
  }

  produced_files = []
  for datasetname in datasetnames:
    print(f"Generating queries for: {projectname}, {datasetname}")

    next_workload_id = get_next_workload_id(dbs)
    queries_output_file_name = f"{DIRECTORY_PATH_QUERY_FILES}/{next_workload_id}_{projectname}_{datasetname}.queries.sql"

    partitioning_pred_per_table = get_table_partitioning_predicates(
        projectname, datasetname, dbs
    )

    parameters = {
        "project_name": projectname,
        "dataset_name": datasetname,
        "dataset_json_input_directory_path": DIRECTORY_PATH_JSON_FILES,
        "query_file_output_path": queries_output_file_name,
        "partitioning_predicate_per_table": partitioning_pred_per_table,
        "allowed_predicate_operators": [
            query_generator.Operator.NEQ,
            query_generator.Operator.EQ,
            query_generator.Operator.LEQ,
            query_generator.Operator.GEQ,
            query_generator.Operator.IS_NOT_NULL,
            query_generator.Operator.IS_NULL,
        ],
        "allowed_aggregate_functions": [],
        "num_queries_to_generate": 35000,
        "max_nunmber_joins": 7,
        "max_number_filter_predicates": 2,
        "max_number_aggregates": 0,
        "max_number_group_by": 0,
        "max_cols_per_agg": 0,
        "group_by_threshold": 0,
        "int_neq_predicate_threshold": 100,
        "seed": 0,
        "complex_predicates": False,
        "recreate_query_file_if_exist": True,
        "always_create_the_maximum_number_of_joins": False,
        "always_create_the_maximum_number_of_aggregates": False,
        "always_create_the_maximum_number_of_predicates": False,
        "always_create_the_maximum_number_of_group_bys": False,
        "left_outer_join_ratio": 0,
        "groupby_limit_probability": 0.0,
        "groupby_having_probability": 0.0,
        "exists_predicate_probability": 0.0,
        "max_no_exists": 0,
        "outer_groupby_probability": 0.0,
        "min_number_joins": 1,
        "min_number_predicates": 0,
        "min_grouping_cols": (
            0
        ),  # not implemented as a check in the query generator
        "min_aggregation_cols": 0,  # not implemented in the query generator
    }

    save_workload_information(
        projectname,
        datasetname,
        next_workload_id,
        queries_output_file_name,
        parameters,
        dbs,
    )
    quote_table_sql_string = ""
    if dbs["data_dbtype"] == DBType.BIGQUERY:
      quote_table_sql_string = "`"
    try:
      generate_queries(
          table_identifier=f"{projectname}.{datasetname}.",
          quote_table_sql_string=quote_table_sql_string,
          dataset_name=parameters["dataset_name"],
          dataset_json_input_directory_path=parameters[
              "dataset_json_input_directory_path"
          ],
          query_file_output_path=parameters["query_file_output_path"],
          partitioning_predicate_per_table=parameters[
              "partitioning_predicate_per_table"
          ],
          allowed_predicate_operators=parameters["allowed_predicate_operators"],
          allowed_aggregate_functions=parameters["allowed_aggregate_functions"],
          num_queries_to_generate=parameters["num_queries_to_generate"],
          max_nunmber_joins=parameters["max_nunmber_joins"],
          max_number_filter_predicates=parameters[
              "max_number_filter_predicates"
          ],
          max_number_aggregates=parameters["max_number_aggregates"],
          max_number_group_by=parameters["max_number_group_by"],
          max_cols_per_agg=parameters["max_cols_per_agg"],
          group_by_threshold=parameters["group_by_threshold"],
          int_neq_predicate_threshold=parameters["int_neq_predicate_threshold"],
          seed=parameters["seed"],
          complex_predicates=parameters["complex_predicates"],
          recreate_query_file_if_exists=parameters[
              "recreate_query_file_if_exist"
          ],
          always_create_the_maximum_number_of_joins=parameters[
              "always_create_the_maximum_number_of_joins"
          ],
          always_create_the_maximum_number_of_aggregates=parameters[
              "always_create_the_maximum_number_of_aggregates"
          ],
          always_create_the_maximum_number_of_predicates=parameters[
              "always_create_the_maximum_number_of_predicates"
          ],
          always_create_the_maximum_number_of_group_bys=parameters[
              "always_create_the_maximum_number_of_group_bys"
          ],
          left_outer_join_ratio=parameters["left_outer_join_ratio"],
          groupby_limit_probability=parameters["groupby_limit_probability"],
          groupby_having_probability=parameters["groupby_having_probability"],
          min_number_joins=parameters["min_number_joins"],
          min_number_predicates=parameters["min_number_predicates"],
          min_grouping_cols=parameters["min_grouping_cols"],
          min_aggregation_cols=parameters["min_aggregation_cols"],
      )
      produced_files.append(
          [datasetname, next_workload_id, queries_output_file_name]
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN GENERATING :" + datasetname)
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> error: " + str(e))
      print(traceback.print_exc())
      produced_files.append([datasetname, "failed", "failed"])

  print("Produced files summary:")
  for info in produced_files:
    print(f"{info[0]} {info[1]} {info[2]}")

if __name__ == "__main__":
  app.run(generate_queries_and_save_to_file)
