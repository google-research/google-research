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

"""Helper functions to generate sql queries and write them to a file."""

import datetime
from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers
from CardBench_zero_shot_cardinality_training.generate_queries_library import query_generator_v2


TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
run_query = database_connector.run_query
create_database_connection = database_connector.create_database_connection
get_query_result_first_row = database_connector.get_query_result_first_row
DBType = database_connector.DBType
build_partitioned_predicate = helpers.build_partitioned_predicate
DIRECTORY_PATH_QUERY_FILES = configuration.DIRECTORY_PATH_QUERY_FILES
DIRECTORY_PATH_JSON_FILES = configuration.DIRECTORY_PATH_JSON_FILES
Filter_Predicate_Operator = query_generator_v2.Operator
Aggregation_Function = query_generator_v2.Aggregator
generate_queries = query_generator_v2.generate_queries
WOKRLOAD_DEFINITION_TABLE = configuration.WORKLOAD_DEFINITION_TABLE


def template_query_generation_parameters(
    projectname,
    datasetname,
    num_queries_to_generate,
    queries_output_file_name,
    partitioning_pred_per_table,
    use_partitioning_predicate = False,
):
  return {
      "project_name": projectname,
      "dataset_name": datasetname,
      "dataset_json_input_directory_path": DIRECTORY_PATH_JSON_FILES,
      "query_file_output_path": queries_output_file_name,
      "partitioning_predicate_per_table": partitioning_pred_per_table,
      "use_partitioning_predicate": use_partitioning_predicate,
      "recreate_query_file_if_exist": True,
      "seed": 0,
      # query number
      "num_queries_to_generate": num_queries_to_generate,
      # filter parameters
      "min_number_filter_predicates_per_table": 0,
      "max_number_filter_predicates_per_table": 0,
      "always_create_the_maximum_number_of_predicates": False,
      "allowed_predicate_operators": [
          query_generator_v2.Operator.NEQ,
          query_generator_v2.Operator.EQ,
          query_generator_v2.Operator.LEQ,
          query_generator_v2.Operator.GEQ,
          query_generator_v2.Operator.IS_NOT_NULL,
          query_generator_v2.Operator.IS_NULL,
      ],
      "allowed_predicate_column_types": [  # TODO(chronis): use an enum
          "int",
          "int64",
          "uint64",
          "int32",
          "uint32",
          "string",
          "float",
          "float64",
          "double",
      ],
      "eq_and_neq_unique_value_threshold_for_integer_columns": 1000,
      # join parameters
      "min_number_joins": 0,
      "max_nunmber_joins": 0,
      "always_create_the_maximum_number_of_joins": False,
      # grouping parameters
      "min_number_group_bys_cols": 0,
      "max_number_group_by_cols": 0,
      "allowed_grouping_column_types": ["int", "string"],
      "always_create_the_maximum_number_of_group_bys": False,
      "group_by_col_unqiue_vals_treshold": 500,
      # aggregation parameters
      "min_aggregation_fns": 0,
      "max_aggregation_fns": 0,
      "always_create_the_maximum_number_of_aggregate_fns": False,
      "allowed_aggregate_functions": 0,
      "max_cols_per_agg": 0,
      "allowed_aggregation_column_types": [],
      # disallow lists
      "column_disallow_list_for_predicates_aggs_and_group_bys": [],
      "table_name_disallow_list_for_predicates_aggs_and_group_bys": [],
  }


# BEGIN_GOOGLE_INTERNAL


# This functions creates the parameter specification for the single table
# queries used to create the CardBench single table queries.
def cardbench_single_table_query_generation_parameters(
    projectname,
    datasetname,
    num_queries_to_generate,
    queries_output_file_name,
    partitioning_pred_per_table,
    use_partitioning_predicate = False,
):
  return {
      "project_name": projectname,
      "dataset_name": datasetname,
      "dataset_json_input_directory_path": DIRECTORY_PATH_JSON_FILES,
      "query_file_output_path": queries_output_file_name,
      "partitioning_predicate_per_table": partitioning_pred_per_table,
      "use_partitioning_predicate": use_partitioning_predicate,
      "recreate_query_file_if_exist": True,
      "seed": 0,
      # query number
      "num_queries_to_generate": num_queries_to_generate,
      # filter parameters
      "min_number_filter_predicates_per_table": 1,
      "max_number_filter_predicates_per_table": 4,
      "always_create_the_maximum_number_of_predicates": False,
      "allowed_predicate_operators": [
          query_generator_v2.Operator.NEQ,
          query_generator_v2.Operator.EQ,
          query_generator_v2.Operator.LEQ,
          query_generator_v2.Operator.GEQ,
          query_generator_v2.Operator.IS_NOT_NULL,
          query_generator_v2.Operator.IS_NULL,
      ],
      "allowed_predicate_column_types": [
          "int",
          "int64",
          "uint64",
          "int32",
          "uint32",
          "string",
          "float",
          "float64",
          "double",
      ],
      "eq_and_neq_unique_value_threshold_for_integer_columns": 10000,
      # join parameters
      "min_number_joins": 0,
      "max_nunmber_joins": 0,
      "always_create_the_maximum_number_of_joins": False,
      # grouping parameters
      "min_number_group_bys_cols": 0,
      "max_number_group_by_cols": 0,
      "allowed_grouping_column_types": ["int", "string"],
      "always_create_the_maximum_number_of_group_bys": False,
      "group_by_col_unqiue_vals_treshold": 5000,
      # aggregation parameters
      "min_aggregation_fns": 0,
      "max_aggregation_fns": 0,
      "always_create_the_maximum_number_of_aggregate_fns": False,
      "allowed_aggregate_functions": 0,
      "max_cols_per_agg": 0,
      "allowed_aggregation_column_types": [],
      # disallow lists
      "column_disallow_list_for_predicates_aggs_and_group_bys": [],
      "table_name_disallow_list_for_predicates_aggs_and_group_bys": [],
  }


# This functions creates the parameter specification for the binary join queries
# used to create the CardBench binary join queries.
def cardbench_binary_join_query_generation_parameters(
    projectname,
    datasetname,
    num_queries_to_generate,
    queries_output_file_name,
    partitioning_pred_per_table,
    use_partitioning_predicate = False,
):
  return {
      "project_name": projectname,
      "dataset_name": datasetname,
      "dataset_json_input_directory_path": DIRECTORY_PATH_JSON_FILES,
      "query_file_output_path": queries_output_file_name,
      "partitioning_predicate_per_table": partitioning_pred_per_table,
      "use_partitioning_predicate": use_partitioning_predicate,
      "recreate_query_file_if_exist": True,
      "seed": 0,
      # query number
      "num_queries_to_generate": num_queries_to_generate,
      # filter parameters
      "min_number_filter_predicates_per_table": 0,
      "max_number_filter_predicates_per_table": 3,
      "always_create_the_maximum_number_of_predicates": False,
      "allowed_predicate_operators": [
          query_generator_v2.Operator.NEQ,
          query_generator_v2.Operator.EQ,
          query_generator_v2.Operator.LEQ,
          query_generator_v2.Operator.GEQ,
          query_generator_v2.Operator.IS_NOT_NULL,
          query_generator_v2.Operator.IS_NULL,
      ],
      "allowed_predicate_column_types": [
          "int",
          "int64",
          "uint64",
          "int32",
          "uint32",
          "string",
          "float",
          "float64",
          "double",
      ],
      "eq_and_neq_unique_value_threshold_for_integer_columns": 1000,
      # join parameters
      "min_number_joins": 1,
      "max_nunmber_joins": 1,
      "always_create_the_maximum_number_of_joins": True,
      # grouping parameters
      "min_number_group_bys_cols": 0,
      "max_number_group_by_cols": 0,
      "allowed_grouping_column_types": ["int", "string"],
      "always_create_the_maximum_number_of_group_bys": False,
      "group_by_col_unqiue_vals_treshold": 5000,
      # aggregation parameters
      "min_aggregation_fns": 0,
      "max_aggregation_fns": 0,
      "always_create_the_maximum_number_of_aggregate_fns": False,
      "allowed_aggregate_functions": 0,
      "max_cols_per_agg": 0,
      "allowed_aggregation_column_types": [],
      # disallow lists
      "column_disallow_list_for_predicates_aggs_and_group_bys": [],
      "table_name_disallow_list_for_predicates_aggs_and_group_bys": [],
  }


# This functions creates the parameter specification for the multi-join queries
# used to create the CardBench multi-join queries.
def cardbench_multi_join_query_generation_parameters(
    projectname,
    datasetname,
    num_queries_to_generate,
    queries_output_file_name,
    partitioning_pred_per_table,
    use_partitioning_predicate = False,
):
  return {
      "project_name": projectname,
      "dataset_name": datasetname,
      "dataset_json_input_directory_path": DIRECTORY_PATH_JSON_FILES,
      "query_file_output_path": queries_output_file_name,
      "partitioning_predicate_per_table": partitioning_pred_per_table,
      "use_partitioning_predicate": use_partitioning_predicate,
      "recreate_query_file_if_exist": True,
      "seed": 0,
      # query number
      "num_queries_to_generate": num_queries_to_generate,
      # filter parameters
      "min_number_filter_predicates_per_table": 0,
      "max_number_filter_predicates_per_table": 2,
      "always_create_the_maximum_number_of_predicates": False,
      "allowed_predicate_operators": [
          query_generator_v2.Operator.NEQ,
          query_generator_v2.Operator.EQ,
          query_generator_v2.Operator.LEQ,
          query_generator_v2.Operator.GEQ,
          query_generator_v2.Operator.IS_NOT_NULL,
          query_generator_v2.Operator.IS_NULL,
      ],
      "allowed_predicate_column_types": [
          "int",
          "int64",
          "uint64",
          "int32",
          "uint32",
          "string",
          "float",
          "float64",
          "double",
      ],
      "eq_and_neq_unique_value_threshold_for_integer_columns": 10000,
      # join parameters
      "min_number_joins": 1,
      "max_nunmber_joins": 7,
      "always_create_the_maximum_number_of_joins": False,
      # grouping parameters
      "min_number_group_bys_cols": 0,
      "max_number_group_by_cols": 0,
      "allowed_grouping_column_types": ["int", "string"],
      "always_create_the_maximum_number_of_group_bys": False,
      "group_by_col_unqiue_vals_treshold": 50000,
      # aggregation parameters
      "min_aggregation_fns": 0,
      "max_aggregation_fns": 0,
      "always_create_the_maximum_number_of_aggregate_fns": False,
      "allowed_aggregate_functions": 0,
      "max_cols_per_agg": 0,
      "allowed_aggregation_column_types": [],
      # disallow lists
      "column_disallow_list_for_predicates_aggs_and_group_bys": [],
      "table_name_disallow_list_for_predicates_aggs_and_group_bys": [],
  }


# This functions creates the parameter specification for the aggregation queries
# used to create the CardBench aggregation queries.
def cardbench_aggregation_query_generation_parameters(
    projectname,
    datasetname,
    num_queries_to_generate,
    queries_output_file_name,
    partitioning_pred_per_table,
    use_partitioning_predicate = False,
):
  return {
      "project_name": projectname,
      "dataset_name": datasetname,
      "dataset_json_input_directory_path": DIRECTORY_PATH_JSON_FILES,
      "query_file_output_path": queries_output_file_name,
      "partitioning_predicate_per_table": partitioning_pred_per_table,
      "use_partitioning_predicate": use_partitioning_predicate,
      "recreate_query_file_if_exist": True,
      "seed": 0,
      # query number
      "num_queries_to_generate": num_queries_to_generate,
      # filter parameters
      "min_number_filter_predicates_per_table": 0,
      "max_number_filter_predicates_per_table": 2,
      "always_create_the_maximum_number_of_predicates": False,
      "allowed_predicate_operators": [
          query_generator_v2.Operator.NEQ,
          query_generator_v2.Operator.EQ,
          query_generator_v2.Operator.LEQ,
          query_generator_v2.Operator.GEQ,
          query_generator_v2.Operator.IS_NOT_NULL,
          query_generator_v2.Operator.IS_NULL,
      ],
      "allowed_predicate_column_types": [
          "int",
          "int64",
          "uint64",
          "int32",
          "uint32",
          "string",
          "float",
          "float64",
          "double",
      ],
      "eq_and_neq_unique_value_threshold_for_integer_columns": 1000,
      # join parameters
      "min_number_joins": 1,
      "max_nunmber_joins": 7,
      "always_create_the_maximum_number_of_joins": False,
      # grouping parameters
      "min_number_group_bys_cols": 1,
      "max_number_group_by_cols": 3,
      "allowed_grouping_column_types": [
          "int",
          "int64",
          "uint64",
          "int32",
          "uint32",
          "string",
          "date",
      ],
      "always_create_the_maximum_number_of_group_bys": False,
      "group_by_col_unqiue_vals_treshold": 5000,
      # aggregation parameters
      "min_aggregation_fns": 0,
      "max_aggregation_fns": 0,
      "always_create_the_maximum_number_of_aggregate_fns": False,
      "allowed_aggregate_functions": 0,
      "max_cols_per_agg": 0,
      "allowed_aggregation_column_types": [],
      # disallow lists
      "column_disallow_list_for_predicates_aggs_and_group_bys": [],
      "table_name_disallow_list_for_predicates_aggs_and_group_bys": [],
  }


# def innovation_day_query_generation_parameters(
#     projectname: str,
#     datasetname: str,
#     num_queries_to_generate: int,
#     queries_output_file_name: str,
#     partitioning_pred_per_table: str,
#     use_partitioning_predicate: bool = False,
# ):
#   return {
#       "project_name": projectname,
#       "dataset_name": datasetname,
#       "dataset_json_input_directory_path": DIRECTORY_PATH_JSON_FILES,
#       "query_file_output_path": queries_output_file_name,
#       "partitioning_predicate_per_table": partitioning_pred_per_table,
#       "use_partitioning_predicate": use_partitioning_predicate,
#       "allowed_predicate_operators": [
#           query_generator_v2.Operator.NEQ,
#           query_generator_v2.Operator.EQ,
#           query_generator_v2.Operator.LEQ,
#           query_generator_v2.Operator.GEQ,
#           query_generator_v2.Operator.IS_NOT_NULL,
#           query_generator_v2.Operator.IS_NULL,
#       ],
#       "allowed_aggregate_functions": [],
#       "num_queries_to_generate": num_queries_to_generate,
#       "max_nunmber_joins": 0,
#       "max_number_filter_predicates": 0,
#       "max_number_aggregates": 0,
#       "max_number_group_by": 6,
#       "max_cols_per_agg": 0,
#       "group_by_threshold": 0,
#       "int_neq_predicate_threshold": 1000000000000,
#       "seed": 0,
#       "complex_predicates": False,
#       "recreate_query_file_if_exist": True,
#       "always_create_the_maximum_number_of_joins": False,
#       "always_create_the_maximum_number_of_aggregates": False,
#       "always_create_the_maximum_number_of_predicates": True,
#       "always_create_the_maximum_number_of_group_bys": False,
#       "left_outer_join_ratio": 0,
#       "groupby_limit_probability": 0.0,
#       "groupby_having_probability": 0.0,
#       "exists_predicate_probability": 0.0,
#       "max_no_exists": 0,
#       "outer_groupby_probability": 0.0,
#       "min_number_joins": 0,
#       "min_number_predicates": 0,
#       "min_grouping_cols": 1,
#       "min_aggregation_cols": 0,  # not implemented in the query generator
#   }


# END_GOOGLE_INTERNAL


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
