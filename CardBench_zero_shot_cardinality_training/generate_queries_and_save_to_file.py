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

"""Generate sql queries and write them to a file."""

from collections.abc import Sequence
import traceback

from absl import app

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers
from CardBench_zero_shot_cardinality_training.generate_queries_library import generate_queries_and_save_to_file_helpers
from CardBench_zero_shot_cardinality_training.generate_queries_library import query_generator
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
template_query_generation_parameters = (
    generate_queries_and_save_to_file_helpers.template_query_generation_parameters
)
cardbench_single_table_query_generation_parameters = (
    generate_queries_and_save_to_file_helpers.cardbench_single_table_query_generation_parameters
)
cardbench_multi_join_query_generation_parameters = (
    generate_queries_and_save_to_file_helpers.cardbench_multi_join_query_generation_parameters
)
cardbench_binary_join_query_generation_parameters = (
    generate_queries_and_save_to_file_helpers.cardbench_binary_join_query_generation_parameters
)
get_next_workload_id = (
    generate_queries_and_save_to_file_helpers.get_next_workload_id
)
save_workload_information = (
    generate_queries_and_save_to_file_helpers.save_workload_information
)
get_table_partitioning_predicates = (
    generate_queries_and_save_to_file_helpers.get_table_partitioning_predicates
)
generate_queries_old = query_generator.generate_queries


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
    parameters = cardbench_single_table_query_generation_parameters(
        projectname,
        datasetname,
        35000,
        queries_output_file_name,
        partitioning_pred_per_table,
        False,
    )

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
    sql_table_prefix = f"{projectname}.{datasetname}."
    try:
      generate_queries(
          sql_table_prefix,
          quote_table_sql_string,
          dataset_name=parameters["dataset_name"],
          dataset_json_input_directory_path=parameters[
              "dataset_json_input_directory_path"
          ],
          query_file_output_path=parameters["query_file_output_path"],
          partitioning_predicate_per_table=parameters[
              "partitioning_predicate_per_table"
          ],
          use_partitioning_predicate=parameters["use_partitioning_predicate"],
          recreate_query_file_if_exists=parameters[
              "recreate_query_file_if_exist"
          ],
          seed=parameters["seed"],
          # query number
          num_queries_to_generate=parameters["num_queries_to_generate"],
          # filter parameters
          min_number_filter_predicates_per_table=parameters[
              "min_number_filter_predicates_per_table"
          ],
          max_number_filter_predicates_per_table=parameters[
              "max_number_filter_predicates_per_table"
          ],
          always_create_the_maximum_number_of_predicates=parameters[
              "always_create_the_maximum_number_of_predicates"
          ],
          allowed_predicate_operators=parameters["allowed_predicate_operators"],
          allowed_predicate_column_types=parameters[
              "allowed_predicate_column_types"
          ],
          eq_and_neq_unique_value_threshold_for_integer_columns=parameters[
              "eq_and_neq_unique_value_threshold_for_integer_columns"
          ],
          # join parameters
          min_number_joins=parameters["min_number_joins"],
          max_nunmber_joins=parameters["max_nunmber_joins"],
          always_create_the_maximum_number_of_joins=parameters[
              "always_create_the_maximum_number_of_joins"
          ],
          # grouping parameters
          min_number_group_bys_cols=parameters["min_number_group_bys_cols"],
          max_number_group_by_cols=parameters["max_number_group_by_cols"],
          allowed_grouping_column_types=parameters[
              "allowed_grouping_column_types"
          ],
          always_create_the_maximum_number_of_group_bys=parameters[
              "always_create_the_maximum_number_of_group_bys"
          ],
          group_by_col_unqiue_vals_treshold=parameters[
              "group_by_col_unqiue_vals_treshold"
          ],
          # aggregation parameters
          min_aggregation_fns=parameters["min_aggregation_fns"],
          max_aggregation_fns=parameters["max_aggregation_fns"],
          always_create_the_maximum_number_of_aggregate_fns=parameters[
              "always_create_the_maximum_number_of_aggregate_fns"
          ],
          allowed_aggregate_functions=parameters["allowed_aggregate_functions"],
          max_cols_per_agg=parameters["max_cols_per_agg"],
          allowed_aggregation_column_types=parameters[
              "allowed_aggregation_column_types"
          ],
          # disallow lists
          column_disallow_list_for_predicates_aggs_and_group_bys=parameters[
              "column_disallow_list_for_predicates_aggs_and_group_bys"
          ],
          table_name_disallow_list_for_predicates_aggs_and_group_bys=parameters[
              "table_name_disallow_list_for_predicates_aggs_and_group_bys"
          ],
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
