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

"""Generate training graphs for a given workload id and query run id."""

from collections.abc import Sequence
import traceback
from typing import Any

from absl import app

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import convert_query_plan_to_graph
from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import convert_relational_operators_to_query_plan
from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import convert_sql_to_relational_operators
from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import create_sparse_deferred_graph_struct_object
from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import generate_training_querygraphs_helpers
from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import validate_query_plan_and_graph

QUERY_RUN_INFORMATION_TABLE = configuration.QUERY_RUN_INFORMATION_TABLE
WORKLOAD_DEFINITION_TABLE = configuration.WORKLOAD_DEFINITION_TABLE
DIRECTORY_TRAINING_QUERYGRAPH_OUTPUT = (
    configuration.DIRECTORY_TRAINING_QUERYGRAPH_OUTPUT
)
printif = generate_training_querygraphs_helpers.printif
DBType = database_connector.DBType
create_database_connection = database_connector.create_database_connection
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
write_sparse_deferred_graph_object_file = (
    create_sparse_deferred_graph_struct_object.write_sparse_deferred_graph_object_file
)
write_tf_graph_object_file = (
    generate_training_querygraphs_helpers.write_tf_graph_object_file
)
write_graph_schema_file = (
    generate_training_querygraphs_helpers.write_graph_schema_file
)
convert_sql_to_relational_operators = (
    convert_sql_to_relational_operators.convert_sql_to_relational_operators
)
convert_relational_operators_to_query_plan = (
    convert_relational_operators_to_query_plan.convert_relational_operators_to_query_plan
)
convert_query_plan_to_graph = (
    convert_query_plan_to_graph.convert_query_plan_to_graph
)
create_tf_graph_object = (
    generate_training_querygraphs_helpers.create_tf_graph_object
)
create_sparse_deferred_graph_struct_object = (
    create_sparse_deferred_graph_struct_object.create_sparse_deferred_graph_struct_object
)
validate_query_plan_and_graph = (
    validate_query_plan_and_graph.validate_query_plan_and_graph
)
find_unique_and_non_zero_cardinality_queries = (
    generate_training_querygraphs_helpers.find_unique_and_non_zero_cardinality_queries
)


def print_progress(
    processed_queries,
    queries_information,
    query_ids_to_save,
    zero_cardinality_queries,
    skipped_queries_with_no_operators,
):
  print(
      f"Progress: {processed_queries}/{len(queries_information)} queries"
      f" Failed: {processed_queries - len(query_ids_to_save)} Successful:"
      f" {len(query_ids_to_save)}  Zero-Cardinality:"
      f" {zero_cardinality_queries} Queries with no operators (skipped):"
      f" {skipped_queries_with_no_operators}."
  )


def load_dataset_name(
    workload_id,
    metadata_dbtype,
    metadata_dbclient,
):
  """Load the dataset name for a given workload id."""
  query = (
      "SELECT dataset_name FROM"
      f" `{WORKLOAD_DEFINITION_TABLE}` WHERE workload_id = {workload_id}"
  )
  try:
    queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
    dataset_name = get_query_result_first_row(metadata_dbtype, queryjob)[
        "dataset_name"
    ]
    return dataset_name
  except Exception as e:  # pylint: disable=broad-except
    print("Error in query:", query)
    print("Error in query:", str(e))


def load_queries_information(
    workload_id,
    query_run_id,
    metadata_dbtype,
    metadata_dbclient,
):
  """Load the queries information for a given workload id and query run id."""
  query_information = {}
  query = (
      "SELECT database_query_id, cardinality, query_string FROM"
      f" `{QUERY_RUN_INFORMATION_TABLE}` WHERE workload_id = {workload_id} and"
      f" query_run_id = {query_run_id}"
  )
  try:
    queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
    for row in queryjob:
      query_information[row["database_query_id"]] = {
          "query_string": row["query_string"],
          "cardinality": int(row["cardinality"]),
      }
  except Exception as e:  # pylint: disable=broad-except
    print("Error in query:", query)
    print(e)
  return query_information


def generate_training_querygraphs_and_save_to_file(argv):
  """Generate training graphs for a given workload id and query run id."""
  if len(argv) < 3:
    print(
        "No arguments were provided. Please specify the workload id and"
        " query run id."
    )
    return
  workload_id = int(argv[1])
  query_run_id = int(argv[2])
  print(
      f"Generating training query graphs for workload id: {workload_id} and"
      f" query run id: {query_run_id}"
  )

  dbs = {
      # used to stored the collected statistics
      "metadata_dbtype": DBType.BIGQUERY,
      "metadata_dbclient": create_database_connection(DBType.BIGQUERY),
  }

  queries_information = load_queries_information(
      workload_id,
      query_run_id,
      dbs["metadata_dbtype"],
      dbs["metadata_dbclient"],
  )
  dataset_name = load_dataset_name(
      workload_id,
      dbs["metadata_dbtype"],
      dbs["metadata_dbclient"],
  )

  # Cache dataset statistics to avoid loading from the metadata
  # database multiple times.
  query_statistics_caches = {
      "correlation_cache": {},
      "correlation_cache_prefetched": False,
      "column_info_cache": {},
      "table_stats_cache": {},
      "table_stats_cache_prefetched": False,
  }

  querygraph_file_path = f"{DIRECTORY_TRAINING_QUERYGRAPH_OUTPUT}/{dataset_name}_{query_run_id}.npz"
  # BEGIN_GOOGLE_INTERNAL
  tf_query_graphs_file_path = f"{DIRECTORY_TRAINING_QUERYGRAPH_OUTPUT}/{workload_id}_{query_run_id}.tfrecord"
  tf_query_graphs_schema_file_path = f"{DIRECTORY_TRAINING_QUERYGRAPH_OUTPUT}/{workload_id}_{query_run_id}_graph_schema.pbtxt"
  # END_GOOGLE_INTERNAL
  debug = False
  processed_queries = 0

  annotated_query_plans = {}
  sparse_deferred_graph_objects = {}
  # BEGIN_GOOGLE_INTERNAL
  tf_objects = {}
  # END_GOOGLE_INTERNAL
  query_ids_to_save = []
  zero_cardinality_queries = 0
  skipped_queries_with_no_operators = 0
  printif(debug, f"Processing {len(queries_information)} queries...")
  for database_query_id in queries_information:
    processed_queries += 1
    query_sql_string = "no query string"
    try:
      query_information = queries_information[database_query_id]
      query_sql_string = query_information["query_string"]
      printif(debug, query_sql_string)
      cardinality = query_information["cardinality"]
      if cardinality == 0:
        zero_cardinality_queries += 1
        continue
      top_level_query_information = {
          "query_string": query_sql_string,
          "cardinality": cardinality,
          "database_query_id": database_query_id,
          "workload_id": workload_id,
          "query_run_id": query_run_id,
      }
      # filter queries that do not join or have a where clause
      if (
          ("JOIN" not in query_sql_string)
          and ("WHERE" not in query_sql_string)
          and ("GROUP BY" not in query_sql_string)
      ):
        print("Query has no join or where clause: ", query_sql_string)
        continue

      printif(debug, "QUERY <><><><><<><><><><><><><><><> ")
      printif(debug, top_level_query_information["cardinality"])
      printif(debug, top_level_query_information["query_string"])
      # Parse sql into a plan. The plan is a graph of JOIN and SCAN operators.
      parsed_query = convert_sql_to_relational_operators(
          query_information, debug
      )
      printif(debug, "convert_sql_to_relational_operators successful")
      printif(debug, "PARSED QUERY <><><><><<><><><><><><><><><> ")
      for op in parsed_query["ops"]:
        printif(debug, op)

      # Convert relational operators to query plan annotated with statistics.
      # The query plan is a graph.
      annotated_query_plan = convert_relational_operators_to_query_plan(
          parsed_query,
          dbs["metadata_dbtype"],
          dbs["metadata_dbclient"],
          query_statistics_caches,
          debug,
      )
      if not annotated_query_plan:
        skipped_queries_with_no_operators += 1
        printif(debug, "convert_relational_operators_to_query_plan failed")
        continue
      printif(debug, "convert_relational_operators_to_query_plan successful")
      annotated_query_plans[database_query_id] = annotated_query_plan
      printif(debug, "ANNOTATED QUERY PLAN <><><><><<><><><><><><><><><> ")
      for node in annotated_query_plan["nodes"]:
        printif(debug, node)
      for edge in annotated_query_plan["edges"]:
        printif(debug, edge)
      printif(debug, "ANNOTATED QUERY PLAN <><><><><<><><><><><><><><><> ")

      # Convert the query plan into a generic graph representation.
      query_graph = {"nodes": None, "edges": None}
      query_graph["nodes"], query_graph["edges"] = convert_query_plan_to_graph(
          annotated_query_plan,
          debug,
      )
      validate_query_plan_and_graph(
          annotated_query_plan, query_graph, query_sql_string
      )
      printif(debug, "validate_query_plan_and_graph successful")
      # BEGIN_GOOGLE_INTERNAL
      tf_objects[database_query_id] = create_tf_graph_object(
          query_graph, top_level_query_information
      )
      # END_GOOGLE_INTERNAL
      # Convert the graph represenation to the sparse deferred graph
      # struct object
      sparse_deferred_graph_objects[database_query_id] = (
          create_sparse_deferred_graph_struct_object(
              query_graph, top_level_query_information
          )
      )
    except Exception as e:  # pylint: disable=broad-except
      print("\n\n <><><><><><><><><><><><> ")
      print("Error in generating training query graph:", str(e))
      print(traceback.format_exc())
      print(query_sql_string)
      print("<><><><><><><><><><><><>\n\n")
    else:
      query_ids_to_save.append(database_query_id)
      print_progress(
          processed_queries,
          queries_information,
          query_ids_to_save,
          zero_cardinality_queries,
          skipped_queries_with_no_operators,
      )
      printif(debug, "\n\n")

  database_query_ids_unique_queries, _ = (
      find_unique_and_non_zero_cardinality_queries(
          annotated_query_plans, queries_information, query_ids_to_save
      )
  )
  print_progress(
      processed_queries,
      queries_information,
      query_ids_to_save,
      zero_cardinality_queries,
      skipped_queries_with_no_operators,
  )
  print(f"Zero cardinality queries: {zero_cardinality_queries}")
  print(
      "Unique non-zero cardinality queries:"
      f" {len(database_query_ids_unique_queries)}"
  )

  if not database_query_ids_unique_queries:
    print("No unique non-zero cardinality queries found.")
    print("No output files will be generated.")
    return

  # BEGIN_GOOGLE_INTERNAL
  tf_objects_unique_queries = []
  for database_query_id in database_query_ids_unique_queries:
    tf_objects_unique_queries.append(tf_objects[database_query_id])
  write_tf_graph_object_file(
      tf_objects_unique_queries, tf_query_graphs_file_path
  )
  write_graph_schema_file(
      tf_query_graphs_schema_file_path, tf_objects_unique_queries
  )
  # END_GOOGLE_INTERNAL

  sparse_deferred_graph_objects_unique_queries = []
  for database_query_id in database_query_ids_unique_queries:
    sparse_deferred_graph_objects_unique_queries.append(
        sparse_deferred_graph_objects[database_query_id]
    )
  write_sparse_deferred_graph_object_file(
      sparse_deferred_graph_objects_unique_queries, querygraph_file_path
  )


if __name__ == "__main__":
  app.run(generate_training_querygraphs_and_save_to_file)
