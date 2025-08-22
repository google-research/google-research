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

"""Convert a graph to a sparse deferred graph struct object."""

import os
from typing import Any
import tensorflow as tf
from sparse_deferred.structs import graph_struct

InMemoryDB = graph_struct.InMemoryDB
GraphStruct = graph_struct.GraphStruct

open_file = open
file_exists = os.path.exists
remove_file = os.remove


def create_sparse_deferred_graph_struct_object(
    guery_graph, top_level_query_information
):
  """Converts the graph into a graph struct object."""
  query_graph_nodes = guery_graph["nodes"]
  query_graph_edges = guery_graph["edges"]

  graph = GraphStruct.new(
      nodes={
          "g": {
              "cardinality": tf.constant(
                  [top_level_query_information["cardinality"]], dtype=tf.int64
              ),
              "database_query_id": tf.constant(
                  [top_level_query_information["database_query_id"]],
                  dtype=tf.string,
              ),
              "query_run_id": tf.constant(
                  [top_level_query_information["query_run_id"]], dtype=tf.int32
              ),
              "workload_id": tf.constant(
                  [top_level_query_information["workload_id"]], dtype=tf.int32
              ),
              "query_string": tf.constant(
                  [top_level_query_information["query_string"]], dtype=tf.string
              ),
          },
          "tables": {
              "rows": tf.constant(
                  query_graph_nodes["tables"]["rows"], tf.int64
              ),
              "name": tf.constant(
                  query_graph_nodes["tables"]["name"], tf.string
              ),
          },
          "attributes": {
              "null_frac": tf.constant(
                  query_graph_nodes["attrs"]["null_frac"], tf.float32
              ),
              "num_unique": tf.constant(
                  query_graph_nodes["attrs"]["num_unique"], tf.int64
              ),
              "data_type": tf.constant(
                  query_graph_nodes["attrs"]["data_type"], tf.string
              ),
              "name": tf.constant(
                  query_graph_nodes["attrs"]["name"], tf.string
              ),
              "percentiles_100_numeric": tf.reshape(
                  tf.constant(
                      query_graph_nodes["attrs"]["percentiles_100_numeric"],
                      tf.float32,
                  ),
                  (-1, 101),
              ),
              "percentiles_100_string": tf.reshape(
                  tf.constant(
                      query_graph_nodes["attrs"]["percentiles_100_string"],
                      tf.string,
                  ),
                  (-1, 101),
              ),
              "min_numeric": tf.constant(
                  query_graph_nodes["attrs"]["min_numeric"], tf.float32
              ),
              "max_numeric": tf.constant(
                  query_graph_nodes["attrs"]["max_numeric"], tf.float32
              ),
              "min_string": tf.constant(
                  query_graph_nodes["attrs"]["min_string"], tf.string
              ),
              "max_string": tf.constant(
                  query_graph_nodes["attrs"]["max_string"], tf.string
              ),
          },
          "predicates": {
              "predicate_operator": tf.constant(
                  query_graph_nodes["predicates"]["operator"], tf.int32
              ),
              "estimated_selectivity": tf.constant(
                  query_graph_nodes["predicates"]["estimated_selectivity"],
                  tf.float32,
              ),
              "offset": tf.reshape(
                  tf.constant(
                      query_graph_nodes["predicates"]["offset"],
                      tf.float32,
                  ),
                  (-1, 6),
              ),
              "constant": tf.constant(
                  query_graph_nodes["predicates"]["constant"], tf.string
              ),
              "encoded_constant": tf.constant(
                  query_graph_nodes["predicates"]["encoded_constant"], tf.int64
              ),
          },
          "ops": {
              "operator": tf.constant(
                  query_graph_nodes["ops"]["type"], tf.string
              ),
          },
          "correlations": {
              "type": tf.constant(
                  query_graph_nodes["correlations"]["type"], tf.string
              ),
              "correlation": tf.constant(
                  query_graph_nodes["correlations"]["correlation"], tf.float32
              ),
              "validity": tf.constant(
                  query_graph_nodes["correlations"]["validity"], tf.string
              ),
          },
      },
      schema={
          "table_to_attr": ("tables", "attributes"),
          "attr_to_pred": ("attributes", "predicates"),
          "pred_to_pred": ("predicates", "predicates"),
          "attr_to_op": ("attributes", "ops"),
          "op_to_op": ("ops", "ops"),
          "pred_to_op": ("predicates", "ops"),
          "attr_to_corr": ("attributes", "correlations"),
          "corr_to_pred": ("correlations", "predicates"),
          "corr_to_op": ("correlations", "ops"),
      },
      edges={
          # edge indices, edge features
          "table_to_attr": (
              (
                  tf.constant(query_graph_edges["table_to_attr"][0], tf.int32),
                  tf.constant(query_graph_edges["table_to_attr"][1], tf.int32),
              ),
              {},
          ),
          "attr_to_pred": (
              (
                  tf.constant(query_graph_edges["attr_to_pred"][0], tf.int32),
                  tf.constant(query_graph_edges["attr_to_pred"][1], tf.int32),
              ),
              {},
          ),
          "pred_to_pred": (
              (
                  tf.constant(query_graph_edges["pred_to_pred"][0], tf.int32),
                  tf.constant(query_graph_edges["pred_to_pred"][1], tf.int32),
              ),
              {},
          ),
          "attr_to_op": (
              (
                  tf.constant(query_graph_edges["attr_to_op"][0], tf.int32),
                  tf.constant(query_graph_edges["attr_to_op"][1], tf.int32),
              ),
              {},
          ),
          "op_to_op": (
              (
                  tf.constant(query_graph_edges["op_to_op"][0], tf.int32),
                  tf.constant(query_graph_edges["op_to_op"][1], tf.int32),
              ),
              {},
          ),
          "pred_to_op": (
              (
                  tf.constant(query_graph_edges["pred_to_op"][0], tf.int32),
                  tf.constant(query_graph_edges["pred_to_op"][1], tf.int32),
              ),
              {},
          ),
          "attr_to_corr": (
              (
                  tf.constant(
                      query_graph_edges["attr_to_correlation"][0], tf.int32
                  ),
                  tf.constant(
                      query_graph_edges["attr_to_correlation"][1], tf.int32
                  ),
              ),
              {},
          ),
          "corr_to_pred": (
              (
                  tf.constant(
                      query_graph_edges["correlation_to_pred"][0], tf.int32
                  ),
                  tf.constant(
                      query_graph_edges["correlation_to_pred"][1], tf.int32
                  ),
              ),
              {},
          ),
          "corr_to_op": (
              (
                  tf.constant(
                      query_graph_edges["corr_to_op"][0], tf.int32
                  ),
                  tf.constant(
                      query_graph_edges["corr_to_op"][1], tf.int32
                  ),
              ),
              {},
          ),
      },
  )
  return graph


def write_sparse_deferred_graph_object_file(
    sparse_deferred_graph_objects_unique_queries,
    querygraph_file_path,
):
  """Writes the sparse deferred graph object file."""
  if file_exists(querygraph_file_path):
    print("opensource_graphs.npz file exists -- recreating")
    remove_file(querygraph_file_path)

  print("Saving graphs to: ", querygraph_file_path)

  db = InMemoryDB()
  for g in sparse_deferred_graph_objects_unique_queries:
    db.add(g)
  db.finalize()
  db.save(querygraph_file_path)
