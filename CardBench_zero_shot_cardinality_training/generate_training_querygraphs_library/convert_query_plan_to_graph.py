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

"""Convert the annotated query plan to a graph."""

from typing import Any

from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import generate_training_querygraphs_helpers

printif = generate_training_querygraphs_helpers.printif


def add_table_node(
    node,
    graph_nodes,
    annotated_query_planid_to_node_per_type_pods,
):
  graph_nodes["tables"]["ids"].append(node["id"])
  graph_nodes["tables"]["rows"].append(node["rows"])
  graph_nodes["tables"]["name"].append(node["name"])
  annotated_query_planid_to_node_per_type_pods[node["id"]] = (
      len(graph_nodes["tables"]["ids"]) - 1
  )


def add_column_node(
    node,
    graph_nodes,
    annotated_query_planid_to_node_per_type_pods,
):
  """Add column node to the graph."""
  graph_nodes["attrs"]["ids"].append(node["id"])
  graph_nodes["attrs"]["name"].append(node["name"])
  graph_nodes["attrs"]["is_clust_attr"].append(node["is_clust_attr"])
  graph_nodes["attrs"]["is_part_attr"].append(node["is_part_attr"])
  graph_nodes["attrs"]["null_frac"].append(node["null_frac"])
  graph_nodes["attrs"]["num_unique"].append(node["num_unique"])
  graph_nodes["attrs"]["data_type"].append(node["column_type"])
  if node["column_type"] in [
      "INT64",
      "NUMERIC",
      "BIGNUMERIC",
      "FLOAT64",
      "DECIMAL",
      "BIGDECIMAL",
  ]:
    graph_nodes["attrs"]["min_numeric"].append(float(node["min_val"]))
    graph_nodes["attrs"]["max_numeric"].append(float(node["max_val"]))
  else:
    graph_nodes["attrs"]["min_numeric"].append(float(-1))
    graph_nodes["attrs"]["max_numeric"].append(float(-1))

  graph_nodes["attrs"]["min_string"].append(str(node["min_val"]))
  graph_nodes["attrs"]["max_string"].append(str(node["max_val"]))

  # handle percentiles of variable size
  padded_percentiles = [-1] * 65
  if node["column_type"] in [
      "INT64",
      "NUMERIC",
      "BIGNUMERIC",
      "FLOAT64",
      "DECIMAL",
      "BIGDECIMAL",
  ]:
    if node["percentiles"]:
      for i in range(len(node["percentiles"])):
        padded_percentiles[i] = float(node["percentiles"][i])

  graph_nodes["attrs"]["percentiles"].append(padded_percentiles)

  # handle percentiles 100
  perc100_float = []
  perc100_string = []

  perc1000_float = []
  perc1000_string = []

  perc100_encoded = []

  if (
      node["column_type"]
      in [
          "INT64",
          "NUMERIC",
          "BIGNUMERIC",
          "FLOAT64",
          "DECIMAL",
          "BIGDECIMAL",
      ]
      and node["percentiles_100"]
  ):
    for i in node["percentiles_100"]:
      perc100_float.append(float(i))
  else:
    for _ in range(101):
      perc100_float.append(float(-1.0))
    for _ in range(1001):
      perc1000_float.append(float(-1.0))

  if node["percentiles_100"]:
    for i in node["percentiles_100"]:
      perc100_string.append(str(i))
  else:
    for _ in range(101):
      perc100_string.append(str(-1.0))
    for _ in range(1001):
      perc1000_string.append(str(-1.0))

  if "percentiles_100_encoded" in node and node["percentiles_100_encoded"]:
    for i in node["percentiles_100_encoded"]:
      perc100_encoded.append(i)
  else:
    perc100_encoded.append(-1)

  graph_nodes["attrs"]["percentiles_100_numeric"].append(perc100_float)
  graph_nodes["attrs"]["percentiles_100_string"].append(perc100_string)
  graph_nodes["attrs"]["encoded_percentiles"].append(perc100_encoded)

  annotated_query_planid_to_node_per_type_pods[node["id"]] = (
      len(graph_nodes["attrs"]["ids"]) - 1
  )


def add_predicate_operator_node(
    node,
    graph_nodes,
    annotated_query_planid_to_node_per_type_pods,
):
  """Add predicate operator node to the graph_nodes object.

  Args:
    node: the node to be added
    graph_nodes: the graph_nodes object
    annotated_query_planid_to_node_per_type_pods: a map from node id to the
      index of the node in the graph_nodes object
  """
  graph_nodes["predicates"]["ids"].append(node["id"])
  graph_nodes["predicates"]["operator"].append(node["operator"])
  graph_nodes["predicates"]["estimated_selectivity"].append(
      node["estimated_selectivity"]
  )
  if "encoded_constant" in node:
    graph_nodes["predicates"]["encoded_constant"].append(
        node["encoded_constant"]
    )
  else:
    graph_nodes["predicates"]["encoded_constant"].append(-1)

  if "offset" in node:
    graph_nodes["predicates"]["offset"].append(node["offset"])
  else:
    graph_nodes["predicates"]["offset"].append([-1.0] * 6)
  if "constant" in node:
    graph_nodes["predicates"]["constant"].append(node["constant"])
  else:
    graph_nodes["predicates"]["constant"].append("")

  annotated_query_planid_to_node_per_type_pods[node["id"]] = (
      len(graph_nodes["predicates"]["ids"]) - 1
  )


def add_operator_node(
    node,
    graph_nodes,
    annotated_query_planid_to_node_per_type_pods,
):
  """Add scan/join/groupy node to the graph_nodes object.

  Args:
    node: the node to be added
    graph_nodes: the graph_nodes object
    annotated_query_planid_to_node_per_type_pods: a map from node id to the
      index of the node in the graph_nodes object
  """
  graph_nodes["ops"]["ids"].append(node[0])
  graph_nodes["ops"]["type"].append(node[1])
  annotated_query_planid_to_node_per_type_pods[node[0]] = (
      len(graph_nodes["ops"]["ids"]) - 1
  )


def add_correlation_node(
    node,
    graph_nodes,
    annotated_query_planid_to_node_per_type_pods,
):
  """Adds correlation node to the graph."""
  invalid_correlation_value_to_reason = {
      # this is an error of the graph creation, should not appear in graphs
      -100: "not_initialized",
      -10: "missing",  # this is an error of data collection
      -20: "nan",  # this is valid value of the bq pearson function
      -30: "null",  # this is valid value of the bq pearson function
      -40: "none",  # this is valid value of the bq pearson function
      # this singifies that the types of the columns are not supported
      # by the bq pearson function
      -50: "invalidtypes",
  }

  graph_nodes["correlations"]["ids"].append(node["id"])
  graph_nodes["correlations"]["type"].append(node["type"])
  graph_nodes["correlations"]["correlation"].append(node["correlation"])

  validity_feature = "valid"
  if node["correlation"] in invalid_correlation_value_to_reason:
    validity_feature = invalid_correlation_value_to_reason[node["correlation"]]
  graph_nodes["correlations"]["validity"].append(validity_feature)

  annotated_query_planid_to_node_per_type_pods[node["id"]] = (
      len(graph_nodes["correlations"]["ids"]) - 1
  )


def print_graph(
    graph_nodes, graph_edges, debug
):
  """Prints the graph."""
  printif(debug, "")
  printif(debug, "")
  printif(debug, "tables")
  printif(debug, "\t" + str(graph_nodes["tables"]["ids"]))
  printif(debug, "\t" + str(graph_nodes["tables"]["rows"]))
  printif(debug, "\t" + str(graph_nodes["tables"]["name"]))
  printif(debug, "attrs")
  printif(debug, "\t" + str(graph_nodes["attrs"]["ids"]))
  printif(debug, "\t" + str(graph_nodes["attrs"]["name"]))
  printif(debug, "\t" + str(graph_nodes["attrs"]["is_clust_attr"]))
  printif(debug, "\t" + str(graph_nodes["attrs"]["is_part_attr"]))
  printif(debug, "\t" + str(graph_nodes["attrs"]["null_frac"]))
  printif(debug, "\t" + str(graph_nodes["attrs"]["num_unique"]))
  printif(debug, "\t" + str(graph_nodes["attrs"]["data_type"]))
  printif(debug, "\t" + str(graph_nodes["attrs"]["percentiles"]))
  printif(debug, "\t" + str(graph_nodes["attrs"]["percentiles_100_string"]))
  printif(debug, "\t" + str(graph_nodes["attrs"]["percentiles_100_numeric"]))
  printif(debug, "predicates")
  printif(debug, "\t" + str(graph_nodes["predicates"]["ids"]))
  printif(debug, "\t" + str(graph_nodes["predicates"]["operator"]))
  printif(debug, "\t" + str(graph_nodes["predicates"]["estimated_selectivity"]))
  printif(debug, "\t" + str(graph_nodes["predicates"]["offset"]))
  printif(debug, "\t" + str(graph_nodes["predicates"]["constant"]))
  printif(debug, "ops")
  printif(debug, "\t" + str(graph_nodes["ops"]["ids"]))
  printif(debug, "\t" + str(graph_nodes["ops"]["type"]))

  printif(debug, "")
  printif(debug, "")
  printif(debug, "table_to_attr " + str(graph_edges["table_to_attr"]))
  printif(debug, "attr_to_pred " + str(graph_edges["attr_to_pred"]))
  printif(debug, "pred_to_pred " + str(graph_edges["pred_to_pred"]))
  printif(debug, "attr_to_op " + str(graph_edges["attr_to_op"]))
  printif(debug, "op_to_op " + str(graph_edges["op_to_op"]))
  printif(debug, "pred_to_op " + str(graph_edges["pred_to_op"]))
  printif(debug, "attrs_to_corr " + str(graph_edges["attr_to_correlation"]))
  printif(debug, "corr_to_op " + str(graph_edges["corr_to_op"]))
  printif(
      debug,
      "corr_to_predicates " + str(graph_edges["correlation_to_pred"]),
  )


def add_graph_edges(
    annotated_query_plan,
    annotated_query_planid_to_node_per_type_pods,
    id_to_type,
    graph_edges,
):
  """Add edges to the graph.

  Args:
    annotated_query_plan: the annotated query plan
    annotated_query_planid_to_node_per_type_pods: a map from node id to the
      index of the node in the graph_nodes object
    id_to_type: a map from node id to the type of the node
    graph_edges: the graph_edges object
  """
  # add edges, each edge type is handled separately
  for e in annotated_query_plan["edges"]:
    f = e["from"]
    t = e["to"]
    ft = id_to_type[f]
    et = id_to_type[t]
    fnew = annotated_query_planid_to_node_per_type_pods[f]
    tnew = annotated_query_planid_to_node_per_type_pods[t]
    if ft == "table" and et == "column":
      edge_type = "table_to_attr"
    elif ft == "column" and et == "predicate_operator":
      edge_type = "attr_to_pred"
    elif ft == "predicate_operator" and et == "predicate_operator":
      edge_type = "pred_to_pred"
    elif ft == "column" and et in ["scan", "join", "groupby"]:
      edge_type = "attr_to_op"
    elif ft in ["scan", "join"] and et in ["scan", "join", "groupby"]:
      edge_type = "op_to_op"
    elif ft == "predicate_operator" and et in ["scan", "join"]:
      edge_type = "pred_to_op"
    elif ft == "correlation" and et == "predicate_operator":
      edge_type = "correlation_to_pred"
    elif ft == "column" and et == "correlation":
      edge_type = "attr_to_correlation"
    elif ft == "correlation" and et in ["scan", "join", "groupby"]:
      edge_type = "corr_to_op"
    else:
      raise ValueError("edge_type not handled", ft, et)

    graph_edges[edge_type][0].append(fnew)
    graph_edges[edge_type][1].append(tnew)


# Given a annotated_query_plan convert it into the tf graph spec format


def convert_query_plan_to_graph(
    annotated_query_plan, debug
):
  """Helper data structures to convert the plan to the graph format."""
  graph_nodes = {
      "tables": {"ids": [], "rows": [], "name": []},
      "attrs": {
          "ids": [],
          "name": [],
          "is_clust_attr": [],  # int feature 0,1
          "is_part_attr": [],  # int feature 0,1
          "null_frac": [],  # float feature 0 to 1
          "num_unique": [],  # int
          "data_type": [],  # string
          "percentiles": [],
          "percentiles_100_numeric": [],  # float,
          "percentiles_100_string": [],  # string,
          "encoded_percentiles": [],  # int
          "min_numeric": [],
          "max_numeric": [],
          "max_string": [],
          "min_string": [],
      },
      "predicates": {
          "ids": [],
          "operator": [],  # string =, >, <, >=, <=, and, or, between
          "estimated_selectivity": [],  # float 0-1
          "offset": [],
          "constant": [],  # string
          "encoded_constant": [],  # int
      },
      "ops": {
          "ids": [],
          "type": [],  # string scan, join
      },
      "correlations": {
          "ids": [],  # int
          "type": [],  # string
          "correlation": [],  # float
          "validity": [],  # string
      },
  }

  graph_edges = {
      "table_to_attr": [[], [], "tables", "attrs"],
      "attr_to_pred": [[], [], "attrs", "predicates"],
      "pred_to_pred": [[], [], "predicates", "predicates"],
      "table_to_op": [[], [], "tables", "ops"],
      "attr_to_op": [[], [], "attrs", "ops"],
      "op_to_op": [[], [], "ops", "ops"],
      "pred_to_op": [[], [], "predicates", "ops"],
      "attr_to_correlation": [[], [], "attrs", "correlations"],
      "correlation_to_pred": [[], [], "correlations", "predicates"],
      "corr_to_op": [[], [], "correlations", "ops"],
  }

  ### helpers for the conversion
  id_to_type = {}
  annotated_query_planid_to_node_per_type_pods = {}

  groupbyops = []
  joinops = []
  scanops = []

  ## parse the annotated_query_plan and covert to a graph_nodes object
  for node in annotated_query_plan["nodes"]:
    id_to_type[node["id"]] = node["nodetype"]
    if node["nodetype"] == "table":
      add_table_node(
          node, graph_nodes, annotated_query_planid_to_node_per_type_pods
      )
    elif node["nodetype"] == "column":
      add_column_node(
          node, graph_nodes, annotated_query_planid_to_node_per_type_pods
      )
    elif node["nodetype"] == "predicate_operator":
      add_predicate_operator_node(
          node, graph_nodes, annotated_query_planid_to_node_per_type_pods
      )
    elif node["nodetype"] in ["scan", "join", "groupby"]:
      # processing of joins and scans is defferred
      if "join" == node["nodetype"]:
        joinops.append([node["id"], node["nodetype"]])
      elif "scan" == node["nodetype"]:
        scanops.append([node["id"], node["nodetype"]])
      elif "groupby" == node["nodetype"]:
        groupbyops.append([node["id"], node["nodetype"]])
      else:
        raise ValueError("nodetype not handled", node["nodetype"])
    elif node["nodetype"] == "correlation":
      add_correlation_node(
          node, graph_nodes, annotated_query_planid_to_node_per_type_pods
      )

  # add groupby, join and scan nodes -- the reason is that we want to
  # add them ordered
  # first the join node and then the scan nodes
  for n in groupbyops:
    add_operator_node(
        n, graph_nodes, annotated_query_planid_to_node_per_type_pods
    )
  # ordered in reverse to help the TFGNN API
  joinops.reverse()
  for n in joinops:
    add_operator_node(
        n, graph_nodes, annotated_query_planid_to_node_per_type_pods
    )
  for n in scanops:
    add_operator_node(
        n, graph_nodes, annotated_query_planid_to_node_per_type_pods
    )

  add_graph_edges(
      annotated_query_plan,
      annotated_query_planid_to_node_per_type_pods,
      id_to_type,
      graph_edges,
  )

  print_graph(graph_nodes, graph_edges, debug)

  return graph_nodes, graph_edges
