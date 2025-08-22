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

"""Validates query plan and query graph."""

from typing import Any

from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import generate_training_querygraphs_helpers

predicate_operator_dict = (
    generate_training_querygraphs_helpers.predicate_operator_dict
)


def validate_query_plan(
    annotated_query_plan, join_count, table_count
):
  """Validates query plan.

  Args:
    annotated_query_plan: annotated query plan.
    join_count: number of joins in the query.
    table_count: number of tables in the query.

  Raises:
    ValueError: if the query plan is invalid.
  """
  count_per_op_type = {
      "join": 0,
      "scan": 0,
      "column": 0,
      "table": 0,
      "predicate_operator": 0,
      "logical_predicate_operator": 0,
      "correlation": 0,
      "groupby": 0,
  }
  nodeid_to_type = {}
  logical_preds = []
  joinpreds = []
  joinlogpreds = []
  for n in annotated_query_plan["nodes"]:
    optype = n["nodetype"]
    count_per_op_type[optype] += 1
    nodeid_to_type[n["id"]] = optype
    if optype == "predicate_operator":
      if "t_is_join_pred_log" in n:
        joinlogpreds.append(n["id"])
      elif "t_is_join_pred" in n:
        joinpreds.append(n["id"])
      elif n["operator"] == predicate_operator_dict["and"]:
        logical_preds.append(n["id"])

  #### very basic plan stats check
  if count_per_op_type["join"] != join_count:
    raise ValueError("Failed validation: one join node per plan")
  if count_per_op_type["scan"] != table_count:
    raise ValueError("Failed validation: two scan nodes per plan")
  if count_per_op_type["table"] != table_count:
    raise ValueError("Failed validation: two table nodes per plan")
  if count_per_op_type["groupby"] > 1:
    raise ValueError("Failed validation: at most one groupby node per plan")

  ######## Extra checks on statistics and graph structure
  e_dict_index_by_from = {}
  e_dict_to_to_from = {}
  for e in annotated_query_plan["edges"]:
    ef = e["from"]
    et = e["to"]
    if ef not in nodeid_to_type:
      raise ValueError("node with id", str(ef), " not found")
    if et not in nodeid_to_type:
      raise ValueError("node with id", str(et), " not found")

    if ef not in e_dict_index_by_from:
      e_dict_index_by_from[ef] = []
    e_dict_index_by_from[ef].append(et)

    if et not in e_dict_to_to_from:
      e_dict_to_to_from[et] = []
    e_dict_to_to_from[et].append(ef)

  count_attrs = 0
  for n in annotated_query_plan["nodes"]:
    optype = n["nodetype"]
    if optype == "column":
      if "min_val" not in n:
        raise ValueError("min_val is required for column", str(n))
      if "max_val" not in n:
        raise ValueError("max_val is required for column", str(n))
      if "percentiles" not in n:
        raise ValueError("percentiles is required for column", str(n))
      if "percentiles_100" not in n:
        raise ValueError("percentiles_100 is required for column", str(n))
      if "column_type" not in n:
        raise ValueError("column_type is required for column", str(n))
      if "name" not in n:
        raise ValueError("name is required for column", str(n))
      if "num_unique" not in n:
        raise ValueError("num_unique is required for column", str(n))
      if "is_clust_attr" not in n:
        raise ValueError("is_clust_attr is required for column", str(n))
      if "is_part_attr" not in n:
        raise ValueError("is_part_attr is required for column", str(n))
      if "null_frac" not in n:
        raise ValueError("null_frac is required for column", str(n))
      if len(n["percentiles_100"]) != 101:
        raise ValueError("percentiles_100 is empty", str(n))

      if len(e_dict_index_by_from[n["id"]]) < 1:
        raise ValueError("at least one outgoing edge from a col", str(n))
      if len(e_dict_to_to_from[n["id"]]) != 1:
        raise ValueError("exactly one incoming edge to a col", str(n))

    if optype == "predicate_operator":
      if "estimated_selectivity" not in n:
        raise ValueError(
            "estimated_selectivity is required for predicate_operator", str(n)
        )
      if (
          "t_is_join_pred" not in n
          and not (
              n["operator"] == predicate_operator_dict["and"]
              or n["operator"] == predicate_operator_dict["or"]
          )
          and n["estimated_selectivity"] == -1
      ):
        raise ValueError(
            "estimated_selectivity must be <> -1 when op is not AND", str(n)
        )
      if len(n["offset"]) != 6:
        raise ValueError("offset is wrong", str(n))

      if len(e_dict_index_by_from[n["id"]]) != 1:
        raise ValueError("exactly one outgoing edge from a predop", str(n))
      if len(e_dict_to_to_from[n["id"]]) < 1 and "t_is_join_pred" not in n:
        raise ValueError("exactly one incoming edge to a predop", str(n))
      if len(e_dict_to_to_from[n["id"]]) < 2 and "t_is_join_pred" in n:
        raise ValueError("exactly 2 incoming edges to a join predop", str(n))

    if optype == "correlation":
      if n["correlation"] in [-10, -100]:
        raise ValueError("corr val should not be -10 or -100", str(n))

      if not (n["correlation"] > -1.01 and n["correlation"] < 1.01) and n[
          "correlation"
      ] not in [-20, -30, -40, -50]:
        raise ValueError("corr val invalid", str(n))

      if len(e_dict_index_by_from[n["id"]]) < 1:
        raise ValueError("at least one outgoing edge from a corr", str(n))
      if len(e_dict_to_to_from[n["id"]]) != 2:
        raise ValueError("exactly two incoming edge to a corr", str(n))

    if optype == "table":
      if n["id"] not in e_dict_index_by_from:
        raise ValueError("at least one outgoing edge from a table", str(n))
      if len(e_dict_index_by_from[n["id"]]) < 1:
        raise ValueError("at least one outgoing edge from a table", str(n))
    if optype == "column":
      count_attrs += 1
    if optype == "groupby":
      if len(e_dict_to_to_from[n["id"]]) < 1:
        raise ValueError("at least one incoming edge to a groupby", str(n))

  if count_attrs == 0:
    raise ValueError("too few attributes", str(count_attrs))


def validate_query_graph(graph_nodes):
  """Validates query graph.

  Args:
    graph_nodes: query graph.

  Raises:
    ValueError: if the query graph is invalid.
  """
  corr_values = graph_nodes["correlations"]["correlation"]
  corr_validities = graph_nodes["correlations"]["validity"]
  if not graph_nodes["attrs"]:
    raise ValueError("no attributes")

  for i in range(len(corr_values)):
    val = corr_values[i]
    validity = corr_validities[i]

    if (
        (val == -100 and validity != "not_initialized")
        or (val == -10 and validity != "missing")
        or (val == -20 and validity != "nan")
        or (val == -30 and validity != "null")
        or (val == -40 and validity != "none")
        or (val == -50 and validity != "invalidtypes")
        or (val >= -1.01 and val <= 1.01 and validity != "valid")
    ):
      raise ValueError(
          "Validity wrong", str(corr_values), str(corr_validities), str(i)
      )


def validate_query_plan_and_graph(
    annotated_query_plan,
    query_graph,
    query_sql_string,
):
  join_count = query_sql_string.count(" JOIN ")
  table_count = join_count + 1
  validate_query_plan(annotated_query_plan, join_count, table_count)
  validate_query_graph(query_graph["nodes"])
