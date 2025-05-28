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

"""Helper functions for generate_training_querygraphs.py."""

import os
from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector


types_to_tables = configuration.TYPES_TO_TABLES
DBType = database_connector.DBType
create_database_connection = database_connector.create_database_connection
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_INFO_TABLE = configuration.COLUMNS_INFO_TABLE
COLUMNS_STATS_TABLE = configuration.COLUMNS_STATS_TABLE
COLUMNS_HISTOGRAM_TABLE = configuration.COLUMNS_HISTOGRAM_TABLE
CORRELATION_TABLE = configuration.CORRELATION_TABLE

open_file = open
file_exists = os.path.exists
remove_file = os.remove


def find_unique_and_non_zero_cardinality_queries(
    annotated_query_plans,
    queries_information,
    query_ids_to_save,
):
  """Finds the non duplicate and non zero cardinality queries."""
  plans_str = set()
  plans_str_no_constants = set()
  count_non_zero_plans = 0
  # find all the qier
  unique_query_plans = []
  unique_query_plans_without_constants = []

  for database_query_id in query_ids_to_save:
    query_cardinality = queries_information[database_query_id]["cardinality"]
    annotated_query_plan = annotated_query_plans[database_query_id]
    if query_cardinality == 0:
      continue
    count_non_zero_plans += 1

    table_preds = []
    table_preds_no_constant = []

    tables = annotated_query_plan["dup_tables"]
    join_preds = annotated_query_plan["dup_join_preds"]
    group_by_cols = annotated_query_plan["dup_groupby_cols"]

    for p in annotated_query_plan["dup_scan_preds"]:
      s1 = p[0] + "_-_" + str(p[1])
      s2 = s1 + "_" + str(p[2])
      table_preds_no_constant.append(s1)
      table_preds.append(s2)

    tables.sort()
    join_preds.sort()
    table_preds.sort()
    table_preds_no_constant.sort()
    group_by_cols.sort()

    p_string = ""
    p_no_constants = ""
    for t in tables:
      p_string += str(t)
      p_no_constants += str(t)
    for p in table_preds:
      p_string += p
    for p in table_preds_no_constant:
      p_no_constants += p
    for p in join_preds:
      p_string += p
      p_no_constants += p
    for p in group_by_cols:
      p_string += p
      p_no_constants += p

    if p_string not in plans_str:
      # finds the unique query plans
      unique_query_plans.append(database_query_id)

    if p_no_constants not in plans_str_no_constants:
      # finds the unique query plans without considering constants
      unique_query_plans_without_constants.append(database_query_id)

    plans_str.add(p_string)
    plans_str_no_constants.add(p_no_constants)

  return unique_query_plans, unique_query_plans_without_constants


def printif(flag, msg):
  if flag:
    print(msg)


predicate_operator_dict_operators_in_text = {
    "and": 1,
    "or": 2,
    "greater_or_equal": 3,
    "less_or_equal": 4,
    "equal": 5,
    "not_equal": 6,
    "greater": 7,
    "less": 8,
    "between": 9,
    "IS": 10,
    "ISNOT": 11,
}

predicate_operator_dict = {
    "and": 1,
    "or": 2,
    ">=": 3,
    "<=": 4,
    "=": 5,
    "!=": 6,
    ">": 7,
    "<": 8,
    "between": 9,
    "is": 10,
    "isnot": 11,
}


def print_queryplan_no_percentiles(
    queryplan, debug
):
  """Prints the query plan without metadata."""
  printif(debug, "<><><><><><><>")
  for n in queryplan["nodes"]:
    exclude_keys = [
        "percentiles",
        "percentiles_100",
    ]
    new_n = {k: n[k] for k in set(list(n.keys())) - set(exclude_keys)}
    printif(debug, str(n["id"]) + str(new_n))
  for e in queryplan["edges"]:
    printif(debug, str(e))
  printif(debug, "<><><><><><><>")


def prefetch_correlations(
    datasetpath,
    metadata_dbclient,
    metadata_dbtype,
    queryplan,
):
  """Prefetch the correlations for a given dataset, cache the results to avoid running this query again."""

  if queryplan["query_statistics_caches"]["correlation_cache_prefetched"]:
    return
  projectname = datasetpath.split(".")[0]
  datasetname = datasetpath.split(".")[1]
  query = (
      " select  distinct dataset_name,"
      " table_name_a,column_name_a,column_name_b,pearson_correlation from"
      f" `{CORRELATION_TABLE}` WHERE"
      f" project_name='{projectname}' AND dataset_name='{datasetname}'"
  )
  try:
    queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
    for row in queryjob:
      dataset_name = row["dataset_name"]
      tbl_name = row["table_name_a"]
      col_a = row["column_name_a"]
      col_b = row["column_name_b"]
      corr = row["pearson_correlation"]
      key = dataset_name + "." + tbl_name + "." + col_a + "." + col_b
      queryplan["query_statistics_caches"]["correlation_cache"][key] = corr
  except Exception as e:  # pylint: disable=broad-except
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + str(e))
  queryplan["query_statistics_caches"]["correlation_cache_prefetched"] = True


# correlation is:
# -100 when the graph node is initialized
# -10 if the correlation does exist in the BQ table
# -20 if the pearson bq function returned nan
# -30 if the pearson bq function returned null
# -40 if the pearson bq function returned none
# -50 if the column types do not allow correlation calculation


def get_correlation(
    table_path,
    col_a,
    col_b,
    col_a_type,
    col_b_type,
    correlation_cache,
):
  """Get the correlation between two columns in a table."""
  allowed_types = [
      "INT64",
      "INT32",
      "NUMERIC",
      "FLOAT64",
      "BIGNUMERIC",
      "DECIMAL",
      "BIGDECIMAL",
  ]
  if col_a_type not in allowed_types or col_b_type not in allowed_types:
    return -50

  dataset_name = table_path.split(".")[1]
  table_name = table_path.split(".")[2]
  key = dataset_name + "." + table_name

  if col_a < col_b:
    key = key + "." + col_a + "." + col_b
  else:
    key = key + "." + col_b + "." + col_a

  result_corr = -10
  if key in correlation_cache:
    result_corr = correlation_cache[key]
  else:
    print("info for missing correlation:", key)
  return result_corr


def get_table_rows(
    metadata_dbtype,
    metadata_dbclient,
    table_path,
    queryplan,
    prefetch = True,
):
  """Get the number of rows in a table, cache the results to avoid running this query again."""
  table_stas_cache = queryplan["query_statistics_caches"]["table_stats_cache"]
  if (
      prefetch
      and not queryplan["query_statistics_caches"][
          "table_stats_cache_prefetched"
      ]
  ):
    table_ids = table_path.split(".")
    query = (
        f" SELECT table_name, row_count FROM `{TABLES_INFO_TABLE}`  WHERE"
        f" project_name='{table_ids[0]}' AND dataset_name='{table_ids[1]}'"
    )
    try:
      queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
      for row in queryjob:
        table_path_temp = f"{table_ids[0]}.{table_ids[1]}.{row['table_name']}"
        table_stas_cache[table_path_temp] = row["row_count"]
        queryplan["query_statistics_caches"][
            "table_stats_cache_prefetched"
        ] = True
    except Exception as e:  # pylint: disable=broad-except
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + str(e))

  if table_path not in table_stas_cache:
    table_ids = table_path.split(".")
    query = (
        " SELECT row_count FROM"
        f" `{TABLES_INFO_TABLE}`  WHERE"
        f" project_name='{table_ids[0]}' AND dataset_name='{table_ids[1]}' AND"
        f" table_name='{table_ids[2]}'"
    )
    try:
      queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
      for row in queryjob:
        table_stas_cache[table_path] = row["row_count"]
    except Exception as e:  # pylint: disable=broad-except
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + str(e))

  rows = table_stas_cache[table_path]
  return rows


def prefetch_column_info(
    metadata_dbtype,
    metadata_dbclient,
    table_path,
    queryplan,
):
  """Prefetch the column info for a given table, cache the results to avoid running this query again."""
  table_path = table_path.replace(";", "")
  table_ids = table_path.split(".")
  query = (
      "SELECT cs.column_type as ct, cs.null_frac as nf, cs.num_unique as nu,"
      " cs.table_name as tn, cs.column_name as cn,  (   SELECT count(*) FROM"
      f" `{TABLES_INFO_TABLE}` as ti  WHERE"
      f" project_name='{table_ids[0]}' AND dataset_name='{table_ids[1]}' AND"
      "  table_name = cs.table_name   and partition_column = cs.column_name"
      "  ) as is_part,  (SELECT count(*) FROM"
      f" `{TABLES_INFO_TABLE}` as ti,"
      f" ti.clustered_columns as cc  WHERE project_name='{table_ids[0]}' AND"
      f" dataset_name='{table_ids[1]}' AND table_name = cs.table_name  and cc ="
      " cs.column_name  ) as is_clust  FROM"
      f" `{COLUMNS_STATS_TABLE}` as cs WHERE "
      f" cs.project_name='{table_ids[0]}' AND"
      f" cs.dataset_name='{table_ids[1]}'AND not"
      " (REGEXP_CONTAINS(cast(cs.column_type as string) ,'ARRAY')) and not"
      " (REGEXP_CONTAINS(cast(cs.column_type as string),'STRUCT')) AND "
      " cs.column_type not in  ('UNKNOWN_TYPE','ENUM','PROTO','JSON', "
      " 'BYTES','INTERVAL','RANGE','BOOLEAN','GEOGRAPHY',  'BOOL') "
  )
  simple_column_information_dict = {}
  try:
    queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
    for row in queryjob:
      ctype = row["ct"]
      nullfrac = row["nf"]
      numunique = row["nu"]
      is_part = row["is_part"]
      is_clust = row["is_clust"]
      table_name = row["tn"]
      col_name = row["cn"]
      columnkey = f"{table_ids[0]}.{table_ids[1]}.{table_name}.{col_name}"
      simple_column_information_dict[columnkey] = (
          ctype,
          nullfrac,
          numunique,
          is_part,
          is_clust,
          table_name,
          col_name,
      )
  except Exception as e:  # pylint: disable=broad-except
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + str(e))

  extra_column_information_dict = {}
  for cctype in types_to_tables.keys():
    query = (
        f"select * from `{types_to_tables[cctype]}` where "
        f" project_name='{table_ids[0]}' AND dataset_name='{table_ids[1]}'"
    )
    try:
      queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
      for row in queryjob:
        minv = row["min_val"]
        maxv = row["max_val"]
        table_name = row["table_name"]
        col_name = row["column_name"]
        columnkey = f"{table_ids[0]}.{table_ids[1]}.{table_name}.{col_name}"
        extra_column_information_dict[columnkey] = [
            minv,
            maxv,
            table_name,
            col_name,
        ]
    except Exception as e:  # pylint: disable=broad-except
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
      print(e)

  column_quantiles_dict = {}
  column_quantiles_dict_100 = {}

  query = (
      f"select * FROM `{COLUMNS_HISTOGRAM_TABLE}` WHERE"
      f" project_name='{table_ids[0]}' AND dataset_name='{table_ids[1]}'"
  )
  try:
    queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
    for row in queryjob:
      table_name = row["table_name"]
      col_name = row["column_name"]
      columnkey = f"{table_ids[0]}.{table_ids[1]}.{table_name}.{col_name}"
      if row["approx_quantiles"]:
        quantiles = row["approx_quantiles"]
        column_quantiles_dict[columnkey] = quantiles
      if row["approx_quantiles_100"]:
        quantiles100 = row["approx_quantiles_100"]
        column_quantiles_dict_100[columnkey] = quantiles100

  except Exception as e:  # pylint: disable=broad-except
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + str(e))

  column_info_cache = queryplan["query_statistics_caches"]["column_info_cache"]
  for key in simple_column_information_dict:
    minv = extra_column_information_dict[key][0]
    maxv = extra_column_information_dict[key][1]
    ctype = simple_column_information_dict[key][0]
    nullfrac = simple_column_information_dict[key][1]
    numunique = simple_column_information_dict[key][2]
    is_part = simple_column_information_dict[key][3]
    is_clust = simple_column_information_dict[key][4]
    quantiles = []
    quantiles100 = []

    if key in column_quantiles_dict.keys():
      quantiles = column_quantiles_dict[key]
    if key in column_quantiles_dict_100.keys():
      quantiles100 = column_quantiles_dict_100[key]

    column_info_cache[key] = {
        "minv": minv,
        "maxv": maxv,
        "ctype": ctype,
        "nullfrac": nullfrac,
        "numunique": numunique,
        "is_part": is_part,
        "is_clust": is_clust,
        "quantiles": quantiles,
        "quantiles100": quantiles100,
    }


def print_myplan(queryplan, debug):
  printif(debug, "<><><><><><><>")
  for n in queryplan["nodes"]:
    printif(debug, str(n["id"]) + str(n))
  for e in queryplan["edges"]:
    printif(debug, str(e))
  printif(debug, "<><><><><><><>")


def print_myplan_no_metadata(queryplan, debug):
  """Prints the query plan without metadata."""
  printif(debug, "<><><><><><><>")
  for n in queryplan["nodes"]:
    exclude_keys = [
        "percentiles",
        "percentiles_100",
        "percentiles_500",
        "percentiles_1000",
    ]
    new_n = {k: n[k] for k in set(list(n.keys())) - set(exclude_keys)}
    printif(debug, str(n["id"]) + str(new_n))
  for e in queryplan["edges"]:
    printif(debug, str(e))
  printif(debug, "<><><><><><><>")


def get_column_info(
    metadata_dbtype,
    metadata_dbclient,
    table_path,
    col_name,
    queryplan,
):
  """Get the column info for a given column, cache the results to avoid running this query again."""
  column_info_cache = queryplan["query_statistics_caches"]["column_info_cache"]
  requested_col_key = f"{table_path}.{col_name}"
  if requested_col_key not in column_info_cache:
    prefetch_column_info(
        metadata_dbtype, metadata_dbclient, table_path, queryplan
    )
  return (
      column_info_cache[requested_col_key]["ctype"],
      column_info_cache[requested_col_key]["nullfrac"],
      column_info_cache[requested_col_key]["numunique"],
      column_info_cache[requested_col_key]["is_part"],
      column_info_cache[requested_col_key]["is_clust"],
      column_info_cache[requested_col_key]["minv"],
      column_info_cache[requested_col_key]["maxv"],
      column_info_cache[requested_col_key]["quantiles"],
      column_info_cache[requested_col_key]["quantiles100"],
  )


# BEGIN_GOOGLE_INTERNAL


def create_tf_graph_object(
    guery_graph, top_level_query_information
):
  """Takes in a graph_spec nodes object and a query_graph_edges object  (defined in create_graph_spec.py) and converts them to a tf graph object.

  The conversion is a 1-1 mapping
  Args:
   guery_graph: a graph_spec nodes object
   top_level_query_information: a dictionary containing the top level query
     information

  Returns:
   a tf graph object
  """
  query_graph_nodes = guery_graph["nodes"]
  query_graph_edges = guery_graph["edges"]

  graph = tfgnn.GraphTensor.from_pieces(
      context=tfgnn.Context.from_fields(
          features={
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
          }
      ),
      node_sets={
          "tables": tfgnn.NodeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_nodes["tables"]["ids"])], tf.int32
              ),
              features={
                  "rows": tf.constant(
                      query_graph_nodes["tables"]["rows"], tf.int64
                  ),
                  "name": tf.constant(
                      query_graph_nodes["tables"]["name"], tf.string
                  ),
              },
          ),
          "attributes": tfgnn.NodeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_nodes["attrs"]["ids"])], tf.int32
              ),
              features={
                  "is_clust_attr": tf.constant(
                      query_graph_nodes["attrs"]["is_clust_attr"], tf.int32
                  ),
                  "is_part_attr": tf.constant(
                      query_graph_nodes["attrs"]["is_part_attr"], tf.int32
                  ),
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
                  "percentiles": tf.constant(
                      query_graph_nodes["attrs"]["percentiles"], tf.float32
                  ),
                  "percentiles_100_numeric": tf.constant(
                      query_graph_nodes["attrs"]["percentiles_100_numeric"],
                      tf.float32,
                  ),
                  "percentiles_100_string": tf.constant(
                      query_graph_nodes["attrs"]["percentiles_100_string"],
                      tf.string,
                  ),
                  "encoded_percentiles": tf.constant(
                      query_graph_nodes["attrs"]["encoded_percentiles"],
                      tf.int64,
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
          ),
          "predicates": tfgnn.NodeSet.from_fields(
              sizes=tf.constant([len(query_graph_nodes["predicates"]["ids"])]),
              features={
                  "predicate_operator": tf.constant(
                      query_graph_nodes["predicates"]["operator"], tf.int32
                  ),
                  "estimated_selectivity": tf.constant(
                      query_graph_nodes["predicates"]["estimated_selectivity"],
                      tf.float32,
                  ),
                  "offset": tf.constant(
                      query_graph_nodes["predicates"]["offset"],
                      tf.float32,
                  ),
                  "constant": tf.constant(
                      query_graph_nodes["predicates"]["constant"], tf.string
                  ),
                  "encoded_constant": tf.constant(
                      query_graph_nodes["predicates"]["encoded_constant"],
                      tf.int64,
                  ),
              },
          ),
          "ops": tfgnn.NodeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_nodes["ops"]["ids"])], tf.int32
              ),
              features={
                  "operator": tf.constant(
                      query_graph_nodes["ops"]["type"], tf.string
                  ),
              },
          ),
          "correlations": tfgnn.NodeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_nodes["correlations"]["ids"])], tf.int32
              ),
              features={
                  "type": tf.constant(
                      query_graph_nodes["correlations"]["type"], tf.string
                  ),
                  "correlation": tf.constant(
                      query_graph_nodes["correlations"]["correlation"],
                      tf.float32,
                  ),
                  "validity": tf.constant(
                      query_graph_nodes["correlations"]["validity"], tf.string
                  ),
              },
          ),
      },
      edge_sets={
          "table_to_attr": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_edges["table_to_attr"][0])], tf.int32
              ),
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(
                      "tables",
                      tf.constant(
                          query_graph_edges["table_to_attr"][0], tf.int32
                      ),
                  ),
                  target=(
                      "attributes",
                      tf.constant(
                          query_graph_edges["table_to_attr"][1], tf.int32
                      ),
                  ),
              ),
          ),
          "attr_to_pred": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_edges["attr_to_pred"][0])], tf.int32
              ),
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(
                      "attributes",
                      tf.constant(
                          query_graph_edges["attr_to_pred"][0], tf.int32
                      ),
                  ),
                  target=(
                      "predicates",
                      tf.constant(
                          query_graph_edges["attr_to_pred"][1], tf.int32
                      ),
                  ),
              ),
          ),
          "pred_to_pred": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_edges["pred_to_pred"][0])], tf.int32
              ),
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(
                      "predicates",
                      tf.constant(
                          query_graph_edges["pred_to_pred"][0], tf.int32
                      ),
                  ),
                  target=(
                      "predicates",
                      tf.constant(
                          query_graph_edges["pred_to_pred"][1], tf.int32
                      ),
                  ),
              ),
          ),
          "attr_to_op": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_edges["attr_to_op"][0])], tf.int32
              ),
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(
                      "attributes",
                      tf.constant(query_graph_edges["attr_to_op"][0], tf.int32),
                  ),
                  target=(
                      "ops",
                      tf.constant(query_graph_edges["attr_to_op"][1], tf.int32),
                  ),
              ),
          ),
          "op_to_op": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_edges["op_to_op"][0])], tf.int32
              ),
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(
                      "ops",
                      tf.constant(query_graph_edges["op_to_op"][0], tf.int32),
                  ),
                  target=(
                      "ops",
                      tf.constant(query_graph_edges["op_to_op"][1], tf.int32),
                  ),
              ),
          ),
          "pred_to_op": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_edges["pred_to_op"][0])], tf.int32
              ),
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(
                      "predicates",
                      tf.constant(query_graph_edges["pred_to_op"][0], tf.int32),
                  ),
                  target=(
                      "ops",
                      tf.constant(query_graph_edges["pred_to_op"][1], tf.int32),
                  ),
              ),
          ),
          "attr_to_corr": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_edges["attr_to_correlation"][0])], tf.int32
              ),
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(
                      "attributes",
                      tf.constant(
                          query_graph_edges["attr_to_correlation"][0], tf.int32
                      ),
                  ),
                  target=(
                      "correlations",
                      tf.constant(
                          query_graph_edges["attr_to_correlation"][1], tf.int32
                      ),
                  ),
              ),
          ),
          "corr_to_pred": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_edges["correlation_to_pred"][0])], tf.int32
              ),
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(
                      "correlations",
                      tf.constant(
                          query_graph_edges["correlation_to_pred"][0], tf.int32
                      ),
                  ),
                  target=(
                      "predicates",
                      tf.constant(
                          query_graph_edges["correlation_to_pred"][1], tf.int32
                      ),
                  ),
              ),
          ),
          "corr_to_op": tfgnn.EdgeSet.from_fields(
              sizes=tf.constant(
                  [len(query_graph_edges["corr_to_op"][0])], tf.int32
              ),
              adjacency=tfgnn.Adjacency.from_indices(
                  source=(
                      "correlations",
                      tf.constant(query_graph_edges["corr_to_op"][0], tf.int32),
                  ),
                  target=(
                      "ops",
                      tf.constant(query_graph_edges["corr_to_op"][1], tf.int32),
                  ),
              ),
          ),
      },
  )

  return graph


def write_tf_graph_object_file(graphs, file_path):
  if file_exists(file_path):
    print("tfrecord file exists -- recreating:", file_path)
    remove_file(file_path)
  with tf.io.TFRecordWriter(file_path) as writer:
    for graph in graphs:
      example = tfgnn.write_example(graph)
      writer.write(example.SerializeToString())
    print(f"{len(graphs)} graphs saved to {file_path}")


def write_graph_schema_file(
    tf_query_graphs_schema_file_path,
    tf_objects_unique_queries,
):
  tf_graph_random = tf_objects_unique_queries[0]
  graph_schema = tfgnn.create_schema_pb_from_graph_spec(tf_graph_random)
  tfgnn.write_schema(graph_schema, tf_query_graphs_schema_file_path)


