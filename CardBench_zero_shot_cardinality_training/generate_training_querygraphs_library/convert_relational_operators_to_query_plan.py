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

"""Converts relational operators to query plan annotated with statistics."""

import datetime
import itertools
from typing import Any

from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import estimate_selectivity
from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import generate_training_querygraphs_helpers

printif = generate_training_querygraphs_helpers.printif
get_table_rows = generate_training_querygraphs_helpers.get_table_rows
get_column_info = generate_training_querygraphs_helpers.get_column_info

DBType = database_connector.DBType
create_database_connection = database_connector.create_database_connection
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
prefetch_correlations = (
    generate_training_querygraphs_helpers.prefetch_correlations
)
predicate_operator_dict = (
    generate_training_querygraphs_helpers.predicate_operator_dict
)
print_queryplan_no_percentiles = (
    generate_training_querygraphs_helpers.print_queryplan_no_percentiles
)

print_myplan_no_metadata = (
    generate_training_querygraphs_helpers.print_myplan_no_metadata
)

get_correlation = generate_training_querygraphs_helpers.get_correlation

estimate_selectivity = estimate_selectivity.estimate_selectivity


def add_metadata_to_nodes(
    queryplan,
    metadata_dbtype,
    metadata_dbclient,
):
  """Adds metadata to nodes in the query plan."""
  for node in queryplan["nodes"]:
    if node["nodetype"] == "table":
      node["rows"] = get_table_rows(
          metadata_dbtype, metadata_dbclient, node["name"], queryplan
      )
    elif node["nodetype"] == "column":
      tablenodeid = queryplan["colnametotablenode_ids"][node["name"]]
      tablepath = ""
      for n in queryplan["nodes"]:
        if n["id"] == tablenodeid:
          tablepath = n["name"]
      if not tablepath:
        print("ERROR tablepath empty")
      (
          ctype,
          nullfrac,
          numunique,
          is_part,
          is_clust,
          minv,
          maxv,
          quantiles,
          quantiles100,
      ) = get_column_info(
          metadata_dbtype, metadata_dbclient, tablepath, node["name"], queryplan
      )
      node["is_clust_attr"] = is_clust
      node["is_part_attr"] = is_part
      node["null_frac"] = nullfrac
      node["num_unique"] = numunique
      node["column_type"] = ctype
      node["min_val"] = minv
      node["max_val"] = maxv
      node["percentiles"] = quantiles
      node["percentiles_100"] = quantiles100


def is_part_pred(column_type, pred_operator, constant):
  """Returns true if the predicate is a partitioning predicate."""
  if (
      column_type == "DATE"
      and pred_operator == ">"
      and constant == "'1000-01-01'"
  ):
    return True
  elif (
      column_type == "TIMESTAMP"
      and pred_operator == ">"
      and constant == "'1000-01-01 00:00:00'"
  ):
    return True
  elif (
      column_type == "DATETIME"
      and pred_operator == ">"
      and constant == "'1000-01-01 00:00:00'"
  ):
    return True
  elif (
      column_type in ["INT32", "INT64", "UINT32", "UINT64"]
      and pred_operator == "<="
      and constant == "cast('+inf' as float64)"
  ):
    return True
  return False


def is_column(colname, table_aliases_and_names):
  for alias in table_aliases_and_names:
    if alias + "." in colname:
      return True
  return False


def split_column_name(column_path):
  table_alias = column_path.split(".")[0].strip()
  column_name = column_path.split(".")[1].strip()
  return table_alias, column_name


def parse_single_col_pred(
    pred, table_aliases
):
  """Parses a single column predicate."""
  operand1 = pred["operand1"]
  operator = pred["operator"].lower()
  operand2 = pred["operand2"]

  if is_column(operand1, table_aliases):
    table_alias1, column_name1 = split_column_name(operand1)
    constant = operand2
  elif is_column(operand2, table_aliases):
    table_alias1, column_name1 = split_column_name(operand2)
    constant = operand1
  else:
    raise ValueError("no columns detected in" + str(pred), str(table_aliases))

  return table_alias1, column_name1, operator, constant


def parse_two_col_pred(
    pred, table_aliases
):
  """Parses a two column predicate."""
  operand1 = pred["operand1"]
  operator = pred["operator"]
  operand2 = pred["operand2"]

  if not is_column(operand1, table_aliases) or not is_column(
      operand2, table_aliases
  ):
    raise ValueError("constant detected in" + str(pred), str(table_aliases))

  table_alias1, column_name1 = split_column_name(operand1)
  table_alias2, column_name2 = split_column_name(operand2)
  return table_alias1, column_name1, table_alias2, column_name2, operator


def get_partitioning_column_info(queryplan):
  """Returns a dictionary of table alias to partitioning column name and type."""
  part_attrs_bytablenodeid = {}
  column_name_to_type = {}
  for n in queryplan["nodes"]:
    if n["nodetype"] == "column":
      column_name_to_type[n["name"]] = n["column_type"]
      if n["is_part_attr"] == 1:
        tablenodeid = queryplan["colnametotablenode_ids"][n["name"]]
        part_attrs_bytablenodeid[tablenodeid] = n["name"]

  table_alias_to_part_col_name_and_type = {}

  # return part_attrs_bytablenodeid
  if part_attrs_bytablenodeid:
    for n in queryplan["nodes"]:
      if n["nodetype"] == "table" and n["id"] in part_attrs_bytablenodeid:
        part_col_name = part_attrs_bytablenodeid[n["id"]]
        column_type = column_name_to_type[part_col_name]
        table_alias_to_part_col_name_and_type[n["alias"]] = [
            part_col_name,
            column_type,
        ]
  return table_alias_to_part_col_name_and_type


def remove_partitioning_predicate(
    preds,
    queryplan,
    table_alias_to_part_col_name_and_type,
    debug,
):
  """Remove partitioning predicate from the predicate tree."""

  # The partitioning predicate is an artifact of the partitioning of the table
  # and does not affect the semantics of the query or the cardinality.
  if not preds:
    return preds
  elif preds["operator"] == "AND":
    # when the predicate to be remove is part of the tree
    # we need to fix the tree
    if not remove_partitioning_predicate(
        preds["operand1"],
        queryplan,
        table_alias_to_part_col_name_and_type,
        debug,
    ):
      return preds["operand2"]
    if not remove_partitioning_predicate(
        preds["operand2"],
        queryplan,
        table_alias_to_part_col_name_and_type,
        debug,
    ):
      return preds["operand1"]
  else:
    # case where the preds is a single column predicate not a pred tree
    table_alias, column_name, operator, constant = parse_single_col_pred(
        preds, queryplan["table_aliases_and_names"]
    )
    if column_name == table_alias_to_part_col_name_and_type[table_alias][0]:
      column_type = table_alias_to_part_col_name_and_type[table_alias][1]
      if is_part_pred(column_type, operator, constant):
        return {}
      else:
        return preds


def create_pred_tree_of_single_column_predicates(
    preds,
    queryplan,
    cols_referenced,
    debug,
):
  """Creates a predicate tree of single column predicates."""
  # preds is already a tree
  # we need to convert it to the format of queryplan
  if not preds:
    return None, [], []
  elif preds["operator"] == "AND" or preds["operator"] == "OR":
    predopid1, col_list1, colslist1_dup = (
        create_pred_tree_of_single_column_predicates(
            preds["operand1"],
            queryplan,
            cols_referenced,
            debug,
        )
    )
    predopid2, col_list2, colslist2_dup = (
        create_pred_tree_of_single_column_predicates(
            preds["operand2"],
            queryplan,
            cols_referenced,
            debug,
        )
    )

    predopid = queryplan["nextmyplanopid"]
    queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
    predop = {
        "id": predopid,
        "nodetype": "predicate_operator",
        "operator": predicate_operator_dict[preds["operator"].lower()],
        "estimated_selectivity": -1,
        "constant": "",
        "offset": [-1.0] * 6,
    }

    queryplan["nodes"].append(predop)
    queryplan["nodedict"][predopid] = predop
    queryplan["edges"].append({"from": predopid1, "to": predopid})
    queryplan["edges"].append({"from": predopid2, "to": predopid})

    ## add empty correlation nodes
    for col1 in col_list1:
      for col2 in col_list2:
        if col1 == col2:
          continue
        col1_opid = col1[2]
        col2_opid = col2[2]
        key = (
            str(min(col1_opid, col2_opid))
            + "_"
            + str(max(col1_opid, col2_opid))
        )
        if key in queryplan["added_corr_from_to"]:
          corr_opid = queryplan["added_corr_from_to"][key]
          queryplan["edges"].append({"from": corr_opid, "to": predopid})
        else:
          corr_opid = queryplan["nextmyplanopid"]
          queryplan["added_corr_from_to"][key] = corr_opid
          corr_op = {
              "id": corr_opid,
              "nodetype": "correlation",
              "correlation": -100,
              "temp_colid1": col1_opid,
              "temp_colid2": col2_opid,
          }
          queryplan["nodes"].append(corr_op)
          queryplan["nodedict"][corr_opid] = corr_op

          queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
          queryplan["edges"].append({"from": col1_opid, "to": corr_opid})
          queryplan["edges"].append({"from": col2_opid, "to": corr_opid})
          queryplan["edges"].append({"from": corr_opid, "to": predopid})

    col_list1.extend(col_list2)
    colslist1_dup.extend(colslist2_dup)
    return predopid, col_list1, colslist1_dup
  else:
    table_alias, column_name, operator, constant = parse_single_col_pred(
        preds, queryplan["table_aliases_and_names"]
    )
    if column_name in cols_referenced:
      cols_referenced.remove(column_name)
    col_opid = queryplan["name_to_opid"][table_alias + "." + column_name]
    predopid = queryplan["nextmyplanopid"]
    queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
    predop = {
        "id": predopid,
        "nodetype": "predicate_operator",
        "operator": predicate_operator_dict[operator],
        "constant": constant,
        "estimated_selectivity": -1,
        "offset": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "temp_colid": col_opid,
    }
    queryplan["nodes"].append(predop)
    queryplan["nodedict"][predopid] = predop
    queryplan["edges"].append({"from": col_opid, "to": predopid})
    return (
        predopid,
        [[table_alias, column_name, col_opid]],
        [[
            table_alias + " __ " + column_name,
            str(predicate_operator_dict[operator]),
            str(constant),
        ]],
    )


def create_predicate_tree_from_single_column_predicates(
    preds,
    myplan,
    cols_referenced,
    debug,
):
  """Creates a predicate tree from single column predicates."""
  # preds is already a tree
  # we need to convert it to the format of myplan
  if not preds or not preds or not preds:
    return None, [], []
  elif preds["operator"] == "AND" or preds["operator"] == "OR":
    predopid1, col_list1, colslist1_dup = (
        create_predicate_tree_from_single_column_predicates(
            preds["operand1"],
            myplan,
            cols_referenced,
            debug,
        )
    )
    predopid2, col_list2, colslist2_dup = (
        create_predicate_tree_from_single_column_predicates(
            preds["operand2"],
            myplan,
            cols_referenced,
            debug,
        )
    )

    predopid = myplan["nextmyplanopid"]
    myplan["nextmyplanopid"] = myplan["nextmyplanopid"] + 1
    predop = {
        "id": predopid,
        "nodetype": "predicate_operator",
        "operator": predicate_operator_dict[preds["operator"].lower()],
        "estimated_selectivity": -1,
        "constant": "",
        "offset": [-1.0] * 6,
    }

    myplan["nodes"].append(predop)
    myplan["nodedict"][predopid] = predop
    myplan["edges"].append({"from": predopid1, "to": predopid})
    myplan["edges"].append({"from": predopid2, "to": predopid})

    ## add empty correlation nodes
    for col1 in col_list1:
      for col2 in col_list2:
        if col1 == col2:
          continue
        col1_opid = col1[2]
        col2_opid = col2[2]
        key = (
            str(min(col1_opid, col2_opid))
            + "_"
            + str(max(col1_opid, col2_opid))
        )
        if key in myplan["added_corr_from_to"]:
          corr_opid = myplan["added_corr_from_to"][key]
          myplan["edges"].append({"from": corr_opid, "to": predopid})
        else:
          corr_opid = myplan["nextmyplanopid"]
          myplan["added_corr_from_to"][key] = corr_opid
          corr_op = {
              "id": corr_opid,
              "nodetype": "correlation",
              "correlation": -100,
              "temp_colid1": col1_opid,
              "temp_colid2": col2_opid,
          }
          myplan["nodes"].append(corr_op)
          myplan["nodedict"][corr_opid] = corr_op

          myplan["nextmyplanopid"] = myplan["nextmyplanopid"] + 1
          myplan["edges"].append({"from": col1_opid, "to": corr_opid})
          myplan["edges"].append({"from": col2_opid, "to": corr_opid})
          myplan["edges"].append({"from": corr_opid, "to": predopid})

    col_list1.extend(col_list2)
    colslist1_dup.extend(colslist2_dup)
    return predopid, col_list1, colslist1_dup

  else:
    table_alias, column_name, operator, constant = parse_single_col_pred(
        preds, myplan["table_aliases_and_names"]
    )
    if column_name in cols_referenced:
      cols_referenced.remove(column_name)
    col_opid = myplan["name_to_opid"][table_alias + "." + column_name]
    predopid = myplan["nextmyplanopid"]
    myplan["nextmyplanopid"] = myplan["nextmyplanopid"] + 1
    predop = {
        "id": predopid,
        "nodetype": "predicate_operator",
        "operator": predicate_operator_dict[operator],
        "constant": constant,
        "estimated_selectivity": -1,
        "offset": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "temp_colid": col_opid,
    }
    myplan["nodes"].append(predop)
    myplan["nodedict"][predopid] = predop
    myplan["edges"].append({"from": col_opid, "to": predopid})
    return (
        predopid,
        [[table_alias, column_name, col_opid]],
        [[
            table_alias + " __ " + column_name,
            str(predicate_operator_dict[operator]),
            str(constant),
        ]],
    )


def get_col_id_name_pred(
    queryplan, pred_id, nodedict
):
  for e in queryplan["edges"]:
    if e["to"] == pred_id:
      columnnode = nodedict[e["from"]]
      if columnnode["nodetype"] == "column":
        column_id = e["from"]
        column_name = columnnode["name"]
        return [column_id, column_name]
  return []


def get_col_names_nested_pred(
    queryplan, nested_pred_id, nodedict
):
  """Returns the column names of the nested predicate."""
  result = []
  children_ids = []
  for e in queryplan["edges"]:
    if e["to"] == nested_pred_id:
      children_ids.append(e["from"])

  for child_id in children_ids:
    child_pred_node = nodedict[child_id]
    if (
        nodedict[child_pred_node["id"]]["operator"]
        == predicate_operator_dict["and"]
        or nodedict[child_pred_node["id"]]["operator"]
        == predicate_operator_dict["or"]
    ):
      result.extend(
          get_col_names_nested_pred(queryplan, child_pred_node, nodedict)
      )
    else:
      result.extend(get_col_id_name_pred(queryplan, child_pred_node, nodedict))

  return result


def add_correlation_val(
    queryplan,
    tableid_to_name,
    colid_to_col_name,
):
  """Adds correlation values to correlation nodes."""
  for n in queryplan["nodes"]:
    if n["nodetype"] == "correlation":
      colid1 = n["temp_colid1"]
      colid2 = n["temp_colid2"]
      table_id = queryplan["colid_to_table_id"][colid1]
      table_id2 = queryplan["colid_to_table_id"][colid2]
      if table_id != table_id2:
        raise ValueError(
            "table ids should be the same"
            + str(table_id)
            + " != "
            + str(table_id2)
        )
      col1type = queryplan["nodedict"][colid1]["column_type"]
      col2type = queryplan["nodedict"][colid2]["column_type"]
      table_name = tableid_to_name[table_id]
      colname1 = colid_to_col_name[colid1]
      colname2 = colid_to_col_name[colid2]

      corr = get_correlation(
          table_name,
          colname1,
          colname2,
          col1type,
          col2type,
          queryplan["query_statistics_caches"]["correlation_cache"],
      )
      n["correlation"] = corr
      n["type"] = "pearson"


def add_correlation_nodes_for_colid_join_pred(
    colid, predid, queryplan
):
  """Adds correlation nodes for a given column id and predicate id."""
  table_id = queryplan["colid_to_table_id"][colid]
  cols_ids = queryplan["tableopid_to_column_ids"][table_id]

  for col_id in cols_ids:
    if col_id == colid:
      continue
    key = str(min(col_id, colid)) + "_" + str(max(col_id, colid))
    if key not in queryplan["added_corr_from_to"]:
      corr_opid = queryplan["nextmyplanopid"]
      queryplan["added_corr_from_to"][key] = corr_opid
      corr_op = {
          "id": queryplan["nextmyplanopid"],
          "nodetype": "correlation",
          "correlation": -100,
          "temp_colid1": col_id,
          "temp_colid2": colid,
      }

      queryplan["nodes"].append(corr_op)
      queryplan["nodedict"][corr_opid] = corr_op

      queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
      queryplan["edges"].append({"from": col_id, "to": corr_opid})
      queryplan["edges"].append({"from": colid, "to": corr_opid})
      queryplan["edges"].append({"from": corr_opid, "to": predid})
    else:
      corr_opid = queryplan["added_corr_from_to"][key]
      queryplan["edges"].append({"from": corr_opid, "to": predid})


def create_pred_tree_join(
    queryplan, preds, joinop, debug
):
  """Creates a predicate tree for join predicates."""
  if not preds:
    return None, []
  elif len(preds) == 1:
    predopid = queryplan["nextmyplanopid"]
    queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
    predop = {
        "id": predopid,
        "nodetype": "predicate_operator",
        "operator": predicate_operator_dict["="],
        "t_is_join_pred": True,
        "estimated_selectivity": -1,
        "constant": "",
        "corelation": 10,
        "offset": [-1.0] * 6,
    }

    table_alias1, column_name1, table_alias2, column_name2, _ = (
        parse_two_col_pred(preds[0], queryplan["table_aliases_and_names"])
    )

    col1id = queryplan["name_to_opid"][table_alias1 + "." + column_name1]
    col2id = queryplan["name_to_opid"][table_alias2 + "." + column_name2]
    queryplan["join_col_ids"].append(col1id)
    queryplan["join_col_ids"].append(col2id)
    queryplan["join_col_to_pred_id"][col1id] = predopid
    queryplan["join_col_to_pred_id"][col2id] = predopid

    queryplan["edges"].append({"from": col1id, "to": predopid})
    queryplan["edges"].append({"from": col2id, "to": predopid})

    queryplan["nodes"].append(predop)
    queryplan["nodedict"][predopid] = predop

    add_correlation_nodes_for_colid_join_pred(col1id, predopid, queryplan)
    add_correlation_nodes_for_colid_join_pred(col2id, predopid, queryplan)
    tl = [table_alias1 + "." + column_name1, table_alias2 + "." + column_name2]
    tl.sort()
    return predopid, [tl[0] + "__" + tl[1]]
  else:
    predopid1, listcols1 = create_pred_tree_join(
        queryplan, preds[0:1], joinop, debug
    )
    predopid2, listcols2 = create_pred_tree_join(
        queryplan, preds[1:], joinop, debug
    )

    predopid = queryplan["nextmyplanopid"]
    queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
    predop = {
        "id": predopid,
        "nodetype": "predicate_operator",
        "operator": predicate_operator_dict["and"],
        "estimated_selectivity": -1,
        "t_is_join_pred_log": True,
        "constant": "",
        "offset": [-1.0] * 6,
    }
    queryplan["nodes"].append(predop)
    queryplan["nodedict"][predopid] = predop
    queryplan["edges"].append({"from": predopid1, "to": predopid})
    queryplan["edges"].append({"from": predopid2, "to": predopid})

  listcols = []
  listcols.extend(listcols1)
  listcols.extend(listcols2)
  return predopid, listcols


def convert_string_to_data_type(x, col_type):
  """Converts a string to the data type of the column."""
  if x.count("'") == 2:
    x = x.replace("'", "")
  if col_type == "STRING":
    return x
  elif col_type == "NUMERIC" or col_type == "FLOAT64":
    return float(x)
  elif col_type == "INT64":
    return int(x)
  elif col_type == "BOOL":
    return "true" == x.lower()
  elif col_type == "DATE":
    return datetime.datetime.strptime(x, "%Y-%m-%d").date()
  elif col_type == "DATETIME" or col_type == "TIMESTAMP":
    if "." in x and "+" in x:
      return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f%z").timestamp()
    elif "." in x:
      return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    elif "+" in x:
      return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S%z").timestamp()
    else:
      return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()
  elif col_type == "TIME":
    return datetime.datetime.strptime(x, "%H:%M:%S").time()
  else:
    raise ValueError(col_type + " is not supported")


def convert_strings_to_data_type(
    list_string, col_type
):
  return [convert_string_to_data_type(x, col_type) for x in list_string]


def convert_relational_operators_to_query_plan(
    parsed_query,
    metadata_dbtype,
    metadata_dbclient,
    query_statistics_caches,
    debug,
):
  """Converts relational operators to query plan annotated with statistics."""
  # The query plan is a graph, it has nodes and edges and other information that
  # is helpful during creation of the graph representation.
  queryplan = {
      # basic graph representation
      "nodes": [],
      "edges": [],
      # The fields below are used to create the graph representation
      "nextmyplanopid": 0,
      "colnametotablenode_ids": {},
      "added_corr_from_to": {},
      "table_aliases_and_names": [],
      # for columns key is table_alias + "." + col_name
      "name_to_opid": {},
      "join_col_ids": [],  # unused?
      "join_col_to_pred_id": {},  # unused?
      "colid_to_table_id": {},
      "tableopid_to_column_ids": {},
      "nodedict": {},
      "dup_tables": [],
      "dup_join_preds": [],
      "dup_scan_preds": [],
      "dup_groupby_cols": [],
      "query_statistics_caches": query_statistics_caches,
  }

  ## temp helper structure
  name_to_table_id = {}
  tableid_to_name = {}
  colid_to_col_name = {}
  myplan_table_id_to_sqlplan_table_id = {}
  sql_plan_to_myplan_id = {}

  ## Add SCAN nodes from the parsed query
  for op in parsed_query["ops"]:
    if op["reloptype"] == "SCAN":
      tableopid = queryplan["nextmyplanopid"]
      queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
      queryplan["dup_tables"].append(op["table_name"])
      myplan_table_id_to_sqlplan_table_id[tableopid] = op["relop_id"]
      tableop = {
          "id": tableopid,
          "nodetype": "table",
          "name": op["table_name"],
          "alias": op["table_alias"],
          "temp_scan_preds": op["predicates"],
          "temp_ref_cols": op["columns"],
      }
      ## temp helper structure
      queryplan["table_aliases_and_names"].append(op["table_alias"])
      queryplan["table_aliases_and_names"].append(
          op["table_name"].split(".")[-1].strip()
      )
      ## temp helper structure
      tableid_to_name[tableopid] = op["table_name"]

      queryplan["nodes"].append(tableop)
      queryplan["tableopid_to_column_ids"][tableopid] = []
      for col in op["columns"]:
        colopid = queryplan["nextmyplanopid"]
        queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1

        ## temp helper structure
        queryplan["name_to_opid"][op["table_alias"] + "." + col] = colopid
        name_to_table_id[op["table_alias"] + "." + col] = tableopid
        queryplan["colnametotablenode_ids"][col] = tableopid
        queryplan["tableopid_to_column_ids"][tableopid].append(colopid)
        queryplan["colid_to_table_id"][colopid] = tableopid
        colid_to_col_name[colopid] = col
        colop = {"id": colopid, "nodetype": "column", "name": col}
        queryplan["nodes"].append(colop)
        queryplan["nodedict"][colopid] = colop

        queryplan["edges"].append({"from": tableopid, "to": colopid})
  add_metadata_to_nodes(queryplan, metadata_dbtype, metadata_dbclient)

  table_alias_to_part_col_name_and_type = get_partitioning_column_info(
      queryplan
  )

  for n in queryplan["nodes"]:
    if n["nodetype"] == "table":
      datasetpath = f"{n['name'].split('.')[0]}.{n['name'].split('.')[1]}"
      prefetch_correlations(
          datasetpath, metadata_dbclient, metadata_dbtype, queryplan
      )
      break

  scanopids = []
  # Add SCAN nodes for the table nodes and move predicates to the scan node
  for n in queryplan["nodes"]:
    if n["nodetype"] == "table":
      scanopid = queryplan["nextmyplanopid"]
      queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
      scanop = {
          "id": scanopid,
          "nodetype": "scan",
      }
      sql_plan_to_myplan_id[myplan_table_id_to_sqlplan_table_id[n["id"]]] = (
          scanopid
      )
      scanopids.append(scanopid)
      queryplan["nodes"].append(scanop)
      queryplan["nodedict"][scanopid] = scanop

      cols_referenced = n["temp_ref_cols"]
      # TODO(chronis): update remove remove_partitioning_predicate to remove
      # potential dangling column nodes from the query plan. I think the
      # solution is to count the # a columns is referenced and if when
      # removing the partitioning predicate we end up with 0 references then
      # we remove the column node.
      if n["alias"] in table_alias_to_part_col_name_and_type:
        n["temp_scan_preds"] = remove_partitioning_predicate(
            n["temp_scan_preds"],
            queryplan,
            table_alias_to_part_col_name_and_type,
            debug,
        )
      rootpredopid, _, colslist_dup = (
          create_predicate_tree_from_single_column_predicates(
              n["temp_scan_preds"],
              queryplan,
              cols_referenced,
              debug,
          )
      )
      queryplan["dup_scan_preds"].extend(colslist_dup)
      if rootpredopid is not None:
        queryplan["edges"].append({"from": rootpredopid, "to": scanopid})
      for col in cols_referenced:
        colid = queryplan["name_to_opid"][n["alias"] + "." + col]
        queryplan["edges"].append({"from": colid, "to": scanopid})
      if rootpredopid is None:
        n["no_scan_preds"] = True
      n.pop("temp_ref_cols")
      n.pop("temp_scan_preds")

  ## Add JOIN nodes from the parsed query
  for op in parsed_query["ops"]:
    if op["reloptype"] == "JOIN":
      joinopid = queryplan["nextmyplanopid"]
      queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
      joinop = {
          "id": joinopid,
          "nodetype": "join",
          "joining_preds": str(op["predicates"]),
      }
      queryplan["nodes"].append(joinop)
      queryplan["nodedict"][joinopid] = joinop
      sql_plan_to_myplan_id[op["relop_id"]] = joinopid

      from1 = sql_plan_to_myplan_id[op["input_opid_a"]]
      from2 = sql_plan_to_myplan_id[op["input_opid_b"]]

      queryplan["edges"].append({"from": from1, "to": joinopid})
      queryplan["edges"].append({"from": from2, "to": joinopid})

      root_pred_id, listcols = create_pred_tree_join(
          queryplan, op["predicates"], op, debug
      )
      queryplan["edges"].append({"from": root_pred_id, "to": joinopid})
      queryplan["dup_join_preds"] = listcols

  ## Add GROUPBY nodes from the parsed query
  for op in parsed_query["ops"]:
    if op["reloptype"] == "GROUPBY":
      aggopid = queryplan["nextmyplanopid"]
      queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
      aggop = {
          "id": aggopid,
          "nodetype": "groupby",
          "group_by_cols": str(op["grouping_columns"]),
      }
      queryplan["dup_groupby_cols"].extend(op["grouping_columns"])
      queryplan["nodes"].append(aggop)
      queryplan["nodedict"][aggopid] = aggop
      sql_plan_to_myplan_id[op["relop_id"]] = aggopid
      from1 = sql_plan_to_myplan_id[op["input_opid"]]

      queryplan["edges"].append({"from": from1, "to": aggopid})

      colids = []
      for col in op["grouping_columns"]:
        colid = queryplan["name_to_opid"][col]
        colids.append(colid)
        queryplan["edges"].append({"from": colid, "to": aggopid})

      for comb in itertools.combinations(colids, 2):
        corr_opid = queryplan["nextmyplanopid"]
        queryplan["nextmyplanopid"] = queryplan["nextmyplanopid"] + 1
        corr_op = {
            "id": corr_opid,
            "nodetype": "correlation",
            "correlation": -100,
            "temp_colid1": comb[0],
            "temp_colid2": comb[1],
        }
        queryplan["nodes"].append(corr_op)
        queryplan["nodedict"][corr_opid] = corr_op
        queryplan["edges"].append({"from": comb[0], "to": corr_opid})
        queryplan["edges"].append({"from": comb[1], "to": corr_opid})
        queryplan["edges"].append({"from": corr_opid, "to": aggopid})

  estimate_selectivity(queryplan)

  # traverse the plan and when a correlation node is encountered then add
  # the value
  add_correlation_val(queryplan, tableid_to_name, colid_to_col_name)

  # dedup plan
  edge_dict = {}
  for e in queryplan["edges"]:
    key = str(e["from"]) + "-" + str(e["to"])
    if key not in edge_dict:
      edge_dict[key] = 1
    else:
      queryplan["edges"].remove(e)

  print_myplan_no_metadata(queryplan, debug)
  count_join = 0
  count_filters = 0
  count_group_bys = 0
  for n in queryplan["nodes"]:
    if n["nodetype"] == "join":
      count_join += 1
    elif n["nodetype"] == "table" and "no_scan_preds" not in n:
      count_filters += 1
    elif n["nodetype"] == "groupby":
      count_group_bys += 1
  if count_join + count_filters + count_group_bys == 0:
    return {}
  return queryplan
