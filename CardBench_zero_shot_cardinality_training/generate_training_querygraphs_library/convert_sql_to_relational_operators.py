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

"""Convert a SQL query to relational operators."""

import re
from typing import Any

from CardBench_zero_shot_cardinality_training.generate_training_querygraphs_library import generate_training_querygraphs_helpers


printif = generate_training_querygraphs_helpers.printif


def find_referenced_columns_join_preds(
    preds, alias, referenced_cols
):
  stringtomatch = alias + "."
  cols = []
  for pred in preds:
    if pred["operand1"].startswith(stringtomatch):
      cols.append(pred["operand1"])
    if pred["operand2"].startswith(stringtomatch):
      cols.append(pred["operand2"])
  for col in cols:
    referenced_cols.append(col.split(".")[1])


def find_closing_paranthesis(pred):
  """Find the closing paranthesis in a predicate string."""
  stack = []
  closing_paranthesis_index = -1
  stack.append("(")
  # find closing parenthesis
  for i in range(1, len(pred)):
    if pred[i] == "(":
      if stack[-1] == ")":
        stack.pop()
      else:
        stack.append("(")
    elif pred[i] == ")":
      if stack[-1] == "(":
        stack.pop()
      else:
        stack.append(")")
    if not stack:
      closing_paranthesis_index = i
      break
  if closing_paranthesis_index == -1:
    raise ValueError("unmatched parenthesis in " + pred)
  return closing_paranthesis_index


def split_by_paranthesis(preds):
  """Split a predicate string by paranthesis."""
  parts = []
  first = preds.find("(")
  if first == -1:
    return [preds]
  parts.extend(split_by_paranthesis(preds[:first]))
  # there is a paranthesis in the string
  if first != -1:
    i = find_closing_paranthesis(preds[first:])
    parts.append(preds[first : first + i + 1])
    after = preds[first + i + 1 :]
    if after:
      parts.extend(split_by_paranthesis(after))
  return parts


def parse_pred_rec(
    pred,
    referenced_columns_per_table,
    debug = False,
):
  """Recursively parse a predicate string into a predicate tree."""
  if not pred.strip():
    return {}
  if pred.count(" AND ") + pred.count(" OR ") == 0:
    conjucts_split = re.findall(r"'[^']*'$|'[^']*' |[^ ]+", pred)
    pred_node = {
        "operand1": conjucts_split[0].strip(),
        "operator": conjucts_split[1].strip(),
        "operand2": conjucts_split[2].strip(),
    }
    table, col = get_table_of_the_predicate(pred_node)
    if table not in referenced_columns_per_table:
      referenced_columns_per_table[table] = []
    referenced_columns_per_table[table].append(col)
    return pred_node

  closing_paranthesis_index = -1
  if pred[0] == "(":
    stack = []
    stack.append("(")
    # find closing parenthesis
    for i in range(1, len(pred)):
      if pred[i] == "(":
        if stack[-1] == ")":
          stack.pop()
        else:
          stack.append("(")
      elif pred[i] == ")":
        if stack[-1] == "(":
          stack.pop()
        else:
          stack.append(")")
      if not stack:
        closing_paranthesis_index = i
        break
    pred = (
        pred[1:closing_paranthesis_index]
        + pred[closing_paranthesis_index + 1 :]
    )
    if closing_paranthesis_index == len(pred):
      raise ValueError("unmatched parenthesis in " + pred)

  splitpartnethesis = split_by_paranthesis(pred)
  operands_temp = []
  for operand in splitpartnethesis:
    if operand[0] == "(" and operand[-1] == ")":
      operands_temp.append(operand)
    else:
      dd = re.split(r"( (OR|AND) (?![^(]*\)))", operand)
      operands_temp.extend(dd)
  operands = operands_temp

  # hacky way to remove duplicate match groups from the result of re.split
  # there is probably a better way to do this
  operands_temp = [operands[0]]
  for i in range(1, len(operands)):
    if operands[i].strip() != operands_temp[-1].strip():
      operands_temp.append(operands[i].strip())
  operands = operands_temp
  operands = [x for x in operands if x]

  root_node = {
      "operand1": parse_pred_rec(
          operands[0], referenced_columns_per_table, debug
      ),
      "operator": operands[1].strip(),
      "operand2": parse_pred_rec(
          operands[2], referenced_columns_per_table, debug
      ),
  }
  for i in range(4, len(operands), 2):
    np = parse_pred_rec(operands[i], referenced_columns_per_table, debug)
    new_root = {
        "operand1": root_node,
        "operator": operands[i - 1].strip(),
        "operand2": np,
    }
    root_node = new_root

  return root_node


def get_table_of_the_predicate(pred):
  """Find the table name in a predicate operand."""
  # One operand is a column name and the other is a constant
  if "." in pred["operand1"]:
    return pred["operand1"].split(".")[0], pred["operand1"].split(".")[1]
  else:
    return pred["operand2"].split(".")[0], pred["operand2"].split(".")[1]


def is_pred_subtree_of_one_table(pred_tree):
  """Are all the predicates of a predicate tree operating on collumns of one table."""
  if pred_tree["operator"] != "AND" and pred_tree["operator"] != "OR":
    table, _ = get_table_of_the_predicate(pred_tree)
    return True, table
  else:
    flag1, table1 = is_pred_subtree_of_one_table(pred_tree["operand1"])
    flag2, table2 = is_pred_subtree_of_one_table(pred_tree["operand2"])
    if flag1 and flag2 and table1 == table2:
      return True, table1
    else:
      return False, None


def split_tree_per_table(
    pred_tree, pred_per_table
):
  """For a pred tree split it in conjuctions per table."""
  if pred_tree["operator"] != "AND" and pred_tree["operator"] != "OR":
    table, _ = get_table_of_the_predicate(pred_tree)
    if table not in pred_per_table:
      pred_per_table[table] = []
    pred_per_table[table].append(pred_tree)
  else:
    # is the predicate tree using only one table?
    only_one_table_referenced, table = is_pred_subtree_of_one_table(pred_tree)
    if only_one_table_referenced:
      pred_per_table[table] = [pred_tree]
      return
    else:
      split_tree_per_table(pred_tree["operand1"], pred_per_table)
      split_tree_per_table(pred_tree["operand2"], pred_per_table)


def create_conjunction_tree(
    pred_per_table,
):
  """Merge a list of of predicates in a single conjunction tree."""
  for table in pred_per_table:
    if isinstance(pred_per_table[table], dict):
      pred_per_table[table] = [pred_per_table[table]]

  for table in pred_per_table:
    if len(pred_per_table[table]) > 1:
      # create a conjuction tree
      root_pred = {
          "operator": "AND",
          "operand1": pred_per_table[table][0],
          "operand2": pred_per_table[table][1],
      }
      for i in range(2, len(pred_per_table[table])):
        new_root = {
            "operator": "AND",
            "operand1": root_pred,
            "operand2": pred_per_table[table][i],
        }
        root_pred = new_root
      pred_per_table[table] = root_pred
    elif len(pred_per_table[table]) == 1:
      pred_per_table[table] = pred_per_table[table][0]
    else:
      pred_per_table[table] = {}


def parse_pred(
    pred, debug = False
):
  """Parse a predicate string into a predicate tree."""

  pred_per_table = {}
  referenced_columns_per_table = {}
  pred = pred.strip()
  # Make all operators into a single word
  pred = pred.replace("IS NOT NULL", "!= NULL")
  pred = pred.replace("IS NULL", "= NULL")
  pred = pred.replace(";", "")

  pred_tree = parse_pred_rec(pred, referenced_columns_per_table, debug)
  split_tree_per_table(pred_tree, pred_per_table)

  # if a table has more than one pred_tree, create a conjuction of them
  create_conjunction_tree(pred_per_table)
  return pred_per_table, referenced_columns_per_table


def process_select_clause(select_clause):
  """Parse the select clause of a query into aggregate functions."""
  aggregate_functions = []
  select_clause = select_clause.strip()
  items = select_clause.split(",")
  if len(items) == 1:
    return aggregate_functions
  for item in items:
    if "rwcnt" in item:
      continue
    if "as" in item:
      item = item.split("as")[0].strip()
    agg_fn = item.split("(")[0].strip()
    agg_fn_args = item.split("(")[1].split(")")[0].strip()
    columns = []
    for agg_fn_arg in agg_fn_args.split(","):
      columns.append(agg_fn_arg.strip())
    aggregate_functions.append([agg_fn, columns])
  return aggregate_functions


def process_from_clause(
    from_clause, debug = False
):
  """Parse the from clause of a query into tables, aliases and join predicates."""
  from_clause = from_clause.strip().replace(";", "")
  from_clause = from_clause.replace("AS", "as")
  tables = []
  aliases = []
  join_preds_ret = []

  # From clase is of the form:
  # `table1` as t1 JOIN `table2` as t2 ON t1.column1 = table2.column2 JOIN
  # `table3` as t3 ON table2.column3 = table3.column4
  splitfrom = from_clause.split(" JOIN ")
  alias_a = splitfrom[0].split(" as ")[1]
  table_a = splitfrom[0].split(" as ")[0]
  table_a = table_a.replace("`", "")
  table_a = table_a.strip()
  printif(debug, f"alias_a: {alias_a}")
  printif(debug, f"table_a: {table_a}")

  tables.append(table_a)
  aliases.append(alias_a)

  for split_part in splitfrom[1:]:
    ss = split_part.split(" ON ")[0]
    alias_b = ss.split(" as ")[1]
    table_b = ss.split(" as ")[0]
    table_b = table_b.strip()
    table_b = table_b[1:-1]
    printif(debug, f"alias_b: {alias_b}")
    printif(debug, f"table_b: {table_b}")
    tables.append(table_b)
    aliases.append(alias_b)

    join_preds = []
    join_preds_all = split_part.split(" ON ")[1]
    for jp in join_preds_all.split(", "):
      for jpp in jp.split(" AND "):
        jpp = jpp.strip()
        jppsplit = jpp.split(" ")
        o1 = jppsplit[0].strip()
        o2 = jppsplit[2].strip()
        o1re = re.findall(r"[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+", o1)
        o2re = re.findall(r"[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+", o2)
        if len(o1re) < 1 or len(o2re) < 1:
          continue
        if o1 != o1re[0] or o2 != o2re[0]:
          continue
        join_preds.append({
            "operand1": jppsplit[0].strip(),
            "operator": jppsplit[1].strip(),
            "operand2": jppsplit[2].strip(),
            "concat": jpp,
        })

    printif(debug, f"join_preds: {str(join_preds)}")
    join_preds_ret.append(join_preds)

  return tables, aliases, join_preds_ret


def process_where_clause(
    where_clause, debug = False
):
  """Parse the where clause string into predicates per table and referenced columns per table."""
  # Remove dummy predicate that is part of some queries to bypass specific
  # database ssytem quirks
  where_clause = where_clause.replace("AND 1 = 1", "")
  where_clause = where_clause.replace("1 = 1", "")
  where_clause = where_clause.strip()
  if not where_clause:
    return {}, {}
  predicates_per_table, referenced_columns_per_table = parse_pred(
      where_clause, debug
  )
  for alias in predicates_per_table:
    printif(
        debug, f"For alias:{alias} preds: {str(predicates_per_table[alias])}"
    )
  return predicates_per_table, referenced_columns_per_table


def process_group_by_clause(
    group_by_clause, debug = False
):
  """Parse the group by clause string into grouping columns."""
  grouping_columns = []
  if not group_by_clause:
    return grouping_columns
  group_by_clause = group_by_clause.replace(";", "")
  for g in group_by_clause.split(","):
    g = g.strip()
    grouping_columns.append(g)
  printif(debug, f"grouping_columns: {grouping_columns}")
  return grouping_columns


def convert_sql_to_relational_operators(
    query_information, debug = False
):
  """Parse query sql string and convert to relational operators."""

  # The parsed_query contains SCAN and JOIN operators, the query cardinality and
  # the sql query string.
  parsed_query = {}
  parsed_query["ops"] = []
  query_sql_string = query_information["query_string"]

  # Split query string into select, from and where clauses
  select_clause = query_sql_string.split("FROM")[0]
  if "WHERE" in query_sql_string:
    from_clause = query_sql_string.split("FROM")[1].split("WHERE")[0]
  elif "GROUP BY" in query_sql_string:
    from_clause = query_sql_string.split("FROM")[1].split("GROUP BY")[0]
  else:
    from_clause = query_sql_string.split("FROM")[1]
  if "WHERE" in query_sql_string:
    if "GROUP BY" in query_sql_string:
      where_clause = (
          query_sql_string.split("FROM")[1]
          .split("WHERE")[1]
          .split("GROUP BY")[0]
      )
    else:
      where_clause = query_sql_string.split("FROM")[1].split("WHERE")[1]
  else:
    where_clause = ""
  if "GROUP BY" in query_sql_string:
    group_by_clause = query_sql_string.split("GROUP BY")[1]
  else:
    group_by_clause = ""

  printif(debug, f"select_clause: {select_clause}")
  printif(debug, f"from_clause: {from_clause}")
  printif(debug, f"where_clause: {where_clause}")
  printif(debug, f"group_by_clause: {group_by_clause}")

  tables, table_aliases, join_preds = process_from_clause(from_clause, debug)

  # referenced_columns_per_table contains all the columns of a table that are
  # used by the query. This includes filter and join predicates.
  predicates_per_table, referenced_columns_per_table = process_where_clause(
      where_clause, debug
  )

  grouping_columns = process_group_by_clause(group_by_clause, debug)
  aggregate_functions = process_select_clause(select_clause)

  printif(debug, f"aggregate_functions: {aggregate_functions}")
  printif(debug, f"grouping_columns: {grouping_columns}")

  # predicates_per_table and referenced_columns_per_table must contains all
  # the aliases
  for alias in table_aliases:
    if alias not in predicates_per_table:
      predicates_per_table[alias] = []
    if alias not in referenced_columns_per_table:
      referenced_columns_per_table[alias] = []

  # For each join predicate find all the referenced columns
  for alias in table_aliases:
    for join_pred in join_preds:
      find_referenced_columns_join_preds(
          join_pred, alias, referenced_columns_per_table[alias]
      )

  for grouping_column in grouping_columns:
    table, col = grouping_column.split(".")
    table = table.strip()
    col = col.strip()
    referenced_columns_per_table[table].append(col)

  for alias in table_aliases:
    referenced_columns_per_table[alias] = list(
        dict.fromkeys(referenced_columns_per_table[alias])
    )

  return convert_parsed_string_to_relational_operators(
      parsed_query,
      tables,
      table_aliases,
      join_preds,
      predicates_per_table,
      referenced_columns_per_table,
      grouping_columns,
      debug,
  )


def convert_parsed_string_to_relational_operators(
    parsed_query,
    tables,
    table_aliases,
    join_preds,
    predicates_per_table,
    referenced_columns_per_table,
    grouping_columns,
    debug = False,
):
  """Create the SCAN and JOIN operators of the plan from the parsed query."""

  # Each operator has a unique id
  nextopid = 0

  # Create scan operators
  for i in range(len(tables)):
    table_alias = table_aliases[i]
    scanop = {}
    scanop["relop_id"] = nextopid
    scanop["reloptype"] = "SCAN"
    scanop["table_name"] = tables[i]
    scanop["table_alias"] = table_alias
    scanop["columns"] = referenced_columns_per_table[table_alias]
    scanop["predicates"] = predicates_per_table[table_alias]
    parsed_query["ops"].append(scanop)
    nextopid += 1

  # Create join operators, creates a left-deep join tree
  for i in range(len(join_preds)):
    if i == 0:  # join between base tables
      input_opid_a = 0  # the id of the first scan operator
      input_opid_b = 1  # the id of the second scan operator
    else:  # join between base table and join operator
      input_opid_a = i + 1
      input_opid_b = i + len(join_preds)
    joinop = {
        "relop_id": nextopid,
        "reloptype": "JOIN",
        "predicates": join_preds[i],
        "input_opid_a": input_opid_a,
        "input_opid_b": input_opid_b,
    }
    parsed_query["ops"].append(joinop)
    nextopid += 1

  if grouping_columns:
    aggregateop = {
        "relop_id": nextopid,
        "reloptype": "GROUPBY",
        "input_opid": nextopid - 1,
        "grouping_columns": grouping_columns,
    }
    parsed_query["ops"].append(aggregateop)
    nextopid += 1

  printif(debug, "---- Query plan start <><><><>")
  for k in parsed_query["ops"]:
    printif(debug, k)
  printif(debug, "----")
  printif(debug, "---- Query plan end <><><><>")

  return parsed_query
