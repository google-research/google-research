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

"""Query generator from https://github.com/DataManagementLab/zero-shot-cost-estimation/tree/main, minor changes."""

import collections
import enum
import math
import os
import sys

import numpy as np

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training.generate_queries_library import load_jsons_utilities

load_schema_json = load_jsons_utilities.load_schema_json
load_column_statistics = load_jsons_utilities.load_column_statistics
load_string_statistics = load_jsons_utilities.load_string_statistics

Datatype = configuration.Datatype
open_file = open
file_exists = os.path.exists
remove_file = os.remove
Enum = enum.Enum


class LogicalOperator(Enum):
  AND = 'AND'
  OR = 'OR'

  def __str__(self):
    return self.value


class Operator(Enum):
  """Filter predicate operators."""

  NEQ = '!='
  EQ = '='
  LEQ = '<='
  GEQ = '>='
  LIKE = 'LIKE'
  NOT_LIKE = 'NOT LIKE'
  IS_NOT_NULL = 'IS NOT NULL'
  IS_NULL = 'IS NULL'
  IN = 'IN'
  BETWEEN = 'BETWEEN'

  def __str__(self):
    return self.value


class Aggregator(Enum):
  """Aggregation functions."""

  AVG = 'AVG'
  SUM = 'SUM'
  COUNT = 'COUNT'
  MIN = 'MIN'
  MAX = 'MAX'

  def __str__(self):
    return self.value


def rand_choice(randstate, l, no_elements=None, replace=False):
  if no_elements is None:
    idx = randstate.randint(0, len(l))
    return l[idx]
  else:
    idxs = randstate.choice(range(len(l)), no_elements, replace=replace)
    return [l[i] for i in idxs]


#### JOIN GENERATION ##########################################################


def find_possible_joins(join_tables, relationships_table):
  possible_joins = list()
  for t in join_tables:
    for column_l, table_r, column_r in relationships_table[t]:
      if table_r in join_tables:
        continue
      possible_joins.append((t, column_l, table_r, column_r))
  return possible_joins


def sample_acyclic_join(no_joins, relationships_table, schema, randstate):
  """Sample acyclic join.

  Args:
    no_joins: The number of joins.
    relationships_table: The relationships table.
    schema: The schema.
    randstate: The random state.

  Returns:
    The start table, joins, and join tables.
  """
  # randomly sample join
  joins = list()
  start_t = rand_choice(randstate, schema.tables)
  join_tables = {start_t}

  for _ in range(no_joins):
    possible_joins = find_possible_joins(join_tables, relationships_table)

    # randomly select one join
    if possible_joins:
      t, column_l, table_r, column_r = rand_choice(randstate, possible_joins)
      join_tables.add(table_r)

      joins.append((t, column_l, table_r, column_r))
    else:
      break
  return start_t, joins, join_tables


#### PREDICATE GENERATION #####################################################


class PredicateOperator:
  """Predicate operator."""

  def __init__(self, logical_op, children=None):
    self.logical_op = logical_op
    if children is None:
      children = []
    self.children = children

  def __str__(self):
    return self.to_sql(top_operator=True)

  def string_for_dedup(self):
    # liststrs
    pred_sorted = []
    for c in self.children:
      pred_sorted.extend(c.string_for_dedup())
    if LogicalOperator.OR == self.logical_op:
      pred_sorted.sort()
      return [f'{self.logical_op}_{"_".join(pred_sorted)}']
    else:
      return pred_sorted

  def to_sql(self, top_operator=False):
    """Convert to sql string."""
    sql = ''
    if self.children:
      # if len(self.children) == 1:
      #     return self.children[0].to_sql(top_operator=top_operator)

      predicates_str_list = [c.to_sql() for c in self.children]
      sql = f' {str(self.logical_op)} '.join(predicates_str_list)

      if top_operator:
        sql = f' WHERE {sql}'
      elif len(self.children) > 1:
        sql = f'({sql})'

    return sql


class ColumnPredicate:
  """Column predicate."""

  def __init__(self, table, col_name, operator, literal):
    self.table = table
    self.col_name = col_name
    self.operator = operator
    self.literal = literal

  def __str__(self):
    return self.to_sql(top_operator=True)

  def string_for_dedup(self):
    if self.operator == Operator.IS_NOT_NULL:
      return [f'{self.table}.{self.col_name} IS NOT NULL']
    elif self.operator == Operator.IS_NULL:
      return [f'{self.table}.{self.col_name} IS NULL']
    else:
      return [
          f'{self.table}.{self.col_name}_{str(self.operator)}_{str(self.literal)}'
      ]

  def to_sql(self, top_operator=False):
    """Covert to sql string."""
    if self.operator == Operator.IS_NOT_NULL:
      predicates_str = f'{self.table}.{self.col_name} IS NOT NULL'
    elif self.operator == Operator.IS_NULL:
      predicates_str = f'{self.table}.{self.col_name} IS NULL'
    else:
      predicates_str = (
          f'{self.table}.{self.col_name} {str(self.operator)} {self.literal}'
      )

    if top_operator:
      predicates_str = f' WHERE {predicates_str}'

    return predicates_str


def sample_literal_from_percentiles(percentiles, randstate):
  """Sample literal from percentiles.

  Args:
    percentiles: The percentiles.
    randstate: The random state.

  Returns:
    The literal.
  """
  if not percentiles:
    return None
  if isinstance(percentiles[0], str):
    percentiles = [int(x) for x in percentiles]
  if np.all(np.isnan(percentiles)):
    return None
  percentiles_non_nan = [x for x in percentiles if not math.isnan(x)]
  start_idx = randstate.randint(0, len(percentiles_non_nan) - 1)
  l = percentiles_non_nan[start_idx]
  h = percentiles_non_nan[start_idx + 1]

  if np.isinf(h):
    h = sys.maxsize
  literal = randstate.uniform(l, h)
  literal = int(literal)
  return literal


def analyze_columns(
    column_stats,
    group_by_col_unqiue_vals_treshold,
    join_tables,
    allowed_predicate_column_types,
    allowed_grouping_column_types,
    allowed_aggregation_column_types,
    column_disallow_list_for_predicates_aggs_and_group_bys,
    table_name_disallow_list_for_predicates_aggs_and_group_bys,
):
  """Analyze columns to identify possible columns for predicates, group bys, and aggregations."""
  # find possible columns for predicates
  possible_columns = []
  possible_group_by_columns = []
  possbile_aggregation_columns = []
  # also track how many columns we have per table to reweight them
  for t in join_tables:
    if t not in vars(column_stats):
      continue
    for col_name, col_stats in vars(vars(column_stats)[t]).items():
      if col_name in column_disallow_list_for_predicates_aggs_and_group_bys:
        continue
      if t in table_name_disallow_list_for_predicates_aggs_and_group_bys:
        continue
      # predicate columns
      if col_stats.datatype in [
          str(d).lower() for d in allowed_predicate_column_types
      ]:
        possible_columns.append([t, col_name])
      # group by columns
      if (
          col_stats.datatype
          in [str(d).lower() for d in allowed_grouping_column_types]
          and col_stats.num_unique < group_by_col_unqiue_vals_treshold
      ):
        possible_group_by_columns.append([t, col_name, col_stats.num_unique])

      # numerical aggregation columns
      if col_stats.datatype in [
          str(d) for d in allowed_aggregation_column_types
      ]:
        possbile_aggregation_columns.append([t, col_name])
  return (
      possible_columns,
      possible_group_by_columns,
      possbile_aggregation_columns,
  )


def sample_predicate(
    column_stats,
    t,
    col_name,
    eq_and_neq_unique_value_threshold_for_integer_columns,
    randstate,
    allowed_predicate_operators,
    p_is_not_null=0.01,
):
  """Sample a predicate.

  Args:
    column_stats: The column statistics.
    t: The table.
    col_name: The column name.
    eq_and_neq_unique_value_threshold_for_integer_columns: The threshold for int
      neq predicate.
    randstate: The random state.
    allowed_predicate_operators: The allowed predicate operators.
    p_is_not_null: The probability of sampling IS NOT NULL or IS NULL.

  Returns:
    The predicate.
  """
  literal = None
  col_stats = vars(vars(column_stats)[t]).get(col_name)
  reasonable_ops = set()

  if col_stats.nan_ratio > 0 and randstate.uniform() < p_is_not_null:
    reasonable_ops.add(Operator.IS_NOT_NULL)
    reasonable_ops.add(Operator.IS_NULL)

  if col_stats.datatype.lower() in [
      'int',
      'int64',
      'uint64',
      'int32',
      'uint32',
  ]:
    reasonable_ops.add(Operator.LEQ)
    reasonable_ops.add(Operator.GEQ)
    if (
        col_stats.num_unique
        < eq_and_neq_unique_value_threshold_for_integer_columns
    ):
      reasonable_ops.add(Operator.EQ)
      reasonable_ops.add(Operator.NEQ)
    literal = sample_literal_from_percentiles(col_stats.percentiles, randstate)
    if literal is None:
      return None

  elif col_stats.datatype.lower() in ['float64', 'float', 'double']:
    reasonable_ops.add(Operator.LEQ)
    reasonable_ops.add(Operator.GEQ)
    literal = sample_literal_from_percentiles(col_stats.percentiles, randstate)
    # nan comparisons only produce errors
    # happens when column is all nan
    if literal is None:
      return None
  elif col_stats.datatype in [str(Datatype.STRING), str(Datatype.CATEGORICAL)]:
    reasonable_ops.add(Operator.EQ)
    reasonable_ops.add(Operator.NEQ)
    possible_literals = []
    if hasattr(col_stats, 'unique_vals'):
      possible_literals = [
          v
          for v in col_stats.unique_vals
          if v is not None
          and not (isinstance(v, float) and np.isnan(v))
          and '(' not in v
          and ')' not in v
      ]
    if not possible_literals:
      return None
    literal = rand_choice(randstate, possible_literals)
    literal = f"'{literal}'"
  else:
    reasonable_ops = []

  # only allow reasonable ops from allow list
  reasonable_ops = [
      x for x in reasonable_ops if x in allowed_predicate_operators
  ]
  if not reasonable_ops:
    return None

  operator = rand_choice(randstate, reasonable_ops)

  return ColumnPredicate(t, col_name, operator, literal)


def sample_group_bys(no_group_bys, possible_group_by_columns, randstate):
  """Sample group bys.

  Args:
    no_group_bys: The number of group bys.
    possible_group_by_columns: The possible group by columns.
    randstate: The random state.

  Returns:
    The group by columns
  """
  group_bys = []
  if no_group_bys > 0:
    no_group_bys = min(no_group_bys, len(possible_group_by_columns))
    group_bys = rand_choice(
        randstate,
        possible_group_by_columns,
        no_elements=no_group_bys,
        replace=False,
    )
  return group_bys


def sample_predicates(
    column_stats,
    eq_and_neq_unique_value_threshold_for_integer_columns,
    no_predicates_per_table_list,
    possible_columns,
    randstate,
    allowed_predicate_operators,
):
  """Sample predicates.

  Args:
    column_stats: The column statistics.
    eq_and_neq_unique_value_threshold_for_integer_columns: The threshold for int
      neq predicate.
    no_predicates_per_table_list: The list of the number of predicates per
      table.
    possible_columns: The possible columns.
    randstate: The random state.
    allowed_predicate_operators: The allowed predicate operators.

  Returns:
    The predicates and the count of predicates per table.
  """
  # if not possible_columns:
  #   return None
  predicates = []
  count_preds_per_table = {}

  possible_columns_per_table = {}
  for t, col_name in possible_columns:
    if t not in possible_columns_per_table:
      possible_columns_per_table[t] = []
    possible_columns_per_table[t].append([t, col_name])

  i = 0
  no_predicates_per_table_dict = {}
  for t in possible_columns_per_table:
    num_possible_columns = len(possible_columns_per_table[t])
    no_predicates_per_table_dict[t] = min(
        no_predicates_per_table_list[i], num_possible_columns
    )
    i += 1

  for t in possible_columns_per_table:
    count_preds_per_table[t] = 0
    predicate_col_ids = randstate.choice(
        range(len(possible_columns_per_table[t])),
        no_predicates_per_table_dict[t],
        replace=False,
    )
    predicate_columns = [
        possible_columns_per_table[t][i] for i in predicate_col_ids
    ]
    for [t, col_name] in predicate_columns:
      p = sample_predicate(
          column_stats,
          t,
          col_name,
          eq_and_neq_unique_value_threshold_for_integer_columns,
          randstate,
          allowed_predicate_operators,
      )
      if p is not None:
        predicates.append(p)
        count_preds_per_table[p.table] += 1

  return (
      PredicateOperator(LogicalOperator.AND, predicates),
      count_preds_per_table,
  )


def generate_predicates(
    no_predicates_per_table,
    column_stats,
    eq_and_neq_unique_value_threshold_for_integer_columns,
    join_tables,
    randstate,
    allowed_predicate_operators,
    group_by_col_unqiue_vals_treshold,
    allowed_predicate_column_types,
    allowed_grouping_column_types,
    allowed_aggregation_column_types,
    column_disallow_list_for_predicates_aggs_and_group_bys,
    table_name_disallow_list_for_predicates_aggs_and_group_bys,
):
  """Generate predicates."""
  (
      possible_columns,
      possible_group_by_columns,
      possbile_aggregation_columns,
  ) = analyze_columns(
      column_stats,
      group_by_col_unqiue_vals_treshold,
      join_tables,
      allowed_predicate_column_types,
      allowed_grouping_column_types,
      allowed_aggregation_column_types,
      column_disallow_list_for_predicates_aggs_and_group_bys,
      table_name_disallow_list_for_predicates_aggs_and_group_bys,
  )
  # TODO(chronis): add support for disjunctions in the future here
  predicates, count_preds_per_table = sample_predicates(
      column_stats,
      eq_and_neq_unique_value_threshold_for_integer_columns,
      no_predicates_per_table,
      possible_columns,
      randstate,
      allowed_predicate_operators,
  )
  return (
      possbile_aggregation_columns,
      possible_group_by_columns,
      predicates,
      count_preds_per_table,
  )


def sample_query(
    max_cols_per_agg,
    min_aggregation_fns,
    max_aggregation_fns,
    min_number_joins,
    max_nunmber_joins,
    min_number_filter_predicates_per_table,
    max_number_filter_predicates_per_table,
    relationships_table,
    schema,
    min_number_group_bys_cols,
    max_number_group_by_cols,
    always_create_the_maximum_number_of_joins,
    always_create_the_maximum_number_of_aggregate_fns,
    always_create_the_maximum_number_of_predicates,
    always_create_the_maximum_number_of_group_bys,
    allowed_aggregate_functions,
    column_stats,
    eq_and_neq_unique_value_threshold_for_integer_columns,
    randstate,
    allowed_predicate_operators,
    group_by_col_unqiue_vals_treshold,
    allowed_predicate_column_types,
    allowed_grouping_column_types,
    allowed_aggregation_column_types,
    column_disallow_list_for_predicates_aggs_and_group_bys,
    table_name_disallow_list_for_predicates_aggs_and_group_bys,
):
  """Sample a query.

  Args:
    max_cols_per_agg: The maximum number of columns per aggregate.
    min_aggregation_fns: The minimum number of aggregation functions.
    max_aggregation_fns: The maximum number of aggregation functions.
    min_number_joins: The minimum number of joins.
    max_nunmber_joins: The maximum number of joins.
    min_number_filter_predicates_per_table: The minimum number of filter
      predicates per table.
    max_number_filter_predicates_per_table: The maximum number of filter
      predicates per table.
    relationships_table: The relationships table.
    schema: The schema.
    min_number_group_bys_cols: The minimum number of group bys columns.
    max_number_group_by_cols: The maximum number of group bys columns.
    always_create_the_maximum_number_of_joins: Whether to always create the
      maximum number of joins.
    always_create_the_maximum_number_of_aggregate_fns: Whether to always create
      the maximum number of aggregate functions.
    always_create_the_maximum_number_of_predicates: Whether to always create the
      maximum number of predicates.
    always_create_the_maximum_number_of_group_bys: Whether to always create the
      maximum number of group bys.
    allowed_aggregate_functions: The allowed aggregate functions.
    column_stats: The column statistics.
    eq_and_neq_unique_value_threshold_for_integer_columns: The threshold for int
      neq predicate.
    randstate: The random state.
    allowed_predicate_operators: The allowed predicate operators.
    group_by_col_unqiue_vals_treshold: The threshold for group by column unique
      values.
    allowed_predicate_column_types: The allowed predicate column types.
    allowed_grouping_column_types: The allowed grouping column types.
    allowed_aggregation_column_types: The allowed aggregation column types.
    column_disallow_list_for_predicates_aggs_and_group_bys: The disallow list
      for predicates, aggs, and group bys.
    table_name_disallow_list_for_predicates_aggs_and_group_bys: The disallow
      list for predicates, aggs, and group bys.

  Returns:
    The query and the count of predicates per table.
  """

  if max_nunmber_joins == 0:
    no_joins = 0
  elif always_create_the_maximum_number_of_joins:
    no_joins = max_nunmber_joins
  else:
    no_joins = randstate.randint(min_number_joins, max_nunmber_joins + 1)

  if max_number_filter_predicates_per_table == 0:
    no_predicates_per_table = [0] * (no_joins + 1)
  elif always_create_the_maximum_number_of_predicates:
    no_predicates_per_table = [max_number_filter_predicates_per_table] * (
        no_joins + 1
    )
  else:
    no_predicates_per_table = []
    for _ in range(no_joins + 1):
      no_predicates_per_table.append(
          randstate.randint(
              min_number_filter_predicates_per_table,
              max_number_filter_predicates_per_table + 1,
          )
      )

  if max_aggregation_fns == 0:
    no_aggregates = 0
  elif always_create_the_maximum_number_of_aggregate_fns:
    no_aggregates = max_aggregation_fns
  else:
    no_aggregates = randstate.randint(
        min_aggregation_fns, max_aggregation_fns + 1
    )

  if max_number_group_by_cols == 0:
    no_group_bys = 0
  elif always_create_the_maximum_number_of_group_bys:
    no_group_bys = max_number_group_by_cols
  else:
    no_group_bys = randstate.randint(
        min_number_group_bys_cols, max_number_group_by_cols + 1
    )

  start_t, joins, join_tables = sample_acyclic_join(
      no_joins, relationships_table, schema, randstate
  )

  (
      possbile_aggregation_columns,
      possible_group_by_columns,
      predicates,
      count_preds_per_table,
  ) = generate_predicates(
      no_predicates_per_table,
      column_stats,
      eq_and_neq_unique_value_threshold_for_integer_columns,
      join_tables,
      randstate,
      allowed_predicate_operators,
      group_by_col_unqiue_vals_treshold,
      allowed_predicate_column_types,
      allowed_grouping_column_types,
      allowed_aggregation_column_types,
      column_disallow_list_for_predicates_aggs_and_group_bys,
      table_name_disallow_list_for_predicates_aggs_and_group_bys,
  )

  group_bys = sample_group_bys(
      no_group_bys, possible_group_by_columns, randstate
  )
  aggregations = sample_aggregations(
      max_cols_per_agg,
      no_aggregates,
      possbile_aggregation_columns,
      randstate,
      allowed_aggregate_functions,
  )

  q = GenQuery(
      aggregations,
      group_bys,
      joins,
      predicates,
      start_t,
      list(join_tables),
  )
  return q, count_preds_per_table


def sample_aggregations(
    max_cols_per_agg,
    no_aggregates,
    numerical_aggregation_columns,
    randstate,
    allowed_aggregate_functions,
):
  """Sample aggregations.

  Args:
    max_cols_per_agg: The maximum number of columns per aggregate.
    no_aggregates: The number of aggregates.
    numerical_aggregation_columns: The numerical aggregation columns.
    randstate: The random state.
    allowed_aggregate_functions: The allowed aggregate functions.

  Returns:
    The aggregations.
  """
  aggregations = []
  already_produced_count = False
  if no_aggregates > 0:
    for _ in range(no_aggregates):
      no_agg_cols = min(
          randstate.randint(1, max_cols_per_agg + 1),
          len(numerical_aggregation_columns),
      )
      l = list(allowed_aggregate_functions)
      agg = rand_choice(randstate, l)
      cols = rand_choice(
          randstate,
          numerical_aggregation_columns,
          no_elements=no_agg_cols,
          replace=False,
      )
      if agg == Aggregator.COUNT and already_produced_count:
        continue
      if agg == Aggregator.COUNT:
        already_produced_count = True
        cols = []
      if no_agg_cols == 0 and agg != Aggregator.COUNT:
        continue
      aggregations.append((agg, cols))
    if not aggregations:
      aggregations.append((Aggregator.COUNT, []))
  return aggregations


################################################################################


class GenQuery:
  """Generate SQL query."""

  def __init__(
      self,
      aggregations,
      group_bys,
      joins,
      predicates,
      start_t,
      join_tables,
      alias_dict=None,
      inner_groupby=None,
      subquery_alias=None,
      limit=None,
      having_clause=None,
  ):
    if alias_dict is None:
      alias_dict = dict()
    self.aggregations = aggregations
    self.group_bys = group_bys
    self.joins = joins
    self.predicates = predicates
    self.start_t = start_t
    self.join_tables = join_tables
    self.alias_dict = alias_dict
    self.exists_predicates = []
    self.inner_groupby = inner_groupby
    self.subquery_alias = subquery_alias
    self.limit = limit
    self.having_clause = having_clause
    if self.inner_groupby is not None:
      self.alias_dict = {t: subquery_alias for t in self.join_tables}

  def append_exists_predicate(self, q_rec, not_exist):
    self.exists_predicates.append((q_rec, not_exist))

  def generate_sql_query(
      self,
      sql_table_prefix,
      quote_table_sql_string,
      partitioning_predicate_per_table,
      use_partitioning_predicate,
      dedup_query_strs,
      semicolon=True,
  ):
    """Generate SQL query.

    Args:
      sql_table_prefix: The SQL table prefix.
      quote_table_sql_string: The SQL string to quote table names.
      partitioning_predicate_per_table: The partitioning predicate per table.
      use_partitioning_predicate: Whether to use partitioning predicate.
      dedup_query_strs: The dedup query strs.
      semicolon: Whether to add semicolon.

    Returns:
      The SQL query.
    """

    # group_bys
    group_by_str = ''
    order_by_str = ''

    group_by_cols = []
    if self.group_bys:
      group_by_cols = [
          f'{table}.{column}' for table, column, _ in self.group_bys
      ]
      group_by_col_str = ', '.join(group_by_cols)
      group_by_str = f' GROUP BY {group_by_col_str}'
      # order_by_str = f' ORDER BY {group_by_col_str}'

    # aggregations
    aggregation_str_list = []
    aggregation_str = ''
    for _, (aggregator, columns) in enumerate(self.aggregations):
      if aggregator == Aggregator.COUNT:
        aggregation_str_list.append('COUNT(*)')
      else:
        agg_cols = ' + '.join([f'{table}.{col}' for table, col in columns])
        aggregation_str_list.append(f'{str(aggregator)}({agg_cols})')
      aggregation_str = ', '.join(
          group_by_cols
          + [f'{agg} as agg_{i}' for i, agg in enumerate(aggregation_str_list)]
      )

    agg_query = False
    if group_by_cols:
      aggregation_str = 'count(*)'
      agg_query = True
    elif not aggregation_str:
      aggregation_str = 'count(*) as rwcnt'

    # having clause
    having_str = ''
    if self.having_clause is not None:
      idx, literal, op = self.having_clause
      having_str = f' HAVING {aggregation_str_list[idx]} {str(op)} {literal}'

    # predicates
    predicate_str = str(self.predicates)

    # other parts can simply be replaced with aliases
    for t, alias_t in self.alias_dict.items():
      predicate_str = predicate_str.replace(f'{t}', alias_t)
      aggregation_str = aggregation_str.replace(f'{t}', alias_t)
      group_by_str = group_by_str.replace(f'{t}', alias_t)
      order_by_str = order_by_str.replace(f'{t}', alias_t)
      having_str = having_str.replace(f'{t}', alias_t)

    # join
    def repl_alias(t, no_alias_intro=False):
      if t in self.alias_dict:
        alias_t = self.alias_dict[t]
        if t in already_repl or no_alias_intro:
          return alias_t
        else:
          return f'{t} {alias_t}'
      return f'{t}'

    join_tables = set()
    join_conds = []
    if self.inner_groupby is not None:
      join_str = (
          f'({self.inner_groupby.generate_sql_query(sql_table_prefix, quote_table_sql_string, partitioning_predicate_per_table, use_partitioning_predicate, semicolon=False)})'
          f' {self.subquery_alias}'
      )
    else:
      already_repl = set()

      join_str = (
          f'{quote_table_sql_string}{sql_table_prefix}{repl_alias(self.start_t)}{quote_table_sql_string} as'
          f' {repl_alias(self.start_t)}'
      )

      join_tables.add(self.start_t)
      for (
          table_l,
          column_l,
          table_r,
          column_r,
      ) in self.joins:
        join_kw = 'JOIN'
        join_str += (
            f' {join_kw} {quote_table_sql_string}{sql_table_prefix}{repl_alias(table_r)}{quote_table_sql_string} as'
            f' {repl_alias(table_r)}'
        )
        join_cond = ' AND '.join([
            f'{repl_alias(table_l, no_alias_intro=True)}.{col_l} = '
            f'{repl_alias(table_r, no_alias_intro=True)}.{col_r}'
            for col_l, col_r in zip(column_l, column_r)
        ])
        join_conds.append(join_cond)
        join_str += f' ON {join_cond}'
        join_tables.add(table_l)
        join_tables.add(table_r)

    limit_str = ''
    if self.limit is not None:
      limit_str = f' LIMIT {self.limit}'

    partitioning_preds = ''
    if use_partitioning_predicate:
      for table in join_tables:
        if repl_alias(table) in partitioning_predicate_per_table:
          part_pred = partitioning_predicate_per_table[repl_alias(table)]
          if part_pred is None:
            continue
          if part_pred in partitioning_preds:
            continue
          partitioning_preds += f' AND {part_pred}'
      if partitioning_preds and (predicate_str == 'None' or not predicate_str):
        predicate_str = 'WHERE '
        partitioning_preds = partitioning_preds.replace('AND', '', 1)

    if agg_query:
      sql_query = (
          f'select count(*) as rwcnt from (SELECT {aggregation_str} FROM'
          f' {join_str} {predicate_str}{partitioning_preds}{group_by_str}{having_str}{order_by_str}{limit_str})'
          .strip()
      )
    else:
      sql_query = (
          f'SELECT {aggregation_str} FROM'
          f' {join_str} {predicate_str}{partitioning_preds}{group_by_str}{having_str}{order_by_str}{limit_str}'
          .strip()
      )

    if semicolon:
      sql_query += ';'

    # print('sql_query', sql_query)
    dedup_str = ''
    join_conds.sort()
    dedup_preds = self.predicates.string_for_dedup()
    dedup_preds.sort()
    dedup_groupby = group_by_cols
    dedup_groupby.sort()

    for t in join_tables:
      dedup_str += str(t) + '_'
    dedup_str += '_'
    for p in dedup_preds:
      dedup_str += str(p)
    dedup_str += '_'
    for p in join_conds:
      dedup_str += str(p)
    dedup_str += '_'
    for p in dedup_groupby:
      dedup_str += str(p)

    if dedup_str in dedup_query_strs:
      # print('dedup', dedup_str, len(dedup_query_strs))
      return None
    else:
      dedup_query_strs.append(dedup_str)
      return sql_query

    # if dedup_str not in dedup_query_strs:
    #   dedup_query_strs.append(dedup_str)
    # return sql_query


###############################################################################


def check_num_in_spec(num, min_num, max_num, always_create_the_max):
  if always_create_the_max and num < max_num:
    return False
  if num < min_num:
    return False
  if num > max_num:
    return False
  return True


def check_matches_criteria(
    q,
    count_preds_per_table,
    min_number_filter_predicates_per_table,
    max_number_filter_predicates_per_table,
    always_create_the_maximum_number_of_predicates,
    min_number_joins,
    max_nunmber_joins,
    always_create_the_maximum_number_of_joins,
    min_number_group_bys_cols,
    max_number_group_by_cols,
    always_create_the_maximum_number_of_group_bys,
    min_aggregation_fns,
    max_aggregation_fns,
    always_create_the_maximum_number_of_aggregate_fns,
):
  """Check if the query matches the criteria.

  Args:
    q: The query.
    count_preds_per_table: The count of predicates per table.
    min_number_filter_predicates_per_table: The minimum number of filter
      predicates per table.
    max_number_filter_predicates_per_table: The maximum number of filter
      predicates per table.
    always_create_the_maximum_number_of_predicates: Whether to always create the
      maximum number of predicates.
    min_number_joins: The minimum number of joins.
    max_nunmber_joins: The maximum number of joins.
    always_create_the_maximum_number_of_joins: Whether to always create the
      maximum number of joins.
    min_number_group_bys_cols: The minimum number of group bys columns.
    max_number_group_by_cols: The maximum number of group bys columns.
    always_create_the_maximum_number_of_group_bys: Whether to always create the
      maximum number of group bys.
    min_aggregation_fns: The minimum number of aggregation functions.
    max_aggregation_fns: The maximum number of aggregation functions.
    always_create_the_maximum_number_of_aggregate_fns: Whether to always create
      the maximum number of aggregate functions.

  Returns:
    Whether the query matches the criteria.
  """
  desired_query = True
  no_joins = len(q.joins)
  no_aggregates = len(q.aggregations)
  no_group_bys = len(q.group_bys)

  desired_query &= check_num_in_spec(
      no_joins,
      min_number_joins,
      max_nunmber_joins,
      always_create_the_maximum_number_of_joins,
  )
  desired_query &= check_num_in_spec(
      no_aggregates,
      min_aggregation_fns,
      max_aggregation_fns,
      always_create_the_maximum_number_of_aggregate_fns,
  )
  desired_query &= check_num_in_spec(
      no_group_bys,
      min_number_group_bys_cols,
      max_number_group_by_cols,
      always_create_the_maximum_number_of_group_bys,
  )
  for _, count in count_preds_per_table.items():
    if always_create_the_maximum_number_of_predicates:
      if count != max_number_filter_predicates_per_table:
        desired_query = False
        break
    else:
      if count < min_number_filter_predicates_per_table:
        desired_query = False
        break
      if count > max_number_filter_predicates_per_table:
        desired_query = False
        break

  return desired_query


def generate_queries(
    sql_table_prefix,
    quote_table_sql_string,
    dataset_name,
    dataset_json_input_directory_path,
    query_file_output_path,
    partitioning_predicate_per_table,
    use_partitioning_predicate,
    recreate_query_file_if_exists,
    seed,
    # query number
    num_queries_to_generate,
    # filter parameters
    min_number_filter_predicates_per_table,
    max_number_filter_predicates_per_table,
    always_create_the_maximum_number_of_predicates,
    allowed_predicate_operators,
    allowed_predicate_column_types,
    eq_and_neq_unique_value_threshold_for_integer_columns,
    # join parameters
    min_number_joins,
    max_nunmber_joins,
    always_create_the_maximum_number_of_joins,
    # grouping parameters
    min_number_group_bys_cols,
    max_number_group_by_cols,
    allowed_grouping_column_types,
    always_create_the_maximum_number_of_group_bys,
    group_by_col_unqiue_vals_treshold,
    # aggregation parameters
    min_aggregation_fns,
    max_aggregation_fns,
    always_create_the_maximum_number_of_aggregate_fns,
    allowed_aggregate_functions,
    max_cols_per_agg,
    allowed_aggregation_column_types,
    # disallow lists
    column_disallow_list_for_predicates_aggs_and_group_bys,
    table_name_disallow_list_for_predicates_aggs_and_group_bys,
):
  """Generate SQL queries."""
  randstate = np.random.RandomState(seed)

  print('Generate workload: ', query_file_output_path)

  if file_exists(query_file_output_path) and not recreate_query_file_if_exists:
    print('Workload already generated')
    return query_file_output_path

  # read the schema file
  column_stats = load_column_statistics(
      dataset_json_input_directory_path, dataset_name
  )
  # string_stats = load_string_statistics(
  #     dataset_json_input_directory_path, dataset_name
  # )
  schema = load_schema_json(dataset_json_input_directory_path, dataset_name)

  # build index of join relationships∏
  relationships_table = collections.defaultdict(list)
  for table_l, column_l, table_r, column_r in schema.relationships:
    if not isinstance(column_l, list):
      column_l = [column_l]
    if not isinstance(column_r, list):
      column_r = [column_r]

    relationships_table[table_l].append([column_l, table_r, column_r])
    relationships_table[table_r].append([column_r, table_l, column_l])

  queries = []
  dedup_query_strs = []
  count_queries = 0
  failed_queries = 0
  total_tries = 0
  count_1 = 0
  count_2 = 0
  count_3 = 0

  for _ in range(num_queries_to_generate):
    # sample query as long as it does not meet requirements
    tries = 0
    desired_query = False
    while not desired_query:
      total_tries += 1
      q, count_preds_per_table = sample_query(
          max_cols_per_agg,
          min_aggregation_fns,
          max_aggregation_fns,
          min_number_joins,
          max_nunmber_joins,
          min_number_filter_predicates_per_table,
          max_number_filter_predicates_per_table,
          relationships_table,
          schema,
          min_number_group_bys_cols,
          max_number_group_by_cols,
          always_create_the_maximum_number_of_joins,
          always_create_the_maximum_number_of_aggregate_fns,
          always_create_the_maximum_number_of_predicates,
          always_create_the_maximum_number_of_group_bys,
          allowed_aggregate_functions,
          column_stats,
          eq_and_neq_unique_value_threshold_for_integer_columns,
          randstate,
          allowed_predicate_operators,
          group_by_col_unqiue_vals_treshold,
          allowed_predicate_column_types,
          allowed_grouping_column_types,
          allowed_aggregation_column_types,
          column_disallow_list_for_predicates_aggs_and_group_bys,
          table_name_disallow_list_for_predicates_aggs_and_group_bys,
      )
      desired_query |= check_matches_criteria(
          q,
          count_preds_per_table,
          min_number_filter_predicates_per_table,
          max_number_filter_predicates_per_table,
          always_create_the_maximum_number_of_predicates,
          min_number_joins,
          max_nunmber_joins,
          always_create_the_maximum_number_of_joins,
          min_number_group_bys_cols,
          max_number_group_by_cols,
          always_create_the_maximum_number_of_group_bys,
          min_aggregation_fns,
          max_aggregation_fns,
          always_create_the_maximum_number_of_aggregate_fns,
      )
      sql_query = None
      if desired_query:
        sql_query = q.generate_sql_query(
            sql_table_prefix,
            quote_table_sql_string,
            partitioning_predicate_per_table,
            use_partitioning_predicate,
            dedup_query_strs,
        )
        count_1 += 1
        if sql_query is None:
          count_2 += 1
          desired_query = False

      # check that query performs a join a filter or a group by
      if sql_query and (
          'WHERE' not in sql_query
          and 'GROUP BY' not in sql_query
          and 'JOIN' not in sql_query
      ):
        count_3 += 1
        desired_query = False

      if desired_query:
        # print('sucess !!',)
        queries.append(sql_query)
      else:
        failed_queries += 1
        # print('fail !!')
        tries += 1
        if tries > 10000:
          raise ValueError(
              'Did not find a valid query after 10000 trials. '
              'Please check if your conditions can be fulfilled'
          )
      count_queries += 1
  print('total_tries', total_tries)
  print('counts', count_1, count_2, count_3)
  print('failed_queries', failed_queries)
  print('dedup queries', len(dedup_query_strs))

  with open_file(query_file_output_path, 'w') as text_file:
    text_file.write('\n'.join(queries))

  return query_file_output_path
