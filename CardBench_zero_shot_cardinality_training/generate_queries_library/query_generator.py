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


def sample_acyclic_aggregation_query(
    column_stats,
    string_stats,
    group_by_threshold,
    int_neq_predicate_threshold,
    max_cols_per_agg,
    max_number_aggregates,
    max_number_group_by,
    max_nunmber_joins,
    max_number_filter_predicates,
    relationships_table,
    schema,
    randstate,
    complex_predicates,
    always_create_the_maximum_number_of_joins,
    always_create_the_maximum_number_of_aggregates,
    always_create_the_maximum_number_of_predicates,
    always_create_the_maximum_number_of_group_bys,
    left_outer_join_ratio,
    groupby_limit_probability,
    groupby_having_probability,
    allowed_predicate_operators,
    allowed_aggregate_functions,
):
  """Sample acyclic aggregation query.

  Args:
    column_stats: The column statistics.
    string_stats: The string statistics.
    group_by_threshold: The group by threshold.
    int_neq_predicate_threshold: The int neq predicate threshold.
    max_cols_per_agg: The maximum number of columns per aggregate.
    max_number_aggregates: The maximum number of aggregates.
    max_number_group_by: The maximum number of group bys.
    max_nunmber_joins: The maximum number of joins.
    max_number_filter_predicates: The maximum number of filter predicates.
    relationships_table: The relationships table.
    schema: The schema.
    randstate: The random state.
    complex_predicates: Whether the predicates are complex.
    always_create_the_maximum_number_of_joins: Whether to always create the
      maximum number of joins.
    always_create_the_maximum_number_of_aggregates: Whether to always create the
      maximum number of aggregates.
    always_create_the_maximum_number_of_predicates: Whether to always create the
      maximum number of predicates.
    always_create_the_maximum_number_of_group_bys: Whether to always create the
      maximum number of group bys.
    left_outer_join_ratio: The ratio of left outer joins.
    groupby_limit_probability: The probability of group by limit.
    groupby_having_probability: The probability of group by having.
    allowed_predicate_operators: The allowed predicate operators.
    allowed_aggregate_functions: The allowed aggregate functions.

  Returns:
    The query.
  """
  if max_nunmber_joins == 0:
    no_joins = 0
  else:
    no_joins = randstate.randint(0, max_nunmber_joins + 1)

  if max_number_filter_predicates == 0:
    no_predicates = 0
  else:
    no_predicates = randstate.randint(1, max_number_filter_predicates + 1)

  if max_number_aggregates == 0:
    no_aggregates = 0
  else:
    no_aggregates = randstate.randint(1, max_number_aggregates + 1)

  if max_number_group_by == 0:
    no_group_bys = 0
  else:
    no_group_bys = randstate.randint(0, max_number_group_by + 1)

  if always_create_the_maximum_number_of_joins:
    no_joins = max_nunmber_joins
  if always_create_the_maximum_number_of_predicates:
    no_predicates = max_number_filter_predicates
  if always_create_the_maximum_number_of_aggregates:
    no_aggregates = max_number_aggregates
  if always_create_the_maximum_number_of_group_bys:
    no_group_bys = max_number_group_by

  start_t, joins, join_tables = sample_acyclic_join(
      no_joins, relationships_table, schema, randstate, left_outer_join_ratio
  )

  numerical_aggregation_columns, possible_group_by_columns, predicates = (
      generate_predicates(
          column_stats,
          complex_predicates,
          group_by_threshold,
          int_neq_predicate_threshold,
          join_tables,
          no_predicates,
          randstate,
          string_stats,
          allowed_predicate_operators,
      )
  )
  limit = None
  if randstate.rand() < groupby_limit_probability:
    limit = randstate.choice([10, 100, 1000])

  group_bys = sample_group_bys(
      no_group_bys, possible_group_by_columns, randstate
  )
  aggregations = sample_aggregations(
      max_cols_per_agg,
      no_aggregates,
      numerical_aggregation_columns,
      randstate,
      allowed_aggregate_functions,
  )
  having_clause = None
  if randstate.rand() < groupby_having_probability:
    idx = randstate.randint(0, len(aggregations))
    _, cols = aggregations[idx]
    literal = sum(
        [vars(vars(column_stats)[col[0]])[col[1]].mean for col in cols]
    )
    op = rand_choice(randstate, [Operator.LEQ, Operator.GEQ, Operator.NEQ])
    having_clause = (idx, literal, op)
  q = GenQuery(
      aggregations,
      group_bys,
      joins,
      predicates,
      start_t,
      list(join_tables),
      limit=limit,
      having_clause=having_clause,
  )
  return q


def generate_predicates(
    column_stats,
    complex_predicates,
    group_by_threshold,
    int_neq_predicate_threshold,
    join_tables,
    no_predicates,
    randstate,
    string_stats,
    allowed_predicate_operators,
):
  """Generate predicates."""
  (
      numerical_aggregation_columns,
      possible_columns,
      possible_string_columns,
      possible_group_by_columns,
      table_predicates,
      string_table_predicates,
  ) = analyze_columns(
      column_stats,
      group_by_threshold,
      join_tables,
      string_stats,
      complex_predicates,
  )
  if complex_predicates:
    predicates = sample_complex_predicates(
        column_stats,
        string_stats,
        int_neq_predicate_threshold,
        no_predicates,
        possible_columns,
        possible_string_columns,
        table_predicates,
        string_table_predicates,
        randstate,
        allowed_predicate_operators,
    )
  else:
    predicates = sample_predicates(
        column_stats,
        int_neq_predicate_threshold,
        no_predicates,
        possible_columns,
        table_predicates,
        randstate,
        allowed_predicate_operators,
    )
  return numerical_aggregation_columns, possible_group_by_columns, predicates


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
      table_identifier,
      quote_table_sql_string,
      partitioning_predicate_per_table,
      semicolon=True,
  ):
    """Generate SQL query.

    Args:
      table_identifier: id for table to be used in the sql query like
        table_identifiertable_name
      quote_table_sql_string: The string to quote table names in SQL.
      partitioning_predicate_per_table: The partitioning predicate per table.
      semicolon: Whether to end the query with a semicolon.

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
    if not aggregation_str:
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
    if self.inner_groupby is not None:
      join_str = (
          f'({self.inner_groupby.generate_sql_query(table_identifier, quote_table_sql_string, partitioning_predicate_per_table, semicolon=False)})'
          f' {self.subquery_alias}'
      )
    else:
      already_repl = set()

      join_str = (
          f'{quote_table_sql_string}{table_identifier}{repl_alias(self.start_t)}{quote_table_sql_string} as'
          f' {repl_alias(self.start_t)}'
      )

      join_tables.add(self.start_t)
      for table_l, column_l, table_r, column_r, left_outer in self.joins:
        join_kw = 'JOIN'
        if left_outer:
          join_kw = 'LEFT OUTER JOIN'
        join_str += (
            f' {join_kw} {quote_table_sql_string}{table_identifier}{repl_alias(table_r)}{quote_table_sql_string} as'
            f' {repl_alias(table_r)}'
        )
        join_cond = ' AND '.join([
            f'{repl_alias(table_l, no_alias_intro=True)}.{col_l} = '
            f'{repl_alias(table_r, no_alias_intro=True)}.{col_r}'
            for col_l, col_r in zip(column_l, column_r)
        ])
        join_str += f' ON {join_cond}'
        join_tables.add(table_l)
        join_tables.add(table_r)

    limit_str = ''
    if self.limit is not None:
      limit_str = f' LIMIT {self.limit}'

    partitioning_preds = ''

    for table in join_tables:
      if repl_alias(table) in partitioning_predicate_per_table:
        part_pred = partitioning_predicate_per_table[repl_alias(table)]
        if part_pred is None:
          continue
        if part_pred in partitioning_preds:
          continue
        partitioning_preds += f' AND {part_pred}'
    if partitioning_preds and predicate_str == 'None':
      predicate_str = 'WHERE '
      partitioning_preds = partitioning_preds.replace('AND', '', 1)
    sql_query = (
        f'SELECT {aggregation_str} FROM'
        f' {join_str} {predicate_str}{partitioning_preds}{group_by_str}{having_str}{order_by_str}{limit_str}'
        .strip()
    )

    if semicolon:
      sql_query += ';'

    return sql_query


def generate_queries(
    table_identifier,
    quote_table_sql_string,
    dataset_name,
    dataset_json_input_directory_path,
    query_file_output_path,
    partitioning_predicate_per_table,
    allowed_predicate_operators,
    allowed_aggregate_functions,
    num_queries_to_generate,
    max_nunmber_joins,
    max_number_filter_predicates,
    max_number_aggregates,
    max_number_group_by,
    max_cols_per_agg,
    group_by_threshold,
    int_neq_predicate_threshold,
    seed,
    complex_predicates,
    recreate_query_file_if_exists,
    always_create_the_maximum_number_of_joins,
    always_create_the_maximum_number_of_aggregates,
    always_create_the_maximum_number_of_predicates,
    always_create_the_maximum_number_of_group_bys,
    left_outer_join_ratio,
    groupby_limit_probability,
    groupby_having_probability,
    min_number_joins,
    min_number_predicates,
    min_grouping_cols,
    min_aggregation_cols,
):
  """Generate SQL queries.

  Args:
    table_identifier: id for table to be used in the sql query like
      table_identifiertable_name
    quote_table_sql_string: The string to quote table names in SQL.
    dataset_name: The dataset name.
    dataset_json_input_directory_path: The dataset json input directory path.
    query_file_output_path: The query file output path.
    partitioning_predicate_per_table: The partitioning predicate per table.
    allowed_predicate_operators: The allowed predicate operators.
    allowed_aggregate_functions: The allowed aggregate functions.
    num_queries_to_generate: The number of queries to generate.
    max_nunmber_joins: The maximum number of joins.
    max_number_filter_predicates: The maximum number of filter predicates.
    max_number_aggregates: The maximum number of aggregates.
    max_number_group_by: The maximum number of group bys.
    max_cols_per_agg: The maximum number of columns per aggregate.
    group_by_threshold: The group by threshold.
    int_neq_predicate_threshold: The int neq predicate threshold.
    seed: The seed.
    complex_predicates: Whether the predicates are complex.
    recreate_query_file_if_exists: Whether to recreate the query file if it
      exists.
    always_create_the_maximum_number_of_joins: Whether to always create the
      maximum number of joins.
    always_create_the_maximum_number_of_aggregates: Whether to always create the
      maximum number of aggregates.
    always_create_the_maximum_number_of_predicates: Whether to always create the
      maximum number of predicates.
    always_create_the_maximum_number_of_group_bys: Whether to always create the
      maximum number of group bys.
    left_outer_join_ratio: The ratio of left outer joins.
    groupby_limit_probability: The probability of group by limit.
    groupby_having_probability: The probability of group by having.
    min_number_joins: The minimum number of joins.
    min_number_predicates: The minimum number of predicates.
    min_grouping_cols: minimum number of grouping columns.
    min_aggregation_cols: minimum number of aggregation columns.

  Returns:
    The query file output path.
  """

  if min_grouping_cols > 0 or min_aggregation_cols > 0:
    print(
        'Support for min grouping and aggregation columns is not'
        ' implemented yet'
    )

  randstate = np.random.RandomState(seed)

  print('Generate workload: ', query_file_output_path)

  if file_exists(query_file_output_path) and not recreate_query_file_if_exists:
    print('Workload already generated')
    return query_file_output_path

  # read the schema file
  column_stats = load_column_statistics(
      dataset_json_input_directory_path, dataset_name
  )
  string_stats = load_string_statistics(
      dataset_json_input_directory_path, dataset_name
  )
  schema = load_schema_json(dataset_json_input_directory_path, dataset_name)

  # build index of join relationships∏li
  relationships_table = collections.defaultdict(list)
  for table_l, column_l, table_r, column_r in schema.relationships:
    if not isinstance(column_l, list):
      column_l = [column_l]
    if not isinstance(column_r, list):
      column_r = [column_r]

    relationships_table[table_l].append([column_l, table_r, column_r])
    relationships_table[table_r].append([column_r, table_l, column_l])

  queries = []
  failed_queries = 0
  total_tries = 0
  for _ in range(num_queries_to_generate):
    # sample query as long as it does not meet requirements
    tries = 0
    desired_query = False
    while not desired_query:
      total_tries += 1
      q = sample_acyclic_aggregation_query(
          column_stats,
          string_stats,
          group_by_threshold,
          int_neq_predicate_threshold,
          max_cols_per_agg,
          max_number_aggregates,
          max_number_group_by,
          max_nunmber_joins,
          max_number_filter_predicates,
          relationships_table,
          schema,
          randstate,
          complex_predicates,
          always_create_the_maximum_number_of_joins,
          always_create_the_maximum_number_of_aggregates,
          always_create_the_maximum_number_of_predicates,
          always_create_the_maximum_number_of_group_bys,
          left_outer_join_ratio,
          groupby_limit_probability,
          groupby_having_probability,
          allowed_predicate_operators,
          allowed_aggregate_functions,
      )

      # Check if query matches criteria otherwise retry
      desired_query |= check_matches_criteria(
          q,
          complex_predicates,
          max_number_aggregates,
          always_create_the_maximum_number_of_aggregates,
          max_number_group_by,
          always_create_the_maximum_number_of_group_bys,
          max_nunmber_joins,
          always_create_the_maximum_number_of_joins,
          max_number_filter_predicates,
          always_create_the_maximum_number_of_predicates,
          min_number_joins,
          min_number_predicates,
      )

      if desired_query:
        sql_query = q.generate_sql_query(
            table_identifier,
            quote_table_sql_string,
            partitioning_predicate_per_table,
        )
        if 'WHERE' not in sql_query:
          desired_query = False
        else:
          queries.append(sql_query)

      if not desired_query:
        tries += 1
        failed_queries += 1
        if tries > 10000:
          raise ValueError(
              'Did not find a valid query after 10000 trials. '
              'Please check if your conditions can be fulfilled'
          )
  print('total_tries', total_tries)
  print('failed_queries', failed_queries)
  with open_file(query_file_output_path, 'w') as text_file:
    text_file.write('\n'.join(queries))

  return query_file_output_path


def sample_outer_groupby(allowed_aggregate_functions, q, randstate):
  """Sample outer groupby.

  Args:
    allowed_aggregate_functions: The allowed aggregate functions.
    q: The query to sample outer groupby for.
    randstate: The random state.

  Returns:
    The outer groupby query.
  """
  subquery_alias = 'subgb'
  outer_aggs = []
  for i, (_, _) in enumerate(q.aggregations):
    l = list(allowed_aggregate_functions)
    agg_type = rand_choice(randstate, l)
    outer_aggs.append((agg_type, [[subquery_alias, f'agg_{i}']]))
  outer_groupby = []
  if q.group_bys:
    outer_groupby = rand_choice(
        randstate,
        q.group_bys,
        no_elements=randstate.randint(0, len(q.group_bys)),
        replace=False,
    )
    outer_groupby = [(subquery_alias, c, x) for _, c, x in outer_groupby]
  q = GenQuery(
      outer_aggs,
      outer_groupby,
      [],
      PredicateOperator(LogicalOperator.AND, []),
      None,
      q.join_tables,
      inner_groupby=q,
      subquery_alias=subquery_alias,
  )
  return q


def sample_exists_subqueries(
    column_stats,
    complex_predicates,
    exists_ratio,
    group_by_threshold,
    int_neq_predicate_threshold,
    max_no_exists,
    q,
    randstate,
    relationships_table,
    string_stats,
    allowed_predicate_operators,
):
  """Sample exists subqueries.

  Args:
    column_stats: The column statistics.
    complex_predicates: Whether the predicates are complex.
    exists_ratio: The ratio of queries to sample exists subqueries for.
    group_by_threshold: The threshold for group by.
    int_neq_predicate_threshold: The threshold for int neq predicate.
    max_no_exists: The maximum number of exists subqueries.
    q: The query to sample exists subqueries for.
    randstate: The random state.
    relationships_table: The relationships table.
    string_stats: The string statistics.
    allowed_predicate_operators: The allowed predicate operators.
  """
  exists_subquery = randstate.rand() < exists_ratio
  eligible_exist = list(
      set(q.join_tables).intersection(relationships_table.keys())
  )
  if exists_subquery and eligible_exist:

    no_exists = randstate.randint(1, max_no_exists + 1)

    alias_dict = dict()
    exist_tables = []
    chosen_aliases = set()

    for _ in range(no_exists):
      alias_table = randstate.choice(eligible_exist)

      if alias_table not in alias_dict:
        alias_dict[alias_table] = f'{alias_table.lower()}_1'
      chosen_aliases.add(alias_dict[alias_table])

      for i in range(2, int(1e10)):
        subquery_alias = f'{alias_table.lower()}_{i}'
        if subquery_alias not in chosen_aliases:
          rec_alias_dict = {alias_table: subquery_alias}
          exist_tables.append((alias_table, rec_alias_dict))
          chosen_aliases.add(subquery_alias)
          break

    q.alias_dict = alias_dict

    # for each table generate exists subquery
    for t, rec_alias_dict in exist_tables:
      no_rec_pred = randstate.randint(1, 3)
      _, _, predicates = generate_predicates(
          column_stats,
          complex_predicates,
          group_by_threshold,
          int_neq_predicate_threshold,
          [t],
          no_rec_pred,
          randstate,
          string_stats,
          allowed_predicate_operators,
      )
      possible_cols = set()
      for ct, _, _ in relationships_table[t]:
        possible_cols.update(ct)
      if not possible_cols:
        continue
      key_exist_col = randstate.choice(list(possible_cols))

      op = randstate.choice([Operator.EQ, Operator.NEQ])
      self_pred = ColumnPredicate(
          t, key_exist_col, op, f'{alias_dict[t]}."{key_exist_col}"'
      )
      if isinstance(predicates, ColumnPredicate) or predicates.children:
        p = PredicateOperator(LogicalOperator.AND, [predicates, self_pred])
      else:
        p = self_pred

      q_rec = GenQuery([], [], [], p, t, [t], alias_dict=rec_alias_dict)
      q.append_exists_predicate(q_rec, randstate.choice([True, False]))


def check_matches_criteria(
    q,
    complex_predicates,
    max_number_aggregates,
    always_create_the_maximum_number_of_aggregates,
    max_number_group_by,
    always_create_the_maximum_number_of_group_bys,
    max_nunmber_joins,
    always_create_the_maximum_number_of_joins,
    max_number_filter_predicates,
    always_create_the_maximum_number_of_predicates,
    min_number_joins,
    min_number_predicates,
):
  """Check if query matches criteria.

  Args:
    q: The query to check.
    complex_predicates: Whether the predicates are complex.
    max_number_aggregates: The maximum number of aggregates.
    always_create_the_maximum_number_of_aggregates: Whether to always create the
      maximum number of aggregates.
    max_number_group_by: The maximum number of group bys.
    always_create_the_maximum_number_of_group_bys: Whether to always create the
      maximum number of group bys.
    max_nunmber_joins: The maximum number of joins.
    always_create_the_maximum_number_of_joins: Whether to always create the
      maximum number of joins.
    max_number_filter_predicates: The maximum number of filter predicates.
    always_create_the_maximum_number_of_predicates: Whether to always create the
      maximum number of predicates.
    min_number_joins: The minimum number of joins.
    min_number_predicates: The minimum number of predicates.

  Returns:
    Whether the query matches the criteria.
  """
  desired_query = True
  if (
      (
          always_create_the_maximum_number_of_joins
          and len(q.joins) < max_nunmber_joins
      )
      or (
          always_create_the_maximum_number_of_aggregates
          and len(q.aggregations) < max_number_aggregates
      )
      or (
          always_create_the_maximum_number_of_group_bys
          and len(q.group_bys) < max_number_group_by
      )
      or (len(q.joins) < min_number_joins)
  ):
    desired_query = False
  if len(q.joins) < min_number_joins:
    desired_query = False

  if not q.predicates or not q.predicates.children:
    num_preds = 0
  else:
    num_preds = len(q.predicates.children)

  if min_number_predicates > num_preds:
    desired_query = False
  if num_preds and not q.joins:
    desired_query = False
  if always_create_the_maximum_number_of_predicates:
    if complex_predicates:
      raise NotImplementedError('Check not implemented for complex predicates')
    else:
      if num_preds != max_number_filter_predicates:
        desired_query = False
  if min_number_predicates > str(q.predicates).count('AND'):
    desired_query = False
  return desired_query


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


class ColumnPredicate:
  """Column predicate."""

  def __init__(self, table, col_name, operator, literal):
    self.table = table
    self.col_name = col_name
    self.operator = operator
    self.literal = literal

  def __str__(self):
    return self.to_sql(top_operator=True)

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


class LogicalOperator(Enum):
  AND = 'AND'
  OR = 'OR'

  def __str__(self):
    return self.value


class PredicateOperator:
  """Predicate operator."""

  def __init__(self, logical_op, children=None):
    self.logical_op = logical_op
    if children is None:
      children = []
    self.children = children

  def __str__(self):
    return self.to_sql(top_operator=True)

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


class PredicateChain(Enum):
  SIMPLE = 'SIMPLE'
  OR_OR = 'OR_OR'
  OR = 'OR'
  OR_AND = 'OR_AND'

  def __str__(self):
    return self.value


def sample_complex_predicates(
    column_stats,
    string_stats,
    int_neq_predicate_threshold,
    no_predicates,
    possible_columns,
    possible_string_columns,
    table_predicates,
    string_table_predicates,
    randstate,
    allowed_predicate_operators,
    p_or=0.05,
    p_or_or=0.05,
    p_or_and=0.05,
    p_second_column=0.5,
):
  """Creates complex predicares."""
  # weight the prob of being sampled by number of columns in table
  # make sure we do not just have conditions on one table with many columns
  weights = [1 / table_predicates[t] for t, _ in possible_columns]
  weights += [
      1 / string_table_predicates[t] for t, _ in possible_string_columns
  ]
  weights = np.array(weights)
  weights /= np.sum(weights)

  possible_columns += possible_string_columns
  no_predicates = min(no_predicates, len(possible_columns))
  if not possible_columns:
    return None
  predicate_col_idx = randstate.choice(
      range(len(possible_columns)), no_predicates, p=weights, replace=False
  )
  predicate_columns = [possible_columns[i] for i in predicate_col_idx]
  predicates = []
  for [t, col_name] in predicate_columns:

    # sample which predicate chain
    predicate_options = [
        PredicateChain.SIMPLE,
        PredicateChain.OR,
        PredicateChain.OR_OR,
        PredicateChain.OR_AND,
    ]
    pred_weights = [1 - p_or - p_or_or - p_or_and, p_or, p_or_or, p_or_and]
    pred_chain_idx = randstate.choice(
        range(len(predicate_options)), 1, p=pred_weights
    )[0]
    pred_chain = predicate_options[pred_chain_idx]

    # sample first predicate
    p = sample_predicate(
        string_stats,
        column_stats,
        t,
        col_name,
        int_neq_predicate_threshold,
        randstate,
        complex_predicate=True,
        allowed_predicate_operators=allowed_predicate_operators,
    )
    if p is None:
      continue

    if pred_chain == PredicateChain.SIMPLE:
      predicates.append(p)
    else:
      # sample if we use another column condition
      second_column = randstate.uniform() < p_second_column
      if second_column:
        potential_2nd_col = [
            c2
            for t2, c2 in possible_columns
            if t2 == t and c2 != col_name and [t2, c2] not in predicate_columns
        ]
        if not potential_2nd_col:
          continue
        second_col = rand_choice(randstate, potential_2nd_col)
        p2 = sample_predicate(
            string_stats,
            column_stats,
            t,
            second_col,
            int_neq_predicate_threshold,
            randstate,
            complex_predicate=True,
            allowed_predicate_operators=allowed_predicate_operators,
        )
      else:
        p2 = sample_predicate(
            string_stats,
            column_stats,
            t,
            col_name,
            int_neq_predicate_threshold,
            randstate,
            complex_predicate=True,
            allowed_predicate_operators=allowed_predicate_operators,
        )
      if p2 is None:
        continue

      complex_pred = None
      if pred_chain == PredicateChain.OR:
        complex_pred = PredicateOperator(LogicalOperator.OR, [p, p2])
      else:
        p3 = sample_predicate(
            string_stats,
            column_stats,
            t,
            col_name,
            int_neq_predicate_threshold,
            randstate,
            complex_predicate=True,
            allowed_predicate_operators=allowed_predicate_operators,
        )
        if p3 is None:
          complex_pred = PredicateOperator(LogicalOperator.OR, [p, p2])
        else:
          if pred_chain == PredicateChain.OR_OR:
            complex_pred = PredicateOperator(LogicalOperator.OR, [p, p2, p3])
          elif pred_chain == PredicateChain.OR_AND:
            complex_pred = PredicateOperator(
                LogicalOperator.OR,
                [p, PredicateOperator(LogicalOperator.AND, [p2, p3])],
            )
      predicates.append(complex_pred)

  if len(predicates) == 1:
    return predicates[0]

  return PredicateOperator(LogicalOperator.AND, predicates)


def sample_predicate(
    string_stats,
    column_stats,
    t,
    col_name,
    int_neq_predicate_threshold,
    randstate,
    allowed_predicate_operators,
    complex_predicate=False,
    p_like=0.5,
    p_is_not_null=0.01,
    p_in=0.5,
    p_between=0.0,
    p_not_like=0.5,
    p_mid_string_whitespace=0.5,
):
  """Sample predicate.

  Args:
    string_stats: The string statistics.
    column_stats: The column statistics.
    t: The table.
    col_name: The column name.
    int_neq_predicate_threshold: The threshold for int neq predicate.
    randstate: The random state.
    allowed_predicate_operators: The allowed predicate operators.
    complex_predicate: Whether the predicate is complex.
    p_like: The probability of LIKE.
    p_is_not_null: The probability of IS NOT NULL.
    p_in: The probability of IN.
    p_between: The probability of BETWEEN.
    p_not_like: The probability of NOT LIKE.
    p_mid_string_whitespace: The probability of mid string whitespace.

  Returns:
    The predicate.
  """
  literal = None
  col_stats = vars(vars(column_stats)[t]).get(col_name)

  if complex_predicate:
    # LIKE / NOT LIKE
    if (
        Operator.LIKE in allowed_predicate_operators
        or Operator.NOT_LIKE in allowed_predicate_operators
    ):
      str_stats = None
      if string_stats is not None:
        str_stats = vars(vars(string_stats)[t]).get(col_name)
      if (
          col_stats is None
          or col_stats.datatype == str(Datatype.STRING)
          or (str_stats is not None and randstate.uniform() < p_like)
      ):
        freq_words = [w for w in str_stats.freq_str_words if len(w) > 1]
        if not freq_words:
          return None
        literal = rand_choice(randstate, freq_words)

        # additional whitespace in the middle
        if randstate.uniform() < p_mid_string_whitespace:
          split_pos = randstate.randint(1, len(literal))
          literal = literal[:split_pos] + '%' + literal[split_pos:]

        literal = f"'%{literal}%'"
        op = None
        if (
            randstate.uniform() < p_not_like
            and Operator.NOT_LIKE in allowed_predicate_operators
        ):
          op = Operator.NOT_LIKE
        elif Operator.LIKE in allowed_predicate_operators:
          op = Operator.LIKE
        if op is not None:
          return ColumnPredicate(t, col_name, op, literal)

    # IS NOT NULL / IS NULL
    if (
        Operator.IS_NULL in allowed_predicate_operators
        or Operator.IS_NOT_NULL in allowed_predicate_operators
    ):
      if col_stats.nan_ratio > 0 and randstate.uniform() < p_is_not_null:
        if (
            randstate.uniform() < 0.8
            and Operator.IS_NOT_NULL in allowed_predicate_operators
        ):
          return ColumnPredicate(t, col_name, Operator.IS_NOT_NULL, None)
        elif Operator.IS_NULL in allowed_predicate_operators:
          return ColumnPredicate(t, col_name, Operator.IS_NULL, None)

    # IN
    if Operator.IN in allowed_predicate_operators:
      if (
          col_stats.datatype
          in [
              str(Datatype.STRING),
              str(Datatype.INT),
              str(Datatype.FLOAT),
              str(Datatype.CATEGORICAL),
          ]
          and randstate.uniform() < p_in
      ):
        # rand_choice(randstate, l, no_elements=None, replace=False)
        literals = col_stats.unique_vals
        first_cap = min(len(literals), 10)
        literals = literals[:first_cap]

        if len(literals) <= 1:
          return None

        no_in_literals = randstate.randint(1, len(literals))
        literals = rand_choice(
            randstate, literals, no_elements=no_in_literals, replace=False
        )
        literals = ', '.join([f"'{l}'" for l in literals])
        literals = f'({literals})'

        return ColumnPredicate(t, col_name, Operator.IN, literals)

    # BETWEEN
    if Operator.BETWEEN in allowed_predicate_operators:
      if (
          col_stats.datatype
          in [str(Datatype.INT), str(Datatype.FLOAT), str(Datatype.CATEGORICAL)]
          and randstate.uniform() < p_between
      ):
        l1 = sample_literal_from_percentiles(col_stats.percentiles, randstate)
        l2 = sample_literal_from_percentiles(
            col_stats.percentiles,
            randstate,
        )
        if l1 == l2:
          l2 += 1
        literal = f'{min(l1, l2)} AND {max(l1, l2)}'
        return ColumnPredicate(t, col_name, Operator.BETWEEN, literal)

  # simple predicates
  if col_stats.datatype.lower() in [
      'int',
      'int64',
      'uint64',
      'int32',
      'uint32',
  ]:
    reasonable_ops = [
        Operator.LEQ,
        Operator.GEQ,
    ]
    if col_stats.num_unique < int_neq_predicate_threshold:
      reasonable_ops.append(Operator.EQ)
      reasonable_ops.append(Operator.NEQ)
      reasonable_ops.append(Operator.EQ)
      reasonable_ops.append(Operator.IS_NOT_NULL)
      reasonable_ops.append(Operator.IS_NULL)
    literal = sample_literal_from_percentiles(col_stats.percentiles, randstate)
    if literal is None:
      return None

  elif col_stats.datatype.lower() in ['float64', 'float', 'double']:
    reasonable_ops = [
        Operator.LEQ,
        Operator.GEQ,
        Operator.IS_NOT_NULL,
        Operator.IS_NULL,
    ]
    literal = sample_literal_from_percentiles(col_stats.percentiles, randstate)
    # nan comparisons only produce errors
    # happens when column is all nan
    if literal is None:
      return None
  elif col_stats.datatype in [str(Datatype.STRING), str(Datatype.CATEGORICAL)]:
    reasonable_ops = [
        Operator.EQ,
        Operator.NEQ,
        Operator.IS_NOT_NULL,
        Operator.IS_NULL,
    ]
    possible_literals = []
    if hasattr(col_stats, 'unique_vals'):
      possible_literals = [
          v
          for v in col_stats.unique_vals
          if v is not None and not (isinstance(v, float) and np.isnan(v))
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


def sample_predicates(
    column_stats,
    int_neq_predicate_threshold,
    no_predicates,
    possible_columns,
    table_predicates,
    randstate,
    allowed_predicate_operators,
):
  """Sample predicates.

  Args:
    column_stats: The column statistics.
    int_neq_predicate_threshold: The threshold for int neq predicate.
    no_predicates: The number of predicates.
    possible_columns: The possible columns.
    table_predicates: The table predicates.
    randstate: The random state.
    allowed_predicate_operators: The allowed predicate operators.

  Returns:
    The predicates.
  """
  if not possible_columns:
    return None

  # sample random predicates
  # weight the prob of being sampled by number of columns in table
  # make sure we do not just have conditions on one table with many columns
  weights = np.array([1 / table_predicates[t] for t, _ in possible_columns])
  weights /= np.sum(weights)
  # we cannot sample more predicates than available columns
  no_predicates = min(no_predicates, len(possible_columns))
  predicate_col_idx = randstate.choice(
      range(len(possible_columns)), no_predicates, p=weights, replace=False
  )
  predicate_columns = [possible_columns[i] for i in predicate_col_idx]
  predicates = []
  for [t, col_name] in predicate_columns:
    p = sample_predicate(
        None,
        column_stats,
        t,
        col_name,
        int_neq_predicate_threshold,
        randstate,
        allowed_predicate_operators,
        complex_predicate=False,
    )
    if p is not None:
      predicates.append(p)

  return PredicateOperator(LogicalOperator.AND, predicates)


def analyze_columns(
    column_stats,
    group_by_treshold,
    join_tables,
    string_stats,
    complex_predicates,
):
  """Analyze columns to identify possible columns for predicates, group bys, and aggregations."""
  # find possible columns for predicates
  possible_columns = []
  possible_string_columns = []
  possible_group_by_columns = []
  numerical_aggregation_columns = []
  # also track how many columns we have per table to reweight them
  table_predicates = collections.defaultdict(int)
  string_table_predicates = collections.defaultdict(int)
  for t in join_tables:
    if t not in vars(column_stats):
      continue
    for col_name, col_stats in vars(vars(column_stats)[t]).items():
      if col_stats.datatype in {
          str(d).lower() for d in ['int', 'int64', 'uint64', 'int32', 'uint32']
      }:
        possible_columns.append([t, col_name])
        table_predicates[t] += 1

      if string_stats is not None and t in vars(string_stats):
        if complex_predicates and col_name in vars(vars(string_stats)[t]):
          possible_string_columns.append([t, col_name])
          string_table_predicates[t] += 1

      # group by columns
      if (
          col_stats.datatype
          in {str(d) for d in [Datatype.INT, Datatype.STRING]}
          and col_stats.num_unique < group_by_treshold
      ):
        possible_group_by_columns.append([t, col_name, col_stats.num_unique])

      # numerical aggregation columns
      if col_stats.datatype in {str(d) for d in [Datatype.INT, Datatype.FLOAT]}:
        numerical_aggregation_columns.append([t, col_name])
  return (
      numerical_aggregation_columns,
      possible_columns,
      possible_string_columns,
      possible_group_by_columns,
      table_predicates,
      string_table_predicates,
  )


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


def rand_choice(randstate, l, no_elements=None, replace=False):
  if no_elements is None:
    idx = randstate.randint(0, len(l))
    return l[idx]
  else:
    idxs = randstate.choice(range(len(l)), no_elements, replace=replace)
    return [l[i] for i in idxs]


def sample_acyclic_join(
    no_joins, relationships_table, schema, randstate, left_outer_join_ratio
):
  """Sample acyclic join.

  Args:
    no_joins: The number of joins.
    relationships_table: The relationships table.
    schema: The schema.
    randstate: The random state.
    left_outer_join_ratio: The ratio of left outer joins.

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

      left_outer_join = False
      if left_outer_join_ratio > 0 and randstate.rand() < left_outer_join_ratio:
        left_outer_join = True

      joins.append((t, column_l, table_r, column_r, left_outer_join))
    else:
      break
  return start_t, joins, join_tables


def find_possible_joins(join_tables, relationships_table):
  possible_joins = list()
  for t in join_tables:
    for column_l, table_r, column_r in relationships_table[t]:
      if table_r in join_tables:
        continue
      possible_joins.append((t, column_l, table_r, column_r))
  return possible_joins
