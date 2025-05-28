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

"""Calculate column correlation."""

import itertools
import math
from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector


run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
DBType = database_connector.DBType
TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_INFO_TABLE = configuration.COLUMNS_INFO_TABLE
COLUMNS_STATS_TABLE = configuration.COLUMNS_STATS_TABLE
INFO_SCHEMA_COLUMNS = configuration.BQ_INFO_SCHEMA_COLUMNS
TYPES_TO_TABLES = configuration.TYPES_TO_TABLES
CORRELATION_TABLE = configuration.CORRELATION_TABLE
SAMPLE_PROJECTNAME_DATASET_NAME_4K = (
    configuration.SAMPLE_PROJECTNAME_DATASET_NAME_4K
)
get_sql_table_string = database_connector.get_sql_table_string


def writes_correlations(
    work_orders,
    projectname,
    datasetname,
    tablename,
    metadata_dbtype,
    metadata_dbclient,
):
  """Writes calculated correlations to table configuration.CORRELATION_TABLE."""
  insert_query_preamble = (
      f"insert into `{CORRELATION_TABLE}` (project_name, dataset_name,"
      " table_name_a, table_name_b, column_name_a, column_name_b,"
      " pearson_correlation) VALUES "
  )

  insert_query_body = ""
  for work_order in work_orders.values():
    insert_query_body += (
        f"( '{projectname}',  '{datasetname}',  '{tablename}',  '{tablename}', "
        f" '{work_order[0]}',  '{work_order[1]}', {work_order[2]} ),"
    )
    if len(insert_query_preamble) + len(insert_query_body) > 100000:
      query = insert_query_preamble + insert_query_body[:-1]
      try:
        _, _ = run_query(metadata_dbtype, query, metadata_dbclient)
        insert_query_body = ""
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
        print(">>>> " + str(e))
        print(">>>> " + query)

  if insert_query_body:
    query = insert_query_preamble + insert_query_body[:-1]
    try:
      _, _ = run_query(metadata_dbtype, query, metadata_dbclient)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)


def create_work_orders_correlations(
    col_pairs,
    tablename,
    collected_correlations,
):
  """Creates work orders for the given column pairs."""

  # The work orders specify the column pairs to collect correlations, also skips
  # columns pairs if the correlation exists in the
  # configuration.CORRELATION_TABLE table.

  # Returns:
  #   A dictionary of work orders. The key is the work order id, the value is a
  #   list of [column_name_a, column_name_b, pearson_correlation].

  work_orders = {}
  next_key = 0
  for p in col_pairs:
    col1name = p[0].strip()
    col2name = p[1].strip()
    exists_key = tablename + "." + col1name + "." + col2name
    if exists_key not in collected_correlations:
      work_orders[str(next_key)] = [col1name, col2name, -10]
      next_key += 1
  return work_orders


def run_calculate_correlations_query_bq(
    query_corr,
    sample_table_path,
    work_orders,
    data_dbtype,
    data_dbclient,
):
  """Run the query to calculate the correlations, result is stored in work_orders."""
  sample_table_path_sql_string = get_sql_table_string(
      data_dbtype, sample_table_path
  )
  full_query = (
      f"select {query_corr} from {sample_table_path_sql_string} as talias"
  )
  try:
    query_job, _ = run_query(data_dbtype, full_query, data_dbclient)
    for row in query_job:
      for key in row.keys():
        wo_key = key.replace("id_", "")
        corr_val = row[key]
        if str(corr_val).lower() == "nan":
          corr_val = -20
        elif str(corr_val).lower() == "null":
          corr_val = -30
        elif str(corr_val).lower() == "none":
          corr_val = -40

        work_orders[wo_key][2] = corr_val
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + full_query)


def calculate_correlations_bq(
    work_orders,
    sample_table_path,
    data_dbtype,
    data_dbclient,
):
  """Calculate the correlations for the given work orders, result is stored in work_orders."""
  query_corr = ""
  num_correlations_to_collect = 0
  for wo_key in work_orders.keys():
    col1name = work_orders[wo_key][0]
    col2name = work_orders[wo_key][1]
    num_correlations_to_collect += 1

    query_corr += f"corr(talias.{col1name}, talias.{col2name}) as id_{wo_key}, "

    # there is 1M char limit per query
    if len(query_corr) > 900000 or num_correlations_to_collect > 9990:
      run_calculate_correlations_query_bq(
          query_corr, sample_table_path, work_orders, data_dbtype, data_dbclient
      )
      query_corr = ""
      num_correlations_to_collect = 0
  if num_correlations_to_collect > 0:
    run_calculate_correlations_query_bq(
        query_corr, sample_table_path, work_orders, data_dbtype, data_dbclient
    )


# assumes there is a function called corr that calculates pearson correlation
# https://cloud.google.com/bigquery/docs/reference/standard-sql/statistical_aggregate_functions#corr
def calculate_correlations(
    work_orders,
    sample_table_path,
    data_dbtype,
    data_dbclient,
):
  """Calculates pearson correlation for a set of columns, result is stored in work_orders."""

  # Correlation value in the work order  is set to:
  # -100 when the graph node is initialized
  # -10 if the correlation does exist in the BQ table (when checking if previous
  # collected)
  # -20 if the pearson bq function returned nan
  # -30 if the pearson bq function returned null
  # -40 if the pearson bq function returned none
  # -50 if the column types do not allow correlation calculation
  # float if the correlation calculation was successful

  if data_dbtype == DBType.BIGQUERY:
    return calculate_correlations_bq(
        work_orders, sample_table_path, data_dbtype, data_dbclient
    )
  else:
    raise ValueError(f'Dbtype not supported yet: {str("data_dbtype")}')


def get_collected_correlation_for_dataset(
    projectname,
    datasetname,
    metadata_dbtype,
    metadata_dbclient,
):
  """Returns the already collected pearson correlations for a dataset in a set."""
  query = (
      "select table_name_a, column_name_a, column_name_b from"
      f" `{CORRELATION_TABLE}` WHERE project_name = '{projectname}' and"
      f" dataset_name = '{datasetname}'"
  )
  exists = set()
  try:
    queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
    for row in queryjob:
      exists.add(
          f'{row["table_name_a"]}.{row["column_name_a"]}.{row["column_name_b"]}'
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)
  return exists


def number_of_combinations_of_pair_without_order(n):
  if n <= 1:
    return 0
  return math.factorial(n) / (math.factorial(2) * math.factorial(n - 2))


def create_column_pairs(collist):
  """Create the combinations of all columns in the collist."""
  pairs = itertools.combinations(collist, 2)
  # sort lexigocraphically the columns in each pair
  pairs_sorted = []
  for p in pairs:
    ps = list(p)
    ps.sort()
    pairs_sorted.append(ps)

  # Verify that we have the correct number of column pairs
  assert len(pairs_sorted) == number_of_combinations_of_pair_without_order(
      len(collist)
  )
  return pairs_sorted


def is_numeric_type(columntype):
  return (
      columntype == "INT64"
      or columntype == "INT32"
      or columntype == "UINT32"
      or columntype == "UINT64"
      or columntype == "FLOAT64"
      or columntype == "DOUBLE"
      or columntype == "NUMERIC"
      or columntype == "BIGNUMERIC"
      or columntype == "BIGDECIMAL"
      or columntype == "DECIMAL"
  )


def get_cols_per_table_dataset(
    projectname,
    datasetname,
    metadata_dbtype,
    metadata_dbclient,
    data_dbtype,
):
  """Returns the columns of each table in the dataset."""

  if data_dbtype == DBType.BIGQUERY:
    query = (
        "select project_name, dataset_name, table_name, column_name from"
        f" `{COLUMNS_INFO_TABLE}` WHERE project_name = '{projectname}' and"
        f" dataset_name = '{datasetname}' AND column_type in ('INT64', 'INT32',"
        " 'NUMERIC', 'FLOAT64', 'FLOAT', 'DOUBLE',  'BIGNUMERIC',"
        " 'DECIMAL','BIGDECIMAL')"
    )
  else:
    raise ValueError(f'Dbtype not supported yet: {str("data_dbtype")}')
  cols_per_table = {}

  try:
    queryjob, _ = run_query(metadata_dbtype, query, metadata_dbclient)
    for row in queryjob:
      key = f'{row["project_name"]}.{row["dataset_name"]}.{row["table_name"]}'
      if key not in cols_per_table:
        cols_per_table[key] = []
      cols_per_table[key].append(row["column_name"])
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)

  return cols_per_table


def calculate_and_write_pearson_correlation(
    projectname, datasetname, dbs
):
  """Calculate and writes pearson correlation for each pair of columns in each table, the results are stored in table configuration.CORRELATION_TABLE."""

  # Find the column names and types of each table in the dataset
  # Restrict to 'INT64', 'INT32', 'NUMERIC', 'FLOAT64', 'BIGNUMERIC',
  # 'DECIMAL','BIGDECIMAL, UINT32', 'UINT64' types,
  # as the BigQuery function corr() only supports
  # these types.
  cols_per_table = get_cols_per_table_dataset(
      projectname,
      datasetname,
      dbs["metadata_dbtype"],
      dbs["metadata_dbclient"],
      dbs["data_dbtype"],
  )
  # Produce all unique pairs of columns for each table, the columns
  # in each pair are sorted
  table_cols_pair = {}
  total_corr_to_be_collected = 0
  for k in cols_per_table:
    table_cols_pair[k] = create_column_pairs(cols_per_table[k])
    total_corr_to_be_collected += len(table_cols_pair[k])

  if total_corr_to_be_collected == 0:
    return

  print(
      "Total number of correlations to be collected:"
      f" {total_corr_to_be_collected}"
  )
  # Get already calculated pearson correlations for a dataset.
  # Returns a list of strings of the format
  # table_name.column_name_a.columna_name_b
  # column_name_a and column_name_b are lexicographically sorted
  collected_correlations = get_collected_correlation_for_dataset(
      projectname,
      datasetname,
      dbs["metadata_dbtype"],
      dbs["metadata_dbclient"],
  )
  print(
      "Total number of correlations already collected:"
      f" {len(collected_correlations)}"
  )

  # Process each table separately
  for k in table_cols_pair:
    # create a work order for each correlation to be collected
    # each work order is a dictionary with keys
    # table_name, column_name_a, column_name_b, pearson_correlation
    # we calculate the correlation on 4k row samples of the original table
    # for efficiency.
    table_path = k
    table_path_split = table_path.split(".")
    projectname = table_path_split[0].strip()
    datasetname = table_path_split[1].strip()
    tablename = table_path_split[2].strip()

    sample_table_path = f"{SAMPLE_PROJECTNAME_DATASET_NAME_4K}.{projectname}_{datasetname}_{tablename}"

    work_orders = create_work_orders_correlations(
        table_cols_pair[k], tablename, collected_correlations
    )
    print(
        f"For table {tablename} we need to collect"
        f" {len(work_orders)} correlations"
    )

    # now that we now what we need to collect
    # we collect the correlations and insert the in a BQ table
    # (configuration.CORRELATION_TABLE)
    # work_orders[X][2] contain the corr values

    calculate_correlations(
        work_orders,
        sample_table_path,
        dbs["data_dbtype"],
        dbs["data_dbclient"],
    )

    writes_correlations(
        work_orders,
        projectname,
        datasetname,
        tablename,
        dbs["metadata_dbtype"],
        dbs["metadata_dbclient"],
    )
