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

"""Collect column extra statistics."""

import math
from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers


INTERNAL = False

get_partitioning_info = helpers.get_partitioning_info
build_partitioned_predicate = helpers.build_partitioned_predicate
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
DBType = database_connector.DBType
TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_INFO_TABLE = configuration.COLUMNS_INFO_TABLE
COLUMNS_STATS_TABLE = configuration.COLUMNS_STATS_TABLE
BQ_INFO_SCHEMA_COLUMNS = configuration.BQ_INFO_SCHEMA_COLUMNS
TYPES_TO_TABLES = configuration.TYPES_TO_TABLES
get_sql_table_string = database_connector.get_sql_table_string


def calculate_and_write_extra_column_statistics_internal_string(
    projectname,
    column_list,
    extra_stats_table,
    dbs,
):
  """Calculates column statistics and writes for all types except string."""

  # Return the number of rows written to configuration.COLUMNS_STATS_TABLE.
  rows_written_to_extra_stats_table = 0
  for work_item in column_list:
    datasetname, tablename, columnname = work_item
    print("  collecting extra stats for column: ", columnname)
    is_partitioned, partition_column, partition_column_type, _ = (
        get_partitioning_info(
            projectname,
            datasetname,
            tablename,
            dbs["metadata_dbclient"],
            dbs["metadata_dbtype"],
        )
    )
    partitioned_predicate = build_partitioned_predicate(
        is_partitioned, tablename, partition_column, partition_column_type
    )
    rowres = {}
    table_sql_string = get_sql_table_string(
        dbs["data_dbtype"], f"{projectname}.{datasetname}.{tablename}"
    )
    query = (
        f"SELECT count(distinct `{columnname}`) as cnt from"
        f" {table_sql_string} WHERE"
        f" {partitioned_predicate}"
    )

    try:
      queryjob, _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
      rowres = get_query_result_first_row(dbs["data_dbtype"], queryjob)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)
    extra_stats_table_sql_string = get_sql_table_string(
        dbs["metadata_dbtype"], extra_stats_table
    )
    if rowres["cnt"] == 0:
      query = (
          f"INSERT INTO {extra_stats_table_sql_string} (`project_name`,"
          " `dataset_name`,`table_name`, `column_name`, `allnull`) VALUES"
          f" ('{projectname}', '{datasetname}', '{tablename}', '{columnname}',"
          " true)"
      )
      try:
        _, _ = run_query(
            dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
        print(">>>> " + str(e))
        print(">>>> " + query)
      continue

    minval = "NULL"
    maxval = "NULL"
    maxlength = "NULL"
    query = (
        f"SELECT '{projectname}', '{datasetname}',"
        f" '{tablename}', '{columnname}', max(`{columnname}`) as maxval,"
        f" min(`{columnname}`) as minval, false, max(length(`{columnname}`)) as"
        f" maxlength FROM {table_sql_string} WHERE {partitioned_predicate}"
    )
    try:
      queryjob, _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
      firstrow = get_query_result_first_row(dbs["data_dbtype"], queryjob)
      minval = firstrow["minval"]
      maxval = firstrow["maxval"]
      minval = minval.replace("'", "'").replace("\n", "\\n")
      maxval = maxval.replace("'", "'").replace("\n", "\\n")
      maxval = maxval.replace("'", "'")
      maxlength = firstrow["maxlength"]
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)

    query = (
        f"INSERT INTO {extra_stats_table_sql_string} (`project_name`,"
        " `dataset_name`, `table_name`, `column_name`, `min_val`, `max_val`,"
        f" `allnull`,  `max_length`) VALUES ('{projectname}', '{datasetname}',"
        f" '{tablename}', '{columnname}', '{minval}', '{maxval}', false,"
        f" {maxlength})"
    )

    success = True
    try:
      _, _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
      rows_written_to_extra_stats_table += 1
    except Exception as e:  # pylint: disable=broad-exception-caught
      success = False
      print(">>>>>>>>>>>>>>>>>> ERROR IN QUERY RETRYING >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)
    if success:
      rows_written_to_extra_stats_table += 1
  return rows_written_to_extra_stats_table


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


def return_null_if_invalid_value(value, columntype):
  if is_numeric_type(columntype) and math.isinf(value):
    return "NULL"
  if value in ["-inf", "inf", None, "nan"]:
    return "NULL"
  else:
    if value in ["inf", "nan"]:
      return "NULL"
    return value


def calculate_and_write_extra_column_statistics_internal(
    projectname,
    column_list,
    columntype,
    extra_stats_table,
    dbs,
):
  """Calculates column statistics and writes for all types except string."""
  # Return the number of rows written to configuration.COLUMNS_STATS_TABLE.

  if columntype == "STRING":
    return calculate_and_write_extra_column_statistics_internal_string(
        projectname, column_list, extra_stats_table, dbs
    )

  rows_written_to_extra_stats_table = 0
  extra_stats_table_sql_string = get_sql_table_string(
      dbs["metadata_dbtype"], extra_stats_table
  )

  preamble = (
      f"INSERT INTO {extra_stats_table_sql_string} (`project_name`,"
      " `dataset_name`, `table_name`, `column_name`, `min_val`, `max_val` "
  )
  if is_numeric_type(columntype):
    preamble += ", `mean_val`"
  preamble = preamble + ", `allnull` ) VALUES "

  newvals = ""
  count = 0
  for work_item in column_list:
    datasetname, tablename, columnname = work_item
    print("  collecting extra stats for column: ", columnname)
    is_partitioned, partition_column, partition_column_type, _ = (
        get_partitioning_info(
            projectname,
            datasetname,
            tablename,
            dbs["metadata_dbclient"],
            dbs["metadata_dbtype"],
        )
    )
    partitioned_predicate = build_partitioned_predicate(
        is_partitioned, tablename, partition_column, partition_column_type
    )
    rowres = {}
    table_sql_string = get_sql_table_string(
        dbs["data_dbtype"], f"{projectname}.{datasetname}.{tablename}"
    )
    query = (
        f"SELECT count(distinct `{columnname}`) as cnt from"
        f" {table_sql_string} WHERE"
        f" {partitioned_predicate}"
    )
    try:
      queryjob, _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
      rowres = get_query_result_first_row(dbs["data_dbtype"], queryjob)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)

    if rowres["cnt"] == 0:
      allnulls = "true"
      query = (
          f"INSERT INTO {extra_stats_table_sql_string} (`project_name`,"
          " `dataset_name`, `table_name`, `column_name`, `allnull`) VALUES"
          f" ('{projectname}', '{datasetname}', '{tablename}', '{columnname}',"
          " true )"
      )
      try:
        run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
        print(">>>> " + str(e))
        print(">>>> " + query)
      continue

    min_val = None
    max_val = None
    mean_val = None
    allnulls = False

    firstquery = (
        f"SELECT max(`{columnname}`) as maxval, min(`{columnname}`) as minval"
    )
    if is_numeric_type(columntype):
      firstquery = f"{firstquery}, AVG(`{columnname}`) as meanval"
    firstquery = (
        f"{firstquery} FROM {table_sql_string} WHERE {partitioned_predicate}"
    )
    rowres_firstquery = {}
    try:
      queryjob, _ = run_query(
          dbs["data_dbtype"], firstquery, dbs["data_dbclient"]
      )
      rowres_firstquery = get_query_result_first_row(
          dbs["data_dbtype"], queryjob
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + firstquery)

    min_val = rowres_firstquery["minval"]
    max_val = rowres_firstquery["maxval"]
    mean_val = None

    min_val = str(return_null_if_invalid_value(min_val, columntype))
    max_val = str(return_null_if_invalid_value(max_val, columntype))
    if is_numeric_type(columntype):
      mean_val = str(
          return_null_if_invalid_value(rowres_firstquery["meanval"], columntype)
      )
    if mean_val == "NULL" or max_val == "NULL" or min_val == "NULL":
      # if min, max or mean contains then the all null flag is set
      allnulls = True
      mean_val = "NULL"
      min_val = "NULL"
      max_val = "NULL"

    if newvals:
      newvals = newvals + ", "
    newvals = (
        f"{newvals} ('{projectname}', '{datasetname}', '{tablename}',"
        f" '{columnname}', "
    )
    if is_numeric_type(columntype):
      if (
          columntype == "NUMERIC"
          or columntype == "BIGNUMERIC"
          or columntype == "DECIMAL"
          or columntype == "BIGDECIMAL"
      ):
        newvals += "" + min_val
        if "." in min_val:
          newvals += ","
        else:
          newvals += ".0, "
        newvals += "" + max_val
        if "." in max_val:
          newvals += ","
        else:
          newvals += ".0, "
        newvals += "" + mean_val
      else:
        newvals += min_val + ", "
        newvals += max_val + ", "
        newvals += mean_val
    else:
      if columntype in ["DATE", "DATETIME", "TIMESTAMP", "TIME"]:
        newvals += "CAST('" + min_val + "' AS " + columntype + "), "
        newvals += "CAST('" + max_val + "' AS " + columntype + ") "
      else:
        newvals += "'" + min_val + "', "
        newvals += "'" + max_val + "'"
    newvals += ", " + str(allnulls) + " )"
    count += 1
  if count >= 10:
    query = preamble + newvals
    try:
      _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)
    rows_written_to_extra_stats_table += count
    count = 0
    newvals = ""

  success = True
  if newvals:
    query = preamble + newvals
    rows_written_to_extra_stats_table += count
    try:
      _, _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
    except Exception as e:  # pylint: disable=broad-exception-caught
      success = False
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)
  if not success:
    rows_written_to_extra_stats_table = 0
  return rows_written_to_extra_stats_table


def calculate_and_write_extra_column_statistics(
    projectname, datasetname, dbs
):
  """Calculate type specific column statistics, the result is stored in the configuration.COLUMNS_STATS_TABLE table."""

  # create the tasks of calculating extra column statistics per column type
  # each task specifies the column for which to collect statistics
  columns_to_collect_stats_by_type = {}
  for type_to_table in TYPES_TO_TABLES.items():
    # type_to_table[0] is the column type
    # type_to_table[1] is the table name where the statistics are stored
    columns_to_collect_stats_by_type[type_to_table[0]] = {
        "stats_table": type_to_table[1],
        "column_list": [],
    }
    columns_info_table_sql_string = get_sql_table_string(
        dbs["metadata_dbtype"], COLUMNS_INFO_TABLE
    )
    extra_stats_type_table_sql_string = get_sql_table_string(
        dbs["metadata_dbtype"], type_to_table[1]
    )
    query = (
        "SELECT  ci.dataset_name, ci.table_name, ci.column_name,"
        f" ci.column_type FROM {columns_info_table_sql_string} as ci  WHERE"
        f" ci.dataset_name = '{datasetname}' AND ci.project_name  ="
        f" '{projectname}' AND ci.column_type = '{type_to_table[0]}' AND NOT"
        f" EXISTS (select * from {extra_stats_type_table_sql_string} as est"
        " where est.column_name = ci.column_name and est.table_name ="
        " ci.table_name and est.dataset_name = ci.dataset_name AND"
        " est.project_name = ci.project_name)"
    )
    try:
      queryjob, _ = run_query(
          dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
      )
      for row in queryjob:
        columns_to_collect_stats_by_type[type_to_table[0]][
            "column_list"
        ].append([
            row["dataset_name"],
            row["table_name"],
            row["column_name"],
        ])
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)

  # Now that the tasks are created, we calculate the extra column statistics
  num_rows_written_to_extra_column_stats_table = 0
  if not INTERNAL:
    for column_type in columns_to_collect_stats_by_type:
      num_rows_written_to_extra_column_stats_table += (
          calculate_and_write_extra_column_statistics_internal(
              projectname=projectname,
              column_list=columns_to_collect_stats_by_type[column_type][
                  "column_list"
              ],
              columntype=column_type,
              extra_stats_table=columns_to_collect_stats_by_type[column_type][
                  "stats_table"
              ],
              dbs=dbs,
          )
      )
  print(
      "Extra column rows written:", num_rows_written_to_extra_column_stats_table
  )
