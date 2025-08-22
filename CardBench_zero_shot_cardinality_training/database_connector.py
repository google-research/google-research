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

"""Database connector."""

import enum
from typing import Any

from CardBench_zero_shot_cardinality_training.database_connector_library import big_query_database_connector


class DBType(enum.Enum):
  """Database types."""

  BIGQUERY = 1


def get_sql_table_string(db_type, table_name):
  """Gets the SQL table string for the given database type and table name.

  Args:
    db_type: The database type.
    table_name: The table name.

  Returns:
    The SQL table string.
  """
  if db_type == DBType.BIGQUERY:
    return big_query_database_connector.get_sql_table_string_bigquery(
        table_name
    )
  else:
    raise ValueError(f"Unknown database type: {db_type}")


def get_query_result_first_row(
    db_type, queryjob
):
  """Gets the result of the first row of the query.

  Args:
    db_type: The database type.
    queryjob: The query job.

  Returns:
    A dictionary, keys: column name, column value as value.
  """
  if db_type == DBType.BIGQUERY:
    return big_query_database_connector.get_query_result_first_row_bigquery(
        queryjob
    )
  else:
    raise ValueError(f"Unknown database type: {type}")


def create_database_connection(db_type):
  """Creates a database connection.

  Args:
    db_type: The database type.

  Returns:
    The database connection context.
  """
  if db_type == DBType.BIGQUERY:
    return big_query_database_connector.create_bigquery_connection()
  else:
    raise ValueError(f"Unknown database type: {db_type}")


def get_query_cardinality(db_type, queryjob):
  """Gets the cardinality of the query result.

  Args:
    db_type: The database type.
    queryjob: The query job.

  Returns:
    The cardinality of the query result.
  """
  if db_type == DBType.BIGQUERY:
    return big_query_database_connector.get_query_cardinality_bigquery(queryjob)
  else:
    raise ValueError(f"Unknown database type: {db_type}")


def run_query(
    db_type, query, connection_context, timeout = -1
):
  """Runs a query on the database.

  Args:
    db_type: The database type.
    query: The query to run.
    connection_context: The database connection context.
    timeout: The query timeout in seconds.

  Returns:
    A list of dictionaries, each dictionary contains the data of one database
    row, keys: column name, column value as value.
  """
  if db_type == DBType.BIGQUERY:
    return big_query_database_connector.run_query_bigquery(
        query, connection_context, timeout
    )
  else:
    raise ValueError(f"Unknown database type: {db_type}")


def table_exists(
    db_type, table_name, connection_context
):
  """Checks if a table exists.

  Args:
    db_type: The database type.
    table_name: The table name.
    connection_context: The database connection context.

  Returns:
    True if the table exists, False otherwise.
  """
  if db_type == DBType.BIGQUERY:
    return big_query_database_connector.table_exists_bigquery(
        table_name, connection_context
    )
  else:
    raise ValueError(f"Unknown database type: {db_type}")
