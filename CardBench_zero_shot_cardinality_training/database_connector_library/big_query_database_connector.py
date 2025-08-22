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

"""BigQuery database connector."""

from typing import Any
import uuid

from google.cloud import bigquery


def get_sql_table_string_bigquery(table_name):
  return f"`{table_name}`"


def create_bigquery_connection():
  """Creates a BigQuery connection."""
  return bigquery.Client()


def run_query_bigquery(
    query, bqclient, timeout = -1
):
  """Runs a query on BigQuery, the call is blocking."""
  job = bqclient.query(query)
  if timeout != -1:
    job.result(timeout=timeout)
  else:
    job.result()
  return job, str(uuid.uuid4())


def get_query_cardinality_bigquery(queryjob):
  cardinality = 0
  for _ in queryjob:
    cardinality += 1
  return cardinality


def get_query_result_first_row_bigquery(queryjob):
  return next(iter(queryjob))


def table_exists_bigquery(table_name, bqclient):
  """Check if a table exists in BigQuery.

  Args:
    table_name: Name of the table.
    bqclient: BigQuery client.

  Returns:
    True if the table exists in BigQuery.
  """
  sql = """
    SELECT * FROM `{table_name}` LIMIT 1;
  """.format(table_name=table_name)

  try:
    run_query_bigquery(sql, bqclient)
  except Exception as e:  # pylint: disable=broad-exception-caught
    if "Table not found" in str(e):
      return False
  return True
