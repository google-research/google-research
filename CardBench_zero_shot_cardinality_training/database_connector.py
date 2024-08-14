# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

from google.cloud import bigquery


class DBType(enum.Enum):
  """Database types."""

  BIGQUERY = 1


def get_query_result_first_row(
    db_type, queryjob
):
  if db_type == DBType.BIGQUERY:
    return next(iter(queryjob))
  else:
    raise ValueError(f"Unknown database type: {type}")


def create_bigquery_connection():
  """Creates a BigQuery connection."""
  return bigquery.Client()


def create_database_connection(db_type):
  if db_type == DBType.BIGQUERY:
    return create_bigquery_connection()
  else:
    raise ValueError(f"Unknown database type: {type}")


def run_query_bigquery(query, bqclient):
  """Runs a query on BigQuery, the call is blocking."""
  job = bqclient.query(query)
  job.result()
  return job


def run_query(db_type, query, connection_context):
  """Runs a query on the database."""
  if db_type == DBType.BIGQUERY:
    return run_query_bigquery(query, connection_context)
  else:
    raise ValueError(f"Unknown database type: {db_type}")
