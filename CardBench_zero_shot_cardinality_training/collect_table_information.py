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

"""Collect table information."""

import google.cloud.bigquery

tables_info_table = (
    "bq-cost-models-exp.chronis.public-bq-table-info-stats_open_source"
)
columns_info_table = (
    "bq-cost-models-exp.chronis.public-bq-column-info_open_source"
)

info_schema_tables = ".INFORMATION_SCHEMA.TABLES"
info_schema_columns = ".INFORMATION_SCHEMA.COLUMNS"

bq_public_datasets_projectname = "bq-cost-models-exp"
sample_datasets_projectname = "bq-cost-models-exp"


def execute_insert_query(bqclient, query):
  try:
    job = bqclient.query(query)
    job.result()
  except google.cloud.bigquery.exceptions.BigQueryError as e:
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY :" + query)
    print(">>>> " + str(e))
    print(">>>> " + query)


def collect_table_information(bqclient, projectname, datasetname):
  """Collect table information such as name, paritioning and clustering columns.

  using BigQuery's information_schema.

  Args:
    bqclient: initialized Big Query client
    projectname: the GCP project of the table
    datasetname: the GCP dataset of the table
  """
  table_combined_info = {}
  # collect table names, partitioning and clustering information
  query1 = (
      "select * from (  SELECT table_name, max(is_partitioning_column) as"
      " ispart, max(clustering_ordinal_position) as isclust FROM "
      + "`{projectname}.{datasetname}.{info_schema_columns}` "
      + "GROUP BY table_name ) where  NOT EXISTS (select * from "
      + "`{tables_info_table}` as it where  it.project_name = '{projectname}' "
      + "AND it.dataset_name = '{datasetname}' "
      + "' AND it.table_name = table_name)"
  ).format(
      projectname=projectname,
      datasetname=datasetname,
      info_schema_columns=info_schema_columns,
      tables_info_table=tables_info_table,
  )

  queryjob1 = None
  try:
    queryjob1 = bqclient.query(query1)
    queryjob1.result()
  except google.cloud.bigquery.exceptions.BigQueryError as e:
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query1)

  # for each table gather the infrotamtion from query1 and collect row count
  for row in queryjob1:
    ispart = "false"
    if row["ispart"] == "YES":
      ispart = "true"
    isclust = "false"
    if row["isclust"] != "null":
      isclust = "true"

    query2 = (
        "SELECT count(*) as cnt FROM `"
        + projectname
        + "."
        + datasetname
        + "."
        + row["table_name"]
        + "`"
    )
    print(query2)
    rowrescnt = None
    try:
      queryjob2 = bqclient.query(query2)
      rowrescnt = next(iter(queryjob2))
    except google.cloud.bigquery.exceptions.BigQueryError as e:
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query2)

    table_combined_info[row["table_name"]] = {
        "is_partitioned": ispart,
        "is_clustered": isclust,
        "row_count": rowrescnt["cnt"],
        # the following rows will be populated later
        "partition_column": "empty",
        "clustered_columns": [],
    }

  # add results in a big query table
  total_rows_added = 0
  preamble = (
      "INSERT INTO `"
      + tables_info_table
      + "` (`project_name`, `dataset_name`, `table_name`, `row_count`,"
      " `is_partitioned`, `is_clustered`,"
      " `partition_column`, `clustered_columns`) VALUES"
  )
  count = 0
  newvals = ""
  for ti in table_combined_info:
    if newvals:
      newvals = newvals + ", "
    newvals = (
        newvals
        + "( '"
        + projectname
        + "', '"
        + datasetname
        + "', '"
        + str(ti)
        + "', "
        + str(table_combined_info[ti]["row_count"])
        + ", "
        + str(table_combined_info[ti]["is_partitioned"])
        + ", "
        + str(table_combined_info[ti]["is_clustered"])
        + ", '"
        + str(table_combined_info[ti]["partition_column"])
        + "', "
        + str(table_combined_info[ti]["clustered_columns"])
        + ", "
    )
    count += 1
    if count >= 200:
      query = preamble + newvals
      execute_insert_query(bqclient, query)
      total_rows_added = total_rows_added + count
      count = 0
      newvals = ""

  if newvals:
    total_rows_added = total_rows_added + count
    query = preamble + newvals
    execute_insert_query(bqclient, query)

  print(
      "collect_table_information -> Number of rows added to the table info: ",
      total_rows_added,
  )
