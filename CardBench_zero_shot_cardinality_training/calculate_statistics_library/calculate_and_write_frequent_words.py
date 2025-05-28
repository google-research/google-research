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

"""Collect frequent words in string columns."""

from typing import Any
from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import helpers

INTERNAL = False


build_partitioned_predicate = helpers.build_partitioned_predicate
get_partitioning_info = helpers.get_partitioning_info
run_query = database_connector.run_query
get_query_result_first_row = database_connector.get_query_result_first_row
DBType = database_connector.DBType
TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_INFO_TABLE = configuration.COLUMNS_INFO_TABLE
COLUMNS_STATS_TABLE = configuration.COLUMNS_STATS_TABLE
BQ_INFO_SCHEMA_COLUMNS = configuration.BQ_INFO_SCHEMA_COLUMNS
TYPES_TO_TABLES = configuration.TYPES_TO_TABLES
get_sql_table_string = database_connector.get_sql_table_string


def calculate_and_write_frequent_words_internal_bq(
    projectname,
    column_list,
    extra_stats_table,
    max_sample_vas,
    min_str_occ,
    dbs,
):
  """Collects the most frequent words in a string column, Big Query version."""
  rows_updated_with_freq_words = 0
  for task in column_list:
    print(" calculating frequent words for: ", task)
    datasetname, tablename, columnname = task
    table_sql_string = get_sql_table_string(
        dbs["data_dbtype"], f"{projectname}.{datasetname}.{tablename}"
    )
    # get size of table
    query = f"select count(*) as cnt from {table_sql_string}"
    count = -1
    try:
      queryjob, _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
      rowres = get_query_result_first_row(dbs["data_dbtype"], queryjob)
      count = rowres["cnt"]
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)

    # The threshold is the minimum number of times a word must appear in the
    # sample to be considered frequent.
    threshold = int(count * min_str_occ)
    table_sql_string = get_sql_table_string(
        dbs["data_dbtype"], f"{projectname}.{datasetname}.{tablename}"
    )
    if dbs["data_dbtype"] == DBType.BIGQUERY:
      query = (
          "SELECT array_agg(words) AS words FROM   ( SELECT words, COUNT(*) AS"
          f" cnt FROM   ( SELECT SPLIT(`{columnname}`, ' ') AS word FROM"
          f" {table_sql_string} WHERE 1 = 1 ORDER BY rand() LIMIT"
          f" {max_sample_vas} ) AS t CROSS JOIN t.word AS words WHERE words !="
          " '' AND words != ' ' AND words IS NOT NULL AND words != ',' GROUP"
          f" BY words HAVING cnt >= {threshold} )"
      )
    else:
      raise ValueError("Unsupported database type: " + dbs["data_dbtype"])
    feqwords_array = []
    try:
      queryjob, _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
      feqwords_array = get_query_result_first_row(dbs["data_dbtype"], queryjob)[
          "words"
      ]
      rows_updated_with_freq_words += 1
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          ">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY RETRYING"
          " >>>>>>>>>>>>>>>>>>>"
      )
      print(">>>> " + str(e))
      print(">>>>" + query)

    # if we do not find enough frequent words, we pick the most frequent 200
    # words and set the flag
    # did_not_find_enough_for_freq_word_picked_200_most_frequent to true
    if feqwords_array:
      did_not_find_enough_for_freq_word_picked_200_most_frequent = "false"
    else:
      did_not_find_enough_for_freq_word_picked_200_most_frequent = "true"
      if dbs["data_dbtype"] == DBType.BIGQUERY:
        query = (
            "SELECT array_agg(words) AS words FROM   ( SELECT words, COUNT(*)"
            f" AS cnt FROM   ( SELECT SPLIT(`{columnname}`, ' ') AS word FROM"
            f" {table_sql_string} WHERE 1 = 1 ORDER BY rand() LIMIT"
            f" {max_sample_vas} ) AS t CROSS JOIN t.word AS words WHERE words"
            " != '' AND words != ' ' AND words IS NOT NULL AND words != ','"
            " GROUP BY words LIMIT 200 )"
        )
      else:
        raise ValueError("Unsupported database type: " + dbs["data_dbtype"])

      try:
        queryjob, _ = run_query(dbs["data_dbtype"], query, dbs["data_dbclient"])
        feqwords_array = get_query_result_first_row(
            dbs["data_dbtype"], queryjob
        )["words"]
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY RETRYING"
            " >>>>>>>>>>>>>>>>>>>"
        )
        print(">>>> " + str(e))
        print(">>>> " + query)
        if "due to concurrent update" not in str(e):
          break
    if feqwords_array is None:
      feqwords_array = []
    query = (
        f"UPDATE `{extra_stats_table}` SET freq_str_words = {feqwords_array},"
        " did_not_find_enough_for_freq_word_picked_200_most_frequent ="
        f" {did_not_find_enough_for_freq_word_picked_200_most_frequent} WHERE"
        f" project_name='{projectname}' AND dataset_name='{datasetname}' AND"
        f" table_name='{tablename}' AND column_name='{columnname}'"
    )
    try:
      _, _ = run_query(dbs["metadata_dbtype"], query, dbs["metadata_dbclient"])
      rows_updated_with_freq_words += 1
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          ">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY RETRYING"
          " >>>>>>>>>>>>>>>>>>>"
      )
      print(">>>> " + str(e))
      print(">>>> " + query)
      if "due to concurrent update" not in str(e):
        break
      rows_updated_with_freq_words -= 1

  return rows_updated_with_freq_words


def calculate_and_write_frequent_words_internal(
    projectname,
    column_list,
    extra_stats_table,
    max_sample_vas,
    min_str_occ,
    dbs,
):
  """Collects the most frequent words in a string column."""

  if dbs["data_dbtype"] == DBType.BIGQUERY:
    return calculate_and_write_frequent_words_internal_bq(
        projectname,
        column_list,
        extra_stats_table,
        max_sample_vas,
        min_str_occ,
        dbs,
    )
  else:
    raise ValueError("dbtype not supported yet: " + str(dbs["data_dbtype"]))


def calculate_and_write_frequent_words(
    projectname,
    datasetname,
    dbs,
    max_sample_vas = 100000,  # the maximum number of frequent words
    min_str_occ = 0.01,  # the min frequency of a word to be considered
):
  """Collects frequent words for string columns."""
  columns_to_collect_freq_words = []
  count = 0
  sqltype = "STRING"
  type_table = TYPES_TO_TABLES[sqltype]
  columns_stats_table_sql_string = get_sql_table_string(
      dbs["metadata_dbtype"], COLUMNS_STATS_TABLE
  )
  type_table_sql_string = get_sql_table_string(
      dbs["metadata_dbtype"], type_table
  )
  query = (
      "SELECT ci.dataset_name, ci.table_name, ci.column_name, ci.column_type"
      f" FROM {columns_stats_table_sql_string} as ci  WHERE ci.dataset_name ="
      f" '{datasetname}' AND ci.project_name = '{projectname}' AND"
      " ci.column_type = 'STRING' AND EXISTS (select * from"
      f" {type_table_sql_string} as est where est.column_name = ci.column_name"
      " and est.table_name = ci.table_name and est.dataset_name ="
      " ci.dataset_name AND est.project_name = ci.project_name and"
      " array_length(freq_str_words) = 0)"
  )
  queryjob, _ = run_query(
      dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
  )
  for row in queryjob:
    count += 1
    columns_to_collect_freq_words.append(
        [row["dataset_name"], row["table_name"], row["column_name"]]
    )

  rows_updated_with_frequent_words = 0
  if not INTERNAL:
    for column in columns_to_collect_freq_words:
      rows_updated_with_frequent_words += (
          calculate_and_write_frequent_words_internal(
              projectname=projectname,
              column_list=[column],
              extra_stats_table=type_table,
              max_sample_vas=max_sample_vas,
              min_str_occ=min_str_occ,
              dbs=dbs,
          )
      )
  print("rows updated with frequent words:", rows_updated_with_frequent_words)
