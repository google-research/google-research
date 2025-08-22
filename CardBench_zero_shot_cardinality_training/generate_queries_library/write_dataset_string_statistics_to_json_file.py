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

"""Write the string column statistics of a dataset to a json file."""

import json
import os
from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector


TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_STATS_TABLE = configuration.COLUMNS_STATS_TABLE
COLUMNS_STRING_EXTRA_STATS_TABLE = (
    configuration.COLUMNS_STRING_EXTRA_STATS_TABLE
)
is_type_we_dont_collect_stats = configuration.is_type_we_dont_collect_stats
PK_FK_TABLE = configuration.PK_FK_TABLE
TYPE_TO_TYPE_TABLE = configuration.TYPES_TO_TABLES
run_query = database_connector.run_query
DBType = database_connector.DBType
get_query_result_first_row = database_connector.get_query_result_first_row
Datatype = configuration.Datatype
open_file = open
file_exists = os.path.exists
remove_file = os.remove


def write_dataset_string_statistics_to_json_file(
    projectname, datasetname, output_dir, dbs
):
  """Write the string column statistics of a dataset to a json file."""
  filepath = f"{output_dir}/{datasetname}.string_statistics.json"
  print("Saving the string coln of dataset: ", datasetname, " to: ", filepath)

  if gfile.Exists(filepath):
    print("String_statistics json exists -- recreating")
    gfile.Remove(filepath)

  string_statistics_json = {}
  query = (
      "SELECT table_name, column_name, freq_str_words FROM "
      f"`{COLUMNS_STRING_EXTRA_STATS_TABLE}`"
      f" WHERE project_name = '{projectname}'"
      f" AND dataset_name = '{datasetname}'"
      " AND array_length(freq_str_words) > 0"
  )
  queryjob = None
  try:
    queryjob, _ = run_query(
        dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)

  for row in queryjob:
    tablename = row["table_name"]
    columnname = row["column_name"]
    print("Writing information for column... ", columnname)
    if tablename not in string_statistics_json:
      string_statistics_json[tablename] = {}
    string_statistics_json[tablename][columnname] = {
        "freq_str_words": row["freq_str_words"]
    }

  print("Writing column string statistics json file to: ", filepath)
  with gfile.Open(filepath, "wt") as outfile:
    json.dump(string_statistics_json, outfile, default=str)
  return filepath
