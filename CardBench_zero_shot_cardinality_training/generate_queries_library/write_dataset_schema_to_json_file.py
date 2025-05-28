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

"""Write the schema of a dataset to a json file."""

import json
import os
from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector

TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
PK_FK_TABLE = configuration.PK_FK_TABLE
run_query = database_connector.run_query
DBType = database_connector.DBType
open_file = open
file_exists = os.path.exists
remove_file = os.remove


def write_dataset_schema_to_json_file(
    projectname, datasetname, output_dir, dbs
):
  """Write the schema of a dataset to a json file."""
  filepath = f"{output_dir}/{datasetname}.schema.json"
  print("Saving the schema of dataset: ", datasetname, " to: ", filepath)

  if gfile.Exists(filepath):
    print("Schema json exists -- recreating")
    gfile.Remove(filepath)

  schema_json = {}
  schema_json["name"] = datasetname
  schema_json["tables"] = []
  schema_json["table_stats"] = {}

  tablenames = []
  temp_dict = {}

  query = (
      f"SELECT * FROM `{TABLES_INFO_TABLE}` WHERE dataset_name ="
      f" '{datasetname}' and project_name = '{projectname}'"
  )
  try:
    queryjob, _ = run_query(
        dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
    )
    for row in queryjob:
      tablenames.append(row["table_name"])
      temp_dict[row["table_name"]] = {
          "row_count": row["row_count"],
          "data_size_gib": row["data_size_gib"],
          "file_count": row["file_count"],
      }
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)

  tablenames.sort()

  for t in tablenames:
    schema_json["tables"].append(t)
    schema_json["table_stats"][t] = temp_dict[t]
  schema_json["relationships"] = []

  query = (
      "SELECT primary_key_table_name, primary_key_column_name, "
      f" foreign_key_table_name, foreign_key_column_name  FROM `{PK_FK_TABLE}` "
      f" WHERE dataset_name = '{datasetname}' and project_name ="
      f" '{projectname}'"
  )

  try:
    queryjob, _ = run_query(
        dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
    )
    for row in queryjob:
      fk_table = row["foreign_key_table_name"]
      fk_col = row["foreign_key_column_name"]
      pk_table = row["primary_key_table_name"]
      pk_col = row["primary_key_column_name"]
      schema_json["relationships"].append([fk_table, fk_col, pk_table, pk_col])

  except Exception as e:  # pylint: disable=broad-exception-caught
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
    print(">>>> " + str(e))
    print(">>>> " + query)

  print("Writing schema json file to: ", filepath)
  with open_file(filepath, "wt") as outfile:
    json.dump(schema_json, outfile, default=str)
  return filepath
