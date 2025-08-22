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

"""Write the column statistics of a dataset to a json file."""

import json
import os
from typing import Any

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector


TABLES_INFO_TABLE = configuration.TABLES_INFO_TABLE
COLUMNS_STATS_TABLE = configuration.COLUMNS_STATS_TABLE
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


def sql_datatype_to_generator_datatype(datatype):
  match datatype:
    case "NUMERIC" | "BIGNUMERIC":
      return Datatype.NUMERIC
    case "INT64" | "INT32" | "UINT64" | "UINT32":
      return Datatype.INT
    case "FLOAT64" | "DOUBLE":
      return Datatype.FLOAT
    case "STRING":
      return Datatype.STRING
    case "TIME" | "TIMESTAMP" | "DATE" | "DATETIME":
      return Datatype.CATEGORICAL
    case _:
      print("ERROR in datatype conversion", datatype)


def write_dataset_column_statistics_to_json_file(
    projectname: str, datasetname: str, output_dir: str, dbs: dict[str, Any]
) -> str:
  """Write the column statistics of a dataset to a json file."""
  filepath = f"{output_dir}/{datasetname}.column_statistics.json"
  print("Saving the schema of dataset: ", datasetname, " to: ", filepath)

  if gfile.Exists(filepath):
    print("Column dataset json exists -- recreating")
    gfile.Remove(filepath)

  column_statistics_json = {}
  query = (
      f"SELECT * FROM `{COLUMNS_STATS_TABLE}` WHERE project_name ="
      f" '{projectname}' AND dataset_name = '{datasetname}'"
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
    columntype = row["column_type"]
    columntype = columntype.upper()
    skip_column, datatype = is_type_we_dont_collect_stats(columntype)
    if skip_column:
      continue
    tablename = row["table_name"]
    columnname = row["column_name"]
    print("Writing information for column... ", columnname)
    if tablename not in column_statistics_json:
      column_statistics_json[tablename] = {}
    column_statistics_json[tablename][columnname] = {}
    column_statistics_json[tablename][columnname]["sql_datatype"] = columntype
    column_statistics_json[tablename][columnname]["nan_ratio"] = row[
        "null_frac"
    ]
    column_statistics_json[tablename][columnname]["num_unique"] = row[
        "num_unique"
    ]

    extra_stats_table = TYPE_TO_TYPE_TABLE[columntype]
    query = (
        f"SELECT * FROM `{extra_stats_table}` WHERE project_name ="
        f" '{projectname}' AND dataset_name = '{datasetname}' AND table_name"
        f" = '{tablename}' AND column_name = '{columnname}'"
    )
    extra_stats_row = None
    try:
      queryjob, _ = run_query(
          dbs["metadata_dbtype"], query, dbs["metadata_dbclient"]
      )
      extra_stats_row = get_query_result_first_row(
          dbs["metadata_dbtype"], queryjob
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR IN QUERY >>>>>>>>>>>>>>>>>>>")
      print(">>>> " + str(e))
      print(">>>> " + query)

    column_statistics_json[tablename][columnname]["max"] = extra_stats_row[
        "max_val"
    ]
    column_statistics_json[tablename][columnname]["min"] = extra_stats_row[
        "min_val"
    ]

    if (
        columntype == "INT64"
        or columntype == "INT32"
        or columntype == "DOUBLE"
        or columntype == "FLOAT64"
        or columntype == "BIGNUMERIC"
        or columntype == "NUMERIC"
        or columntype == "UINT64"
        or columntype == "UINT32"
    ):
      column_statistics_json[tablename][columnname]["mean"] = extra_stats_row[
          "mean_val"
      ]
      column_statistics_json[tablename][columnname]["percentiles"] = (
          extra_stats_row["percentiles"]
      )
    else:
      if extra_stats_row["uniq_vals"]:
        column_statistics_json[tablename][columnname]["unique_vals"] = (
            extra_stats_row["uniq_vals"]
        )
    column_statistics_json[tablename][columnname]["datatype"] = (
        sql_datatype_to_generator_datatype(columntype)
    )

  print("Writing column statistics json file to: ", filepath)
  with gfile.Open(filepath, "wt") as outfile:
    json.dump(column_statistics_json, outfile, default=str)
  return filepath
