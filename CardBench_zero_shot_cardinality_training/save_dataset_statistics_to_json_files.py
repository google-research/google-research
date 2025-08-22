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

"""Write dataset statistics to json files."""

from collections.abc import Sequence
import traceback

from absl import app

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.generate_queries_library import write_dataset_column_statistics_to_json_file
from CardBench_zero_shot_cardinality_training.generate_queries_library import write_dataset_schema_to_json_file
from CardBench_zero_shot_cardinality_training.generate_queries_library import write_dataset_string_statistics_to_json_file


DBType = database_connector.DBType
create_database_connection = database_connector.create_database_connection
DIRECTORY_PATH_JSON_FILES = configuration.DIRECTORY_PATH_JSON_FILES

write_dataset_schema_to_json_file = (
    write_dataset_schema_to_json_file.write_dataset_schema_to_json_file
)
write_dataset_column_statistics_to_json_file = (
    write_dataset_column_statistics_to_json_file.write_dataset_column_statistics_to_json_file
)
write_dataset_string_statistics_to_json_file = (
    write_dataset_string_statistics_to_json_file.write_dataset_string_statistics_to_json_file
)


def write_dataset_statistics_to_json_files(_):
  """Writes information avout a dataset to json files, the information includes schema, column statistics and string statistics and is read from the metadata database."""
  # The list of datasets of a project to save statistics to json files.
  # The project name and dataset name are Big Query terminology. Each
  # dataset containes multiple tables. The json files are required by
  # the query generator as input.

  projectname = configuration.PROJECT_NAME
  datasetnames = configuration.DATASET_NAMES

  dbs = {
      "metadata_dbtype": configuration.METADATA_DBTYPE,
      "metadata_dbclient": create_database_connection(
          configuration.METADATA_DBTYPE
      ),
  }

  for datasetname in datasetnames:
    try:
      schema_json_path = write_dataset_schema_to_json_file(
          projectname, datasetname, DIRECTORY_PATH_JSON_FILES, dbs
      )
    except Exception as e:  # pylint: disable=broad-except
      print("FAILED saving dataset schema to json file ", datasetname)
      print(">>>> " + str(e))
    else:
      print("SUCCESSFUL schema for ", datasetname, " at: ", schema_json_path)

    try:
      column_statistics_json_path = (
          write_dataset_column_statistics_to_json_file(
              projectname, datasetname, DIRECTORY_PATH_JSON_FILES, dbs
          )
      )
    except Exception as e:  # pylint: disable=broad-except
      print(
          "FAILED saving dataset statistics to json files ", datasetname, "\n"
      )
      print(">>>> " + str(e))
      print(traceback.format_exc())
    else:
      print(
          "SUCCESSFUL schema for ",
          datasetname,
          " at: ",
          column_statistics_json_path,
      )

    try:
      string_statistics_json_path = (
          write_dataset_string_statistics_to_json_file(
              projectname, datasetname, DIRECTORY_PATH_JSON_FILES, dbs
          )
      )
    except Exception as e:  # pylint: disable=broad-except
      print(
          "FAILED saving dataset statistics to json files ", datasetname, "\n"
      )
      print(">>>> " + str(e))
    else:
      print(
          "SUCCESSFUL column statistics for ",
          datasetname,
          " at: ",
          string_statistics_json_path,
      )


if __name__ == "__main__":
  app.run(write_dataset_statistics_to_json_files)
