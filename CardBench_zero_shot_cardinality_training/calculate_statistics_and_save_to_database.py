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

"""Generate statistics for a set of datasets."""

from collections.abc import Sequence

from absl import app

from CardBench_zero_shot_cardinality_training import configuration
from CardBench_zero_shot_cardinality_training import database_connector
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import calculate_and_write_column_histograms
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import calculate_and_write_column_statistics
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import calculate_and_write_extra_column_statistics
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import calculate_and_write_frequent_words
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import calculate_and_write_pearson_correlation
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import calculate_and_write_percentiles
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import calculate_and_write_unique_values
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import collect_and_write_column_information
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import collect_and_write_table_information
from CardBench_zero_shot_cardinality_training.calculate_statistics_library import create_table_samples_fixsize

collect_and_write_table_information = (
    collect_and_write_table_information.collect_and_write_table_information
)
collect_and_write_column_information = (
    collect_and_write_column_information.collect_and_write_column_information
)
calculate_and_write_column_statistics = (
    calculate_and_write_column_statistics.calculate_and_write_column_statistics
)
calculate_and_write_extra_column_statistics = (
    calculate_and_write_extra_column_statistics.calculate_and_write_extra_column_statistics
)
calculate_and_write_percentiles = (
    calculate_and_write_percentiles.calculate_and_write_percentiles
)
calculate_and_write_unique_values = (
    calculate_and_write_unique_values.calculate_and_write_unique_values
)
calculate_and_write_frequent_words = (
    calculate_and_write_frequent_words.calculate_and_write_frequent_words
)
calculate_and_write_column_histograms = (
    calculate_and_write_column_histograms.calculate_and_write_column_histograms
)
create_table_samples_fixsize = (
    create_table_samples_fixsize.create_table_samples_fixsize
)
calculate_and_write_pearson_correlation = (
    calculate_and_write_pearson_correlation.calculate_and_write_pearson_correlation
)

DBType = database_connector.DBType
create_database_connection = database_connector.create_database_connection

sample_projectname_dataset_name_4k = (
    configuration.SAMPLE_PROJECTNAME_DATASET_NAME_4K
)


def calculate_statistics_and_save_to_file(_):
  """Calculate statistics for a set of datasets and write them to the metadata database."""

  # The list of datasets of a project to calculate statistics for
  # The project name and dataset name are Big Query terminology. Each
  # dataset containes multiple tables.

  projectname = configuration.PROJECT_NAME
  datasetnames = configuration.DATASET_NAMES

  # The code uses two databases one that contains the
  # data we are calculating statistics for and one that stores the calculated
  # statistics.

  dbs = {
      # used to query the database
      "data_dbtype": configuration.DATA_DBTYPE,
      "data_dbclient": create_database_connection(configuration.DATA_DBTYPE),
      # used to stored the collected statistics
      "metadata_dbtype": configuration.METADATA_DBTYPE,
      "metadata_dbclient": create_database_connection(
          configuration.METADATA_DBTYPE
      ),
  }

  for datasetname in datasetnames:
    print("\n\n\n<><><><> Calculating statistics for dataset: ", datasetname)
    collect_and_write_table_information(projectname, datasetname, dbs)
    collect_and_write_column_information(projectname, datasetname, dbs)
    calculate_and_write_column_statistics(projectname, datasetname, dbs)
    calculate_and_write_extra_column_statistics(projectname, datasetname, dbs)
    calculate_and_write_percentiles(projectname, datasetname, dbs)
    calculate_and_write_unique_values(projectname, datasetname, dbs)
    calculate_and_write_frequent_words(projectname, datasetname, dbs)
    calculate_and_write_column_histograms(projectname, datasetname, dbs)
    # The table samples are used to calculate the pearson correlations.
    create_table_samples_fixsize(
        projectname,
        datasetname,
        sample_projectname_dataset_name_4k,
        dbs,
        target_row_number=4000,
    )
    calculate_and_write_pearson_correlation(projectname, datasetname, dbs)


if __name__ == "__main__":
  app.run(calculate_statistics_and_save_to_file)
