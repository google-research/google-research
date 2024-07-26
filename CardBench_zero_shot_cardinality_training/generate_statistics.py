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

"""Generate statistics for a set of datasets.

Collect table information for a set of datasets
collected data are store in BQ table
"""
from collections.abc import Sequence

from absl import app
from google.cloud import bigquery

from zero_shot_cardinality_training.collect_table_information import collect_table_information


def generate_stats(_):
  """Collect table information for a set of datasets.
  """
  projectname = "bq_public_datasets_projectname"
  datasetnames = ["usfs_fia", "uspto_oce_claims", "wikipedia"]

  bqclient = bigquery.Client()
  for datasetname in datasetnames:
    collect_table_information(bqclient, projectname, datasetname)

if __name__ == "__main__":
  app.run(generate_stats)
