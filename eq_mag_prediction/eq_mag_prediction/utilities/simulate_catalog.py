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

"""Methods for creating a mock catalog."""

import os
import numpy as np
import pandas as pd


def mock_catalog_dataframe(
    dataframe_len=int(1e4),
    spatial_vector=None,
    beta=np.log(10),
):
  """Generates a mock catalog from random data."""
  rnd_seed = np.random.RandomState(seed=1905)
  time_vec = np.arange(dataframe_len) - (
      rnd_seed.uniform(size=(dataframe_len)) * 0.1 + 1e-4
  )
  if spatial_vector is None:
    spatial_vector = np.ones(dataframe_len)
  catalog_simulation = pd.DataFrame({
      'latitude': spatial_vector * rnd_seed.uniform(
          size=(dataframe_len), low=-30, high=-50
      ),
      'longitude': spatial_vector * rnd_seed.uniform(
          size=(dataframe_len), low=20, high=40
      ),
      'depth': rnd_seed.uniform(size=(dataframe_len), low=-10, high=0),
      'magnitude': np.minimum(
          3 + rnd_seed.exponential(scale=1 / beta, size=(dataframe_len)), 9
      ),
      'time': time_vec,
      'a': rnd_seed.uniform(size=(dataframe_len)),
      'b': rnd_seed.uniform(size=(dataframe_len)),
  })
  return catalog_simulation


def gin_file_for_mock_training(
    mock_df,
    config_template_path=None,
    config_target_path=None,
):
  """Generates a config.gin file for mock catalog using a template."""

  df_len = len(mock_df)
  if config_template_path is None:
    config_template_path = os.path.join(
        os.path.dirname(__file__),
        '../..',
        'results/trained_models/mock/config_template.txt',
    )
  if config_target_path is None:
    config_target_path = os.path.join(
        os.path.dirname(__file__),
        '../..',
        'results/trained_models/mock/config.gin',
    )
  train_start_time = int(mock_df.time.min())
  validation_start_time = int(mock_df.iloc[int(df_len / 3)].time)
  test_start_time = int(mock_df.iloc[int(2 * df_len / 3)].time)
  test_end_time = int(mock_df.time.max())

  # Read the content of the text file
  with open(config_template_path, 'r', encoding='utf-8') as file:
    content = file.read()

  # Insert the variables at their defined locations
  # Assuming the locations are marked by placeholders in the text file
  content = content.replace('TRAIN_START_TIME', str(train_start_time))
  content = content.replace('VALIDATION_START_TIME', str(validation_start_time))
  content = content.replace('TEST_START_TIME', str(test_start_time))
  content = content.replace('TEST_END_TIME', str(test_end_time))

  # Save the modified content as a .gin file
  with open(config_target_path, 'w', encoding='utf-8') as file:
    file.write(content)


def mock_catalog_and_config_ingestion(
    dataframe_len=int(1e4),
    spatial_vector=None,
    beta=np.log(10),
    target_folder=None,
    config_template_path=None,
    config_target_path=None,
):
  """Generates a mock catalog and a config.gin file for training."""
  mock_df = mock_catalog_dataframe(
      dataframe_len,
      spatial_vector,
      beta,
  )
  if target_folder is None:
    target_folder = os.path.join(
        os.path.dirname(__file__), '../..', 'results/catalogs/ingested'
    )
  target_path = os.path.join(target_folder, 'mock.csv')
  mock_df.to_csv(target_path)

  gin_file_for_mock_training(
      mock_df,
      config_template_path,
      config_target_path,
  )
