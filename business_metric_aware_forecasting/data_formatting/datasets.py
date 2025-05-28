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

"""Functions for retrieving datasets.

Datasets include M3 industry montly data, synthetic data, and Favorita data.
"""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

from lib.rolling_evaluation_dataset import RollingEvaluationDatasetFactory
from lib.windowed_sequence_dataset import WindowedSequenceDataset
import numpy as np
import pandas as pd
from utils.preprocessing_utils import get_dataloaders


def get_m3_df(N=None, csv_fpath='../data/m3/m3_industry_monthly.csv',
              idx_range=None):
  """Get M3 dataframe.

  Args:
    N: number of series to include
    csv_fpath: path to raw data
    idx_range: range of indices of series to include. If None, include all.

  Returns:
    dataframe of M3 data
  """

  to_drop = ['Series', 'NF', 'Category', 'Starting Year', 'Starting Month']
  df = pd.read_csv(csv_fpath)
  print('full df length: ', len(df))
  if idx_range is not None:
    df = df[idx_range[0] : min(len(df), idx_range[1])]
  else:
    df = df[:N]
  print('new df length: ', len(df))
  to_drop = list([c for c in df.columns if c in to_drop])
  df = df.drop(to_drop, axis=1)
  df.index = df.index.astype(int)
  return df


def get_m3_data(
    forecasting_horizon,
    minmax_scaling,
    train_prop = 0.7,
    val_prop = 0.1,
    batch_size = 20,
    input_window_size = 36,
    csv_fpath='../data/m3/m3_industry_monthly.csv',
    default_nan_value=1e15,
    rolling_evaluation=True,
    idx_range=None,
    N = 334,
):
  """Get dataloaders for M3 industry monthly dataset.

  Args:
    forecasting_horizon: number of timesteps into the future to make forecasts
    minmax_scaling: whether to scale each time series to between 0 and 1
    train_prop: proportion of data to use as training set
    val_prop: proportion of data to use as validation set
    batch_size: batch size for dataloaders to iterate through data
    input_window_size: number of timesteps to use as input for forecasts
    csv_fpath: filepath to csv file containing M3 data
    default_nan_value: default value to replace nans
    rolling_evaluation: whether training and evaluation will be done in a
      roll-forward manner
    idx_range: if not None, a particular range of data to retrieve, e.g. (0, 10)
      retrieves series 0 thru 10
    N: number of series to include

  Returns:
    train_dataloader: training set dataloader
    val_dataloader: validation set dataloader
    test_dataloder: test set dataloader
  """
  df = get_m3_df(N, csv_fpath=csv_fpath, idx_range=idx_range)
  x = df.drop('N', axis=1).values
  x = x[:, :, np.newaxis]
  lengths = df['N'].values

  if rolling_evaluation:
    dataset_factory = RollingEvaluationDatasetFactory(
        x,
        lengths,
        forecasting_horizon,
        minmax_scaling,
        input_window_size,
        default_nan_value=default_nan_value,
    )
    return dataset_factory
  else:
    dataset = WindowedSequenceDataset(
        x,
        lengths,
        forecasting_horizon,
        minmax_scaling,
        input_window_size,
        default_nan_value=default_nan_value,
    )

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        dataset, train_prop=train_prop, val_prop=val_prop, batch_size=batch_size
    )

    return train_dataloader, val_dataloader, test_dataloader, dataset


def get_favorita_data(
    forecasting_horizon,
    minmax_scaling,
    input_window_size = 36,
    data_fpath='../data/favorita/favorita_tensor_full.npy',
    default_nan_value=1e15,
    rolling_evaluation=True,
    N = 100,
    test_t_max=None,
):
  """Gets the Favorita dataset.

  Args:
    forecasting_horizon: number of timesteps into the future to make forecasts
    minmax_scaling: whether to scale each time series to between 0 and 1
    input_window_size: number of timesteps to use as input for forecasts
    data_fpath: filepath to csv file containing data
    default_nan_value: default value to replace nans
    rolling_evaluation: whether training and evaluation will be done in a
      roll-forward manner
    N: number of series to include
    test_t_max: maximum timepoint to include in data. If None, no max.

  Returns:
    train_dataloader: training set dataloader
    val_dataloader: validation set dataloader
    test_dataloder: test set dataloader
  """

  assert rolling_evaluation
  if data_fpath.endswith('.npy'):
    tensor = np.load(data_fpath)
    lengths = pd.read_csv('../data/favorita/favorita_lengths.csv')
    lengths = lengths['days'].values
  else:
    raise NotImplementedError('unrecognized file type: ', data_fpath)

  if N is not None:
    tensor = tensor[:N]
    lengths = lengths[:N]

  if test_t_max is not None:
    tensor = tensor[:, :test_t_max]
    lengths = np.minimum(lengths, test_t_max)

  print('dataset size: ', tensor.shape, lengths.shape)
  x = np.nan_to_num(tensor, nan=0)

  dataset_factory = RollingEvaluationDatasetFactory(
      x,
      lengths,
      forecasting_horizon,
      minmax_scaling,
      input_window_size,
      default_nan_value=default_nan_value,
      per_series_scaling=False,
  )
  return dataset_factory
