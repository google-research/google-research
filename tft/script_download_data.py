# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Script to download  data for a default experiment.

Only downloads data if the csv files are present, unless the "force_download"
argument is supplied. For new datasets, the download_and_unzip(.) can be reused
to pull csv files from an online repository, but may require subsequent
dataset-specific processing.

Usage:
  python3 script_download_data {EXPT_NAME} {OUTPUT_FOLDER} {FORCE_DOWNLOAD}

Command line args:
  EXPT_NAME: Name of experiment to download data for  {e.g. volatility}
  OUTPUT_FOLDER: Path to folder in which
  FORCE_DOWNLOAD: Whether to force data download from scratch.



"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import gc
import glob
import os
import shutil
import sys
from functools import reduce
from operator import ior

from expt_settings.configs import ExperimentConfig
import numpy as np
import pandas as pd
import pyunpack
import wget


# General functions for data downloading & aggregation.
from libs.data_utils import load_or_read_csv


def download_from_url(url, output_path):
  """Downloads a file froma url."""

  print('Pulling data from {} to {}'.format(url, output_path))
  wget.download(url, output_path)
  print('done')


def recreate_folder(path):
  """Deletes and recreates folder."""

  shutil.rmtree(path)
  os.makedirs(path)


def unzip(zip_path, output_file, data_folder):
  """Unzips files and checks successful completion."""

  print('Unzipping file: {}'.format(zip_path))
  pyunpack.Archive(zip_path).extractall(data_folder)

  # Checks if unzip was successful
  if not os.path.exists(output_file):
    raise ValueError(
        'Error in unzipping process! {} not found.'.format(output_file))


def download_and_unzip(url, zip_path, csv_path, data_folder):
  """Downloads and unzips an online csv file.

  Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
  """

  download_from_url(url, zip_path)

  unzip(zip_path, csv_path, data_folder)

  print('Done.')


# Dataset specific download routines.
def download_volatility(config):
  """Downloads volatility data from OMI website."""

  url = 'https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip'

  data_folder = config.data_folder
  csv_path = os.path.join(data_folder, 'oxfordmanrealizedvolatilityindices.csv')
  zip_path = os.path.join(data_folder, 'oxfordmanrealizedvolatilityindices.zip')

  download_and_unzip(url, zip_path, csv_path, data_folder)

  print('Unzip complete. Adding extra inputs')

  df = pd.read_csv(csv_path, index_col=0)  # no explicit index

  # Adds additional date/day fields
  idx = [str(s).split('+')[0] for s in df.index
        ]  # ignore timezones, we don't need them
  dates = pd.to_datetime(idx)
  df['date'] = dates
  df['days_from_start'] = (dates - pd.datetime(2000, 1, 3)).days
  df['day_of_week'] = dates.dayofweek
  df['day_of_month'] = dates.day
  df['week_of_year'] = dates.weekofyear
  df['month'] = dates.month
  df['year'] = dates.year
  df['categorical_id'] = df['Symbol'].copy()

  # Processes log volatility
  vol = df['rv5_ss'].copy()
  vol.loc[vol == 0.] = np.nan
  df['log_vol'] = np.log(vol)

  # Adds static information
  symbol_region_mapping = {
      '.AEX': 'EMEA',
      '.AORD': 'APAC',
      '.BFX': 'EMEA',
      '.BSESN': 'APAC',
      '.BVLG': 'EMEA',
      '.BVSP': 'AMER',
      '.DJI': 'AMER',
      '.FCHI': 'EMEA',
      '.FTMIB': 'EMEA',
      '.FTSE': 'EMEA',
      '.GDAXI': 'EMEA',
      '.GSPTSE': 'AMER',
      '.HSI': 'APAC',
      '.IBEX': 'EMEA',
      '.IXIC': 'AMER',
      '.KS11': 'APAC',
      '.KSE': 'APAC',
      '.MXX': 'AMER',
      '.N225': 'APAC ',
      '.NSEI': 'APAC',
      '.OMXC20': 'EMEA',
      '.OMXHPI': 'EMEA',
      '.OMXSPI': 'EMEA',
      '.OSEAX': 'EMEA',
      '.RUT': 'EMEA',
      '.SMSI': 'EMEA',
      '.SPX': 'AMER',
      '.SSEC': 'APAC',
      '.SSMI': 'EMEA',
      '.STI': 'APAC',
      '.STOXX50E': 'EMEA'
  }

  df['Region'] = df['Symbol'].apply(lambda k: symbol_region_mapping[k])

  # Performs final processing
  output_df_list = []
  for grp in df.groupby('Symbol'):
    sliced = grp[1].copy()
    sliced.sort_values('days_from_start', inplace=True)
    # Impute log volatility values
    sliced['log_vol'].fillna(method='ffill', inplace=True)
    sliced.dropna()
    output_df_list.append(sliced)

  df = pd.concat(output_df_list, axis=0)

  output_file = config.data_csv_path
  print('Completed formatting, saving to {}'.format(output_file))
  df.to_csv(output_file)

  print('Done.')


def download_electricity(config):
  """Downloads electricity dataset from UCI repository."""

  url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

  data_folder = config.data_folder
  csv_path = os.path.join(data_folder, 'LD2011_2014.txt')
  zip_path = csv_path + '.zip'

  download_and_unzip(url, zip_path, csv_path, data_folder)

  print('Aggregating to hourly data')

  df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
  df.index = pd.to_datetime(df.index)
  df.sort_index(inplace=True)

  # Used to determine the start and end dates of a series
  output = df.resample('1h').mean().replace(0., np.nan)

  earliest_time = output.index.min()

  df_list = []
  for label in output:
    print('Processing {}'.format(label))
    srs = output[label]

    start_date = min(srs.fillna(method='ffill').dropna().index)
    end_date = max(srs.fillna(method='bfill').dropna().index)

    active_range = (srs.index >= start_date) & (srs.index <= end_date)
    srs = srs[active_range].fillna(0.)

    tmp = pd.DataFrame({'power_usage': srs})
    date = tmp.index
    tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (
        date - earliest_time).days * 24
    tmp['days_from_start'] = (date - earliest_time).days
    tmp['categorical_id'] = label
    tmp['date'] = date
    tmp['id'] = label
    tmp['hour'] = date.hour
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month

    df_list.append(tmp)

  output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

  output['categorical_id'] = output['id'].copy()
  output['hours_from_start'] = output['t']
  output['categorical_day_of_week'] = output['day_of_week'].copy()
  output['categorical_hour'] = output['hour'].copy()

  # Filter to match range used by other academic papers
  output = output[(output['days_from_start'] >= 1096)
                  & (output['days_from_start'] < 1346)].copy()

  output.to_csv(config.data_csv_path)

  print('Done.')


def download_traffic(config):
  """Downloads traffic dataset from UCI repository."""

  url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'

  data_folder = config.data_folder
  csv_path = os.path.join(data_folder, 'PEMS_train')
  zip_path = os.path.join(data_folder, 'PEMS-SF.zip')

  download_and_unzip(url, zip_path, csv_path, data_folder)

  print('Aggregating to hourly data')

  def process_list(s, variable_type=int, delimiter=None):
    """Parses a line in the PEMS format to a list."""
    if delimiter is None:
      l = [
          variable_type(i) for i in s.replace('[', '').replace(']', '').split()
      ]
    else:
      l = [
          variable_type(i)
          for i in s.replace('[', '').replace(']', '').split(delimiter)
      ]

    return l

  def read_single_list(filename):
    """Returns single list from a file in the PEMS-custom format."""
    with open(os.path.join(data_folder, filename), 'r') as dat:
      l = process_list(dat.readlines()[0])
    return l

  def read_matrix(filename):
    """Returns a matrix from a file in the PEMS-custom format."""
    array_list = []
    with open(os.path.join(data_folder, filename), 'r') as dat:

      lines = dat.readlines()
      for i, line in enumerate(lines):
        if (i + 1) % 50 == 0:
          print('Completed {} of {} rows for {}'.format(i + 1, len(lines),
                                                        filename))

        array = [
            process_list(row_split, variable_type=float, delimiter=None)
            for row_split in process_list(
                line, variable_type=str, delimiter=';')
        ]
        array_list.append(array)

    return array_list

  shuffle_order = np.array(read_single_list('randperm')) - 1  # index from 0
  train_dayofweek = read_single_list('PEMS_trainlabels')
  train_tensor = read_matrix('PEMS_train')
  test_dayofweek = read_single_list('PEMS_testlabels')
  test_tensor = read_matrix('PEMS_test')

  # Inverse permutate shuffle order
  print('Shuffling')
  inverse_mapping = {
      new_location: previous_location
      for previous_location, new_location in enumerate(shuffle_order)
  }
  reverse_shuffle_order = np.array([
      inverse_mapping[new_location]
      for new_location, _ in enumerate(shuffle_order)
  ])

  # Group and reoder based on permuation matrix
  print('Reodering')
  day_of_week = np.array(train_dayofweek + test_dayofweek)
  combined_tensor = np.array(train_tensor + test_tensor)

  day_of_week = day_of_week[reverse_shuffle_order]
  combined_tensor = combined_tensor[reverse_shuffle_order]

  # Put everything back into a dataframe
  print('Parsing as dataframe')
  labels = ['traj_{}'.format(i) for i in read_single_list('stations_list')]

  hourly_list = []
  for day, day_matrix in enumerate(combined_tensor):

    # Hourly data
    hourly = pd.DataFrame(day_matrix.T, columns=labels)
    hourly['hour_on_day'] = [int(i / 6) for i in hourly.index
                            ]  # sampled at 10 min intervals
    if hourly['hour_on_day'].max() > 23 or hourly['hour_on_day'].min() < 0:
      raise ValueError('Invalid hour! {}-{}'.format(
          hourly['hour_on_day'].min(), hourly['hour_on_day'].max()))

    hourly = hourly.groupby('hour_on_day', as_index=True).mean()[labels]
    hourly['sensor_day'] = day
    hourly['time_on_day'] = hourly.index
    hourly['day_of_week'] = day_of_week[day]

    hourly_list.append(hourly)

  hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True, sort=False)

  # Flatten such that each entitiy uses one row in dataframe
  store_columns = [c for c in hourly_frame.columns if 'traj' in c]
  other_columns = [c for c in hourly_frame.columns if 'traj' not in c]
  flat_df = pd.DataFrame(columns=['values', 'prev_values', 'next_values'] +
                         other_columns + ['id'])

  def format_index_string(x):
    """Returns formatted string for key."""

    if x < 10:
      return '00' + str(x)
    elif x < 100:
      return '0' + str(x)
    elif x < 1000:
      return str(x)

    raise ValueError('Invalid value of x {}'.format(x))

  for store in store_columns:
    print('Processing {}'.format(store))

    sliced = hourly_frame[[store] + other_columns].copy()
    sliced.columns = ['values'] + other_columns
    sliced['id'] = int(store.replace('traj_', ''))

    # Sort by Sensor-date-time
    key = sliced['id'].apply(str) \
      + sliced['sensor_day'].apply(lambda x: '_' + format_index_string(x)) \
        + sliced['time_on_day'].apply(lambda x: '_' + format_index_string(x))
    sliced = sliced.set_index(key).sort_index()

    sliced['values'] = sliced['values'].fillna(method='ffill')
    sliced['prev_values'] = sliced['values'].shift(1)
    sliced['next_values'] = sliced['values'].shift(-1)

    flat_df = flat_df.append(sliced.dropna(), ignore_index=True, sort=False)

  # Filter to match range used by other academic papers
  index = flat_df['sensor_day']
  flat_df = flat_df[index < 173].copy()

  # Creating columns fo categorical inputs
  flat_df['categorical_id'] = flat_df['id'].copy()
  flat_df['hours_from_start'] = flat_df['time_on_day'] \
      + flat_df['sensor_day']*24.
  flat_df['categorical_day_of_week'] = flat_df['day_of_week'].copy()
  flat_df['categorical_time_on_day'] = flat_df['time_on_day'].copy()

  flat_df.to_csv(config.data_csv_path)
  print('Done.')


def process_favorita(config):
  """Processes Favorita dataset.

  Makes use of the raw files should be manually downloaded from Kaggle @
    https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data

  Args:
    config: Default experiment config for Favorita
  """

  url = 'https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data'

  data_folder = config.data_folder

  # Save manual download to root folder to avoid deleting when re-processing.
  zip_file = os.path.join(data_folder, '..',
                          'favorita-grocery-sales-forecasting.zip')

  if not os.path.exists(zip_file):
    raise ValueError(
        'Favorita zip file not found in {}!'.format(zip_file) +
        ' Please manually download data from Kaggle @ {}'.format(url))

  # Unpack main zip file
  outputs_file = os.path.join(data_folder, 'train.csv.7z')
  unzip(zip_file, outputs_file, data_folder)

  # Unpack individually zipped files
  for file in glob.glob(os.path.join(data_folder, '*.7z')):

    csv_file = file.replace('.7z', '')

    unzip(file, csv_file, data_folder)

  print('Unzipping complete, commencing data processing...')

  # Extract only a subset of data to save/process for efficiency
  start_date = pd.datetime(2015, 1, 1)
  end_date = pd.datetime(2016, 6, 1)

  print('Regenerating data...')

  # load temporal data
  temporal = pd.read_csv(os.path.join(data_folder, 'train.csv'), index_col=0)

  store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0)
  oil = pd.read_csv(
      os.path.join(data_folder, 'oil.csv'), index_col=0).iloc[:, 0]
  holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'))
  items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0)
  transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'))

  # Take first 6 months of data
  temporal['date'] = pd.to_datetime(temporal['date'])

  # Filter dates to reduce storage space requirements
  if start_date is not None:
    temporal = temporal[(temporal['date'] >= start_date)]
  if end_date is not None:
    temporal = temporal[(temporal['date'] < end_date)]

  dates = temporal['date'].unique()

  # Add trajectory identifier
  temporal['traj_id'] = temporal['store_nbr'].apply(
      str) + '_' + temporal['item_nbr'].apply(str)
  temporal['unique_id'] = temporal['traj_id'] + '_' + temporal['date'].apply(
      str)

  # Remove all IDs with negative returns
  print('Removing returns data')
  min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
  valid_ids = set(min_returns[min_returns >= 0].index)
  selector = temporal['traj_id'].apply(lambda traj_id: traj_id in valid_ids)
  new_temporal = temporal[selector].copy()
  del temporal
  gc.collect()
  temporal = new_temporal
  temporal['open'] = 1

  # Resampling
  print('Resampling to regular grid')
  resampled_dfs = []
  for traj_id, raw_sub_df in temporal.groupby('traj_id'):
    print('Resampling', traj_id)
    sub_df = raw_sub_df.set_index('date', drop=True).copy()
    sub_df = sub_df.resample('1d').last()
    sub_df['date'] = sub_df.index
    sub_df[['store_nbr', 'item_nbr', 'onpromotion']] \
        = sub_df[['store_nbr', 'item_nbr', 'onpromotion']].fillna(method='ffill')
    sub_df['open'] = sub_df['open'].fillna(
        0)  # flag where sales data is unknown
    sub_df['log_sales'] = np.log(sub_df['unit_sales'])

    resampled_dfs.append(sub_df.reset_index(drop=True))

  new_temporal = pd.concat(resampled_dfs, axis=0)
  del temporal
  gc.collect()
  temporal = new_temporal

  print('Adding oil')
  oil.name = 'oil'
  oil.index = pd.to_datetime(oil.index)
  temporal = temporal.join(
      oil.loc[dates].fillna(method='ffill'), on='date', how='left')
  temporal['oil'] = temporal['oil'].fillna(-1)

  print('Adding store info')
  temporal = temporal.join(store_info, on='store_nbr', how='left')

  print('Adding item info')
  temporal = temporal.join(items, on='item_nbr', how='left')

  transactions['date'] = pd.to_datetime(transactions['date'])
  temporal = temporal.merge(
      transactions,
      left_on=['date', 'store_nbr'],
      right_on=['date', 'store_nbr'],
      how='left')
  temporal['transactions'] = temporal['transactions'].fillna(-1)

  # Additional date info
  temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
  temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
  temporal['month'] = pd.to_datetime(temporal['date'].values).month

  # Add holiday info
  print('Adding holidays')
  holiday_subset = holidays[holidays['transferred'].apply(
      lambda x: not x)].copy()
  holiday_subset.columns = [
      s if s != 'type' else 'holiday_type' for s in holiday_subset.columns
  ]
  holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
  local_holidays = holiday_subset[holiday_subset['locale'] == 'Local']
  regional_holidays = holiday_subset[holiday_subset['locale'] == 'Regional']
  national_holidays = holiday_subset[holiday_subset['locale'] == 'National']

  temporal['national_hol'] = temporal.merge(
      national_holidays, left_on=['date'], right_on=['date'],
      how='left')['description'].fillna('')
  temporal['regional_hol'] = temporal.merge(
      regional_holidays,
      left_on=['state', 'date'],
      right_on=['locale_name', 'date'],
      how='left')['description'].fillna('')
  temporal['local_hol'] = temporal.merge(
      local_holidays,
      left_on=['city', 'date'],
      right_on=['locale_name', 'date'],
      how='left')['description'].fillna('')

  temporal.sort_values('unique_id', inplace=True)

  print('Saving processed file to {}'.format(config.data_csv_path))
  temporal.to_csv(config.data_csv_path)


def any_metric_is_missing(df):
  return reduce(
    ior,
    [
      df[metric].isnull() for metric in [
        'conversions_campaign',
        'spend_campaign',
        'impressions_campaign',
        'clicks_campaign',
      ]
    ]
  )


def preprocess_cg(config):

    cat_cols = [
      'data_connector_id',
      'campaign_id',
      'objective',
      'num_adsets',
      'campaign_age',
    ]

    metric_cols = [
      'conversions_campaign',
      'spend_campaign',
      'impressions_campaign',
      'clicks_campaign',
    ]

    df = load_or_read_csv('preprocessed_camp_stats.csv', config)

    df['date'] = pd.to_datetime(df['date'])
    df['date_first_of_campaign'] = pd.to_datetime(df['date_first_of_campaign'])
    df['campaign_age'] = (df['date'] - df['date_first_of_campaign']).dt.days

    # todo: reconsider this
    df = df.drop(columns=[
      'divisor', 'currency',
      'cum_spend', 'cum_conversions', 'cum_cpa',
      'finish_date', 'date_first_of_campaign', 'date_first_of_dc',
      'end_of_month', 'is_date_end_of_month', 'date_end_of_month',
    ])

    for metric in metric_cols:
      df[metric] = df[metric].fillna(0)

    mean_conversions = (
      df
      .groupby('campaign_id')
      ['conversions_campaign']
      .mean()
    )
    print(f'Before filter by mean conversions: {len(df)} rows, {len(mean_conversions)} campaigns')
    mean_conversions = mean_conversions[mean_conversions > 0.9]
    df = df[df['campaign_id'].isin(mean_conversions.index)]
    print(f'After filter by mean conversions: {len(df)} rows, {len(mean_conversions)} campaigns')

    budget_df = load_or_read_csv('campaign_budgets_processed.csv', config, read_csv_kwargs={'index_col': 0})
    budget_df['date'] = pd.to_datetime(budget_df['date'])
    useful_budget_columns = [
      'campaign_start_time',
      'campaign_stop_time',
      'campaign_daily_budget',
      'campaign_lifetime_budget',
      'campaign_budget_remaining',
      'num_of_adsets_with_lifetime_budget',
      'daily_budget_of_adsets_combined',
      'is_cbo',
    ]

    # fill missing values first
    # df = df.merge(
    #   budget_df[['campaign_id', 'date'] + useful_budget_columns],
    #   on=['campaign_id', 'date'],
    #   how='left',
    # )

    # campaign_stats = (
    #   df
    #   .groupby('campaign_id')
    #   .agg({
    #     'campaign_age': 'max',
    #     'conversions_campaign': 'mean',
    #   })
    #   .rename(columns={
    #     'campaign_age': 'campaign_days_in_data',
    #     'conversions_campaign': 'campaign_cpd',
    #   })
    #   .reset_index()
    # )
    # series_stats = (
    #   df
    #   .groupby(['series_id', 'campaign_id'])
    #   .agg({
    #     'campaign_age': 'max',
    #     'conversions_campaign': 'mean'
    #   })
    #   .rename(columns={
    #     'campaign_age': 'series_days_in_data',
    #     'conversions_campaign': 'series_cpd'
    #   })
    #   .reset_index()
    # )
    # series_stats = series_stats.merge(campaign_stats, on='campaign_id')
    # series_stats.to_csv('series_stats.csv', index=False)

    # Resampling
    print('Resampling to regular grid')
    resampled_dfs = []
    for series_id, sub_df in df.groupby('campaign_id'):
        # Insert missing days in the middle
        sub_df = sub_df.set_index('date', drop=True)
        sub_df = sub_df.resample('1d').last()

        sub_df[cat_cols] = sub_df[cat_cols].fillna(method='ffill')
        sub_df["present"] = 1

        if len(sub_df) < 2:
          continue

        if len(sub_df) < 30:
          present_days_in_train = len(sub_df) // 2
          days_to_add_to_train = 15 - present_days_in_train
          first_train_date = sub_df.index.min() - pd.Timedelta(days=days_to_add_to_train)
          sub_df = sub_df.reindex(pd.date_range(first_train_date, sub_df.index.max(), freq="D", name="date"))
          sub_df[cat_cols] = sub_df[cat_cols].fillna(method='bfill')

          days_to_add_to_test = 30 - len(sub_df)
          last_test_date = sub_df.index.max() + pd.Timedelta(days=days_to_add_to_test)
          sub_df = sub_df.reindex(pd.date_range(sub_df.index.min(), last_test_date, freq="D", name="date"))
          sub_df[cat_cols] = sub_df[cat_cols].fillna(method='ffill')

          sub_df["present"] = sub_df["present"].fillna(0).astype(int)
          assert len(sub_df) == 30

        sub_df = sub_df.reset_index()
        resampled_dfs.append(sub_df)

    df = pd.concat(resampled_dfs, axis=0).reset_index(drop=True)

    df['campaign_id'] = df['campaign_id'].astype(int).astype(str)
    df['data_connector_id'] = df['data_connector_id'].astype(int).astype(str)
    df['num_adsets'] = df['num_adsets'].astype(int)
    df['campaign_age'] = df['campaign_age'].astype(int)

    is_missing = any_metric_is_missing(df)
    df["is_missing"] = is_missing.astype(int)

    for metric in metric_cols:
        df[metric] = df[metric].fillna(0)
        df[f'{metric}_sum_last30d'] = df[metric].rolling(30, min_periods=1).sum()
        df[f'{metric}_mean_last30d'] = df[metric].rolling(30, min_periods=1).mean()
        df[f'log1p_{metric}'] = np.log(1 + df[metric])
        df[f'log1p_{metric}_mean_last30d'] = np.log(1 + df[f'{metric}_mean_last30d'])

    df['day_of_week'] = pd.to_datetime(df['date'].values).dayofweek
    df['day_of_month'] = pd.to_datetime(df['date'].values).day
    df['month'] = pd.to_datetime(df['date'].values).month

    dow_jones_index = load_or_read_csv("Dow Jones Industrial Average Historical Data.csv", config)
    dow_jones_index["dow_jones_index"] = dow_jones_index["Price"].str.replace(',', '').astype(float)
    dow_jones_index["date"] = pd.to_datetime(dow_jones_index["Date"])
    dow_jones_index = dow_jones_index[["date", "dow_jones_index"]].set_index("date")
    dow_jones_index = dow_jones_index.reindex(pd.date_range(
      dow_jones_index.index.min(),
      dow_jones_index.index.max(),
      freq="D",
      name="date",
    ))
    dow_jones_index["dow_jones_index"] = dow_jones_index["dow_jones_index"].fillna(method="ffill").fillna(method="bfill")
    dow_jones_index = dow_jones_index.reset_index()
    df = df.merge(
      dow_jones_index[["date", "dow_jones_index"]],
      on="date",
      how="left",
    )
    # df['conversions_campaign'] = 1 + df['conversions_campaign']

    df.to_csv(config.data_csv_path)


# Core routine.
def main(expt_name, force_download, output_folder):
  """Runs main download routine.

  Args:
    expt_name: Name of experiment
    force_download: Whether to force data download from scratch
    output_folder: Folder path for storing data
  """

  print('#### Running download script ###')

  expt_config = ExperimentConfig(expt_name, output_folder)

  if os.path.exists(expt_config.data_csv_path) and not force_download:
    print('Data has been processed for {}. Skipping download...'.format(
        expt_name))
    sys.exit(0)
  # todo: why to do this?
  # else:
  #   print('Resetting data folder...')
  #   recreate_folder(expt_config.data_folder)

  # Default download functions
  download_functions = {
      'volatility': download_volatility,
      'electricity': download_electricity,
      'traffic': download_traffic,
      'favorita': process_favorita,
      'cg': preprocess_cg,
  }

  if expt_name not in download_functions:
    raise ValueError('Unrecongised experiment! name={}'.format(expt_name))

  download_function = download_functions[expt_name]

  # Run data download
  print('Getting {} data...'.format(expt_name))
  download_function(expt_config)

  print('Download completed.')


if __name__ == '__main__':

  def get_args():
    """Returns settings from command line."""

    experiment_names = ExperimentConfig.default_experiments

    parser = argparse.ArgumentParser(description='Data download configs')
    parser.add_argument(
        'expt_name',
        metavar='e',
        type=str,
        nargs='?',
        choices=experiment_names,
        help='Experiment Name. Default={}'.format(','.join(experiment_names)))
    parser.add_argument(
        'output_folder',
        metavar='f',
        type=str,
        nargs='?',
        default='.',
        help='Path to folder for data download')
    parser.add_argument(
        'force_download',
        metavar='r',
        type=str,
        nargs='?',
        choices=['yes', 'no'],
        default='no',
        help='Whether to re-run data download')

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == '.' else args.output_folder

    return args.expt_name, args.force_download == 'yes', root_folder

  name, force, folder = get_args()
  main(expt_name=name, force_download=force, output_folder=folder)
