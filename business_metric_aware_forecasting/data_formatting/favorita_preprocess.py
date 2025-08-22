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

"""Pre-processes Favorita data.

Borrows logic from:
https://github.com/google-research/google-research/blob/master/tft/data_formatters/favorita.py
"""

import datetime
import gc
import glob
import os
import pickle
import time


from data_formatting.favorita_data_formatter import FavoritaFormatter
import numpy as np
import pandas as pd
import pyunpack


def unzip(zip_path, output_file, data_folder):
  """Unzips files and checks successful completion."""
  print('Unzipping file: {}'.format(zip_path))
  pyunpack.Archive(zip_path).extractall(data_folder)

  # Checks if unzip was successful
  if not os.path.exists(output_file):
    raise ValueError(
        'Error in unzipping process! {} not found.'.format(output_file)
    )


def download_data():
  """Downloads Favorita data from internet if not available locally.

  Returns:
    data folder containing favorita data

  Raises:
    ValueError: if download is unsuccessful
  """
  url = 'https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data'

  data_folder = '../data/favorita/'

  # Save manual download to root folder to avoid deleting when re-processing.
  zip_file = os.path.join(
      data_folder, 'favorita-grocery-sales-forecasting.zip'
  )

  if not os.path.exists(zip_file):
    raise ValueError(
        'Favorita zip file not found in {}!'.format(zip_file)
        + ' Please manually download data from Kaggle @ {}'.format(url)
    )

  # Unpack main zip file
  outputs_file = os.path.join(data_folder, 'train.csv.7z')
  unzip(zip_file, outputs_file, data_folder)

  # Unpack individually zipped files
  for file in glob.glob(os.path.join(data_folder, '*.7z')):
    csv_file = file.replace('.7z', '')

    unzip(file, csv_file, data_folder)

  print('Unzipping complete, commencing data processing...')
  return data_folder


def preprocess_temporal_data(data_folder, output_fpath):
  """Pre-processes temporal data.

  Args:
    data_folder: folder housing favorita dataset
    output_fpath: filepath to output the pre-processed data

  Returns:
    pre-processed temporal data
  """
  # Extract only a subset of data to save/process for efficiency
  start_date = datetime.datetime(2015, 1, 1)
  end_date = datetime.datetime(2016, 6, 1)

  print('Regenerating data...')

  # load temporal data
  temporal = pd.read_csv(os.path.join(data_folder, 'train.csv'), index_col=0)
  store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0)
  oil = pd.read_csv(os.path.join(data_folder, 'oil.csv'), index_col=0).iloc[
      :, 0
  ]
  holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'))
  items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0)
  transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'))

  temporal['date'] = pd.to_datetime(temporal['date'])

  # Filter dates to reduce storage space requirements
  if start_date is not None:
    temporal = temporal[(temporal['date'] >= start_date)]
  if end_date is not None:
    temporal = temporal[(temporal['date'] < end_date)]

  # Add trajectory identifier
  temporal['traj_id'] = (
      temporal['store_nbr'].apply(str) + '_' + temporal['item_nbr'].apply(str)
  )
  temporal['unique_id'] = (
      temporal['traj_id'] + '_' + temporal['date'].apply(str)
  )

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
    sub_df[['store_nbr', 'item_nbr', 'onpromotion']] = sub_df[
        ['store_nbr', 'item_nbr', 'onpromotion']
    ].fillna(method='ffill')
    sub_df['open'] = sub_df['open'].fillna(
        0
    )  # flag where sales data is unknown
    sub_df['log_sales'] = np.log(sub_df['unit_sales'])

    resampled_dfs.append(sub_df.reset_index(drop=True))

  new_temporal = pd.concat(resampled_dfs, axis=0)
  del temporal
  gc.collect()
  temporal = new_temporal

  print('Adding oil')
  oil.name = 'oil'
  oil.index = pd.to_datetime(oil.index)
  temporal = temporal.join(oil.fillna(method='ffill'), on='date', how='left')
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
      how='left',
  )
  temporal['transactions'] = temporal['transactions'].fillna(-1)

  # Additional date info
  temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
  temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
  temporal['month'] = pd.to_datetime(temporal['date'].values).month

  # Add holiday info
  print('Adding holidays')
  holiday_subset = holidays[
      holidays['transferred'].apply(lambda x: not x)
  ].copy()
  holiday_subset.columns = [
      s if s != 'type' else 'holiday_type' for s in holiday_subset.columns
  ]
  holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
  local_holidays = holiday_subset[holiday_subset['locale'] == 'Local']
  regional_holidays = holiday_subset[holiday_subset['locale'] == 'Regional']
  national_holidays = holiday_subset[holiday_subset['locale'] == 'National']

  temporal['national_hol'] = temporal.merge(
      national_holidays, left_on=['date'], right_on=['date'], how='left'
  )['description'].fillna('')
  temporal['regional_hol'] = temporal.merge(
      regional_holidays,
      left_on=['state', 'date'],
      right_on=['locale_name', 'date'],
      how='left',
  )['description'].fillna('')
  temporal['local_hol'] = temporal.merge(
      local_holidays,
      left_on=['city', 'date'],
      right_on=['locale_name', 'date'],
      how='left',
  )['description'].fillna('')
  temporal.sort_values('unique_id', inplace=True)
  temporal.to_csv(output_fpath)
  return temporal


def main():
  start = time.time()

  print(f'unzipping data... ({time.time() - start:.2f} sec elapsed)')
  data_folder = download_data()

  print(f'formatting data... ({time.time() - start:.2f} sec elapsed)')
  data = preprocess_temporal_data(data_folder,
                                  '../data/favorita/temporal_output.csv')
  data_formatter = FavoritaFormatter()
  train, valid, test = data_formatter.split_data(data)
  fixed_params = data_formatter.get_fixed_params()

  train.to_csv('../data/favorita/favorita_train.csv', index=None)
  valid.to_csv('../data/favorita/favorita_valid.csv', index=None)
  test.to_csv('../data/favorita/favorita_test.csv', index=None)
  pickle.dump(data_formatter, open('../data/favorita/data_formatter.pkl', 'wb'))
  pickle.dump(fixed_params, open('../data/favorita/fixed_params.pkl', 'wb'))

  print(f'creating combined tensor... ({time.time() - start:.2f} sec elapsed)')
  combined = pd.concat([train, valid, test], axis=0)
  combined['date'] = pd.to_datetime(combined['date'])
  keep_vars = ['traj_id', 'date']
  data_vars = [
      'unit_sales',
      'transactions',
      'oil',
      'day_of_month',
      'month',
      'open',
      'item_nbr',
      'store_nbr',
      'city',
      'state',
      'type',
      'cluster',
      'family',
      'class',
      'perishable',
      'onpromotion',
      'day_of_week',
      'national_hol',
      'regional_hol',
      'local_hol',
  ]
  df = combined.drop('unique_id', axis=1)[data_vars + ['traj_id', 'date']]
  long = df.melt(id_vars=keep_vars, value_vars=data_vars)
  pivoted_vars = long['variable'].unique()
  traj_ids = np.unique(long.traj_id)
  dates = np.unique(long.date)
  iix = pd.MultiIndex.from_product([traj_ids, dates])
  pivoted0 = long.pivot_table(
      values='value', index=keep_vars, columns='variable', aggfunc='first'
  )
  pivoted0 = pivoted0[data_vars]
  pivoted = (
      pivoted0.reindex(iix).to_numpy().reshape(len(traj_ids), len(dates), -1)
  )

  pivoted_df = pd.DataFrame(
      pivoted[:, :, 0], columns=list(dates), index=list(traj_ids)
  )
  pivoted_df = (
      pivoted_df.reset_index()
      .rename({'index': 'traj_id'}, axis=1)
      .set_index('traj_id')
  )
  pivoted_df.to_csv('../data/favorita/favorita_df.csv')

  np.save(
      open('../data/favorita/favorita_full_pivoted_vars.npy', 'wb'),
      pivoted_vars,
      allow_pickle=True,
  )
  np.save(open('../data/favorita/favorita_full_traj_ids.npy', 'wb'), traj_ids,
          allow_pickle=True)
  np.save(open('../data/favorita/favorita_full_dates.npy', 'wb'), dates,
          allow_pickle=True)
  np.save(open('../data/favorita/favorita_tensor_full.npy', 'wb'), pivoted,
          allow_pickle=False)

  start_date = '2015-01-01'
  end_date = '2016-02-01'

  filtered_long = long[long['date'] <= end_date]
  last_dates = pd.to_datetime(filtered_long.groupby('traj_id')['date'].max())

  lengths = (last_dates - pd.Timestamp(start_date))
  lengths = lengths.reset_index()
  lengths['days'] = lengths['date'].dt.days
  lengths.to_csv('../data/favorita/favorita_lengths.csv', index=False)

  category_ct = {}
  for col in pivoted0.columns:
    category_ct[col] = len(pivoted0[col].unique())
    print(col, category_ct[col])
  pickle.dump(category_ct,
              open('../data/favorita/favorita_full_category_counts.pkl', 'wb'))
  print(f'finished. ({time.time() - start:.2f} sec elapsed)')


if __name__ == '__main__':
  main()
