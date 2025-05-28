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

"""Load M5 data."""

import logging
import os
import numpy as np
import pandas as pd
import sklearn.preprocessing
from src import ROOT_PATH
import tqdm

OneHotEncoder = sklearn.preprocessing.OneHotEncoder
tqdm = tqdm.tqdm

logger = logging.getLogger(__name__)


def process_static_features(
    static_features, drop_first=False
):
  """Global standard normalisation of static features & one hot encoding.

  Args:
      static_features: pd.DataFrame with unprocessed static features
      drop_first: Dropping first class of one-hot-encoded features

  Returns:
      pd.DataFrame with pre-processed static features
  """
  processed_static_features = []
  for feature in static_features.columns:
    if isinstance(static_features[feature].iloc[0], float):
      mean = np.mean(static_features[feature])
      std = np.std(static_features[feature])
      processed_static_features.append((static_features[feature] - mean) / std)
    else:
      one_hot = pd.get_dummies(static_features[feature], drop_first=drop_first)
      processed_static_features.append(one_hot.astype(float))

  static_features = pd.concat(processed_static_features, axis=1)
  return static_features


def get_seq_splits(data_path, min_seq_length, max_seq_length, cat_domains=None):
  """Get sequence splits."""
  df_calendar = pd.read_csv(os.path.join(data_path, '../calendar.csv'))
  df_week_1st_day = df_calendar.groupby('wm_yr_wk')[['d', 'year']].first()

  seq_splits_path = os.path.join(data_path, '../seq_splits.csv')
  if os.path.exists(seq_splits_path):
    seq_splits = pd.read_csv(os.path.join(seq_splits_path))
  else:
    seq_splits = []
    for item_id in tqdm(os.listdir(data_path)):
      for store_id in [
          'CA_1',
          'CA_2',
          'CA_3',
          'TX_1',
          'TX_2',
          'TX_3',
          'WI_1',
          'WI_2',
          'WI_3',
      ]:
        temporal = np.load(
            os.path.join(data_path, item_id, '{}.npz'.format(store_id))
        )
        wm_yr_wks, sizes = temporal['wm_yr_wk'], temporal['size']

        cons_sections = []
        curr_cons_sec = None
        for i in range(len(wm_yr_wks)):
          if sizes[i] < 7:
            continue
          if curr_cons_sec is None:
            curr_cons_sec = [i, i]
          else:
            if (
                df_week_1st_day['year'].loc[wm_yr_wks[i]]
                == df_week_1st_day['year'].loc[wm_yr_wks[curr_cons_sec[1]]]
            ):
              curr_cons_sec[1] = i
            else:
              cons_sections.append(curr_cons_sec[:])
              curr_cons_sec = [i, i]
        cons_sections.append(curr_cons_sec[:])
        for s, e in cons_sections:
          seq_splits.append((item_id, store_id, s, e + 1))
    seq_splits = pd.DataFrame(
        seq_splits, columns=['item_id', 'store_id', 'start', 'end']
    )
    seq_splits.to_csv(seq_splits_path, index=False)
  seq_splits['len'] = seq_splits['end'] - seq_splits['start']
  seq_splits = seq_splits[
      (seq_splits['len'] >= min_seq_length)
      & (seq_splits['len'] <= max_seq_length)
  ].reset_index(drop=True)

  # other conditions
  if cat_domains is not None:
    seq_splits['cat'] = seq_splits['item_id'].apply(lambda x: x.split('_')[0])
    seq_splits = seq_splits[seq_splits['cat'].isin(cat_domains)]

  return seq_splits


def uniform_bucktize_rel_sell_price(rsp):
  if rsp == 0:
    return 0
  elif 0 < rsp and rsp <= 1:
    return int(np.ceil(rsp / 0.1)) * 2 - 1
  elif rsp > 1:
    return 21
  else:
    return (-int(np.floor(rsp / 0.1))) * 2


def quantile_bucktize_rel_sell_price(rsp_arr):
  """Quantile bucket size."""
  ret = np.zeros_like(rsp_arr, dtype=np.int64)
  ret[rsp_arr == 0] = 0

  # 20-quantile bins of all rel_prices
  bins = [
      -0.9987908101571947,
      -0.15853658536585366,
      -0.10986547085201788,
      -0.08279668813247458,
      -0.05813953488372085,
      -0.0327868852459016,
      -0.008375209380234476,
      0.003367003367003295,
      0.02192982456140363,
      0.036649214659685896,
      0.04899135446685877,
      0.06036217303822934,
      0.07194244604316553,
      0.08718395815170009,
      0.10112359550561795,
      0.11654135338345865,
      0.13761467889908247,
      0.16617210682492584,
      0.19395465994962216,
      0.25188916876574297,
      9.0,
  ]
  nonzero_cats = (
      pd.cut(rsp_arr[rsp_arr != 0], bins, include_lowest=True).codes.astype(
          np.int64
      )
      + 1
  )
  ret[rsp_arr != 0] = nonzero_cats

  return ret


def load_m5_data_processed(
    data_path,
    min_seq_length = None,
    max_seq_length = None,
    max_number = None,
    data_seed = 100,
    treatment_bucktize='uniform',
    cat_domains=None,
    **kwargs,
):
  """Load M5 dataset aggregated by week (for real-world experiments).

  Args:
    data_path: Path of M5 dataset aggregated by week (structure:
      item_id/[{store_id}_static.npz, {store_id}_static_text.npz,
      {store_id}.npz])
    min_seq_length: Min sequence lenght in cohort
    max_seq_length: Max sequence lenght in cohort
    max_number: Maximum number of patients in cohort
    data_seed: Seed for random cohort patient selection
    treatment_bucktize: uniform or quantile
    cat_domains: selected item category
    **kwargs: kwargs

  Returns:
    tuple of DataFrames and params (treatments, outcomes, vitals,
    static_features, outcomes_unscaled, scaling_params)
  """

  logger.info('%s', f'Loading M5 Weekly dataset from {data_path}.')
  _ = kwargs

  all_seq_splits = get_seq_splits(
      data_path, min_seq_length, max_seq_length, cat_domains=None
  )
  aggregated = {
      'dept_id': [],
      'cat_id': [],
      'state_id': [],
      'store_id': [],  # ordinal
  }
  for _, row in tqdm(
      all_seq_splits.iterrows(), total=len(all_seq_splits), desc='collect'
  ):
    item_id, store_id = row['item_id'], row['store_id']
    static = np.load(
        os.path.join(data_path, item_id, '{}_static_text.npz'.format(store_id))
    )
    for k in ['dept_id', 'cat_id', 'state_id', 'store_id']:
      aggregated[k].append([static[k].item()])
  id_encoders = {}
  for k in aggregated:
    id_encoders[k] = OneHotEncoder(handle_unknown='ignore').fit(aggregated[k])

  if cat_domains is not None:
    all_seq_splits['cat'] = all_seq_splits['item_id'].apply(
        lambda x: x.split('_')[0]
    )
    valid_seq_splits = all_seq_splits[all_seq_splits['cat'].isin(cat_domains)]
  else:
    valid_seq_splits = all_seq_splits
  if max_number is not None and max_number > 0:
    max_number = min(max_number, len(valid_seq_splits['item_id'].unique()))
    rng = np.random.RandomState(seed=data_seed)
    filtered_items = rng.choice(
        valid_seq_splits['item_id'].unique(), size=max_number, replace=False
    )
    valid_seq_splits = (
        valid_seq_splits.set_index('item_id').loc[filtered_items].reset_index()
    )
  _, max_len = (
      valid_seq_splits['len'].min(),
      valid_seq_splits['len'].max(),
  )

  aggregated = {
      'event_name': [],
      'event_type': [],  # one-hot
      'snap': [],  # numerical
      'month': [],  # numerical
      'sell_price': [],  # numerical
      'rel_sell_price': [],  # numerical
      'cat_rel_sell_price': [],  # ordinal
      'sales': [],  # numerical
      'dept_id': [],
      'cat_id': [],
      'state_id': [],
      'store_id': [],
      'length': [],  # ordinal
  }

  # collect event_name, event_type, snap, month, sell_price, sales
  for _, row in tqdm(
      valid_seq_splits.iterrows(), total=len(valid_seq_splits), desc='collect'
  ):
    item_id, store_id = row['item_id'], row['store_id']
    start, end = row['start'], row['end']
    temporal = np.load(
        os.path.join(data_path, item_id, '{}.npz'.format(store_id))
    )
    sample = {
        'event_name': temporal['event_name_embs'][start:end],
        'event_type': temporal['event_type_embs'][start:end],
        'snap': temporal['snap'][start:end, np.newaxis],
        'month': temporal['month'][start:end],
        'sell_price': temporal['sell_price'][start:end, np.newaxis],
        'sales': temporal['sales'][start:end, np.newaxis],
    }
    rel_sell_price = (sample['sell_price'] - sample['sell_price'][0]) / sample[
        'sell_price'
    ][0]
    sample['rel_sell_price'] = rel_sell_price

    if treatment_bucktize == 'uniform':
      cat_rel_sell_price = []
      for rsp in rel_sell_price:
        rsp = rsp.item()
        cat_rel_sell_price.append(uniform_bucktize_rel_sell_price(rsp))
      one_hot_cate_rel_sell_price = np.zeros((len(cat_rel_sell_price), 22))
    elif treatment_bucktize == 'quantile':
      cat_rel_sell_price = quantile_bucktize_rel_sell_price(
          rel_sell_price.flatten()
      )
      one_hot_cate_rel_sell_price = np.zeros((len(cat_rel_sell_price), 22))
    else:
      raise NotImplementedError()

    one_hot_cate_rel_sell_price[
        np.arange(one_hot_cate_rel_sell_price.shape[0]), cat_rel_sell_price
    ] = 1
    sample['cat_rel_sell_price'] = one_hot_cate_rel_sell_price

    month = sample['month']
    one_hot_month = np.zeros((len(month), 12))
    one_hot_month[np.arange(one_hot_month.shape[0]), month - 1] = 1
    sample['month'] = one_hot_month

    for k in sample:
      sample_k = sample[k]
      if len(sample_k) < max_len:
        if len(sample_k.shape) == 2:
          sample_k = np.concatenate(
              [
                  sample_k,
                  np.zeros(
                      (max_len - len(sample_k), sample_k.shape[1]),
                      dtype=sample_k.dtype,
                  )
                  * np.nan,
              ],
              axis=0,
          )
        else:
          sample_k = np.concatenate(
              [
                  sample_k,
                  np.zeros((max_len - len(sample_k),), dtype=sample_k.dtype)
                  * np.nan,
              ],
              axis=0,
          )
      aggregated[k].append(sample_k)

    sample['length'] = end - start
    aggregated['length'].append(sample['length'])

    static = np.load(
        os.path.join(data_path, item_id, '{}_static_text.npz'.format(store_id))
    )
    for k in ['dept_id', 'cat_id', 'state_id', 'store_id']:
      aggregated[k].append([static[k].item()])

  for k in aggregated:
    if k in ['dept_id', 'cat_id', 'state_id', 'store_id']:
      aggregated[k] = id_encoders[k].transform(aggregated[k]).toarray()
    else:
      aggregated[k] = np.stack(aggregated[k], axis=0)

  snap_mean = np.nanmean(aggregated['snap'].reshape((-1, 1)), axis=0)
  snap_std = np.nanstd(aggregated['snap'].reshape((-1, 1)), axis=0)
  aggregated['snap'] = (aggregated['snap'] - snap_mean) / snap_std

  scaling_params = {
      'output_means': np.nanmean(aggregated['sales'].reshape((-1, 1)), axis=0),
      'output_stds': np.nanstd(aggregated['sales'].reshape((-1, 1)), axis=0),
  }
  # return aggregated, scaling_params

  treatments = aggregated['cat_rel_sell_price']
  outcomes_unscaled = aggregated['sales']
  outcomes = (
      outcomes_unscaled - scaling_params['output_means']
  ) / scaling_params['output_stds']
  vitals = np.concatenate(
      [
          aggregated['event_name'],
          aggregated['event_type'],
          aggregated['snap'],
          aggregated['month'],
      ],
      axis=-1,
  )
  static_features = np.concatenate(
      [
          aggregated['dept_id'],
          aggregated['cat_id'],
          aggregated['state_id'],
          aggregated['store_id'],
      ],
      axis=-1,
  )
  sequence_lengths = aggregated['length']

  return (
      valid_seq_splits,
      treatments,
      outcomes,
      vitals,
      static_features,
      outcomes_unscaled,
      scaling_params,
      sequence_lengths,
  )


if __name__ == '__main__':
  load_m5_data_processed(
      os.path.join(ROOT_PATH, 'data/processed/m5/full'),
      min_seq_length=40,
      max_seq_length=60,
      max_number=500,
      data_seed=10,
      treatment_bucktize='quantile',
  )
