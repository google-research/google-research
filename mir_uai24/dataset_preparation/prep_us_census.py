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

"""US Census data preparation utilities."""

from collections.abc import Sequence
import os
from typing import Hashable, Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import tqdm


_READ_PATH = flags.DEFINE_string(
    'read_path',
    default='mir_uai24/datasets/us_census/raw/usa_00001.csv',
    help='Data read path.')
_BAG_SIZE = flags.DEFINE_integer(
    'bag_size',
    default=25,
    help='Bag size.')
_WRITE_DIR = flags.DEFINE_string(
    'write_dir',
    default='mir_uai24/datasets/us_census/processed',
    help='Data write directory.')


MIN_TRAIN_BAGS_PER_KEY = 3
MIN_N_NON_TRAIN_INSTANCES = 2
MAX_MULTIPLIER = 50


def remove_duplicates(data):
  """Removes duplicate instances."""

  unique_instances = data.groupby(
      ['SEX', 'AGE', 'MARST', 'CHBORN', 'SCHOOL', 'EMPSTAT', 'OCC', 'IND']
  ).groups
  instances_to_remove = []
  random_state = np.random.RandomState(0)
  for _, duplicate_instances in tqdm.tqdm(unique_instances.items()):
    keep = random_state.choice(duplicate_instances)
    data.loc[keep, 'WKSWORK1'] = data.loc[
        duplicate_instances, 'WKSWORK1'
    ].mean()
    instances_to_remove.extend(
        [instance for instance in duplicate_instances if instance != keep]
    )
  data = data.drop(instances_to_remove, axis=0)
  data.reset_index(drop=True, inplace=True)


def create_geographic_bags(
    data,
):
  """Creates bags based on geographic features."""

  geographic_bags = data.groupby(
      ['STATEICP', 'COUNTYICP', 'CNTRY', 'CITY', 'REGION']
  ).groups

  max_train_bags_per_key = MIN_TRAIN_BAGS_PER_KEY * MAX_MULTIPLIER
  max_n_non_train_instances = MIN_N_NON_TRAIN_INSTANCES * MAX_MULTIPLIER
  min_sample_size = (
      MIN_TRAIN_BAGS_PER_KEY * _BAG_SIZE.value + MIN_N_NON_TRAIN_INSTANCES
  )
  max_sample_size = (
      max_train_bags_per_key * _BAG_SIZE.value + max_n_non_train_instances
  )
  random_state = np.random.RandomState(seed=0)

  n_train_bags_per_key_map = {}
  n_non_train_instances_map = {}
  for bag_key in tqdm.tqdm(list(geographic_bags.keys())):
    if len(geographic_bags[bag_key]) < min_sample_size:
      del geographic_bags[bag_key]
    else:
      sample_size = (
          len(geographic_bags[bag_key]) // min_sample_size
      ) * min_sample_size
      sample_size = min(sample_size, max_sample_size)
      n_train_bags_per_key_map[bag_key] = min(
          MIN_TRAIN_BAGS_PER_KEY
          * (len(geographic_bags[bag_key]) // min_sample_size),
          max_train_bags_per_key,
      )
      n_non_train_instances_map[bag_key] = min(
          MIN_N_NON_TRAIN_INSTANCES
          * (len(geographic_bags[bag_key]) // min_sample_size),
          max_n_non_train_instances,
      )
      geographic_bags[bag_key] = random_state.choice(
          geographic_bags[bag_key], size=sample_size, replace=False
      )
  return geographic_bags, n_train_bags_per_key_map, n_non_train_instances_map


def create_splits(
    geographic_bags,
    n_train_bags_per_key_map,
    n_non_train_instances_map
):
  """Creates train, val and test splits."""

  train_bags = []
  val_instances = []
  test_instances = []
  random_state = np.random.RandomState(seed=0)
  for bag_key in tqdm.tqdm(geographic_bags):
    non_train_instances = random_state.choice(
        geographic_bags[bag_key],
        size=n_non_train_instances_map[bag_key],
        replace=False,
    )
    val_instances.extend(
        non_train_instances[: n_non_train_instances_map[bag_key] // 2]
    )
    test_instances.extend(
        non_train_instances[n_non_train_instances_map[bag_key] // 2 :]
    )
    train_bag_instance_candidates = list(
        set(geographic_bags[bag_key]) - set(non_train_instances)
    )
    for _ in range(n_train_bags_per_key_map[bag_key]):
      train_bags.append(
          random_state.choice(
              train_bag_instance_candidates, size=_BAG_SIZE.value, replace=False
          )
      )
  return train_bags, val_instances, test_instances


def select_prime_instances(
    train_bags
):
  """Selects prime instances randomly."""

  prime_bag_map = {}
  random_state = np.random.RandomState(0)
  instance_id_map = {}

  prime_instances = set()
  i = 0
  for i, bag in tqdm.tqdm(enumerate(train_bags)):
    prime_i = random_state.choice(list(bag))
    while prime_i in prime_instances:
      prime_i = random_state.choice(list(bag))
    prime_instances.add(prime_i)
    prime_bag_map[prime_i] = list(bag)
    instance_id_map[prime_i] = i

  for instances in prime_bag_map.values():
    for instance_i in instances:
      if instance_i not in instance_id_map:
        i += 1
        instance_id_map[instance_i] = i
  return prime_bag_map, instance_id_map


def create_dataframes(
    data,
    prime_bag_map,
    instance_id_map,
    val_instances,
    test_instances
):
  """Creates train, val, test and train_instance dataframes."""

  data_arr = data.to_numpy()
  train_df = []
  for prime_id in prime_bag_map:
    row = {'bag_id': instance_id_map[prime_id]}
    row['instance_id'] = np.array(
        list(map(lambda x: instance_id_map[x], prime_bag_map[prime_id])))
    row['bag_id_X_instance_id'] = np.arange(
        instance_id_map[prime_id]*_BAG_SIZE.value,
        (instance_id_map[prime_id]+1)*_BAG_SIZE.value
    )
    row_data = data_arr[prime_bag_map[prime_id]]
    for i, feature in enumerate(data.columns):
      row[feature] = row_data[:, i]
    train_df.append(row)
  train_df = pd.DataFrame(train_df)

  val_df = data.loc[val_instances, :]
  val_df.reset_index(drop=True, inplace=True)
  val_df['bag_id_X_instance_id'] = val_df.index
  val_df['bag_id'] = val_df.index
  val_df['instance_id'] = val_df.index

  test_df = data.loc[test_instances, :]
  test_df.reset_index(drop=True, inplace=True)
  test_df['bag_id_X_instance_id'] = test_df.index
  test_df['bag_id'] = test_df.index
  test_df['instance_id'] = test_df.index

  train_df_instance = train_df.explode(
      column=list(data.columns) + ['instance_id', 'bag_id_X_instance_id'])
  train_df_instance = train_df_instance[
      train_df_instance.bag_id == train_df_instance.instance_id]
  train_df_instance.reset_index(drop=True, inplace=True)
  train_df_instance.bag_id_X_instance_id = train_df_instance.index
  train_df_instance.bag_id = train_df_instance.index
  train_df_instance.instance_id = train_df_instance.index

  return train_df, val_df, test_df, train_df_instance


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Reading data from %s ...', _READ_PATH.value)
  data = pd.read_csv(_READ_PATH.value)
  data.drop(
      [
          'YEAR',
          'SERIAL',
          'GQ',
          'PERNUM',
          'EMPSTATD',
          'VERSIONHIST',
          'HISTID',
          'WKSWORK2',
      ],
      axis=1,
      inplace=True,
  )

  logging.info('Removing duplicate instances ...')
  remove_duplicates(data)

  logging.info('Creating geographic bags ...')
  geographic_bags, n_train_bags_per_key_map, n_non_train_instances_map = (
      create_geographic_bags(data)
  )

  logging.info('Creating splits ...')
  train_bags, val_instances, test_instances = create_splits(
      geographic_bags, n_train_bags_per_key_map, n_non_train_instances_map
  )

  logging.info('Selecting prime instances randomly ...')
  prime_bag_map, instance_id_map = select_prime_instances(train_bags)

  data.drop(
      [
          'REGION',
          'STATEICP',
          'COUNTYICP',
          'CITY',
          'CNTRY',
          'ENUMDIST',
      ],
      axis=1,
      inplace=True,
  )
  data = pd.get_dummies(
      data,
      columns=['SEX', 'MARST', 'CHBORN', 'SCHOOL', 'EMPSTAT', 'OCC', 'IND']
  )
  for col in data.columns:
    data[col] = data[col].astype(np.float32)

  logging.info('Creating dataframes ...')
  train_df, val_df, test_df, train_df_instance = create_dataframes(
      data, prime_bag_map, instance_id_map, val_instances, test_instances)

  logging.info('Writing dataframes to %s ...', _WRITE_DIR.value)
  os.makedirs(_WRITE_DIR.value, exist_ok=True)
  train_df.to_feather(os.path.join(_WRITE_DIR.value, 'bags_train.ftr'))
  val_df.to_feather(os.path.join(_WRITE_DIR.value, 'val.ftr'))
  test_df.to_feather(os.path.join(_WRITE_DIR.value, 'test.ftr'))
  train_df_instance.to_feather(
      os.path.join(_WRITE_DIR.value, 'instance_train.ftr'))


if __name__ == '__main__':
  app.run(main)
