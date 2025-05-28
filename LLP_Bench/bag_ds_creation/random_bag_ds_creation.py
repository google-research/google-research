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

"""Code to create random bags."""
from collections.abc import Sequence

from absl import app
from absl import flags
import creation_constants
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# from absl import logging
_BAG_SIZE = flags.DEFINE_integer('bag_size', 64, 'bag_size?')
_SPLIT = flags.DEFINE_integer('split', 0, 'split?')
_WHICH_DATASET = flags.DEFINE_enum(
    'which_dataset',
    'criteo_ctr',
    ['criteo_ctr', 'criteo_sscl'],
    'Which dataset to preprocess.',
)


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if _WHICH_DATASET.value == 'criteo_ctr':
    list_of_cols_test = creation_constants.LIST_OF_COLS_TEST
    offsets = creation_constants.OFFSETS
    feature_cols = creation_constants.FEATURE_COLS
    criteo_df = pd.read_csv(
        '../data/preprocessed_dataset/preprocessed_criteo.csv',
        usecols=list_of_cols_test,
    )
    for i, col in enumerate(feature_cols):
      criteo_df[col] = criteo_df[col] + offsets[i] + i
  else:
    criteo_df = pd.read_csv(
        '../data/preprocessed_dataset/preprocessed_criteo_sscl.csv',
    )
  results_dir = '../data/bag_ds/split_' + str(_SPLIT.value) + '/'
  if _WHICH_DATASET.value == 'criteo_ctr':
    test_file_path = results_dir + 'test/random.csv'
    train_file_path = (
        results_dir + 'train/random_' + str(_BAG_SIZE.value) + '.ftr'
    )
  else:
    test_file_path = results_dir + 'test/sscl_random.csv'
    train_file_path = (
        results_dir + 'train/sscl_random_' + str(_BAG_SIZE.value) + '.ftr'
    )
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  it = kf.split(criteo_df)
  train_index, test_index = next(
      x for i, x in enumerate(it) if i == _SPLIT.value
  )
  train, test = criteo_df.iloc[train_index].reset_index(
      drop=True
  ), criteo_df.iloc[test_index].reset_index(drop=True)
  if _BAG_SIZE.value == 64:
    if _WHICH_DATASET.value == 'criteo_ctr':
      list_of_cols_test = creation_constants.LIST_OF_COLS_TEST
      test[list_of_cols_test].to_csv(test_file_path, index=False)
    else:
      test.to_csv(test_file_path, index=False)
  len_train_df = len(train)
  train = train.sample(frac=1).reset_index(drop=True)
  train['bag_index'] = np.repeat(
      np.arange(len_train_df // _BAG_SIZE.value + 1), _BAG_SIZE.value
  )[:len_train_df]
  train_agg = train.groupby(by=['bag_index']).agg(list).reset_index()
  # pylint: disable=unnecessary-lambda
  if _WHICH_DATASET.value == 'criteo_ctr':
    label = 'label'
  else:
    label = 'Y'
  train_agg['bag_size'] = train_agg[label].apply(lambda x: len(x))
  if _WHICH_DATASET.value == 'criteo_ctr':
    train_agg['label_count'] = train_agg[label].apply(lambda x: np.sum(x))
  else:
    train_agg['mean_label'] = train_agg['Y'].apply(lambda x: np.mean(x))
  train_agg.reset_index(drop=True, inplace=True)
  train_agg.drop(['bag_index'], axis=1, inplace=True)
  train_agg.to_feather(train_file_path)


if __name__ == '__main__':
  app.run(main)
