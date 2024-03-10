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

"""Code to create feature bag datasets."""
from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import creation_constants
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


_C1 = flags.DEFINE_integer('c1', 1, 'c1?')
_C2 = flags.DEFINE_integer('c2', 2, 'c2?')


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Program Started')
  list_of_cols_test = creation_constants.LIST_OF_COLS_TEST
  offsets = creation_constants.OFFSETS
  feature_cols = creation_constants.FEATURE_COLS
  criteo_df = pd.read_csv(
      '../data/preprocessed_dataset/preprocessed_criteo.csv',
      usecols=list_of_cols_test,
  )
  logging.info('DataFrame Loaded')
  logging.info('Number of rows in criteo df : %d', len(criteo_df))
  c1 = creation_constants.C + str(_C1.value)
  c2 = creation_constants.C + str(_C2.value)
  criteo_df = (
      criteo_df.groupby([c1, c2])
      .filter(lambda x: ((len(x) >= 50) and (len(x) <= 2500)))
      .reset_index(drop=True)
  )
  logging.info('All bags of size <50 and >2500 removed from the DataFrame')
  logging.info(
      'Number of rows in after removing groupings of size < 50 and >2500 : %d',
      len(criteo_df),
  )
  for i, col in enumerate(feature_cols):
    criteo_df[col] = criteo_df[col] + offsets[i] + i
  logging.info('Offsets added to the columns')
  results_dir = '../data/bag_ds/split_'
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  for idx, (train_index, test_index) in enumerate(kf.split(criteo_df)):
    train, test = criteo_df.iloc[train_index].reset_index(
        drop=True
    ), criteo_df.iloc[test_index].reset_index(drop=True)
    logging.info('Number of rows in train df : %d', len(train))
    logging.info('Number of rows in test df : %d', len(test))
    test_file_path = results_dir + str(idx) + '/test/' + c1 + '_' + c2 + '.csv'
    train_file_path = (
        results_dir + str(idx) + '/train/' + c1 + '_' + c2 + '.ftr'
    )
    test[list_of_cols_test].to_csv(test_file_path, index=False)
    logging.info('Test DataFrame stored at : %s', test_file_path)
    train[c1 + '_copy'] = train[c1]
    train[c2 + '_copy'] = train[c2]
    df_aggregated = (
        train.groupby([c1 + '_copy', c2 + '_copy']).agg(list).reset_index()
    )
    # pylint: disable=unnecessary-lambda
    df_aggregated['bag_size'] = df_aggregated['label'].apply(lambda x: len(x))
    df_aggregated['label_count'] = df_aggregated['label'].apply(
        lambda x: np.sum(x)
    )
    df_aggregated.sort_values(
        by=['bag_size', c1 + '_copy', c2 + '_copy'],
        ascending=False,
        inplace=True,
    )
    df_aggregated.reset_index(drop=True, inplace=True)
    df_aggregated.drop([c1 + '_copy', c2 + '_copy'], axis=1, inplace=True)
    df_aggregated.to_feather(train_file_path)
    logging.info('Train DataFrame stored at : %s', train_file_path)
  logging.info('Program Ended')


if __name__ == '__main__':
  app.run(main)
