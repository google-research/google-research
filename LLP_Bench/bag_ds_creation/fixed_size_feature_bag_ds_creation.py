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

"""Code to create fixed size feature bags."""
from collections.abc import Sequence
import random

from absl import app
from absl import flags
import creation_constants
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


_C1 = flags.DEFINE_integer('c1', -1, 'c1?')
_C2 = flags.DEFINE_integer('c2', -1, 'c2?')
_BAG_SIZE = flags.DEFINE_integer('bag_size', -1, 'bag size?')
_SPLIT = flags.DEFINE_integer('split', -1, 'split?')


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  list_of_cols_test = creation_constants.LIST_OF_COLS_TEST
  offsets = creation_constants.OFFSETS
  feature_cols = creation_constants.FEATURE_COLS
  criteo_df = pd.read_csv(
      '../data/preprocessed_dataset/preprocessed_criteo.csv',
      usecols=list_of_cols_test,
  )
  for i, col in enumerate(feature_cols):
    criteo_df[col] = criteo_df[col] + offsets[i] + i
  c1 = creation_constants.C + str(_C1.value)
  c2 = creation_constants.C + str(_C2.value)
  results_dir = '../data/bag_ds/split_' + str(_SPLIT.value) + '/'
  train_file_path = (
      results_dir
      + 'train/feature_random_'
      + str(_BAG_SIZE.value)
      + '_'
      + c1
      + '_'
      + c2
      + '.ftr'
  )
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  it = kf.split(criteo_df)
  train_index, _ = next(x for i, x in enumerate(it) if i == _SPLIT.value)
  train = criteo_df.iloc[train_index].reset_index(drop=True)
  train = train.sample(frac=1)
  train_grouped = train.groupby([c1, c2])
  groups = [df for _, df in train_grouped]
  random.shuffle(groups)
  feature_rand_df = pd.concat(groups).reset_index(drop=True)
  len_df = len(feature_rand_df)
  feature_rand_df['bag_index'] = np.repeat(
      np.arange(len_df // _BAG_SIZE.value + 1), _BAG_SIZE.value
  )[:len_df]
  train_agg = feature_rand_df.groupby(by=['bag_index']).agg(list).reset_index()
  # pylint: disable=unnecessary-lambda
  train_agg['bag_size'] = train_agg['label'].apply(lambda x: len(x))
  train_agg['label_count'] = train_agg['label'].apply(lambda x: np.sum(x))
  train_agg.reset_index(drop=True, inplace=True)
  train_agg.drop(['bag_index'], axis=1, inplace=True)
  train_agg.to_feather(train_file_path)


if __name__ == '__main__':
  app.run(main)
