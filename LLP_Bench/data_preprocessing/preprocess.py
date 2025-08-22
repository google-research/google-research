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

"""Code to preprocess Criteo dataset."""
from collections.abc import Sequence
import math

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import preprocess_constants


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
    dense_cols = ['I' + str(i) for i in range(1, 14)]
    sparse_cols = ['C' + str(i) for i in range(1, 27)]
    df_dense = pd.read_csv(
        '../data/raw_dataset/train_x.txt',
        delimiter=' ',
        header=None,
        usecols=range(13),
    )
    df_dense.columns = dense_cols

    def scale(x):
      if x > 2:
        x = int(math.log(float(x)) ** 2)
      return x

    df_dense = df_dense.applymap(scale)

    df_sparse = pd.read_csv(
        '../data/raw_dataset/train_i.txt',
        delimiter=' ',
        header=None,
        usecols=range(13, 39),
    )

    df_sparse.columns = sparse_cols

    df_combined = pd.concat([df_dense, df_sparse], axis=1)

    all_cols = dense_cols + sparse_cols
    for colname in all_cols:
      min_value = df_combined[colname].min()
      df_combined[colname] = df_combined[colname] - min_value

    df_y = pd.read_csv(
        '../data/raw_dataset/train_y.txt', delimiter=' ', header=None
    )

    df_y.columns = ['label']

    df_combined = pd.concat([df_combined, df_y], axis=1)

    df_combined.to_csv(
        '../data/preprocessed_dataset/preprocessed_criteo.csv', index=False
    )
  else:
    criteo_df = pd.read_csv(
        '../data/raw_dataset/CriteoSearchData', sep='\t', header=None
    )
    criteo_df.columns = preprocess_constants.SSCL_COLUMNS
    criteo_df = criteo_df.drop(columns=['sale', 'click_timestamp'])
    categorical_columns = preprocess_constants.SSCL_CATEGORICAL_COLUMNS
    numerical_columns = preprocess_constants.SSCL_NUMERICAL_COLUMNS
    target_column = preprocess_constants.SSCL_TARGET_COLUMN
    sliced_df = criteo_df[criteo_df[target_column] != -1]
    less_freq_lists = {}
    for col in categorical_columns:
      counts = sliced_df[col].value_counts(dropna=False)
      indices = sliced_df[col].value_counts(dropna=False).index
      less_freq_values = indices[counts <= 5].tolist()
      less_freq_lists[col] = less_freq_values
    sliced_df['product_category_7'] = sliced_df['product_category_7'].map(
        lambda x: {-1: '-1'}.get(x, x)
    )
    sliced_df['product_title'] = sliced_df['product_title'].map(
        lambda x: {np.nan: '-1'}.get(x, x)
    )
    for col in categorical_columns:
      tf_dict = dict(
          zip(
              less_freq_lists[col],
              ['-1' for _ in range(len(less_freq_lists[col]))],
          )
      )
      sliced_df[col] = sliced_df[col].map(lambda x: tf_dict.get(x, x))  # pylint: disable=cell-var-from-loop
    for col in categorical_columns:
      tf_dict = {'-1': None} | dict(
          zip(
              less_freq_lists[col],
              [None for _ in range(len(less_freq_lists[col]))],
          )
      )
      sliced_df[col] = sliced_df[col].map(lambda x: tf_dict.get(x, x))  # pylint: disable=cell-var-from-loop
    for col in categorical_columns:
      tf_dict = dict(
          zip(
              sliced_df[col].unique(),
              range(0, len(sliced_df[col].unique())),
          )
      )
      sliced_df[col] = sliced_df[col].map(lambda x: tf_dict.get(x, x))  # pylint: disable=cell-var-from-loop
    sliced_df['time_delay_for_conversion'] = sliced_df[
        'time_delay_for_conversion'
    ].astype(np.float64)
    sliced_df['nb_clicks_1week'] = sliced_df['nb_clicks_1week'].astype(
        np.float64
    )
    for col in numerical_columns + [target_column]:
      mean = sliced_df[col][sliced_df[col] >= 0.0].mean()
      sliced_df[col] = sliced_df[col].map(lambda x: {-1: mean}.get(x, x))  # pylint: disable=cell-var-from-loop
    for col in numerical_columns + [target_column]:
      sliced_df[col] = sliced_df[col].apply(
          lambda x: np.square(np.log2(x)) if x > 2 else x
      )
    numbered_columns = []
    numbered_columns.append('Y')
    for i in range(len(numerical_columns)):
      numbered_columns.append('N' + str(i + 1))
    for i in range(len(categorical_columns)):
      numbered_columns.append('C' + str(i + 1))
    sliced_df.columns = numbered_columns
    sliced_df.to_csv(
        '../data/preprocessed_dataset/preprocessed_criteo_sscl.csv', index=False
    )


if __name__ == '__main__':
  app.run(main)
