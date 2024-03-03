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

"""Code to preprocess Criteo dataset."""
from collections.abc import Sequence
import math

from absl import app
import pandas as pd


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
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


if __name__ == '__main__':
  app.run(main)
