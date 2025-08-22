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

"""Creates the CoverType Datasets."""
import os
from typing import Sequence

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf

_READ_PATH = '/tmp/data/'
_WRITE_PATH = '/tmp/data/'

_READ_PATH = flags.DEFINE_string('read_path', _READ_PATH,
                                 'Path to find data files.')
_WRITE_PATH = flags.DEFINE_string('write_path', _WRITE_PATH,
                                  'Path to write TF Records.')


NUMERIC_FEATURES = ['Elevation', 'Aspect', 'Slope',
                    'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology',
                    'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                    'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points']

# The categorical features are all binary
CATEGORICAL_FEATURES = ([f'Wilderness_Area_{x}' for x in range(4)]
                        +[f'Soil_Type_{x}' for x in range(40)])

TARGET = 'Cover_Type'

FLAGS = flags.FLAGS


def _make_tf_record(dataframe, name, write_path, dataset_name='covtype'):
  """Writes the TF Record for the given dataframe."""
  tf.io.gfile.makedirs(os.path.join(write_path, dataset_name))
  with tf.io.TFRecordWriter(
      os.path.join(write_path, dataset_name, name + '.tfrecord')
  ) as file_writer:
    for i in range(len(dataframe)):
      row = dataframe.iloc[i]
      features = dict()
      for num_feature in NUMERIC_FEATURES+CATEGORICAL_FEATURES:
        features[num_feature] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[row[num_feature]])
        )
      features[TARGET] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[int(row[TARGET])])
      )
      record_bytes = tf.train.Example(
          features=tf.train.Features(feature=features)
      ).SerializeToString()
      file_writer.write(record_bytes)


def _make_datasets_with_1p_splits(df_normalized, write_path):
  """Creates splits for 1% datasets and saves them."""
  pretext = 113400
  train = 1134
  val = 37800
  pretextval = pretext+val  # Size of test is len(df) - pretextval

  np.random.seed(1)
  pretextval_idx = np.random.choice(len(df_normalized),
                                    pretextval, replace=False)
  pretext_idx = pretextval_idx[:pretext]
  val_idx = pretextval_idx[pretext:]
  train_idx = pretextval_idx[pretext-train:pretext]
  test_idx = np.array(list(set(np.arange(0, len(df_normalized)))
                           - set(pretextval_idx)))

  df_norm_pretext = df_normalized.iloc[pretext_idx]
  df_norm_train = df_normalized.iloc[train_idx]
  df_norm_val = df_normalized.iloc[val_idx]
  df_norm_test = df_normalized.iloc[test_idx]

  dataset_name = 'covtype_new_1p'
  _make_tf_record(
      df_norm_train,
      'train',
      write_path=write_path,
      dataset_name=dataset_name,
  )
  _make_tf_record(
      df_norm_val, 'val', write_path=write_path, dataset_name=dataset_name
  )
  _make_tf_record(
      df_norm_test, 'test', write_path=write_path, dataset_name=dataset_name
  )
  _make_tf_record(
      df_norm_pretext,
      'pretext',
      write_path=write_path,
      dataset_name=dataset_name,
  )


def _make_datasets_with_10p_splits(df_normalized, write_path):
  """Creates splits for 10% datasets and saves them."""
  val = 5810  # this is .01*(len(df))
  pretextval = 464809  # this is len(df)*.8
  pretext = pretextval
  train = int(pretext*.1)
  val = int(.05*pretext)
  test = 116203  # this is len(df)*.2

  np.random.seed(1)
  pretextval_idx = np.random.choice(len(df_normalized),
                                    pretextval, replace=False)
  pretext_idx = pretextval_idx[:pretext]  # take the entire thing as the pretext
  val_idx = pretextval_idx[-val:]  # take the tail of this set as the val
  train_idx = pretextval_idx[:train]  # take the head of this as the train
  test_idx = np.array(list(set(np.arange(0, len(df_normalized)))
                           -set(pretextval_idx)))

  assert len(test_idx) == test
  assert len(pretextval_idx) == pretextval
  assert len(val_idx) == val
  assert len(train_idx) == train
  df_norm_pretext = df_normalized.iloc[pretext_idx]
  df_norm_train = df_normalized.iloc[train_idx]
  df_norm_val = df_normalized.iloc[val_idx]
  df_norm_test = df_normalized.iloc[test_idx]

  _make_tf_record(df_norm_train, 'train',
                  write_path=write_path, dataset_name='covtype_10p')
  _make_tf_record(df_norm_val, 'val',
                  write_path=write_path, dataset_name='covtype_10p')
  _make_tf_record(df_norm_test, 'test',
                  write_path=write_path, dataset_name='covtype_10p')
  _make_tf_record(df_norm_pretext, 'pretext',
                  write_path=write_path, dataset_name='covtype_10p')


def _make_datasets_with_15k_splits(df_normalized, write_path):
  """Creates splits for 15k datasets and saves them."""
  train = 11340
  val = 3780
  trainval = train+val
  test = len(df_normalized)-trainval

  np.random.seed(1)
  trainval_idx = np.random.choice(len(df_normalized), trainval, replace=False)
  train_idx = trainval_idx[:train]
  val_idx = trainval_idx[train:trainval]
  test_idx = np.array(list(set(np.arange(0, len(df_normalized)))
                           -set(trainval_idx)))
  df_norm_trainval = df_normalized.iloc[trainval_idx]
  df_norm_test = df_normalized.iloc[test_idx]
  df_norm_train = df_normalized.iloc[train_idx]
  df_norm_val = df_normalized.iloc[val_idx]

  assert len(df_norm_trainval) == trainval
  assert len(df_norm_test) == test
  assert len(df_norm_val) == val
  assert len(df_norm_train) == train

  _make_tf_record(df_norm_trainval, 'trainval',
                  write_path=write_path, dataset_name='covtype15k')
  _make_tf_record(df_norm_test, 'test',
                  write_path=write_path, dataset_name='covtype15k')
  _make_tf_record(df_norm_train, 'train',
                  write_path=write_path, dataset_name='covtype15k')
  _make_tf_record(df_norm_val, 'val',
                  write_path=write_path, dataset_name='covtype15k')


def make_datasets(read_path, write_path):
  """Reads the original data and creates 4 splits of data."""
  columns = NUMERIC_FEATURES+CATEGORICAL_FEATURES+[TARGET]

  with tf.io.gfile.GFile(
      os.path.join(read_path, 'covtype.data')) as f:
    df = pd.read_csv(f, header=None, names=columns)

  # 1% split
  means = df[NUMERIC_FEATURES].mean()
  stds = df[NUMERIC_FEATURES].std()

  df_normalized = df.copy()
  for col in NUMERIC_FEATURES:
    df_normalized[col] = (df_normalized[col]-means[col])/stds[col]
  df_normalized.head()

  # 1% Datasets
  _make_datasets_with_1p_splits(
      df_normalized, write_path
  )

  # 10% Datasets
  _make_datasets_with_10p_splits(df_normalized, write_path)

  # 15k Datasts
  _make_datasets_with_15k_splits(df_normalized, write_path)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  read_path = _READ_PATH.value
  write_path = _WRITE_PATH.value

  make_datasets(read_path, write_path)


if __name__ == '__main__':
  app.run(main)
