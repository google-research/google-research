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

"""Dataset generator."""
import functools
import random
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import train_constants

tfk = tf.keras
tfkl = tf.keras.layers


def generate_dataset(
    random_bags,
    feature_random_bags,
    bs,
    c1,
    c2,
    split,
    batch_size,
):
  """Generates dataset."""
  data_dir = '../data/preprocessed_dataset/bag_ds/split_'
  if random_bags:
    data_dir = '../data/preprocessed_dataset/bag_ds/split_'
    test_file_path = data_dir + split + '/test/random.csv'
    train_file_path = data_dir + split + '/train/random_' + str(bs) + '.ftr'
  elif feature_random_bags:
    data_dir = '../data/preprocessed_dataset/bag_ds/split_'
    test_file_path = data_dir + split + '/test/random.csv'
    results_dir = '../data/preprocessed_dataset/bag_ds/split_' + split + '/'
    train_file_path = (
        results_dir
        + 'train/feature_random_'
        + str(bs)
        + '_'
        + c1
        + '_'
        + c2
        + '.ftr'
    )
  else:
    test_file_path = data_dir + split + '/test/' + c1 + '_' + c2 + '.csv'
    train_file_path = data_dir + split + '/train/' + c1 + '_' + c2 + '.ftr'
  list_of_cols_test = train_constants.LIST_OF_COLS_TEST
  list_of_cols_train = train_constants.LIST_OF_COLS_TRAIN
  feature_cols = train_constants.FEATURE_COLS
  offsets = train_constants.OFFSETS
  multihot_dim = 1000254 + len(offsets)
  df_test = pd.read_csv(test_file_path, usecols=list_of_cols_test)
  x_test = np.array(df_test.drop(['label'], axis=1))
  y_test = np.array(df_test['label'], dtype=np.float32)
  df_all_bags = pd.read_feather(train_file_path, columns=list_of_cols_train)
  p = np.sum(df_all_bags['label_count']) / np.sum(df_all_bags['bag_size'])

  def generate_indices_seq_randorder(steps_per_epoch, batch_size, len_df):
    """Generates randomly shuffled indices."""
    list_to_choose_from = np.arange(len_df).tolist()
    random.shuffle(list_to_choose_from)
    list_of_indices = list_to_choose_from[0 : batch_size * steps_per_epoch]
    return list_of_indices

  def bag_batch_xy_gen_seq_randorder(batch_size, df_bags, feature_cols):
    """Generates bag batch xy generator function from which the dataset is created."""
    length_of_df = len(df_bags)
    steps_per_epoch = int(length_of_df / batch_size)
    list_of_indices = generate_indices_seq_randorder(
        steps_per_epoch, batch_size, length_of_df
    )
    shuffled_df_bags = df_bags.iloc[list_of_indices].reset_index(drop=True)
    batch_no = 0
    while batch_no < steps_per_epoch:
      list_x = []
      list_y = []
      for bag_no in range(batch_size):
        row = shuffled_df_bags.iloc[[batch_no * batch_size + bag_no]]
        label_count = row['label_count'].tolist()[0]
        bag_size = row['bag_size'].tolist()[0]

        list_col = []

        for colname in feature_cols:
          list_col.append(row[colname].to_list()[0])

        n_array = np.array(list_col, dtype=np.int32)
        bag_x = np.transpose(n_array)

        bag_y = np.zeros(shape=(bag_size, 1), dtype=np.float32)
        bag_y[0, 0] = bag_size
        bag_y[1, 0] = label_count

        list_x.append(bag_x)
        list_y.append(bag_y)

      x = np.vstack(list_x)
      y = np.vstack(list_y)

      yield x, y
      batch_no = batch_no + 1

  f = functools.partial(
      bag_batch_xy_gen_seq_randorder, batch_size, df_all_bags, feature_cols
  )
  train_ds = tf.data.Dataset.from_generator(
      f,
      output_signature=(
          tf.TensorSpec(shape=(None, len(feature_cols)), dtype=tf.int32),
          tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
      ),
  )
  return train_ds, x_test, y_test, multihot_dim, p
