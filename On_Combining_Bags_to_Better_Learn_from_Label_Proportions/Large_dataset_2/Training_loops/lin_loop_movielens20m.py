# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

#!/usr/bin/python
#
# Copyright 2021 The On Combining Bags to Better Learn from
# Label Proportions Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training Loop for Linear log reg. oracle."""
import pathlib
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

rng = np.random.default_rng(67489)

batch_size = 1024

data_dir = (pathlib.Path(__file__).parent /
            '../Dataset_Preprocessing/Dataset/').resolve()

list_of_feature_columns = []

genome_file_cols = ['movieId']

for i in range(1, 1129):
  list_of_feature_columns.append(str(i))
  genome_file_cols.append(str(i))

filtered_rating_cols = ['movieId', 'label']

NUM_TOTAL_FEATURES = 1128


def my_model():
  # pylint: disable=redefined-outer-name
  model = Sequential()
  model.add(Dense(64, input_dim=NUM_TOTAL_FEATURES, activation='relu'))
  model.add(Dense(1))

  return model


epochs = 20

genome_file_to_read = (str(data_dir) + '/movie_genome_tags_rowwise.csv')

genome_df = pd.read_csv(genome_file_to_read, usecols=genome_file_cols)

genome_df.set_index(keys='movieId', inplace=True, verify_integrity=True)

genome_df_for_testing = genome_df.copy(deep=True)

X_genomes_for_test = genome_df[list_of_feature_columns].to_numpy()

auc_metric = tf.keras.metrics.AUC(from_logits=True)

for split in range(5):

  random_seed = rng.integers(low=1000000, size=1)[0]
  numpy_seed = rng.integers(low=1000000, size=1)[0]
  tf_seed = rng.integers(low=1000000, size=1)[0]

  random.seed(random_seed)

  np.random.seed(numpy_seed)

  tf.random.set_seed(tf_seed)

  train_file_to_read = (
      str(data_dir) + '/Split_' + str(split) + '/train_Split_' + str(split) +
      '-filtered_ratings.csv')

  test_file_to_read = (
      str(data_dir) + '/Split_' + str(split) + '/test_Split_' + str(split) +
      '-filtered_ratings.csv')

  result_file = (
      str(data_dir) + '/Split_' + str(split) + '/result_lin_' + str(split))

  df_train = pd.read_csv(train_file_to_read, usecols=filtered_rating_cols)

  df_test = pd.read_csv(test_file_to_read, usecols=filtered_rating_cols)

  num_steps = int(len(df_train.index) / batch_size)

  print('num_steps: ' + str(num_steps))

  model = my_model()
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.BinaryCrossentropy(
          from_logits=True, name='binary_crossentropy'))

  for epoch in range(epochs):
    # shuffle df_train
    df_train_shuffled = df_train.sample(
        frac=1, random_state=rng.integers(low=1000000,
                                          size=1)[0]).reset_index(drop=True)

    for step in range(num_steps):
      print('epoch, step : ' + str(epoch) + ' ' + str(step), end='\r')
      df_train_shuffled_slice = df_train_shuffled.iloc[(step * batch_size):(
          (step + 1) * batch_size), :].reset_index(drop=True)

      df_batch = df_train_shuffled_slice.join(
          other=genome_df, on='movieId', how='left',
          rsuffix='right_').reset_index(drop=True)

      X_batch = df_batch[list_of_feature_columns].to_numpy()

      Y_batch = df_batch['label'].to_numpy()

      model.train_on_batch(x=X_batch, y=Y_batch)

    # now evaluate

    Y_genomes_for_test = model.predict(X_genomes_for_test)

    Y_genomes_for_test = Y_genomes_for_test.flatten()

    genome_df_for_testing['Y_genomes_for_test'] = Y_genomes_for_test.tolist()

    df_test_batch = df_test[['movieId', 'label']].join(
        other=genome_df_for_testing[['Y_genomes_for_test']],
        on='movieId',
        how='left',
        rsuffix='right_').reset_index(drop=True)

    print('len(df_test_batch.index)')
    print(len(df_test_batch.index))

    # report the AUC between label and Y_genomes_for_test

    auc_metric.reset_state()

    auc_metric.update_state(
        df_test_batch['label'].to_numpy().flatten(),
        df_test_batch['Y_genomes_for_test'].to_numpy().flatten())

    results_string = (
        str(split) + ',' + str(epoch) + ',' + str(auc_metric.result().numpy()) +
        '\n')

    print('Test Results: ')
    print(results_string)

    with open(result_file, 'a') as filetoAppend:
      filetoAppend.write(results_string)
