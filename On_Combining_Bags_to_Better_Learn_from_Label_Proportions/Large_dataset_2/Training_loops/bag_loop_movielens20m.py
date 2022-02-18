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

"""Training Loop for Generalized bags and bags methods."""
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

rng = np.random.default_rng(7183932)

data_dir = (pathlib.Path(__file__).parent /
            '../Dataset_Preprocessing/Dataset/').resolve()

num_bag_distns = 12

NUM_TOTAL_FEATURES = 1128

print('NUM_TOTAL_FEATURES')
print(NUM_TOTAL_FEATURES)
print()

list_of_feature_columns = []

genome_file_cols = ['movieId']

for i in range(1, 1129):
  list_of_feature_columns.append(str(i))
  genome_file_cols.append(str(i))

filtered_rating_cols = ['movieId', 'label']


def my_model():
  model = Sequential()
  model.add(Dense(64, input_dim=NUM_TOTAL_FEATURES, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  return model


file_to_read = open(str(data_dir) + '/normalized_W_cvxopt', 'rb')
W = np.array(pickle.load(file_to_read))

W = W + 0.0001 * np.eye(num_bag_distns)

print('W')
print(W)

file_to_read.close()
mean_arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

genome_file_to_read = (str(data_dir) + '/movie_genome_tags_rowwise.csv')

genome_df = pd.read_csv(genome_file_to_read, usecols=genome_file_cols)

genome_df.set_index(keys='movieId', inplace=True, verify_integrity=True)

genome_df_for_testing = genome_df.copy(deep=True)

X_genomes_for_test = genome_df[list_of_feature_columns].to_numpy()

auc_metric = tf.keras.metrics.AUC(from_logits=False)

epochs = 20

for split in range(5):

  random_seed = rng.integers(low=1000000, size=1)[0]
  numpy_seed = rng.integers(low=1000000, size=1)[0]
  tf_seed = rng.integers(low=1000000, size=1)[0]

  random.seed(random_seed)

  np.random.seed(numpy_seed)

  tf.random.set_seed(tf_seed)

  test_file_to_read = (
      str(data_dir) + '/Split_' + str(split) + '/test_Split_' + str(split) +
      '-filtered_ratings.csv')

  bags_file_to_read = (
      str(data_dir) + '/Split_' + str(split) + '/BagTrain_Split_' + str(split) +
      '-filtered_ratings.ftr')

  result_file = (
      str(data_dir) + '/Split_' + str(split) +
      '/result_multi_genbags_movielens' + str(split))

  df_test = pd.read_csv(test_file_to_read, usecols=filtered_rating_cols)

  df_train = pd.read_feather(
      bags_file_to_read,
      columns=[
          'month', 'date', 'ts_mod5', 'label_count', 'movieId', 'bag_size'
      ])

  # remove large and small bags
  df_train = df_train[(df_train['bag_size'] <= 2500)
                      & (df_train['bag_size'] >= 50)]

  print('Total Number of Bags:')
  print(len(df_train.index))

  list_of_bag_df = []

  list_of_sizes = []

  for i in range(1, num_bag_distns + 1):
    df_train_i = df_train[df_train['month'] == i]
    print('Number of bags in Dist ', i)
    print(len(df_train_i.index))
    list_of_bag_df.append(df_train_i)
    list_of_sizes.append(len(df_train_i.index))

  min_df_size = min(list_of_sizes)

  num_steps = int(min_df_size / 4)

  model_ell2_sq = my_model()
  model_ell2_sq.compile(metrics=[
      tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
      tf.keras.metrics.AUC(name='auc')
  ])

  model_ell1 = my_model()
  model_ell1.compile(metrics=[
      tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
      tf.keras.metrics.AUC(name='auc')
  ])

  model_single_bag = my_model()
  model_single_bag.compile(metrics=[
      tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
      tf.keras.metrics.AUC(name='auc')
  ])

  model_KLdiv = my_model()
  model_KLdiv.compile(metrics=[
      tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
      tf.keras.metrics.AUC(name='auc')
  ])

  optimizer = tf.keras.optimizers.Adam()

  kld = tf.keras.losses.KLDivergence()

  for epoch in range(epochs):
    shuffled_bag_df = []
    for j in range(num_bag_distns):
      shuffled_bag_df.append(list_of_bag_df[j].sample(
          frac=1, random_state=rng.integers(low=1000000,
                                            size=1)[0]).reset_index(drop=True))

    for step in range(num_steps):

      with tf.GradientTape(persistent=True) as tape:
        step_loss_ell2_sq = 0
        step_loss_ell1 = 0
        step_loss_single_bag = 0
        # pylint: disable=invalid-name
        step_loss_KLdiv = 0

        for i in range(4):
          index = 4 * step + i

          wts = tf.convert_to_tensor(
              np.random.multivariate_normal(mean_arr, W, size=60),
              dtype=tf.float32)

          list_of_pred_label_diffs_sq_ell2 = []

          list_of_pred_label_diffs_ell1 = []

          for j in range(num_bag_distns):
            print(
                'epoch, step, i, j : ' + str(epoch) + ' ' + str(step) + ' ' +
                str(i) + ' ' + str(j),
                end='\r')

            bag_label_diffs_ell2_sq = 0
            bag_label_diffs_ell1 = 0

            row = shuffled_bag_df[j].iloc[[index]]
            label_count = tf.reshape(
                tf.convert_to_tensor(row['label_count'], dtype=tf.float32), [])
            bag_size = tf.reshape(
                tf.convert_to_tensor(row['bag_size'], dtype=tf.float32), [])
            avg_label = label_count / (bag_size)

            df_this_bag = genome_df.loc[row['movieId'].iloc[0].tolist()]

            tf_n_array = tf.convert_to_tensor(
                df_this_bag[list_of_feature_columns].to_numpy())

            predictions_ell2_sq = model_ell2_sq(tf_n_array, training=True)

            predictions_ell1 = model_ell1(tf_n_array, training=True)

            list_of_pred_label_diffs_sq_ell2.append(
                tf.reduce_sum(predictions_ell2_sq) - label_count)

            list_of_pred_label_diffs_ell1.append(
                tf.reduce_sum(predictions_ell1) - label_count)

            predictions_single_bag = model_single_bag(tf_n_array, training=True)

            predictions_KLdiv = model_KLdiv(tf_n_array, training=True)

            step_loss_single_bag = (
                step_loss_single_bag +
                tf.square(tf.reduce_sum(predictions_single_bag) - label_count))

            step_loss_KLdiv = (
                step_loss_KLdiv +
                kld(y_true=[avg_label, 1 - avg_label],
                    y_pred=[
                        tf.reduce_mean(predictions_KLdiv),
                        1 - tf.reduce_mean(predictions_KLdiv)
                    ]))

          step_loss_ell2_sq = step_loss_ell2_sq + tf.square(
              tf.norm(
                  tf.linalg.matvec(
                      wts,
                      tf.convert_to_tensor(list_of_pred_label_diffs_sq_ell2)),
                  ord=2))

          step_loss_ell1 = step_loss_ell1 + tf.norm(
              tf.linalg.matvec(
                  wts, tf.convert_to_tensor(list_of_pred_label_diffs_ell1)),
              ord=1)

      grads_ell2_sq = tape.gradient(step_loss_ell2_sq,
                                    model_ell2_sq.trainable_weights)

      grads_ell1 = tape.gradient(step_loss_ell1, model_ell1.trainable_weights)

      grads_single_bag = tape.gradient(step_loss_single_bag,
                                       model_single_bag.trainable_weights)

      grads_KLdiv = tape.gradient(step_loss_KLdiv,
                                  model_KLdiv.trainable_weights)

      optimizer.apply_gradients(
          zip(grads_ell2_sq, model_ell2_sq.trainable_weights))

      optimizer.apply_gradients(zip(grads_ell1, model_ell1.trainable_weights))

      optimizer.apply_gradients(
          zip(grads_single_bag, model_single_bag.trainable_weights))

      optimizer.apply_gradients(zip(grads_KLdiv, model_KLdiv.trainable_weights))

    print('training done for epoch ', epoch)

    Y_genomes_for_test = model_ell2_sq.predict(X_genomes_for_test)

    Y_genomes_for_test = Y_genomes_for_test.flatten()

    genome_df_for_testing['Y_genomes_for_test'] = Y_genomes_for_test.tolist()

    df_test_batch = df_test[['movieId', 'label']].join(
        other=genome_df_for_testing[['Y_genomes_for_test']],
        on='movieId',
        how='left',
        rsuffix='right_').reset_index(drop=True)

    auc_metric.reset_state()

    auc_metric.update_state(
        df_test_batch['label'].to_numpy().flatten(),
        df_test_batch['Y_genomes_for_test'].to_numpy().flatten())

    result_ell2_sq = auc_metric.result().numpy()

    print('result_ell2_sq')
    print(str(result_ell2_sq))
    print()

    Y_genomes_for_test = model_ell1.predict(X_genomes_for_test)

    Y_genomes_for_test = Y_genomes_for_test.flatten()

    genome_df_for_testing['Y_genomes_for_test'] = Y_genomes_for_test.tolist()

    df_test_batch = df_test[['movieId', 'label']].join(
        other=genome_df_for_testing[['Y_genomes_for_test']],
        on='movieId',
        how='left',
        rsuffix='right_').reset_index(drop=True)

    auc_metric.reset_state()

    auc_metric.update_state(
        df_test_batch['label'].to_numpy().flatten(),
        df_test_batch['Y_genomes_for_test'].to_numpy().flatten())

    result_ell1 = auc_metric.result().numpy()

    print('result_ell1')
    print(result_ell1)
    print()

    Y_genomes_for_test = model_single_bag.predict(X_genomes_for_test)

    Y_genomes_for_test = Y_genomes_for_test.flatten()

    genome_df_for_testing['Y_genomes_for_test'] = Y_genomes_for_test.tolist()

    df_test_batch = df_test[['movieId', 'label']].join(
        other=genome_df_for_testing[['Y_genomes_for_test']],
        on='movieId',
        how='left',
        rsuffix='right_').reset_index(drop=True)

    auc_metric.reset_state()

    auc_metric.update_state(
        df_test_batch['label'].to_numpy().flatten(),
        df_test_batch['Y_genomes_for_test'].to_numpy().flatten())

    result_single_bag = auc_metric.result().numpy()

    print('result_single_bag')
    print(result_single_bag)
    print()

    Y_genomes_for_test = model_KLdiv.predict(X_genomes_for_test)

    Y_genomes_for_test = Y_genomes_for_test.flatten()

    genome_df_for_testing['Y_genomes_for_test'] = Y_genomes_for_test.tolist()

    df_test_batch = df_test[['movieId', 'label']].join(
        other=genome_df_for_testing[['Y_genomes_for_test']],
        on='movieId',
        how='left',
        rsuffix='right_').reset_index(drop=True)

    auc_metric.reset_state()

    auc_metric.update_state(
        df_test_batch['label'].to_numpy().flatten(),
        df_test_batch['Y_genomes_for_test'].to_numpy().flatten())

    result_KLdiv = auc_metric.result().numpy()

    print('result_KLdiv')
    print(result_KLdiv)
    print()

    results_string = (
        str(split) + ',' + str(epoch) + ',' + str(result_ell2_sq) + ',' +
        str(result_ell1) + ',' + str(result_single_bag) + ',' +
        str(result_KLdiv) + '\n')

    print('Test Results: ')
    print(results_string)

    with open(result_file, 'a') as filetoAppend:
      filetoAppend.write(results_string)
