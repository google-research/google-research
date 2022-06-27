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
from tensorflow.keras.layers import CategoryEncoding
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

rng = np.random.default_rng(7183932)

data_dir = (pathlib.Path(__file__).parent /
            '../Dataset_Preprocessing/Dataset/').resolve()

num_bag_distns = 5

offsets = [
    0, 22, 53, 81, 98, 138, 170, 197, 219, 245, 252, 266, 287, 309, 1768, 2350,
    2654, 2676, 3308, 3310, 8992, 12185, 12194, 17845, 20016, 20018, 20034,
    20048
]

NUM_TOTAL_FEATURES = 20151 + len(offsets)

list_of_cols_train = ([
    'bag_size', 'label_count', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8',
    'N9', 'N10', 'N11', 'N12', 'N13', 'C1', 'C2', 'C5', 'C6', 'C8', 'C9', 'C11',
    'C13', 'C17', 'C18', 'C19', 'C20', 'C22', 'C23', 'C25', 'C14_bucket_index'
])

feature_cols = [
    'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12',
    'N13', 'C1', 'C2', 'C5', 'C6', 'C8', 'C9', 'C11', 'C13', 'C17', 'C18',
    'C19', 'C20', 'C22', 'C23', 'C25'
]

list_of_cols_test = [
    'label', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11',
    'N12', 'N13', 'C1', 'C2', 'C5', 'C6', 'C8', 'C9', 'C11', 'C13', 'C17',
    'C18', 'C19', 'C20', 'C22', 'C23', 'C25'
]


def my_model():
  internal_model = Sequential()
  internal_model.add(
      CategoryEncoding(
          num_tokens=NUM_TOTAL_FEATURES, output_mode='multi_hot', sparse=True))
  internal_model.add(
      Dense(1, input_dim=NUM_TOTAL_FEATURES, activation='sigmoid'))

  return internal_model


file_to_read = open(str(data_dir) + '/normalized_W_C14_bucket_cvxopt', 'rb')
W = np.array(pickle.load(file_to_read))

W = W + 0.0001 * np.eye(5)

print('W')
print(W)

file_to_read.close()
mean_arr = np.array([0, 0, 0, 0, 0])

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
      '-processed-allints_selected_cols_C15_C14_bucket_index_C7_offsets.csv')

  bags_file_to_read = (
      str(data_dir) + '/Split_' + str(split) + '/BagTrain_Split_' + str(split) +
      '-processed-allints_selected_cols_C15_C14_bucket_index_C7_offsets.ftr')

  result_file = (
      str(data_dir) + '/Split_' + str(split) + '/result_multi_genbags_' +
      str(split))

  df_test = pd.read_csv(test_file_to_read, usecols=list_of_cols_test)

  X_test = df_test.drop(['label'], axis=1)
  y_test = df_test['label']

  df_train = pd.read_feather(bags_file_to_read, columns=list_of_cols_train)

  # remove large and small bags
  df_train = df_train[(df_train['bag_size'] <= 2500)
                      & (df_train['bag_size'] >= 50)]

  list_of_bag_df = []

  list_of_sizes = []

  for i in range(num_bag_distns):
    df_train_i = df_train[df_train['C14_bucket_index'] == i]
    list_of_bag_df.append(df_train_i)
    list_of_sizes.append(len(df_train_i.index))

  min_df_size = min(list_of_sizes)

  num_steps = int(min_df_size / 8)

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

        for i in range(8):
          index = 8 * step + i

          wts = tf.convert_to_tensor(
              np.random.multivariate_normal(mean_arr, W, size=25),
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

            list_col = []
            for colname in feature_cols:
              list_col.append(row[colname].to_list()[0])

            n_array = np.array(list_col, dtype=np.int32)

            tf_n_array = tf.transpose(tf.convert_to_tensor(n_array))

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

    result_ell2_sq = model_ell2_sq.test_on_batch(
        x=X_test, y=y_test, reset_metrics=True, return_dict=True)

    print('result_ell2_sq')
    print(result_ell2_sq)
    print()

    result_ell1 = model_ell1.test_on_batch(
        x=X_test, y=y_test, reset_metrics=True, return_dict=True)

    print('result_ell1')
    print(result_ell1)
    print()

    result_single_bag = model_single_bag.test_on_batch(
        x=X_test, y=y_test, reset_metrics=True, return_dict=True)

    print('result_single_bag')
    print(result_single_bag)
    print()

    result_KLdiv = model_KLdiv.test_on_batch(
        x=X_test, y=y_test, reset_metrics=True, return_dict=True)

    print('result_KLdiv')
    print(result_KLdiv)
    print()

    results_string = (
        str(split) + ',' + str(epoch) + ',' + str(result_ell2_sq['auc']) + ',' +
        str(result_ell1['auc']) + ',' + str(result_single_bag['auc']) + ',' +
        str(result_KLdiv['auc']) + '\n')

    print('Test Results: ')
    print(results_string)

    with open(result_file, 'a') as filetoAppend:
      filetoAppend.write(results_string)
