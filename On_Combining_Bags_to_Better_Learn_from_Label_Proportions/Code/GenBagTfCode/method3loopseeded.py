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

"""Tensorflow Code for Generalized Bags Training for Scenario III."""
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

rng = np.random.default_rng(636823)

path_to_root_data_dir = (pathlib.Path(__file__).parent /
                         "../../Data/").resolve()

directory_full_datasets = str(path_to_root_data_dir) + "/FullDatasets/"

root_for_experiments = str(path_to_root_data_dir) + "/"

cluster_bags_method = 3

datasetName_list = ["Ionosphere", "Australian", "Heart"]

n_features_list = [35, 15, 14]

n_clusters = 3

NUM_LABELS = 1
META_BATCH_SIZE = 32
NUM_BATCHES = 1000

T1 = NUM_LABELS - 1
T2 = T1 + 1
T3 = T1 + NUM_LABELS
T4 = T3 + 1
T5 = T4 + 1

mean_arr = np.array([0, 0, 0, 0, 0, 0])

correlation_matrix = np.array([[0.13550136, 0., 0., -1.2195122, 0., 0.],
                               [0., 0.13550136, 0., 0., -1.2195122, 0.],
                               [0., 0., 0.13550136, 0., 0., -1.2195122],
                               [-1.2195122, 0., 0., 10.97560976, 0., 0.],
                               [0., -1.2195122, 0., 0., 10.97560976, 0.],
                               [0., 0., -1.2195122, 0., 0., 10.97560976]])

mse = tf.keras.losses.MeanSquaredError()

mae = tf.keras.losses.MeanAbsoluteError()

kld = tf.keras.losses.KLDivergence()


def custom_loss_ell2sq_genbag(y_true_extended, y_pred):
  """L_2^2 Generalized Bags Loss."""
  y_combined = tf.concat((y_pred, y_true_extended), axis=1)
  cumu_loss = 0
  loss = 0
  # pylint: disable=cell-var-from-loop
  for partition_num in range(META_BATCH_SIZE):
    y_combined_sliced = tf.gather(
        y_combined,
        tf.where(
            tf.equal(y_combined[:, T5],
                     tf.constant(partition_num, dtype=float)))[:, 0])
    y_combined_sliced_pred = tf.gather(
        y_combined_sliced, tf.range(T2), axis=1)
    y_combined_sliced_actual = tf.gather(
        y_combined_sliced, tf.range(T2, T4), axis=1)
    wts = tf.gather(y_combined_sliced, [T4], axis=1)
    wtd_y_combined_sliced_pred = y_combined_sliced_pred * wts
    wtd_y_combined_sliced_actual = y_combined_sliced_actual * wts
    wtd_sum_y_combined_sliced_pred = tf.reduce_sum(
        wtd_y_combined_sliced_pred, axis=0)
    wtd_sum_y_combined_sliced_actual = tf.reduce_sum(
        wtd_y_combined_sliced_actual, axis=0)
    loss = mse(
        y_true=wtd_sum_y_combined_sliced_actual,
        y_pred=wtd_sum_y_combined_sliced_pred)
    cumu_loss = cumu_loss + loss
    print()
  tf.print("Step loss: ", cumu_loss)
  return cumu_loss


def custom_loss_abs_genbag(y_true_extended, y_pred):
  """L_1 Generalized Bags Loss."""
  y_combined = tf.concat((y_pred, y_true_extended), axis=1)
  cumu_loss = 0
  loss = 0
  # pylint: disable=cell-var-from-loop
  for partition_num in range(META_BATCH_SIZE):
    y_combined_sliced = tf.gather(
        y_combined,
        tf.where(
            tf.equal(y_combined[:, T5],
                     tf.constant(partition_num, dtype=float)))[:, 0])
    y_combined_sliced_pred = tf.gather(
        y_combined_sliced, tf.range(T2), axis=1)
    y_combined_sliced_actual = tf.gather(
        y_combined_sliced, tf.range(T2, T4), axis=1)
    wts = tf.gather(y_combined_sliced, [T4], axis=1)
    wtd_y_combined_sliced_pred = y_combined_sliced_pred * wts
    wtd_y_combined_sliced_actual = y_combined_sliced_actual * wts
    wtd_sum_y_combined_sliced_pred = tf.reduce_sum(
        wtd_y_combined_sliced_pred, axis=0)
    wtd_sum_y_combined_sliced_actual = tf.reduce_sum(
        wtd_y_combined_sliced_actual, axis=0)
    loss = mae(
        y_true=wtd_sum_y_combined_sliced_actual,
        y_pred=wtd_sum_y_combined_sliced_pred)
    cumu_loss = cumu_loss + loss
    print()
  tf.print("Step loss: ", cumu_loss)
  return cumu_loss


def custom_loss_kld_genbag(y_true_extended, y_pred):
  """KL-div Bags Loss."""
  y_combined = tf.concat((y_pred, y_true_extended), axis=1)
  cumu_loss = 0
  # pylint: disable=cell-var-from-loop
  for partition_num in range(META_BATCH_SIZE):
    y_combined_sliced = tf.gather(
        y_combined,
        tf.where(
            tf.equal(y_combined[:, T5],
                     tf.constant(partition_num, dtype=float)))[:, 0])
    y_combined_sliced_pred = tf.gather(
        y_combined_sliced, tf.range(T2), axis=1)
    y_combined_sliced_actual = tf.gather(
        y_combined_sliced, tf.range(T2, T4), axis=1)
    wts = tf.gather(y_combined_sliced, [T4], axis=1)
    wtd_y_combined_sliced_pred = y_combined_sliced_pred * wts
    wtd_y_combined_sliced_actual = y_combined_sliced_actual * wts

    def thisbagloss():
      wtd_sum_y_combined_sliced_pred = tf.reduce_sum(
          wtd_y_combined_sliced_pred, axis=0)
      avg_sum_y_combined_sliced_pred = tf.divide(
          wtd_sum_y_combined_sliced_pred, tf.reduce_sum(wts))
      wtd_sum_y_combined_sliced_actual = tf.reduce_sum(
          wtd_y_combined_sliced_actual, axis=0)
      avg_sum_y_combined_sliced_actual = tf.divide(
          wtd_sum_y_combined_sliced_actual, tf.reduce_sum(wts))
      oneminus_avg_sum_y_combined_sliced_pred = (
          -1.0) * avg_sum_y_combined_sliced_pred + 1.0
      oneminus_avg_sum_y_combined_sliced_actual = (
          -1.0) * avg_sum_y_combined_sliced_actual + 1.0
      loss1 = kld(
          y_true=avg_sum_y_combined_sliced_actual,
          y_pred=avg_sum_y_combined_sliced_pred)
      loss2 = kld(
          y_true=oneminus_avg_sum_y_combined_sliced_actual,
          y_pred=oneminus_avg_sum_y_combined_sliced_pred)
      thisloss = loss1 + loss2
      return thisloss

    this_bag_loss = tf.cond(
        tf.reduce_sum(wts) > 0.0, thisbagloss, lambda: 0.0)
    cumu_loss = cumu_loss + this_bag_loss
    print()
  tf.print("Step loss: ", cumu_loss)
  return cumu_loss

binacc = m = tf.keras.metrics.BinaryAccuracy()


def custom_binacc_func():
  """Custom Binary Accuracy using slices."""
  def retfunc(y_true, y_pred):
    return binacc(tf.gather(y_true, [0], axis=1), y_pred)

  retfunc.__name__ = "BinaryAcc"
  return retfunc

custom_binacc = custom_binacc_func()

auc = tf.keras.metrics.AUC()


def custom_auc(y_true, y_pred):
  """Custom AUC using slices."""
  return auc(tf.gather(y_true, [0], axis=1), y_pred)


class Linear(keras.layers.Layer):
  """Linear Classifier."""

  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    self.w = self.add_weight(
        shape=(input_dim, units), initializer="zeros", trainable=True)
    self.b = self.add_weight(
        shape=(units,), initializer="zeros", trainable=True)

  def call(self, inputs):
    return tf.nn.sigmoid(tf.matmul(inputs, self.w) + self.b)

for datasetIndex, datasetName in enumerate(datasetName_list):

  NUM_FEATURES = n_features_list[datasetIndex]

  name_dir = root_for_experiments + datasetName + "/"

  cluster_bags_methodoutfile = name_dir + datasetName + "_TFexpOutputClusterBags_" + str(
      cluster_bags_method)

  for foldnumber in range(1, 6):

    folddir = name_dir + "Fold_" + str(foldnumber) + "/"

    for splitnumber in range(1, 6):

      splitdir = folddir + "Split_" + str(splitnumber) + "/"

      test_data_file = splitdir + datasetName + "_" + str(
          foldnumber) + "_" + str(splitnumber) + "-test.csv"

      train_data_file = splitdir + datasetName + "_" + str(
          foldnumber) + "_" + str(splitnumber) + "-train.csv"

      cluster_dir = splitdir + "ClusterBags_" + str(cluster_bags_method) + "/"

      random_seed = rng.integers(low=1000000, size=1)[0]
      numpy_seed = rng.integers(low=1000000, size=1)[0]
      tf_seed = rng.integers(low=1000000, size=1)[0]

      random.seed(random_seed)

      np.random.seed(numpy_seed)

      tf.random.set_seed(tf_seed)

      print()
      print("*********starting************")
      print(cluster_dir)
      print("random_seed = ", random_seed)
      print("numpy_seed = ", numpy_seed)
      print("tf_seed = ", tf_seed)

      # print(train_data_file)

      train_df = pd.read_csv(train_data_file)

      test_df = pd.read_csv(test_data_file)

      distn_Id_to_bags_list = []

      # first pick up the head
      for i in range(n_clusters):
        distn_Id_to_bags_list.append(
            pickle.load(open(cluster_dir + "head_bags_" + str(i), "rb")))

      # next the tail bags
      for i in range(n_clusters):
        distn_Id_to_bags_list.append(
            pickle.load(open(cluster_dir + "tail_bags_" + str(i), "rb")))

      print("distn_Id_to_bags_list")
      print(distn_Id_to_bags_list)

      print("correlation_matrix")
      print(correlation_matrix)

      def label_map(x):
        if x == -1:
          return 0
        else:
          return 1

      list_of_features = []

      for i in range(1, NUM_FEATURES):
        list_of_features.append("x." + str(i))

      list_of_features.append("constant")

      # pylint: disable=unnecessary-lambda
      train_df["label"] = train_df["label"].map(lambda x: label_map(x))

      # pylint: disable=unnecessary-lambda
      test_df["label"] = test_df["label"].map(lambda x: label_map(x))

      train_bag_counts = train_df["bag"].value_counts()
      myLinearModel_l2_S = keras.Sequential(
          [Linear(input_dim=NUM_FEATURES, units=NUM_LABELS)])

      myLinearModel_l2_S.compile(
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
          loss=custom_loss_ell2sq_genbag)

      myLinearModel_l2_R = keras.Sequential(
          [Linear(input_dim=NUM_FEATURES, units=NUM_LABELS)])

      myLinearModel_l2_R.compile(
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
          loss=custom_loss_ell2sq_genbag)

      myLinearModel_l1_S = keras.Sequential(
          [Linear(input_dim=NUM_FEATURES, units=NUM_LABELS)])

      myLinearModel_l1_S.compile(
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
          loss=custom_loss_abs_genbag)

      myLinearModel_l1_R = keras.Sequential(
          [Linear(input_dim=NUM_FEATURES, units=NUM_LABELS)])

      myLinearModel_l1_R.compile(
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
          loss=custom_loss_abs_genbag)

      myLinearModel_l2_U = keras.Sequential(
          [Linear(input_dim=NUM_FEATURES, units=NUM_LABELS)])

      myLinearModel_l2_U.compile(
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
          loss=custom_loss_ell2sq_genbag)

      myLinearModel_l1_U = keras.Sequential(
          [Linear(input_dim=NUM_FEATURES, units=NUM_LABELS)])

      myLinearModel_l1_U.compile(
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
          loss=custom_loss_abs_genbag)

      myLinearModel_KL_U = keras.Sequential(
          [Linear(input_dim=NUM_FEATURES, units=NUM_LABELS)])

      myLinearModel_KL_U.compile(
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
          loss=custom_loss_kld_genbag)

      for step in range(NUM_BATCHES):
        print("Enter Batch: ", step)
        batch_df = pd.DataFrame()
        batch_rand_df = pd.DataFrame()
        batch_unif_df = pd.DataFrame()
        for genBagNo in range(META_BATCH_SIZE):
          sampled_wts = np.random.multivariate_normal(mean_arr,
                                                      correlation_matrix)
          random_wts = np.random.normal(size=2 * n_clusters)
          unifbag_wts = np.eye(2 * n_clusters)[random.randrange(2 * n_clusters)]
          genBag_df = pd.DataFrame()
          genBag_rand_df = pd.DataFrame()
          genBag_unif_df = pd.DataFrame()
          for i in range(2 * n_clusters):
            this_bag_list = random.sample(distn_Id_to_bags_list[i], 1)
            this_bag_df = train_df.iloc[this_bag_list[0]].copy()
            this_bag_rand_df = train_df.iloc[this_bag_list[0]].copy()
            this_bag_unif_df = train_df.iloc[this_bag_list[0]].copy()
            this_bag_df["wt"] = sampled_wts[i]
            this_bag_rand_df["wt"] = random_wts[i]
            this_bag_unif_df["wt"] = unifbag_wts[i]
            genBag_df = genBag_df.append(this_bag_df, ignore_index=True)
            genBag_rand_df = genBag_rand_df.append(
                this_bag_rand_df, ignore_index=True)
            genBag_unif_df = genBag_unif_df.append(
                this_bag_unif_df, ignore_index=True)

          genBag_df["genBagNo"] = genBagNo
          genBag_rand_df["genBagNo"] = genBagNo
          genBag_unif_df["genBagNo"] = genBagNo

          batch_df = batch_df.append(genBag_df, ignore_index=True)

          batch_rand_df = batch_rand_df.append(
              genBag_rand_df, ignore_index=True)

          batch_unif_df = batch_unif_df.append(
              genBag_unif_df, ignore_index=True)

        features_matrix = batch_df[list_of_features].to_numpy()

        features_matrix_rand = batch_rand_df[list_of_features].to_numpy()

        features_matrix_unif = batch_unif_df[list_of_features].to_numpy()

        label_wt_genBagno_matrix = batch_df[["label", "wt",
                                             "genBagNo"]].to_numpy(dtype=float)

        label_wt_genBagno_matrix_rand = batch_rand_df[[
            "label", "wt", "genBagNo"
        ]].to_numpy(dtype=float)

        label_wt_genBagno_matrix_unif = batch_unif_df[[
            "label", "wt", "genBagNo"
        ]].to_numpy(dtype=float)

        myLinearModel_l2_S.train_on_batch(
            x=features_matrix, y=label_wt_genBagno_matrix)

        myLinearModel_l1_S.train_on_batch(
            x=features_matrix, y=label_wt_genBagno_matrix)

        myLinearModel_l2_R.train_on_batch(
            x=features_matrix_rand, y=label_wt_genBagno_matrix_rand)

        myLinearModel_l1_R.train_on_batch(
            x=features_matrix_rand, y=label_wt_genBagno_matrix_rand)

        myLinearModel_l2_U.train_on_batch(
            x=features_matrix_unif, y=label_wt_genBagno_matrix_unif)

        myLinearModel_l1_U.train_on_batch(
            x=features_matrix_unif, y=label_wt_genBagno_matrix_unif)

        myLinearModel_KL_U.train_on_batch(
            x=features_matrix_unif, y=label_wt_genBagno_matrix_unif)

        print("After training step: ", step)

      print("Testing on Full Dataset")

      myLinearModel_l2_S.compile(metrics=[custom_auc])

      myLinearModel_l1_S.compile(metrics=[custom_auc])

      myLinearModel_l2_R.compile(metrics=[custom_auc])

      myLinearModel_l1_R.compile(metrics=[custom_auc])

      myLinearModel_l2_U.compile(metrics=[custom_auc])

      myLinearModel_l1_U.compile(metrics=[custom_auc])

      myLinearModel_KL_U.compile(metrics=[custom_auc])

      test_df["wt"] = 0

      test_df["genBagNo"] = 0

      features_matrix_test = test_df[list_of_features].to_numpy()

      label_wt_genBagno_matrix_test = test_df[["label", "wt", "genBagNo"
                                              ]].to_numpy(dtype=float)

      results_linear_l2_S = myLinearModel_l2_S.test_on_batch(
          x=features_matrix_test, y=label_wt_genBagno_matrix_test)

      results_linear_l1_S = myLinearModel_l1_S.test_on_batch(
          x=features_matrix_test, y=label_wt_genBagno_matrix_test)

      results_linear_l2_R = myLinearModel_l2_R.test_on_batch(
          x=features_matrix_test, y=label_wt_genBagno_matrix_test)

      results_linear_l1_R = myLinearModel_l1_R.test_on_batch(
          x=features_matrix_test, y=label_wt_genBagno_matrix_test)

      results_linear_l2_U = myLinearModel_l2_U.test_on_batch(
          x=features_matrix_test, y=label_wt_genBagno_matrix_test)

      results_linear_l1_U = myLinearModel_l1_U.test_on_batch(
          x=features_matrix_test, y=label_wt_genBagno_matrix_test)

      results_linear_KL_U = myLinearModel_KL_U.test_on_batch(
          x=features_matrix_test, y=label_wt_genBagno_matrix_test)

      results_string = (
          datasetName + ", " + str(cluster_bags_method) + ", " +
          str(splitnumber) + ",  " + str(foldnumber) + ", " +
          str(results_linear_l2_S[1]) + ", " + str(results_linear_l1_S[1]) +
          ", " + str(results_linear_l2_R[1]) + ", " +
          str(results_linear_l1_R[1]) + ", " + str(results_linear_l2_U[1]) +
          ", " + str(results_linear_l1_U[1]) + ", " +
          str(results_linear_KL_U[1]) + "\n")

      print("Test Results: ")
      print(results_string)

      with open(cluster_bags_methodoutfile, "a") as filetoAppend:
        filetoAppend.write(results_string)
