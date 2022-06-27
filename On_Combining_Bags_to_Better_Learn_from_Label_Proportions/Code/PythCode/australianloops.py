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

"""LMMCM Loops for Australian Dataset."""
from pathlib import Path  # pylint: disable=g-importing-member
import random
from models.kernel_model import *  # pylint: disable=wildcard-import
import pandas as pd
import utils.bag_utils
import utils.data_utils
from utils.model_utils import *  # pylint: disable=wildcard-import

rng = np.random.default_rng(74682303)  # pylint: disable=undefined-variable

path_to_root_data_dir = (Path(__file__).parent / "../../Data/").resolve()

root_for_experiments = str(path_to_root_data_dir) + "/"

dataset_name = "Australian"

NUM_FEATURES = 15

name_dir = root_for_experiments + dataset_name + "/"

regularizer_param = 1e0

regularizer_param_string = "1e0"


class Helperdict(dict):
  """Helperdict class."""

  def __key(self):
    return tuple((m, self[m]) for m in sorted(self))

  def __hash__(self):
    return hash(self.__key())

  def __eq__(self, other):
    return self.__key() == other.__key()  # pylint: disable=protected-access


for cluster_bags_method in range(1, 9):

  if cluster_bags_method > 5:
    cluster_bags_method_to_use = cluster_bags_method - 1
  else:
    cluster_bags_method_to_use = cluster_bags_method

  cluster_bags_methodoutfile = name_dir + dataset_name + "_" + regularizer_param_string + "_MutConOutputClusterBags_" + str(
      cluster_bags_method_to_use)

  for foldnumber in range(1, 6):
    folddir = name_dir + "Fold_" + str(foldnumber) + "/"

    for splitnumber in range(1, 6):

      splitdir = folddir + "Split_" + str(splitnumber) + "/"

      testfile = splitdir + dataset_name + "_" + str(foldnumber) + "_" + str(
          splitnumber) + "-test.csv"

      cluster_dir = splitdir + "ClusterBags_" + str(
          cluster_bags_method_to_use) + "/"

      trainfile = cluster_dir + "full_train.csv"

      random_seed = rng.integers(low=1000000, size=1)[0]
      numpy_seed = rng.integers(low=1000000, size=1)[0]

      if cluster_bags_method == 5:
        continue

      random.seed(random_seed)

      np.random.seed(numpy_seed)  # pylint: disable=undefined-variable

      print()
      print("*********starting************")
      print(dataset_name + "_" + regularizer_param_string + "_Fold_" +
            str(foldnumber) + "_Split_" + str(splitnumber))
      print("random_seed = ", random_seed)
      print("numpy_seed = ", numpy_seed)

      list_of_features = []

      for i in range(1, NUM_FEATURES):
        list_of_features.append("x." + str(i))

      list_of_features.append("constant")

      full_train_df = pd.read_csv(trainfile)

      test_df = pd.read_csv(testfile)

      # print('experiment starts on %s' % data_path)
      # experiment_start = time.time()

      num_bags = len(pd.unique(full_train_df["bag"]))

      if num_bags % 2 == 1:
        bag_to_drop = random.choice(pd.unique(full_train_df["bag"]))
        full_train_df = full_train_df[
            full_train_df["bag"] != bag_to_drop].reset_index()

      train_X = full_train_df[list_of_features].copy()
      train_y = full_train_df["label"].copy()
      test_X = test_df[list_of_features].copy()
      test_y = test_df["label"].copy()
      bag_id = full_train_df["bag"].to_numpy()

      cont_col = list_of_features

      prop_dict, size_dict = utils.bag_utils.compute_label_proportion(train_y,
                                                                      bag_id)

      print("train_X")
      print(train_X.head(4).to_string())
      print(train_X.tail(4).to_string())

      print()
      print("train_y")
      print(train_y.head(4).to_string())
      print(train_y.tail(4).to_string())

      print()
      print("test_X")
      print(test_X.head(4).to_string())
      print(test_X.tail(4).to_string())

      print()
      print("test_y")
      print(test_y.head(4).to_string())
      print(test_y.tail(4).to_string())

      print()
      print("bag_id")
      print(bag_id)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("prop_dict")
      print(prop_dict)

      print()
      print("size_dict")
      print(size_dict)

      # exit()

      # set up parameters

      KERNEL_PARAMS_direct_fixed = {
          "kernel": "rbf",
          "gamma": "scale",
          "loss": LogisticLoss  # pylint: disable=undefined-variable
      }

      options_direct_fixed = ("L-BFGS-B",
                              Helperdict({
                                  "ftol": 1e-5,
                                  "maxiter": 100,
                                  "maxcor": 80
                              }))

      TRAIN_PARAMS_direct_fixed = {
          "method": options_direct_fixed,
          "exclusion_param": 0
      }

      # initialize the model
      model = KernelizedMethod(  # pylint: disable=undefined-variable
          **KERNEL_PARAMS_direct_fixed, regularizer=regularizer_param)

      train_X_engineered, test_X_engineered = utils.data_utils.feature_engineering_cont(
          train_X, test_X, cont_col)

      print()
      print("train_X_engineered")
      print(train_X_engineered)
      print(train_X_engineered.shape)

      print()
      print("test_X_engineered")
      print(test_X_engineered)
      print(test_X_engineered.shape)

      # exit()

      model.fit(train_X_engineered, test_X_engineered, test_y, bag_id,
                prop_dict, size_dict, **TRAIN_PARAMS_direct_fixed)

      pred = model.predict(test_X_engineered)

      area, fprs, tprs, thresholds = model.get_roc(test_X_engineered, test_y)

      print("auc = ", area)
      results_string = (
          dataset_name + "_" + regularizer_param_string + ", " +
          str(cluster_bags_method_to_use) + ", " + str(splitnumber) + ",  " +
          str(foldnumber) + ", " + str(area) + "\n")

      with open(cluster_bags_methodoutfile, "a") as fileto_append:
        fileto_append.write(results_string)

rng = np.random.default_rng(82525478)  # pylint: disable=undefined-variable

regularizer_param = 1e-1

regularizer_param_string = "1e-1"

for cluster_bags_method in range(1, 9):

  if cluster_bags_method > 5:
    cluster_bags_method_to_use = cluster_bags_method - 1
  else:
    cluster_bags_method_to_use = cluster_bags_method

  cluster_bags_methodoutfile = name_dir + dataset_name + "_" + regularizer_param_string + "_MutConOutputClusterBags_" + str(
      cluster_bags_method_to_use)

  for foldnumber in range(1, 6):
    folddir = name_dir + "Fold_" + str(foldnumber) + "/"

    for splitnumber in range(1, 6):

      splitdir = folddir + "Split_" + str(splitnumber) + "/"

      testfile = splitdir + dataset_name + "_" + str(foldnumber) + "_" + str(
          splitnumber) + "-test.csv"

      cluster_dir = splitdir + "ClusterBags_" + str(
          cluster_bags_method_to_use) + "/"

      trainfile = cluster_dir + "full_train.csv"

      random_seed = rng.integers(low=1000000, size=1)[0]
      numpy_seed = rng.integers(low=1000000, size=1)[0]

      if cluster_bags_method == 5:
        continue

      random.seed(random_seed)

      np.random.seed(numpy_seed)  # pylint: disable=undefined-variable

      print()
      print("*********starting************")
      print(dataset_name + "_" + regularizer_param_string + "_Fold_" +
            str(foldnumber) + "_Split_" + str(splitnumber))
      print("random_seed = ", random_seed)
      print("numpy_seed = ", numpy_seed)

      list_of_features = []

      for i in range(1, NUM_FEATURES):
        list_of_features.append("x." + str(i))

      list_of_features.append("constant")

      full_train_df = pd.read_csv(trainfile)

      test_df = pd.read_csv(testfile)

      # print('experiment starts on %s' % data_path)
      # experiment_start = time.time()

      num_bags = len(pd.unique(full_train_df["bag"]))

      if num_bags % 2 == 1:
        bag_to_drop = random.choice(pd.unique(full_train_df["bag"]))
        full_train_df = full_train_df[
            full_train_df["bag"] != bag_to_drop].reset_index()

      train_X = full_train_df[list_of_features].copy()
      train_y = full_train_df["label"].copy()
      test_X = test_df[list_of_features].copy()
      test_y = test_df["label"].copy()
      bag_id = full_train_df["bag"].to_numpy()

      cont_col = list_of_features

      prop_dict, size_dict = utils.bag_utils.compute_label_proportion(train_y,
                                                                      bag_id)

      print("train_X")
      print(train_X.head(4).to_string())
      print(train_X.tail(4).to_string())

      print()
      print("train_y")
      print(train_y.head(4).to_string())
      print(train_y.tail(4).to_string())

      print()
      print("test_X")
      print(test_X.head(4).to_string())
      print(test_X.tail(4).to_string())

      print()
      print("test_y")
      print(test_y.head(4).to_string())
      print(test_y.tail(4).to_string())

      print()
      print("bag_id")
      print(bag_id)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("prop_dict")
      print(prop_dict)

      print()
      print("size_dict")
      print(size_dict)

      # exit()

      # set up parameters

      KERNEL_PARAMS_direct_fixed = {
          "kernel": "rbf",
          "gamma": "scale",
          "loss": LogisticLoss  # pylint: disable=undefined-variable
      }

      options_direct_fixed = ("L-BFGS-B",
                              Helperdict({
                                  "ftol": 1e-5,
                                  "maxiter": 100,
                                  "maxcor": 80
                              }))

      TRAIN_PARAMS_direct_fixed = {
          "method": options_direct_fixed,
          "exclusion_param": 0
      }

      # initialize the model
      model = KernelizedMethod(  # pylint: disable=undefined-variable
          **KERNEL_PARAMS_direct_fixed, regularizer=regularizer_param)

      train_X_engineered, test_X_engineered = utils.data_utils.feature_engineering_cont(
          train_X, test_X, cont_col)

      print()
      print("train_X_engineered")
      print(train_X_engineered)
      print(train_X_engineered.shape)

      print()
      print("test_X_engineered")
      print(test_X_engineered)
      print(test_X_engineered.shape)

      # exit()

      model.fit(train_X_engineered, test_X_engineered, test_y, bag_id,
                prop_dict, size_dict, **TRAIN_PARAMS_direct_fixed)

      pred = model.predict(test_X_engineered)

      area, fprs, tprs, thresholds = model.get_roc(test_X_engineered, test_y)

      print("auc = ", area)
      results_string = (
          dataset_name + "_" + regularizer_param_string + ", " +
          str(cluster_bags_method_to_use) + ", " + str(splitnumber) + ",  " +
          str(foldnumber) + ", " + str(area) + "\n")

      with open(cluster_bags_methodoutfile, "a") as fileto_append:
        fileto_append.write(results_string)

rng = np.random.default_rng(937271326)  # pylint: disable=undefined-variable

regularizer_param = 1e-2

regularizer_param_string = "1e-2"

for cluster_bags_method in range(1, 9):

  if cluster_bags_method > 5:
    cluster_bags_method_to_use = cluster_bags_method - 1
  else:
    cluster_bags_method_to_use = cluster_bags_method

  cluster_bags_methodoutfile = name_dir + dataset_name + "_" + regularizer_param_string + "_MutConOutputClusterBags_" + str(
      cluster_bags_method_to_use)

  for foldnumber in range(1, 6):
    folddir = name_dir + "Fold_" + str(foldnumber) + "/"

    for splitnumber in range(1, 6):

      splitdir = folddir + "Split_" + str(splitnumber) + "/"

      testfile = splitdir + dataset_name + "_" + str(foldnumber) + "_" + str(
          splitnumber) + "-test.csv"

      cluster_dir = splitdir + "ClusterBags_" + str(
          cluster_bags_method_to_use) + "/"

      trainfile = cluster_dir + "full_train.csv"

      random_seed = rng.integers(low=1000000, size=1)[0]
      numpy_seed = rng.integers(low=1000000, size=1)[0]

      if cluster_bags_method == 5:
        continue

      random.seed(random_seed)

      np.random.seed(numpy_seed)  # pylint: disable=undefined-variable

      print()
      print("*********starting************")
      print(dataset_name + "_" + regularizer_param_string + "_Fold_" +
            str(foldnumber) + "_Split_" + str(splitnumber))
      print("random_seed = ", random_seed)
      print("numpy_seed = ", numpy_seed)

      list_of_features = []

      for i in range(1, NUM_FEATURES):
        list_of_features.append("x." + str(i))

      list_of_features.append("constant")

      full_train_df = pd.read_csv(trainfile)

      test_df = pd.read_csv(testfile)

      # print('experiment starts on %s' % data_path)
      # experiment_start = time.time()

      num_bags = len(pd.unique(full_train_df["bag"]))

      if num_bags % 2 == 1:
        bag_to_drop = random.choice(pd.unique(full_train_df["bag"]))
        full_train_df = full_train_df[
            full_train_df["bag"] != bag_to_drop].reset_index()

      train_X = full_train_df[list_of_features].copy()
      train_y = full_train_df["label"].copy()
      test_X = test_df[list_of_features].copy()
      test_y = test_df["label"].copy()
      bag_id = full_train_df["bag"].to_numpy()

      cont_col = list_of_features

      prop_dict, size_dict = utils.bag_utils.compute_label_proportion(train_y,
                                                                      bag_id)

      print("train_X")
      print(train_X.head(4).to_string())
      print(train_X.tail(4).to_string())

      print()
      print("train_y")
      print(train_y.head(4).to_string())
      print(train_y.tail(4).to_string())

      print()
      print("test_X")
      print(test_X.head(4).to_string())
      print(test_X.tail(4).to_string())

      print()
      print("test_y")
      print(test_y.head(4).to_string())
      print(test_y.tail(4).to_string())

      print()
      print("bag_id")
      print(bag_id)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("prop_dict")
      print(prop_dict)

      print()
      print("size_dict")
      print(size_dict)

      # exit()

      # set up parameters

      KERNEL_PARAMS_direct_fixed = {
          "kernel": "rbf",
          "gamma": "scale",
          "loss": LogisticLoss  # pylint: disable=undefined-variable
      }

      options_direct_fixed = ("L-BFGS-B",
                              Helperdict({
                                  "ftol": 1e-5,
                                  "maxiter": 100,
                                  "maxcor": 80
                              }))

      TRAIN_PARAMS_direct_fixed = {
          "method": options_direct_fixed,
          "exclusion_param": 0
      }

      # initialize the model
      model = KernelizedMethod(  # pylint: disable=undefined-variable
          **KERNEL_PARAMS_direct_fixed, regularizer=regularizer_param)

      train_X_engineered, test_X_engineered = utils.data_utils.feature_engineering_cont(
          train_X, test_X, cont_col)

      print()
      print("train_X_engineered")
      print(train_X_engineered)
      print(train_X_engineered.shape)

      print()
      print("test_X_engineered")
      print(test_X_engineered)
      print(test_X_engineered.shape)

      # exit()

      model.fit(train_X_engineered, test_X_engineered, test_y, bag_id,
                prop_dict, size_dict, **TRAIN_PARAMS_direct_fixed)

      pred = model.predict(test_X_engineered)

      area, fprs, tprs, thresholds = model.get_roc(test_X_engineered, test_y)

      print("auc = ", area)
      results_string = (
          dataset_name + "_" + regularizer_param_string + ", " +
          str(cluster_bags_method_to_use) + ", " + str(splitnumber) + ",  " +
          str(foldnumber) + ", " + str(area) + "\n")

      with open(cluster_bags_methodoutfile, "a") as fileto_append:
        fileto_append.write(results_string)

rng = np.random.default_rng(1093832)  # pylint: disable=undefined-variable

regularizer_param = 1e-3

regularizer_param_string = "1e-3"

for cluster_bags_method in range(1, 9):

  if cluster_bags_method > 5:
    cluster_bags_method_to_use = cluster_bags_method - 1
  else:
    cluster_bags_method_to_use = cluster_bags_method

  cluster_bags_methodoutfile = name_dir + dataset_name + "_" + regularizer_param_string + "_MutConOutputClusterBags_" + str(
      cluster_bags_method_to_use)

  for foldnumber in range(1, 6):
    folddir = name_dir + "Fold_" + str(foldnumber) + "/"

    for splitnumber in range(1, 6):

      splitdir = folddir + "Split_" + str(splitnumber) + "/"

      testfile = splitdir + dataset_name + "_" + str(foldnumber) + "_" + str(
          splitnumber) + "-test.csv"

      cluster_dir = splitdir + "ClusterBags_" + str(
          cluster_bags_method_to_use) + "/"

      trainfile = cluster_dir + "full_train.csv"

      random_seed = rng.integers(low=1000000, size=1)[0]
      numpy_seed = rng.integers(low=1000000, size=1)[0]

      if cluster_bags_method == 5:
        continue

      random.seed(random_seed)

      np.random.seed(numpy_seed)  # pylint: disable=undefined-variable

      print()
      print("*********starting************")
      print(dataset_name + "_" + regularizer_param_string + "_Fold_" +
            str(foldnumber) + "_Split_" + str(splitnumber))
      print("random_seed = ", random_seed)
      print("numpy_seed = ", numpy_seed)

      list_of_features = []

      for i in range(1, NUM_FEATURES):
        list_of_features.append("x." + str(i))

      list_of_features.append("constant")

      full_train_df = pd.read_csv(trainfile)

      test_df = pd.read_csv(testfile)

      # print('experiment starts on %s' % data_path)
      # experiment_start = time.time()

      num_bags = len(pd.unique(full_train_df["bag"]))

      if num_bags % 2 == 1:
        bag_to_drop = random.choice(pd.unique(full_train_df["bag"]))
        full_train_df = full_train_df[
            full_train_df["bag"] != bag_to_drop].reset_index()

      train_X = full_train_df[list_of_features].copy()
      train_y = full_train_df["label"].copy()
      test_X = test_df[list_of_features].copy()
      test_y = test_df["label"].copy()
      bag_id = full_train_df["bag"].to_numpy()

      cont_col = list_of_features

      prop_dict, size_dict = utils.bag_utils.compute_label_proportion(train_y,
                                                                      bag_id)

      print("train_X")
      print(train_X.head(4).to_string())
      print(train_X.tail(4).to_string())

      print()
      print("train_y")
      print(train_y.head(4).to_string())
      print(train_y.tail(4).to_string())

      print()
      print("test_X")
      print(test_X.head(4).to_string())
      print(test_X.tail(4).to_string())

      print()
      print("test_y")
      print(test_y.head(4).to_string())
      print(test_y.tail(4).to_string())

      print()
      print("bag_id")
      print(bag_id)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("prop_dict")
      print(prop_dict)

      print()
      print("size_dict")
      print(size_dict)

      # exit()

      # set up parameters

      KERNEL_PARAMS_direct_fixed = {
          "kernel": "rbf",
          "gamma": "scale",
          "loss": LogisticLoss  # pylint: disable=undefined-variable
      }

      options_direct_fixed = ("L-BFGS-B",
                              Helperdict({
                                  "ftol": 1e-5,
                                  "maxiter": 100,
                                  "maxcor": 80
                              }))

      TRAIN_PARAMS_direct_fixed = {
          "method": options_direct_fixed,
          "exclusion_param": 0
      }

      # initialize the model
      model = KernelizedMethod(  # pylint: disable=undefined-variable
          **KERNEL_PARAMS_direct_fixed, regularizer=regularizer_param)

      train_X_engineered, test_X_engineered = utils.data_utils.feature_engineering_cont(
          train_X, test_X, cont_col)

      print()
      print("train_X_engineered")
      print(train_X_engineered)
      print(train_X_engineered.shape)

      print()
      print("test_X_engineered")
      print(test_X_engineered)
      print(test_X_engineered.shape)

      # exit()

      model.fit(train_X_engineered, test_X_engineered, test_y, bag_id,
                prop_dict, size_dict, **TRAIN_PARAMS_direct_fixed)

      pred = model.predict(test_X_engineered)

      area, fprs, tprs, thresholds = model.get_roc(test_X_engineered, test_y)

      print("auc = ", area)
      results_string = (
          dataset_name + "_" + regularizer_param_string + ", " +
          str(cluster_bags_method_to_use) + ", " + str(splitnumber) + ",  " +
          str(foldnumber) + ", " + str(area) + "\n")

      with open(cluster_bags_methodoutfile, "a") as fileto_append:
        fileto_append.write(results_string)

rng = np.random.default_rng(2895347)  # pylint: disable=undefined-variable

regularizer_param = 1e-4

regularizer_param_string = "1e-4"

for cluster_bags_method in range(1, 9):

  if cluster_bags_method > 5:
    cluster_bags_method_to_use = cluster_bags_method - 1
  else:
    cluster_bags_method_to_use = cluster_bags_method

  cluster_bags_methodoutfile = name_dir + dataset_name + "_" + regularizer_param_string + "_MutConOutputClusterBags_" + str(
      cluster_bags_method_to_use)

  for foldnumber in range(1, 6):
    folddir = name_dir + "Fold_" + str(foldnumber) + "/"

    for splitnumber in range(1, 6):

      splitdir = folddir + "Split_" + str(splitnumber) + "/"

      testfile = splitdir + dataset_name + "_" + str(foldnumber) + "_" + str(
          splitnumber) + "-test.csv"

      cluster_dir = splitdir + "ClusterBags_" + str(
          cluster_bags_method_to_use) + "/"

      trainfile = cluster_dir + "full_train.csv"

      random_seed = rng.integers(low=1000000, size=1)[0]
      numpy_seed = rng.integers(low=1000000, size=1)[0]

      if cluster_bags_method == 5:
        continue

      random.seed(random_seed)

      np.random.seed(numpy_seed)  # pylint: disable=undefined-variable

      print()
      print("*********starting************")
      print(dataset_name + "_" + regularizer_param_string + "_Fold_" +
            str(foldnumber) + "_Split_" + str(splitnumber))
      print("random_seed = ", random_seed)
      print("numpy_seed = ", numpy_seed)

      list_of_features = []

      for i in range(1, NUM_FEATURES):
        list_of_features.append("x." + str(i))

      list_of_features.append("constant")

      full_train_df = pd.read_csv(trainfile)

      test_df = pd.read_csv(testfile)

      # print('experiment starts on %s' % data_path)
      # experiment_start = time.time()

      num_bags = len(pd.unique(full_train_df["bag"]))

      if num_bags % 2 == 1:
        bag_to_drop = random.choice(pd.unique(full_train_df["bag"]))
        full_train_df = full_train_df[
            full_train_df["bag"] != bag_to_drop].reset_index()

      train_X = full_train_df[list_of_features].copy()
      train_y = full_train_df["label"].copy()
      test_X = test_df[list_of_features].copy()
      test_y = test_df["label"].copy()
      bag_id = full_train_df["bag"].to_numpy()

      cont_col = list_of_features

      prop_dict, size_dict = utils.bag_utils.compute_label_proportion(train_y,
                                                                      bag_id)

      print("train_X")
      print(train_X.head(4).to_string())
      print(train_X.tail(4).to_string())

      print()
      print("train_y")
      print(train_y.head(4).to_string())
      print(train_y.tail(4).to_string())

      print()
      print("test_X")
      print(test_X.head(4).to_string())
      print(test_X.tail(4).to_string())

      print()
      print("test_y")
      print(test_y.head(4).to_string())
      print(test_y.tail(4).to_string())

      print()
      print("bag_id")
      print(bag_id)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("cont_col")
      print(cont_col)

      print()
      print("prop_dict")
      print(prop_dict)

      print()
      print("size_dict")
      print(size_dict)

      # exit()

      # set up parameters

      KERNEL_PARAMS_direct_fixed = {
          "kernel": "rbf",
          "gamma": "scale",
          "loss": LogisticLoss  # pylint: disable=undefined-variable
      }

      options_direct_fixed = ("L-BFGS-B",
                              Helperdict({
                                  "ftol": 1e-5,
                                  "maxiter": 100,
                                  "maxcor": 80
                              }))

      TRAIN_PARAMS_direct_fixed = {
          "method": options_direct_fixed,
          "exclusion_param": 0
      }

      # initialize the model
      model = KernelizedMethod(  # pylint: disable=undefined-variable
          **KERNEL_PARAMS_direct_fixed, regularizer=regularizer_param)

      train_X_engineered, test_X_engineered = utils.data_utils.feature_engineering_cont(
          train_X, test_X, cont_col)

      print()
      print("train_X_engineered")
      print(train_X_engineered)
      print(train_X_engineered.shape)

      print()
      print("test_X_engineered")
      print(test_X_engineered)
      print(test_X_engineered.shape)

      # exit()

      model.fit(train_X_engineered, test_X_engineered, test_y, bag_id,
                prop_dict, size_dict, **TRAIN_PARAMS_direct_fixed)

      pred = model.predict(test_X_engineered)

      area, fprs, tprs, thresholds = model.get_roc(test_X_engineered, test_y)

      print("auc = ", area)
      results_string = (
          dataset_name + "_" + regularizer_param_string + ", " +
          str(cluster_bags_method_to_use) + ", " + str(splitnumber) + ",  " +
          str(foldnumber) + ", " + str(area) + "\n")

      with open(cluster_bags_methodoutfile, "a") as fileto_append:
        fileto_append.write(results_string)


