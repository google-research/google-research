# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python3
"""Loads Adult dataset, processes it and saves it to output folder.

Adapted from code provided by Heinrich Jiang (heinrichj@google.com). The Adult
train and test data files can be downloaded from:
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import pandas as pd
from sklearn import model_selection


flags.DEFINE_float("vali_fraction", 1.0 / 3.0,
                   "Fraction of train set that constitutes the validation set.")
flags.DEFINE_string("train_file", "adult.data",
                    "Path to Adult train data file.")
flags.DEFINE_string("test_file", "adult.test",
                    "Path to Adult test data file.")
flags.DEFINE_string("output_directory", "datasets/",
                    "Path to store processed dataset.")


FLAGS = flags.FLAGS


def bucketize_continuous_column(
    input_train_df, input_test_df, continuous_column_name, num_quantiles=None,
    bins=None):
  """Bucketize continuous columns using either bin cut-points or quantiles."""
  assert (num_quantiles is None or bins is None)
  if num_quantiles is not None:
    # Compute quantile cut-points and bucketize.
    _, bins_quantized = pd.qcut(
        input_train_df[continuous_column_name], num_quantiles, retbins=True,
        labels=False)
    input_train_df[continuous_column_name] = pd.cut(
        input_train_df[continuous_column_name], bins_quantized, labels=False)
    input_test_df[continuous_column_name] = pd.cut(
        input_test_df[continuous_column_name], bins_quantized, labels=False)
  elif bins is not None:
    # Bucketize using provided bin cut-points.
    input_train_df[continuous_column_name] = pd.cut(
        input_train_df[continuous_column_name], bins, labels=False)
    input_test_df[continuous_column_name] = pd.cut(
        input_test_df[continuous_column_name], bins, labels=False)


def binarize_categorical_columns(
    input_train_df, input_test_df, categorical_columns):
  """Function to converting categorical features to one-hot encodings."""

  # Binarize categorical columns.
  binarized_train_df = pd.get_dummies(
      input_train_df, columns=categorical_columns)
  binarized_test_df = pd.get_dummies(
      input_test_df, columns=categorical_columns)

  # Make sure the train and test dataframes have the same binarized columns.
  # Identify columns in train set not in test set and fill them in test set.
  test_df_missing_cols = set(binarized_train_df.columns) - set(
      binarized_test_df.columns)
  for c in test_df_missing_cols:
    binarized_test_df[c] = 0
  # Identify columns in test set not in train set and fill them in train set.
  train_df_missing_cols = set(binarized_test_df.columns) - set(
      binarized_train_df.columns)
  for c in train_df_missing_cols:
    binarized_train_df[c] = 0
  # Just to be sure that both train and test df"s have same columns.
  binarized_train_df = binarized_train_df[binarized_test_df.columns]

  return binarized_train_df, binarized_test_df


def load_data():
  """Load and process train and test datasets at provided data paths."""
  columns = [
      "age", "workclass", "fnlwgt", "education", "education_num",
      "marital_status", "occupation", "relationship", "race", "gender",
      "capital_gain", "capital_loss", "hours_per_week", "native_country",
      "income_bracket"]
  train_df_raw = pd.read_csv(
      FLAGS.train_file, names=columns, skipinitialspace=True)
  test_df_raw = pd.read_csv(
      FLAGS.test_file, names=columns, skipinitialspace=True, skiprows=1)

  # Identify relevant continuous and categorical features.
  continuous_features = [
      "age", "capital_gain", "capital_loss", "hours_per_week", "education_num"]
  categorical_features = [
      "workclass", "education", "marital_status", "occupation", "relationship",
      "race", "gender", "native_country"]

  # Add label column: income > 50K.
  train_df_raw["label"] = (
      train_df_raw["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
  test_df_raw["label"] = (
      test_df_raw["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

  # Disable pandas warning.
  pd.options.mode.chained_assignment = None  # default="warn"

  # Filter out all features except the ones specified.
  train_df = train_df_raw[
      categorical_features + continuous_features + ["label"]]
  test_df = test_df_raw[categorical_features + continuous_features + ["label"]]

  # Bucketize continuous features.
  bucketize_continuous_column(train_df, test_df, "age", num_quantiles=4)
  bucketize_continuous_column(
      train_df, test_df, "capital_gain", bins=[-1, 1, 4000, 10000, 100000])
  bucketize_continuous_column(
      train_df, test_df, "capital_loss", bins=[-1, 1, 1800, 1950, 4500])
  bucketize_continuous_column(
      train_df, test_df, "hours_per_week", bins=[0, 39, 41, 50, 100])
  bucketize_continuous_column(
      train_df, test_df, "education_num", bins=[0, 8, 9, 11, 16])

  # Binarize both the categorical features and bucketized continuous features.
  train_df, test_df = binarize_categorical_columns(
      train_df, test_df, categorical_features + continuous_features)

  # Separate labels for train and test set.
  labels_train = train_df["label"].to_numpy()
  labels_test = test_df["label"].to_numpy()

  # Take "gender_Female" as protected group.
  groups_train = train_df["gender_Female"].to_numpy()
  groups_test = test_df["gender_Female"].to_numpy()

  # Include all columns other than "label" as features.
  feature_names = list(train_df.keys())
  feature_names.remove("label")
  features_train = train_df[feature_names].to_numpy()
  features_test = test_df[feature_names].to_numpy()

  # Return train and test set tuples.
  train_set = (features_train, labels_train, groups_train)
  test_set = (features_test, labels_test, groups_test)

  return train_set, test_set


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Load and pre-process Adult train and test datasets.
  train_set, test_set = load_data()
  x_train, y_train, z_train = test_set

  # Split train data into train and validation sets.
  train_indices, vali_indices = model_selection.train_test_split(
      np.arange(x_train.shape[0]), test_size=FLAGS.vali_fraction)

  x_vali = x_train[vali_indices, :]
  y_vali = y_train[vali_indices]
  z_vali = z_train[vali_indices]

  x_train = x_train[train_indices, :]
  y_train = y_train[train_indices]
  z_train = z_train[train_indices]

  train_set = x_train, y_train, z_train
  vali_set = x_vali, y_vali, z_vali

  # Save processed dataset.
  with open(FLAGS.output_directory + "Adult.npy", "wb") as f:
    np.save(f, (train_set, vali_set, test_set), allow_pickle=True)

if __name__ == "__main__":
  app.run(main)
