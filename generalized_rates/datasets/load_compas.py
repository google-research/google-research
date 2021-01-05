# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Loads COMPAS dataset, processes it and saves it to output folder.

Adapted from code provided by Heinrich Jiang (heinrichj@google.com). The COMPAS
data file can be downloaded from:
https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import pandas as pd
from sklearn import model_selection


flags.DEFINE_float("test_fraction", 1.0 / 3.0,
                   "Fraction of overall dataset that constitutes the test set.")
flags.DEFINE_float("vali_fraction", 1.0 / 3.0,
                   "Fraction of train set that constitutes the validation set.")
flags.DEFINE_string("data_file", "compas-scores-two-years.csv",
                    "Path to COMPAS dataset csv file.")
flags.DEFINE_string("output_directory", "datasets/",
                    "Path to store processed dataset.")


FLAGS = flags.FLAGS


def load_data():
  """Load and process dataset from provided data path."""
  df = pd.read_csv(FLAGS.data_file)

  # Filter relevant features.
  features = [
      "age", "c_charge_degree", "race", "score_text", "sex", "priors_count",
      "days_b_screening_arrest", "decile_score", "is_recid", "two_year_recid"]
  df = df[features]
  df = df[df.days_b_screening_arrest <= 30]
  df = df[df.days_b_screening_arrest >= -30]
  df = df[df.is_recid != -1]
  df = df[df.c_charge_degree != "O"]
  df = df[df.score_text != "N/A"]

  # Divide features into: continuous, categorical and those that are continuous,
  # but need to be converted to categorical.
  categorical_features = ["c_charge_degree", "race", "score_text", "sex"]
  continuous_to_categorical_features = ["age", "decile_score", "priors_count"]

  # Bucketize features in continuous_to_categorical_features.
  for feature in continuous_to_categorical_features:
    if feature == "priors_count":
      bins = list(np.percentile(df[feature], [0, 50, 70, 80, 90, 100]))
    else:
      bins = [0] + list(np.percentile(df[feature], [20, 40, 60, 80, 90, 100]))
    df[feature] = pd.cut(df[feature], bins, labels=False)

  # Binarize all categorical features (including the ones bucketized above).
  df = pd.get_dummies(df, columns=categorical_features +
                      continuous_to_categorical_features)

  # Fill values for decile scores and prior counts feature buckets.
  to_fill = [u"decile_score_0", u"decile_score_1", u"decile_score_2",
             u"decile_score_3", u"decile_score_4", u"decile_score_5"]
  for i in range(len(to_fill) - 1):
    df[to_fill[i]] = df[to_fill[i:]].max(axis=1)
  to_fill = [u"priors_count_0.0", u"priors_count_1.0", u"priors_count_2.0",
             u"priors_count_3.0", u"priors_count_4.0"]
  for i in range(len(to_fill) - 1):
    df[to_fill[i]] = df[to_fill[i:]].max(axis=1)

  # Get the labels (two year recidivism) and groups (female defendants).
  labels = df["two_year_recid"]
  groups = df["sex_Female"]

  # Retain all features other than "two_year_recid" and "is_recid".
  df.drop(columns=["two_year_recid", "is_recid"], inplace=True)

  return df.to_numpy(), labels.to_numpy(), groups.to_numpy()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Load and pre-process COMPAS dataset.
  features, labels, groups = load_data()

  # Split dataset indices into train and test.
  train_indices, test_indices = model_selection.train_test_split(
      np.arange(features.shape[0]), test_size=FLAGS.test_fraction)
  # Split train indices further into train and validation,
  train_indices, vali_indices = model_selection.train_test_split(
      train_indices, test_size=FLAGS.vali_fraction)

  # Split features, labels and groups for train, test and validation sets,
  x_train = features[train_indices, :]
  y_train = labels[train_indices]
  z_train = groups[train_indices]

  x_test = features[test_indices, :]
  y_test = labels[test_indices]
  z_test = groups[test_indices]

  x_vali = features[vali_indices, :]
  y_vali = labels[vali_indices]
  z_vali = groups[vali_indices]

  train_set = x_train, y_train, z_train
  vali_set = x_vali, y_vali, z_vali
  test_set = x_test, y_test, z_test

  # Save processed dataset.
  with open(FLAGS.output_directory + "COMPAS.npy", "wb") as f:
    np.save(f, (train_set, vali_set, test_set), allow_pickle=True)

if __name__ == "__main__":
  app.run(main)
