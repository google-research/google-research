# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Data reader for UCI adult dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow.compat.v1 as tf

from tensorflow.contrib import lookup as contrib_lookup

IPS_WITH_LABEL_TARGET_COLUMN_NAME = "IPS_example_weights_with_label"
IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME = "IPS_example_weights_without_label"
SUBGROUP_TARGET_COLUMN_NAME = "subgroup"


class UCIAdultInput():
  """Data reader for UCI Adult dataset."""

  def __init__(self,
               dataset_base_dir,
               train_file=None,
               test_file=None):
    """Data reader for UCI Adult dataset.

    Args:

      dataset_base_dir: (string) directory path.
      train_file: string list of training data paths.
      test_file: string list of evaluation data paths.

      dataset_base_sir must contain the following files in the dir:
      - train.csv: comma separated training data without header.
        Column order must match the order specified in self.feature_names.
      - test.csv: comma separated training data without header.
        Column order must match the order specified in self.feature_names.
      - mean_std.json: json dictionary of the format feature_name: [mean, std]},
        containing mean and std for numerical features. For example,
        "hours-per-week": [40.437455852092995, 12.347428681731843],...}.
      - vocabulary.json: json dictionary of the format {feature_name:
        [feature_vocabulary]}, containing vocabulary for categorical features.
        For example, {sex": ["Female", "Male"],...}.
      - IPS_example_weights_with_label.json: json dictionary of the format
        {subgroup_id : inverse_propensity_score,...}. For example,
        {"0": 2.34, ...}.
      - IPS_example_weights_without_label.json: json dictionary of the format
        {subgroup_id : inverse_propensity_score,...}. For example,
        {"0": 2.34, ...}.
    """
    # pylint: disable=long-line,line-too-long
    self._dataset_base_dir = dataset_base_dir

    if train_file:
      self._train_file = train_file
    else:
      self._train_file = ["{}/adult.data".format(self._dataset_base_dir)]

    if test_file:
      self._test_file = test_file
    else:
      self._test_file = ["{}/adult.test".format(self._dataset_base_dir)]

    self._mean_std_file = "{}/mean_std.json".format(self._dataset_base_dir)
    self._vocabulary_file = "{}/vocabulary.json".format(self._dataset_base_dir)
    self._ips_with_label_file = "{}/IPS_example_weights_with_label.json".format(
        self._dataset_base_dir)
    self._ips_without_label_file = "{}/IPS_example_weights_without_label.json".format(self._dataset_base_dir)
    # pylint: enable=long-line,line-too-long

    self.feature_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income"]

    self.RECORD_DEFAULTS = [[0.0], ["?"], [0.0], ["?"], [0.0], ["?"], ["?"],  # pylint: disable=invalid-name
                            ["?"], ["?"], ["?"], [0.0], [0.0], [0.0], ["?"],
                            ["?"]]

    # Initializing variable names specific to UCI Adult dataset input_fn
    self.target_column_name = "income"
    self.target_column_positive_value = ">50K"
    self.sensitive_column_names = ["sex", "race"]
    self.sensitive_column_values = ["Female", "Black"]
    self.weight_column_name = "instance_weight"

  def get_input_fn(self, mode, batch_size=128):
    """Gets input_fn for UCI census income data.

    Args:
      mode: The execution mode, as defined in tf.estimator.ModeKeys.
      batch_size: An integer specifying batch_size.

    Returns:
      An input_fn.
    """

    def _input_fn():
      """Input_fn for the dataset."""
      if mode == tf.estimator.ModeKeys.TRAIN:
        filename_queue = tf.train.string_input_producer(self._train_file)
      elif mode == tf.estimator.ModeKeys.EVAL:
        filename_queue = tf.train.string_input_producer(self._test_file)

      # Extracts basic features and targets from filename_queue
      features, targets = self.extract_features_and_targets(
          filename_queue, batch_size)

      # Adds subgroup information to targets. Used to plot metrics.
      targets = self._add_subgroups_to_targets(features, targets)

      # Adds ips_example_weights to targets
      targets = self._add_ips_example_weights_to_targets(targets)

      # Unused in robust_learning models. Adding it for min-diff approaches.
      # Adding instance weight to features.
      features[self.weight_column_name] = tf.ones_like(
          targets[self.target_column_name], dtype=tf.float32)

      return features, targets

    return _input_fn

  def extract_features_and_targets(self, filename_queue, batch_size):
    """Extracts features and targets from filename_queue."""
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    feature_list = tf.decode_csv(value, record_defaults=self.RECORD_DEFAULTS)

    # Setting features dictionary.
    features = dict(zip(self.feature_names, feature_list))
    features = self._binarize_protected_features(features)
    features = tf.train.batch(features, batch_size)

    # Setting targets dictionary.
    targets = {}
    targets[self.target_column_name] = tf.reshape(
        tf.cast(
            tf.equal(
                features.pop(self.target_column_name),
                self.target_column_positive_value), tf.float32), [-1, 1])
    return features, targets

  def _binarize_protected_features(self, features):
    """Processes protected features and binarize them."""
    for sensitive_column_name, sensitive_column_value in zip(
        self.sensitive_column_names, self.sensitive_column_values):
      features[sensitive_column_name] = tf.cast(
          tf.equal(
              features.pop(sensitive_column_name), sensitive_column_value),
          tf.float32)
    return features

  def _add_subgroups_to_targets(self, features, targets):
    """Adds subgroup information to targets dictionary."""
    for sensitive_column_name in self.sensitive_column_names:
      targets[sensitive_column_name] = tf.reshape(
          tf.identity(features[sensitive_column_name]), [-1, 1])
    return targets

  def _load_json_dict_into_hashtable(self, filename):
    """Load json dictionary into a HashTable."""
    with tf.gfile.Open(filename, "r") as filename:
      # pylint: disable=g-long-lambda
      temp_dict = json.load(
          filename,
          object_hook=lambda d:
          {int(k) if k.isdigit() else k: v for k, v in d.items()})
      # pylint: enable=g-long-lambda

    keys = list(temp_dict.keys())
    values = [temp_dict[k] for k in keys]
    feature_names_to_values = contrib_lookup.HashTable(
        contrib_lookup.KeyValueTensorInitializer(
            keys, values, key_dtype=tf.int64, value_dtype=tf.float32), -1)
    return feature_names_to_values

  def _add_ips_example_weights_to_targets(self, targets):
    """Add ips_example_weights to targets. Used in ips baseline model."""

    # Add subgroup information to targets
    target_subgroups = (targets[self.target_column_name],
                        targets[self.sensitive_column_names[1]],
                        targets[self.sensitive_column_names[0]])
    targets[SUBGROUP_TARGET_COLUMN_NAME] = tf.map_fn(
        lambda x: (2 * x[1]) + (1 * x[2]), target_subgroups, dtype=tf.float32)

    # Load precomputed IPS weights into a HashTable.
    ips_with_label_table = self._load_json_dict_into_hashtable(self._ips_with_label_file)  # pylint: disable=line-too-long
    ips_without_label_table = self._load_json_dict_into_hashtable(self._ips_without_label_file)  # pylint: disable=line-too-long

    # Adding IPS example weights to targets
    # pylint: disable=g-long-lambda
    targets[IPS_WITH_LABEL_TARGET_COLUMN_NAME] = tf.map_fn(
        lambda x: ips_with_label_table.lookup(
            tf.cast((4 * x[0]) + (2 * x[1]) + (1 * x[2]), dtype=tf.int64)),
        target_subgroups,
        dtype=tf.float32)
    targets[IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME] = tf.map_fn(
        lambda x: ips_without_label_table.lookup(
            tf.cast((2 * x[1]) + (1 * x[2]), dtype=tf.int64)),
        target_subgroups,
        dtype=tf.float32)
    # pylint: enable=g-long-lambda

    return targets

  def get_feature_columns(self,
                          embedding_dimension=0,
                          include_sensitive_columns=True):
    """Return feature columns and weight_column_name for census data.

    Categorical features are encoded as categorical columns with vocabulary list
    (given by vocabulary in vocabulary_file), and saved as either a
    embedding_column or indicator_column. All numerical columns are normalized
    (given by mean and std in mean_std_file).

    Args:
      embedding_dimension: (int) dimension of the embedding column. If set to 0
        a multi-hot representation using tf.feature_column.indicator_column is
        created. If not, a representation using
        tf.feature_column.embedding_column is created. Consider using
        embedding_column if the number of buckets (unique values) are large.
      include_sensitive_columns: boolean flag. If set, sensitive attributes are
        included in feature_columns.

    Returns:
      feature_columns: list of feature_columns.
      weight_column_name: (string) name of the weight column.
      feature_names: list of feature_columns.
      target_column_name: (string) name of the target variable column.
    """
    # Load precomputed mean and standard deviation values for features.
    with tf.gfile.Open(self._mean_std_file, "r") as mean_std_file:
      mean_std_dict = json.load(mean_std_file)
    with tf.gfile.Open(self._vocabulary_file, "r") as vocabulary_file:
      vocab_dict = json.load(vocabulary_file)

    feature_columns = []
    for i in range(0, len(self.feature_names)):
      if (self.feature_names[i] in [
          self.weight_column_name, self.target_column_name
      ]):
        continue
      elif self.feature_names[i] in self.sensitive_column_names:
        if include_sensitive_columns:
          feature_columns.append(
              tf.feature_column.numeric_column(self.feature_names[i]))
        else:
          continue
      elif self.RECORD_DEFAULTS[i][0] == "?":
        sparse_column = tf.feature_column.categorical_column_with_vocabulary_list(
            self.feature_names[i], vocab_dict[self.feature_names[i]])
        if embedding_dimension > 0:
          feature_columns.append(
              tf.feature_column.embedding_column(sparse_column,
                                                 embedding_dimension))
        else:
          feature_columns.append(
              tf.feature_column.indicator_column(sparse_column))
      else:
        mean, std = mean_std_dict[self.feature_names[i]]
        feature_columns.append(
            tf.feature_column.numeric_column(
                self.feature_names[i],
                normalizer_fn=(lambda x, m=mean, s=std: (x - m) / s)))
    return feature_columns, self.weight_column_name, self.sensitive_column_names, self.target_column_name

