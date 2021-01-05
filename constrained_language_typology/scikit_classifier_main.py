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

r"""Simple tool for training and evaluation with scikit-learn.

Examples:
---------

(1) To run pure cross-validation and save the best configurations:

  > python3 scikit_classifier_main.py \
      --training_data_file /tmp/train_dev.csv \
      --dev_data_file /tmp/dev.csv \
      --data_info_file /tmp/data_info_train_dev.json.gz \
      --target_feature Order_of_Subject,_Object_and_Verb \
      --cross_validate --cv_num_folds 5 --cv_num_repeats 10 \
      --best_configurations_file /tmp/best_configs.json

The above run will save the resulting configurations in `best_configs.json`.

(2) To run simple evaluation of several classifiers for one WALS feature:

  > python3 scikit_classifier_main.py \
      --training_data_file /tmp/train.csv \
      --dev_data_file /tmp/dev.csv \
      --data_info_file /tmp/data_info_train_dev.json.gz \
      --classifiers=RidgeRegression,SVM,DNN,LogisticRegression,RandomForest \
      --target_feature Order_of_Subject,_Object_and_Verb \
      --association_dir data/train
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import json

from absl import app
from absl import flags
from absl import logging

import build_feature_matrix as feature_lib
import compute_associations  # pylint: disable=[unused-import]
import constants as const
import numpy as np
import scikit_classifier as classifier_lib
from sklearn import metrics

# pylint: disable=invalid-name

flags.DEFINE_string(
    "dev_data_file", "",
    "Development data file produced by `sigtyp_reader_main.py`.")

flags.DEFINE_string(
    "training_data_file", "",
    "Training data file produced by `sigtyp_reader_main.py`.")

flags.DEFINE_string(
    "data_info_file", "",
    "Training info file produced by `sigtyp_reader_main.py`.")

flags.DEFINE_string(
    "target_feature", "",
    "Name of the WALS feature we are training/evaluating.")

flags.DEFINE_string(
    "target_feature_file", "",
    "If specified overrides any setting of target_feature "
    "and reads the features to be modeled from a file, "
    "one per line.")

flags.DEFINE_boolean(
    "catch_exceptions", True, "If True, catches failure exceptions.")

flags.DEFINE_boolean(
    "cross_validate", False,
    "Enables cross-validation.")

flags.DEFINE_integer(
    "cv_num_folds", 5,
    "Number of splits/folds ($k$) for $k$-fold cross-validation.")

flags.DEFINE_integer(
    "cv_num_repeats", 10,
    "Number of times to repeat $k$-fold cross-validation.")

FLAGS = flags.FLAGS


def _train_and_evaluate_model(feature_name, classifier_name, X_train, y_train,
                              X_dev, y_truth):
  """Trains and evaluates the supplied model."""
  model = classifier_lib.train_classifier(feature_name, classifier_name,
                                          X_train, y_train)
  y_predicted = model.predict(X_dev)
  accuracy = metrics.accuracy_score(y_predicted, y_truth)
  f1_score = metrics.f1_score(y_predicted, y_truth, average="micro")
  print("=== [{}] {}: Dev set: Accuracy {}, F1: {}".format(
      feature_name, classifier_name, accuracy, f1_score))
  print("=== [{}] {}: Confusion matrix:\n{}".format(
      feature_name, classifier_name, metrics.confusion_matrix(
          y_predicted, y_truth, labels=np.unique(y_train))))
  return accuracy


def _train_and_evaluate(feature_maker, feature_names):
  """Train and evaluate a particular feature.

  Please note: This mode is more suitable for proper evaluation rather than a
  lengthy cross-validation-based training.

  Args:
    feature_maker: (object) Feature builder.
    feature_names: (list) List of WALS feature names (strings).
  """
  for feature_name in feature_names:
    try:
      # Process training and dev data for the feature.
      X_train, y_train, X_dev, y_dev, _, _ = classifier_lib.prepare_data(
          feature_maker, feature_name,
          use_implicationals=FLAGS.use_implicationals)

      # Train and evaluate models.
      best_acc = 0
      best_classifier = ""
      for classifier_name in FLAGS.classifiers:
        acc = _train_and_evaluate_model(feature_name, classifier_name,
                                        X_train, y_train, X_dev, y_dev)
        if acc > best_acc:
          best_acc = acc
          best_classifier = classifier_name
      print("=== [{}] {}: Dev set: Best Accuracy {}".format(
          feature_name, best_classifier, best_acc))

    except Exception:  # pylint: disable=broad-except
      if not FLAGS.catch_exceptions:
        raise
      logging.warning("Problem with processing feature: %s", feature_name)


def _cross_validation_training(feature_maker, feature_names):
  """Finds the best models for features by cross-validation."""
  use_implicationals_values = [True, False]
  best_configs = collections.defaultdict(lambda: collections.defaultdict(str))
  total_num_configs = len(feature_names) * 2  # This excludes the models.
  n = 1
  for feature_name in feature_names:
    best_models = []
    for use_implicationals in use_implicationals_values:
      logging.info("CV training: %d/%d", n, total_num_configs)
      # Process training and dev data for the feature.
      X_train, y_train, _, _, _ = classifier_lib.prepare_data(
          feature_maker, feature_name,
          use_implicationals=use_implicationals)

      # Find the best model for the feature.
      best_model_info = classifier_lib.select_best_model(
          classifier_lib.ALL_MODELS, feature_name, X_train, y_train,
          FLAGS.cv_num_folds, FLAGS.cv_num_repeats)
      best_models.append((use_implicationals, best_model_info))
      n += 1

    # Select the best model with or without implicationals.
    best_models = sorted(
        best_models,
        key=lambda info: info[1][classifier_lib.MODEL_INFO_SCORE_KEY],
        reverse=True)
    best_configs[feature_name]["use_implicationals"] = best_models[0][0]
    best_configs[feature_name]["model"] = best_models[0][1]

  return best_configs


def main(unused_argv):
  if not FLAGS.dev_data_file:
    raise ValueError("Specify --dev_data_file")
  if not FLAGS.training_data_file:
    raise ValueError("Specify --training_data_file")
  if not FLAGS.data_info_file:
    raise ValueError("Specify --data_info_file")
  if not (FLAGS.target_feature or FLAGS.target_feature_file):
    raise ValueError("Specify --target_feature or --target_feature_file")
  if FLAGS.cross_validate and not FLAGS.best_configurations_file:
    raise ValueError("Specify --best_configurations_file in cross-validation "
                     "mode")

  features = []
  if FLAGS.target_feature_file:
    with open(FLAGS.target_feature_file) as s:
      for line in s:
        features.append(line.strip())
  else:
    features = [FLAGS.target_feature]

  # Process features.
  feature_maker = feature_lib.FeatureMaker(
      FLAGS.training_data_file,
      FLAGS.dev_data_file,
      FLAGS.data_info_file)

  # Perform cross-validation to establish the best configurations of models
  # and features or simply train and evaluate.
  if FLAGS.cross_validate:
    best_configs = _cross_validation_training(feature_maker, features)
    logging.info("Saving best configs to \"%s\" ...",
                 FLAGS.best_configurations_file)
    with open(FLAGS.best_configurations_file, "w",
              encoding=const.ENCODING) as f:
      json.dump(best_configs, f)
  else:
    _train_and_evaluate(feature_maker, features)


if __name__ == "__main__":
  app.run(main)
