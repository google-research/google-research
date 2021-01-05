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

"""Simple sckit-learn classification utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import pickle

from absl import flags
from absl import logging
import numpy as np
from sklearn import model_selection
from sklearn.compose import make_column_transformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# pylint: disable=invalid-name

flags.DEFINE_boolean(
    "transform_inputs", True,
    "If enabled, will scale the numeric features and convert categorical "
    "features to one-hot encoding.")

flags.DEFINE_list(
    "classifiers", ["LogisticRegression"],
    "Type of the classifier. One of: \"LogisticRegression\", \"SVM\", "
    "\"RidgeRegression\", \"RandomForest\", \"AdaBoost\", \"LDA\", \"QDA\", "
    "\"GaussianProcess\", \"DecisionTree\", \"DNN\", \"GaussianNaiveBayes\", "
    "\"BaggingEnsemble\".")

flags.DEFINE_boolean(
    "use_implicationals", True, "If True, use the implicational features.")

flags.DEFINE_string(
    "best_configurations_file", "",
    "File containing the JSON dictionary from feature names to the "
    "respective best model and data configurations. When `--cross_validate` "
    "is enabled, this is the output file to be generated. In all other modes "
    "this is an input file.")

FLAGS = flags.FLAGS

# List of all supported classifiers.
ALL_MODELS = [
    "AdaBoost", "DNN", "DecisionTree", "GaussianProcess", "LDA",
    "LogisticRegression", "QDA", "RandomForest", "RidgeRegression", "SVM",
    "GaussianNaiveBayes", "BaggingEnsemble"
]

# Model information keys.
MODEL_INFO_NAME_KEY = "name"
MODEL_INFO_SPARSITY_KEY = "no_cv"  # Not enough data.
MODEL_INFO_SCORE_KEY = "accuracy"
MODEL_INFO_CANDIDATES_KEY = "candidates"

# Random seed.
_RANDOM_STATE = 4611170

# WALS language code.
_LANGUAGE_CODE = "wals_code"


def _prepare_data(input_df):
  """Splits data into features and labels."""
  class_label = "target_value"
  y = input_df[class_label].copy()
  X_columns_to_drop = [class_label, _LANGUAGE_CODE, "target_feature"]
  X = input_df.drop(columns=X_columns_to_drop)
  return X, y


def _split_into_features_and_labels(feature_name, feature_maker,
                                    training_df, dev_df,
                                    transform_inputs):
  """Preprocesses the data and returns the features and labels."""
  # Get the label class counts for the training data.
  train_class_counts = training_df.target_value.value_counts()
  train_class_counts = list(zip(train_class_counts.index,
                                train_class_counts.values))
  logging.info("%s: Class counts: %s", feature_name, train_class_counts)

  # Perform the split into features and labels of the training set.
  X_train, y_train = _prepare_data(training_df)
  logging.info("%s: Input feature dimensions: %s", feature_name,
               X_train.shape[1])

  # Split dev set.
  X_dev, y_dev = _prepare_data(dev_df)

  # Numeric columns are transformed using standard scaler and categorical
  # columns are converted to one-hot.
  if transform_inputs:
    numeric_cols = ["latitude", "longitude"]
    categorical_cols = []
    for col_name in X_train.columns:
      if (col_name in feature_maker.prob_features or
          col_name in feature_maker.count_features):
        numeric_cols.append(col_name)  # Counts, probabilities.
      elif col_name in feature_maker.categorical_features:
        categorical_cols.append(col_name)  # Categorical feature values.
    inputs_transformer = make_column_transformer(
        (StandardScaler(), numeric_cols),
        (OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        remainder="passthrough")
    X_train = inputs_transformer.fit_transform(X_train)
    if X_dev.shape[0]:  # Do we have enough samples?
      X_dev = inputs_transformer.transform(X_dev)
    else:
      logging.warning("Feature %s not found in the dev set. This is likely to "
                      "crash the evaluation mode!", feature_name)
  else:
    # Transform data frames to Numpy. The input transformer in the branch above
    # returns Numpy arrays.
    X_train = X_train.to_numpy()
    X_dev = X_dev.to_numpy()

  return (
      X_train, y_train.to_numpy(), X_dev, y_dev.to_numpy(), train_class_counts)


def prepare_data(feature_maker, feature_name, use_implicationals=True,
                 prediction_mode=False):
  """Prepares the features and labels for the given WALS feature name."""
  # Process training and dev data for the feature. Store the WALS language codes
  # for the development set aside.
  training_df, dev_df = feature_maker.process_data(
      feature_name, prediction_mode=prediction_mode)
  assert _LANGUAGE_CODE in dev_df.columns
  dev_language_codes = list(dev_df[_LANGUAGE_CODE].values)
  if not use_implicationals:
    logging.info("Discarding implicational features")
    training_df = feature_maker.select_columns(training_df,
                                               discard_implicationals=True)
    dev_df = feature_maker.select_columns(dev_df,
                                          discard_implicationals=True)

  # Split the data into features and labels.
  X_train, y_train, X_dev, y_dev, train_class_counts = (
      _split_into_features_and_labels(
          feature_name, feature_maker, training_df, dev_df,
          FLAGS.transform_inputs))
  return X_train, y_train, X_dev, y_dev, dev_language_codes, train_class_counts


def _make_classifier(classifier_name):
  """Classifier factory."""
  # Class weights: if you set this to None, you'd get much better accuracies,
  # but it's likely that the classifier will be overpredicting the majority
  # class.
  class_weight_strategy = None  # Note: this may set "balanced" as default.
  max_iters = 10000
  if classifier_name == "AdaBoost":
    model = AdaBoostClassifier(n_estimators=100)
  elif classifier_name == "LogisticRegression":
    model = LogisticRegression(max_iter=max_iters,
                               class_weight=class_weight_strategy)
  elif classifier_name == "LDA":
    model = LinearDiscriminantAnalysis(tol=1E-6)
  elif classifier_name == "QDA":
    model = QuadraticDiscriminantAnalysis()
  elif classifier_name == "DNN":
    model = MLPClassifier(random_state=_RANDOM_STATE,
                          hidden_layer_sizes=[200])
  elif classifier_name == "DecisionTree":
    model = DecisionTreeClassifier(random_state=_RANDOM_STATE,
                                   min_samples_leaf=3,
                                   criterion="entropy",
                                   class_weight="balanced")
  elif classifier_name == "GaussianProcess":
    model = GaussianProcessClassifier(random_state=_RANDOM_STATE,
                                      max_iter_predict=200)
  elif classifier_name == "RandomForest":
    model = RandomForestClassifier(n_estimators=200,
                                   random_state=_RANDOM_STATE,
                                   min_samples_leaf=3,
                                   criterion="entropy",
                                   class_weight="balanced_subsample")
  elif classifier_name == "RidgeRegression":
    model = RidgeClassifier(normalize=True, tol=1E-5,
                            class_weight=class_weight_strategy)
  elif classifier_name == "SVM":
    model = LinearSVC(max_iter=max_iters, class_weight=class_weight_strategy)
  elif classifier_name == "GaussianNaiveBayes":
    model = GaussianNB()
  elif classifier_name == "BaggingEnsemble":
    model = BaggingClassifier(random_state=_RANDOM_STATE)
  else:
    raise ValueError("Unsupported classifier: %s" % classifier_name)
  return model


def cross_validate(feature_name, classifier_name, X, y,
                   cv_num_folds, cv_num_repeats):
  """Runs repeated stratified $k$-fold cross-validation.

  Returns multiple cross-validation metrics as a dictionary, where for each
  metric mean and variance across multiple repeats and folds is summarized.

  Args:
    feature_name: (string) Name of the WALS feature.
    classifier_name: (string) Classifier name.
    X: (numpy array) Input features.
    y: (numpy array) Labels.
    cv_num_folds: (int) Number of folds ($k$).
    cv_num_repeats: (int) Number of repetitions.

  Returns:
    Dictionary containing cross-validation scores and stats.
  """
  model = _make_classifier(classifier_name)
  scoring = ["f1_micro", "precision_micro", "recall_micro", "accuracy"]
  try:
    # Really primitive logic to figure out class distribution.
    _, y_counts = np.unique(y, return_counts=True)
    y_max_freq = np.max(y_counts)

    # Check if the class counts are not reliable to run cross-validation.
    if y_max_freq < cv_num_folds:
      logging.warning("[%s] %s: Not enough data. Fitting the model instead "
                      "of running CV", feature_name, classifier_name)
      # Simply fit the model.
      model.fit(X, y)
      cv_scores = {}
      cv_scores["accuracy"] = (model.score(X, y), 0.0)
      cv_scores[MODEL_INFO_SPARSITY_KEY] = True
      return cv_scores
    else:
      logging.info("[%s] Running cross-validation of %s (k=%d, n=%d) ...",
                   feature_name, classifier_name, cv_num_folds, cv_num_repeats)
      # Run cross-validation.
      cv = RepeatedStratifiedKFold(n_splits=cv_num_folds,
                                   n_repeats=cv_num_repeats,
                                   random_state=_RANDOM_STATE)
      cv_scores = model_selection.cross_validate(
          model, X, y, cv=cv, scoring=scoring, n_jobs=cv_num_folds)
      cv_scores[MODEL_INFO_SPARSITY_KEY] = False
  except Exception as e:  # pylint: disable=broad-except
    logging.error("[%s] %s: CV: Exception: %s", feature_name, classifier_name,
                  e)
    return None

  del cv_scores["fit_time"]
  del cv_scores["score_time"]
  for score_name in scoring:
    scores_vec_key = "test_" + score_name
    cv_scores[score_name] = (np.mean(cv_scores[scores_vec_key]),
                             np.var(cv_scores[scores_vec_key]))
    del cv_scores[scores_vec_key]
  # Sanity check.
  if math.isnan(cv_scores["accuracy"][0]):
    return None
  logging.info("[train] %s: CV scores for %s: %s", feature_name,
               classifier_name, cv_scores)
  return cv_scores


def train_classifier(feature_name, classifier_name, X, y, model_path=None):
  """Trains classifier."""
  model = _make_classifier(classifier_name)
  logging.info("%s: Fitting %s model ...",
               feature_name, classifier_name)
  model.fit(X, y)
  logging.info("%s: %s: Score: %s", feature_name, classifier_name,
               model.score(X, y))
  if model_path:
    logging.info("Saving model to \"%s\" ...", model_path)
    pickle.dump(model, open(model_path, "wb"))
  return model


def select_best_model(classifiers, feature_name, X_train, y_train,
                      cv_num_folds, cv_num_repeats):
  """Performs cross-validation of various classifiers for a given feature.

  Returns a dictionary with the best classifier name, its score and the number
  of candidates it was selected from.

  Args:
    classifiers: (list) Names of the classifiers to choose from.
    feature_name: (string) WALS feature name.
    X_train: (numpy array) Training features.
    y_train: (numpy array) Training labels.
    cv_num_folds: (int) Number of folds ($k$).
    cv_num_repeats: (int) Number of repetitions.

  Returns:
    Dictionary containing best configuration.
  """
  scores = []
  for classifier_name in classifiers:
    clf_scores = cross_validate(feature_name, classifier_name, X_train, y_train,
                                cv_num_folds, cv_num_repeats)
    if clf_scores:  # Cross-validation may fail for some settings.
      scores.append((classifier_name, clf_scores))
  # Sort the scores by the highest accuracy mean. For some reason F1 and
  # accuracy are the same (as is the precision and recall). Investigate.
  scores = sorted(scores, key=lambda score: score[1]["accuracy"][0],
                  reverse=True)
  if len(scores) < 5:
    raise ValueError("Expected at least five candidate classifiers!")
  best_model = scores[0]
  return {
      MODEL_INFO_NAME_KEY: best_model[0],  # Model name.
      # Accuracy mean.
      MODEL_INFO_SCORE_KEY: best_model[1]["accuracy"][0],
      # Boolean sparsity marker.
      MODEL_INFO_SPARSITY_KEY: best_model[1][MODEL_INFO_SPARSITY_KEY],
      # Overall number of successful evals.
      MODEL_INFO_CANDIDATES_KEY: len(scores)
  }
