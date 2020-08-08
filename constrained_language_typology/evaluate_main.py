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

r"""Utility for evaluating the models.

We distinguish between two modes of operation: evaluation and prediction. In
evaluation mode the predicted values are known in advance, so the prediction
quality can be objectively evaluated. In pure prediction mode, no truth values
are available. If prediction mode is enabled, we simply fill the requested
features with their predicted values.

Example:
--------
(1) Evaluation mode:

Basic KNN based on geo distances:

  > PREPROCESSED_DATA_DIR=...  # Data created by "sigtyp_reader".
  > python3 evaluate_main.py \
      --sigtyp_dir ${ST2020_DIR}/data \
      --algorithm BasicHaversineKNN \
      --training_data_dir ${PREPROCESSED_DATA_DIR}

(2) Prediction mode:

  > PREPROCESSED_DATA_DIR=...  # Data created by "sigtyp_reader".
  > python3 evaluate_main.py \
      --sigtyp_dir ${ST2020_DIR}/data \
      --algorithm BasicExperimentalNemo \
      --training_data_dir ${PREPROCESSED_DATA_DIR} \
      --test_set_name test_blinded \
      --prediction_mode \
      --output_sigtyp_predictions_file /tmp/results.txt

(3) Evaluation mode with model-specific flags:

  > PREPROCESSED_DATA_DIR=...  # Data created by "sigtyp_reader".
  > python3 evaluate_main.py \
      --sigtyp_dir ${ST2020_DIR}/data \
      --training_data_dir /tmp \
      --train_set_name train \
      --test_set_name dev \
      --association_dir data/train \
      --algorithm NemoModel --num_workers 10 \
      --force_classifier LogisticRegression

IMPORTANT: We assume that each row in train/development and test corresponds to
unique language. In other words, language information is not spread across
different sets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import os

from absl import app
from absl import flags
from absl import logging

import basic_models as basic
import compute_associations  # pylint: disable=[unused-import]
import constants as const
import nemo_model as nemo
import pandas as pd
import sigtyp_reader as sigtyp

# pylint: disable=g-long-lambda

flags.DEFINE_string(
    "sigtyp_dir", "",
    "Directory containing SIGTYP original test data.")

flags.DEFINE_string(
    "test_set_name", const.DEV_FILENAME,
    "Name of the test set. Currently defaults to development set.")

flags.DEFINE_string(
    "train_set_name", const.TRAIN_FILENAME,
    "Name of the training set. Defaults to the original training data.")

flags.DEFINE_string(
    "training_data_dir", "",
    "Directory where the training data resides. This data has to be in the "
    "format generated from the original SIGTYP data by the "
    "\"sigtyp_reader_main\" tool.")

flags.DEFINE_string(
    "algorithm", "BasicHaversineKNN",
    "Name of the algorithm. Supported types: \"BasicMajorityClass\", "
    "\"BasicCladeMajorityClass\", \"BasicHaversineKNN\", "
    "\"BasicHaversineKNNWithClades\", \"BasicExperimentalNemo\", "
    "\"NemoModel\".")

flags.DEFINE_integer(
    "num_best", 1,
    "Number of best hypotheses to return.")

flags.DEFINE_boolean(
    "include_countries", False,
    "Include country code information in the models, if possible.")

flags.DEFINE_boolean(
    "prediction_mode", False,
    "If enabled, assume that test dataset is truly blind and we are simply "
    "required to fill in the missing feature values.")

flags.DEFINE_string(
    "output_sigtyp_predictions_file", "",
    "When prediction mode is enabled, specifying this file will output all "
    "languages in the original test set with missing feature values filled in.")

FLAGS = flags.FLAGS


def _value_is_valid(value):
  """Returns true if the value is known and valid."""
  return not pd.isnull(value) and value != const.UNKNOWN_FEATURE_VALUE


def _prepare_context_features(feature_values, feature_to_predict):
  """Converts a list of feature/value tuples to a context feature dictionary."""
  known_features = [(feature, value) for feature, value in feature_values
                    if _value_is_valid(value)]
  known_features_dict = dict(known_features)
  if feature_to_predict in known_features_dict:
    # Make sure the feature we're trying to predict is not in context.
    del known_features_dict[feature_to_predict]
  return known_features_dict


def _features_to_predict(test_df):
  """Returns the list of feature names that we need to predict."""
  test_feature_names = set()
  for _, test_language_df in test_df.iterrows():
    feature_values = sigtyp.get_feature_values(test_language_df)
    for name, value in feature_values:
      if ((FLAGS.prediction_mode and value == const.UNKNOWN_FEATURE_VALUE) or
          (not FLAGS.prediction_mode and _value_is_valid(value))):
        test_feature_names.add(name)
  logging.info("====> %d features to evaluate/predict", len(test_feature_names))
  return list(test_feature_names)


def _evaluate_language(language_df, model, feature_counts,
                       feature_value_counts):
  """Performs prediction and/or evaluation of a single language.

  Returns the total number of evaluations/predictions and number of correct
  predictions (in evaluation mode). In prediction mode returns all the features
  with missing values filled in (in evaluation mode this list will be empty).

  Args:
    language_df: (pandas) Dataframe representing a single language.
    model: (object) Model to evaluate language with.
    feature_counts: (dict) Counters for the language features.
    feature_value_counts: (dict) Counters for feature values (classes).

  Returns:
    A triple representing total number of evaluations, number of correct
    evaluations and the predicted features in SIGTYP format.
  """
  num_evals = 0
  num_correct = 0
  feature_values = sigtyp.get_feature_values(language_df)
  predictions = [language_df[col_id] for col_id in
                 range(sigtyp.NUM_COLUMNS - 1)]
  predicted_feature_values = []
  for feature, value in feature_values:
    if FLAGS.prediction_mode and _value_is_valid(value):
      # In prediction mode we update the predictions list and actually skip the
      # prediction stage for this particular feature.
      predicted_feature_values.append("%s=%s" % (feature, value))
      continue
    if not FLAGS.prediction_mode and not _value_is_valid(value):
      # In evaluation mode, don't evaluate on empty features.
      continue

    if feature not in feature_counts:
      feature_counts[feature]["correct"] = 0
      feature_counts[feature]["total"] = 0

    # In "pure" prediction mode we don't have the truth values to compare
    # against.
    unknown_feature = False
    if value == const.UNKNOWN_FEATURE_VALUE:
      unknown_feature = True
    context_features = _prepare_context_features(feature_values, feature)
    predicted_nbest_values = model.predict(language_df, context_features,
                                           feature)
    single_best = predicted_nbest_values[0]
    if FLAGS.prediction_mode:
      predicted_feature_values.append("%s=%s" % (feature, single_best))
    if not unknown_feature and single_best == value:  # Correct prediction.
      num_correct += 1
      feature_counts[feature]["correct"] += 1
    elif single_best != value:
      # Accumulate false positives (for precision) and false negatives (for
      # recall).
      feature_value_counts[feature][single_best]["fp"] += 1
      feature_value_counts[feature][value]["fn"] += 1

    if not unknown_feature or FLAGS.prediction_mode:
      num_evals += 1
      feature_counts[feature]["total"] += 1

  if FLAGS.prediction_mode:
    predictions.append("|".join(predicted_feature_values))
  return num_evals, num_correct, predictions


def _feature_accuracy(feature_stats):
  """Computes accuracy from the supplied counters."""
  return (feature_stats["correct"] / feature_stats["total"] * 100.0
          if feature_stats["total"] != 0.0 else 0.0)


def _feature_micro_precision(feature_stats, value_stats):
  """Computes micro-averaged precision from the supplied counts."""
  num_all_positives = feature_stats["correct"]
  for value in value_stats:
    num_all_positives += value_stats[value]["fp"]
  return (feature_stats["correct"] / num_all_positives * 100.0
          if num_all_positives != 0.0 else 0.0)


def _feature_micro_recall(feature_stats, value_stats):
  """Computes micro-averaged recall from the supplied counts."""
  num_actual_positives = feature_stats["correct"]
  for value in value_stats:
    num_actual_positives += value_stats[value]["fn"]
  return (feature_stats["correct"] / num_actual_positives * 100.0
          if num_actual_positives != 0.0 else 0.0)


def _evaluate(test_df, test_data_info, model):
  """Evaluates the model on the supplied dataframe."""
  # Prepare the test languages. This lets the model precompute some of the
  # information that is based solely on the language context, not the features.
  logging.info("Preparing test languages ...")
  test_languages = test_df.to_dict(orient="row")
  for test_language_df in test_languages:
    model.prepare_target(test_language_df)

  # Run actual evaluation.
  mode_info = "Prediction" if FLAGS.prediction_mode else "Evaluation"
  logging.info("[%s] Running over %d languages ...", mode_info, len(test_df))
  total_num_evals = 0
  total_num_correct = 0.0
  feature_counts = collections.defaultdict(lambda: collections.defaultdict(int))
  feature_value_counts = collections.defaultdict(
      lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
  all_languages_predictions = []
  for _, test_language_df in test_df.iterrows():
    lang_num_evals, lang_num_correct, predictions = _evaluate_language(
        test_language_df, model, feature_counts, feature_value_counts)
    total_num_evals += lang_num_evals
    total_num_correct += lang_num_correct
    all_languages_predictions.append(predictions)

  logging.info("Total number of evals: %d", total_num_evals)
  if total_num_evals == 0:
    logging.warning("No features to predict found. You are probably not using "
                    "the blind test set and should either use it or switch the "
                    "prediction mode off with '--noprediction_mode'.")
  if not FLAGS.prediction_mode:
    logging.info("Global Accuracy: %f%%",
                 total_num_correct / total_num_evals * 100.0)
  for feature in sorted(feature_counts):
    stats = feature_counts[feature]
    value_stats = feature_value_counts[feature]
    if not FLAGS.prediction_mode:
      logging.info("%s: [n=%d] Accuracy: %f%%, Precision: %f%%, Recall: %f%%",
                   feature, stats["total"], _feature_accuracy(stats),
                   _feature_micro_precision(stats, value_stats),
                   _feature_micro_recall(stats, value_stats))
    else:
      logging.info("%s: [%d predictions]", feature, stats["total"])

  # Save the test dataset with all the features filled in.
  if not FLAGS.prediction_mode:
    return
  logging.info("Saving predictions to \"%s\" ...",
               FLAGS.output_sigtyp_predictions_file)
  columns = ["wals_code", "name", "latitude", "longitude",
             "genus", "family", "countrycodes", "features"]
  result_test_df = pd.DataFrame(all_languages_predictions, columns=columns)
  result_test_df.to_csv(FLAGS.output_sigtyp_predictions_file,
                        index=False, sep="\t",
                        encoding=const.ENCODING, quotechar='"')

  # Sanity check. Read the data back in and make sure the values are sane.
  logging.info("Sanity check ...")
  read_result_test_df, _, read_result_data_info = sigtyp.read(
      FLAGS.output_sigtyp_predictions_file, categorical_as_ints=False,
      verbose=False)
  if len(test_df) != len(read_result_test_df):
    raise ValueError("Expected %s languages in the resulting file!" %
                     len(test_df))
  vanilla_num_feats = len(test_data_info[const.DATA_KEY_FEATURES])
  read_num_feats = len(read_result_data_info[const.DATA_KEY_FEATURES])
  if vanilla_num_feats > read_num_feats:
    raise ValueError("Expected same or larger number of feature to be present: "
                     "Original %d, read %d" % (vanilla_num_feats,
                                               read_num_feats))
  test_df_nonzero_values = test_df.count(axis=1).sum()
  result_df_non_zero_values = read_result_test_df.count(axis=1).sum()
  if test_df_nonzero_values != result_df_non_zero_values:
    raise ValueError("Expected same number of non-zero values in predictions!")


def _make_model(features_to_predict):
  """Factory for making a model based on its name."""
  if FLAGS.algorithm == "BasicHaversineKNN":
    model = basic.BasicHaversineKNN(FLAGS.num_best)
  elif FLAGS.algorithm == "BasicMajorityClass":
    model = basic.BasicMajorityClass()
  elif FLAGS.algorithm == "BasicCladeMajorityClass":
    model = basic.BasicCladeMajorityClass(
        include_countries=FLAGS.include_countries)
  elif FLAGS.algorithm == "BasicHaversineKNNWithClades":
    model = basic.BasicHaversineKNNWithClades(FLAGS.num_best)
  elif FLAGS.algorithm == "BasicExperimentalNemo":  # In flux.
    model = basic.BasicExperimentalNemo(FLAGS.num_best)
    model.init(FLAGS.training_data_dir, FLAGS.train_set_name,
               FLAGS.test_set_name)
    return model
  elif FLAGS.algorithm == "NemoModel":  # In flux.
    model = nemo.NemoModel(FLAGS.prediction_mode)
    model.init(FLAGS.training_data_dir, FLAGS.train_set_name,
               FLAGS.test_set_name, features_to_predict)
    return model
  else:
    raise ValueError("Unsupported algorithm: \"%s\"" % FLAGS.algorithm)
  model.init(FLAGS.training_data_dir, FLAGS.train_set_name)
  return model


def main(unused_argv):
  # Check flags.
  if not FLAGS.sigtyp_dir:
    raise ValueError("Specify --sigtyp_dir for input data!")
  if not FLAGS.training_data_dir:
    raise ValueError("Specify --training_data_dir!")
  if FLAGS.prediction_mode and not FLAGS.output_sigtyp_predictions_file:
    raise ValueError("In prediction mode specify "
                     "--output_sigtyp_predictions_file!")

  # Load the test data (note: we read the vanilla test data in SIGTYP format).
  filename = os.path.join(FLAGS.sigtyp_dir, FLAGS.test_set_name + ".csv")
  vanilla_test_df, _, vanilla_data_info = sigtyp.read(filename,
                                                      categorical_as_ints=False)
  if not len(vanilla_test_df):  # pylint: disable=g-explicit-length-test
    raise ValueError("Test dataset is empty!")
  if len(vanilla_test_df.columns) != 8:
    raise ValueError("Wrong number of columns: %d" % len(
        vanilla_test_df.columns))

  # Run evaluation/prediction.
  features_to_predict = _features_to_predict(vanilla_test_df)
  _evaluate(vanilla_test_df.copy(), vanilla_data_info,
            _make_model(features_to_predict))


if __name__ == "__main__":
  app.run(main)
