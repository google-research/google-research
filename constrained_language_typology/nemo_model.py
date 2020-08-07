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

"""This model relies on whichever model performed best during cross-validation.

The models are simple ML models trained using scikit-learn.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=g-importing-member
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
# pylint: enable=g-importing-member

import json

from absl import flags
from absl import logging

import basic_models
import constants as const
import data_info as data_info_lib
import scikit_classifier as classifier_lib

# pylint: disable=invalid-name

flags.DEFINE_string(
    "force_classifier", "",
    "Name of the classifier to use instead of configuration file specifying "
    "the best models (`--best_configurations_file`).")

flags.DEFINE_integer(
    "num_workers", 1,
    "Number of training pipelines to execute concurrently.")

FLAGS = flags.FLAGS

# Uses this classifier by default.
_DEFAULT_CLASSIFIER_NAME = "RidgeRegression"

# Threshold below which the model is declared unreliable.
_BAD_ACCURACY_THRESHOLD = 0.4


class NemoModel(object):
  """NEMO Model.

  Ideally, should make use information from areas, phylogenetics and
  implicationals.
  """

  def __init__(self, prediction_mode):
    self._name = "NemoModel"
    self._prediction_mode = prediction_mode
    self._df = None
    self._data_info = None
    self._configs = {}    # Cross-validation configuration.
    self._eval_data = {}  # Input features for evaluation (per feature).
    self._models = {}     # Trained models (per feature).
    self._ymax_freq = {}  # Frequency of the most frequent training class.

    if not FLAGS.best_configurations_file and not FLAGS.force_classifier:
      raise ValueError(
          "NEMO model requires model configuration file specified with "
          "`--best_configurations_file`. This can be found in `data/runs` "
          "directory. Alternatively, you can use single classifier for all the "
          "predictions specified with `--force_classifier`.")

  def _prepare_data_worker(self, feature_name):
    """Prepares data for individual feature.

    Decision to use or ignore the implicationals is taken based on
    cross-validation configuration. Returns a set of input features
    for prediction and corresponding language codes.

    Args:
      feature_name: (string) Name of the feature.

    Returns:
      A triple consisting of name of the WALS feature (string), the evalation
      (or test) input features for the classifier (numpy array) and a list of
      language codes (WALS codes), where each code corresponds to a single
      row in eval feature data.
    """
    use_implicationals = True
    if feature_name in self._configs:
      use_implicationals = self._configs[feature_name]["use_implicationals"]
    _, _, X_dev, _, eval_language_codes, _ = classifier_lib.prepare_data(
        self._feature_maker, feature_name,
        use_implicationals=use_implicationals,
        prediction_mode=self._prediction_mode)
    if X_dev.shape[0] != len(eval_language_codes):
      raise ValueError("Number of eval examples (%d) mismatches number of "
                       "languages (%d)!" % (X_dev.shape[0],
                                            len(eval_language_codes)))
    return feature_name, X_dev, eval_language_codes

  def _prepare_data(self, features_to_predict):
    """Prepares the training/evaluation data."""
    num_features = len(features_to_predict)
    logging.info("[%s] Preparing data for %d features ...",
                 self._name, num_features)
    with ThreadPoolExecutor(max_workers=FLAGS.num_workers) as executor:
      future_to_model = {
          executor.submit(self._prepare_data_worker, feature_name):
          feature_name for feature_name in features_to_predict
      }
      for future in as_completed(future_to_model):
        try:
          feature_name, eval_X, eval_language_codes = future.result()
        except Exception as exc:  # pylint: disable=broad-except
          logging.error("Exception occurred: %s!", exc)
        else:
          self._eval_data[feature_name] = (eval_X, eval_language_codes)

  def _train_model_worker(self, feature_name):
    """Train individual classifier in a single thread."""
    model_is_reliable = True
    if not FLAGS.force_classifier:
      # Select classifiers from the best configuration.
      model_name = _DEFAULT_CLASSIFIER_NAME
      if feature_name in self._configs:
        assert "model" in self._configs[feature_name]
        model_config = self._configs[feature_name]["model"]
        model_name = model_config[classifier_lib.MODEL_INFO_NAME_KEY]
        should_ignore = model_config[classifier_lib.MODEL_INFO_SPARSITY_KEY]
        score = model_config[classifier_lib.MODEL_INFO_SCORE_KEY]
        if should_ignore or score < _BAD_ACCURACY_THRESHOLD:
          # Not enough training data or low CV accuracy score. Fall back to
          # search-based approaches.
          logging.warning("[%s] No reliable models found", feature_name)
          model_is_reliable = False
    else:
      # Use single classifier for everything.
      model_name = FLAGS.force_classifier

    if model_is_reliable:
      # Train the model. Please note, the training features have already been
      # constructed and cached by the feature maker during the data preparation
      # step preceding the training.
      logging.info("[%s] %s: \"%s\" ...", self._name, feature_name, model_name)
      use_implicationals = True
      if not FLAGS.force_classifier:
        use_implicationals = self._configs[feature_name]["use_implicationals"]
      X_train, y_train, _, _, _, train_class_counts = (
          classifier_lib.prepare_data(
              self._feature_maker, feature_name,
              use_implicationals=use_implicationals,
              prediction_mode=self._prediction_mode))
      ymax_freq = train_class_counts[0][1]  # Highest frequency.
      model = classifier_lib.train_classifier(feature_name, model_name,
                                              X_train, y_train)
    return feature_name, model, ymax_freq

  def _train_models(self, features_to_predict):
    """Trains classifiers for all features in the list."""
    num_features = len(features_to_predict)
    logging.info("[%s] Training classifiers for %d features ...",
                 self._name, num_features)
    with ThreadPoolExecutor(max_workers=FLAGS.num_workers) as executor:
      future_to_model = {
          executor.submit(self._train_model_worker, feature_name):
          feature_name for feature_name in features_to_predict
      }
      for future in as_completed(future_to_model):
        try:
          feature_name, model, ymax_freq = future.result()
        except Exception as exc:  # pylint: disable=broad-except
          logging.error("Exception occurred: %s!", exc)
        else:
          self._models[feature_name] = model
          self._ymax_freq[feature_name] = ymax_freq

  def init(self, training_data_dir, train_set_name, dev_set_name,
           features_to_predict):
    """Initializes the model."""
    # Load the training set and the data info. Make sure all features (but not
    # necessarily their values) we predict are present in the data info mapping.
    self._df = basic_models.load_training_data(self._name, training_data_dir,
                                               train_set_name)
    self._data_info = data_info_lib.load_data_info(
        data_info_lib.data_info_path_for_testing(training_data_dir))
    for feature_name in features_to_predict:
      if feature_name not in self._data_info[const.DATA_KEY_FEATURES]:
        raise ValueError("Feature \"%s\" unseen in training data!" %
                         feature_name)

    # Load the associations computed from the training data (family, genus,
    # neighborhood and implicationals). We are not using them yet.
    self._feature_maker = basic_models.make_feature_maker(
        self._name, training_data_dir, train_set_name, dev_set_name)

    # Compute majority class stats.
    self._global_majority_class, _, _, _ = basic_models.collect_majority_stats(
        self._name, self._df)

    # Read results of cross-validation.
    if FLAGS.best_configurations_file:
      logging.info("Reading cross-validation configuration from \"%s\" ...",
                   FLAGS.best_configurations_file)
      with open(FLAGS.best_configurations_file, "r",
                encoding=const.ENCODING) as f:
        self._configs = json.load(f)
      logging.info("Read configurations for %d features.", len(self._configs))

    # Prepare the evaluation data for all the features. This also prepares the
    # training data for the next step.
    self._prepare_data(features_to_predict)

    # Train the models. This will use the training data cached by the previous
    # step.
    self._train_models(features_to_predict)

  def prepare_target(self, target_df):
    """Precomputes the language-specific information."""
    pass

  def predict(self, target_df, context_features, feature_name):
    """Predicts the feature given the context.

    The core information (geo location and phylogenetics) is provided
    a fixed set of features in `target_df`.
    The (valid) context features are provided in the `context_features`
    mapping from feature names to their valid values for this language.

    Args:
      target_df: (pandas) Dataframe representing language.
      context_features: (dict) Contextual features.
      feature_name: (string) WALS feature name.

    Returns:
      N-best list with a single top candidate.
    """
    del context_features  # These are currently unused.

    language_code = target_df["wals_code"]
    if feature_name in self._models:
      # Fetch the pre-build eval input features for the one example for this
      # feature and language.
      if feature_name not in self._eval_data:
        raise ValueError("No eval data for feature \"%s\"!" % feature_name)
      eval_X, eval_language_codes = self._eval_data[feature_name]
      if language_code not in eval_language_codes:
        raise ValueError("[%s] Evaluation language \"%s\" not found in eval "
                         "data (languages: %s)!" % (feature_name,
                                                    language_code,
                                                    eval_language_codes))
      x_id = list(eval_language_codes).index(language_code)
      x = eval_X[x_id, :].reshape(1, -1)

      # Perform the prediction and map it back to the resulting string value.
      model = self._models[feature_name]
      y = model.predict(x)[0]  # One prediction.
      feature_values = self._data_info[const.DATA_KEY_FEATURES][
          feature_name]["values"]
      if y < 1 or y > len(feature_values):
        raise ValueError("Invalid predicted class: %d!" % y)
      return [feature_values[y - 1]]
    else:
      # For now simply return majority class.
      return [self._global_majority_class[feature_name]]
