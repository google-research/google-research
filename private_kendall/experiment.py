# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Code for running private linear regression methods.

Adapted in part from
https://github.com/google-research/google-research/blob/master/dp_regression/experiment.py.
"""


import enum

import numpy as np
import sklearn.model_selection

from private_kendall import boosted_adassp
from private_kendall import kendall
from private_kendall import lasso
from private_kendall import metrics
from private_kendall import regression
from private_kendall import tukey


class FeatureSelectionMethod(enum.Enum):
  NONE = 1
  LASSO = 2
  KENDALL = 3


class RegressionMethod(enum.Enum):
  BAS = 1
  TUKEY = 2


def run_nondp(use_lasso, features, labels, train_frac, num_trials):
  """Returns 0.25, 0.5, and 0.75 R^2 quantiles from num_trials non-DP models.

  Args:
    use_lasso: Whether or not to use Lasso regression.
    features: Matrix of feature vectors where each row is an example. Assumed to
      have intercept feature in the last column.
    labels: Vector of labels.
    train_frac: Fraction of data to use for training.
    num_trials: Number of trials to run.

  Returns:
    0.25, 0.5, and 0.75 R^2 quantiles from num_trials non-DP models.
  """
  r2s = np.zeros(num_trials)
  for trial in range(num_trials):
    train_features, test_features, train_labels, test_labels = (
        sklearn.model_selection.train_test_split(
            features, labels, test_size=1 - train_frac
        )
    )
    model = regression.nondp(use_lasso, train_features, train_labels)
    r2s[trial] = metrics.r_squared_from_model(model, test_features, test_labels)
  return np.quantile(r2s, [0.25, 0.5, 0.75])


def compute_private_n(n, epsilon, eta):
  """Privately estimates a lower bound for number of examples n.

  Args:
    n: Number of examples in data.
    epsilon: The output will satisfy epsilon-DP.
    eta: Failure probability eta for private lower bound.

  Returns:
    A private 1-eta probability (over the added noise) lower bound for n. See
    discussion in Experiments section of the paper for details.
  """
  private_n = n + np.random.laplace(scale=1 / epsilon)
  return max(1, int((private_n - np.log(1 / (2 * eta)) / epsilon)))


def run_feature_selection(
    features,
    labels,
    feature_selection_method,
    feature_selection_epsilon,
    feature_selection_k,
    num_models
):
  """Returns selected features according to feature_selection_method.

  Args:
    features: Matrix of feature vectors where each row is an example. Assumed to
      have intercept feature in the last column.
    labels: Vector of labels.
    feature_selection_method: See FeatureSelectionMethod enum.
    feature_selection_epsilon: The feature selection step will be
      feature_selection_epsilon-DP.
    feature_selection_k: Number of features to select.
    num_models: Number of models to use, if feature_selection_method is
    FeatureSelectionMethod.LASSO. This parameter is not used for any other
    setting of feature_selection_method.

  Returns:
    Vector of k features selected according to feature_selection_method (or the
    initial set of features if feature_selection_method =
    FeatureSelectionMethod.NONE).
  """
  if feature_selection_method == FeatureSelectionMethod.KENDALL:
    top_indices = kendall.dp_kendall_feature_selection(
        features,
        labels,
        feature_selection_k,
        feature_selection_epsilon,
    )
  elif feature_selection_method == FeatureSelectionMethod.LASSO:
    top_indices = lasso.dp_lasso_features(
        features,
        labels,
        feature_selection_k,
        num_models,
        feature_selection_epsilon,
    )
  else:
    top_indices = range(len(features[0]))
  return top_indices


def run_private_regression(
    features,
    labels,
    train_frac,
    regression_method,
    feature_selection_method,
    feature_selection_epsilon,
    feature_selection_k,
    compute_n_epsilon,
    compute_n_eta,
    regression_epsilon,
    regression_delta,
    num_trials,
):
  """Returns 0.25, 0.5, 0.75 R^2 quantiles from trials of specified algorithm.

  Args:
    features: Matrix of feature vectors where each row is an example. Assumed to
      have intercept feature in the last column.
    labels: Vector of labels.
    train_frac: Fraction of data to use for training.
    regression_method: See RegressionMethod enum.
    feature_selection_method: See FeatureSelectionMethod enum.
    feature_selection_epsilon: The feature selection step will be
      feature_selection_epsilon-DP.
    feature_selection_k: Number of features to select.
    compute_n_epsilon: The private lower bound for n will be compute_n-DP.
    compute_n_eta: Failure probability eta for computing valid lower bound of n.
    regression_epsilon: The DP regression step will be (regression_epsilon,
      regression_delta)-DP.
    regression_delta: The DP regression step will be (regression_epsilon,
      regression_delta)-DP.
    num_trials: Number of trials to run.
  """
  r2s = np.zeros(num_trials)
  for trial in range(num_trials):
    train_features, test_features, train_labels, test_labels = (
        sklearn.model_selection.train_test_split(
            features, labels, test_size=1 - train_frac
        )
    )
    n, d = train_features.shape
    num_models = -1
    # We only need to privately estimate n if we're using it to set num_models,
    # either for Lasso feature selection or Tukey regression
    if (
        feature_selection_method == FeatureSelectionMethod.LASSO
        or regression_method == RegressionMethod.TUKEY
    ):
      n_estimate = compute_private_n(n, compute_n_epsilon, compute_n_eta)
      # run_feature_selection only uses num_models for Lasso, which uses all d
      # features
      num_models = int(n_estimate / d)
    top_indices = run_feature_selection(
        train_features,
        train_labels,
        feature_selection_method,
        feature_selection_epsilon,
        feature_selection_k,
        num_models,
    )
    train_features = train_features[:, top_indices]
    test_features = test_features[:, top_indices]
    # We only need to set regression_num_models if we're applying Tukey
    # regression and feature selection occurred, so num_models needs to be
    # updated with feature_selection_k rather than d
    if (
        regression_method == RegressionMethod.TUKEY
        and feature_selection_method != FeatureSelectionMethod.NONE
    ):
      num_models = int(n_estimate / (feature_selection_k + 1))
    if regression_method == RegressionMethod.TUKEY:
      models = tukey.multiple_regressions(
          train_features, train_labels, num_models, False
      )
      private_model = tukey.tukey(models, regression_epsilon, regression_delta)
    else:
      # We set Boosted AdaSSP's number of boosting rounds to 100 and both clip
      # norms to 1 throughout
      private_model = boosted_adassp.boosted_adassp(
          train_features,
          train_labels,
          100,
          1,
          1,
          regression_epsilon,
          regression_delta,
      )
      # Boosted AdaSSP learns a model on clipped features, so its R^2 only makes
      # sense with similarly clipped features
      test_features = boosted_adassp.clip(test_features, 1)
    # Handle case when Tukey returns "ptr fail"
    if isinstance(private_model, str):
      r2s[trial] = -np.inf
    else:
      r2s[trial] = metrics.r_squared_from_model(
          private_model, test_features, test_labels
      )
  # quantile nans given an array of infs, so we handle it manually
  non_inf_result = 0
  for entry in r2s:
    non_inf_result = max(non_inf_result, entry != -np.inf)
  if non_inf_result:
    return np.quantile(r2s, [0.25, 0.5, 0.75])
  else:
    return -np.inf, -np.inf, -np.inf
