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

"""Utilities for defining metrics on Earthquakes results."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import base
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils import fixes
from tensorflow.compat.v1 import keras
import xarray as xr


# We use variable X, y to conform to sklearn's naming conventions.
# pylint: disable=invalid-name
def _model_probs_sklearn(model, X):
  return model.predict_proba(X)[:, 1]


def _model_probs_keras(model, X):
  return model.predict(X)


def get_model_probabilities(model, X):
  """Retrieves the probabilities the model gives for the positive label on X."""
  if isinstance(model, base.BaseEstimator):
    return _model_probs_sklearn(model, X)
  if isinstance(model, keras.Model):
    return _model_probs_keras(model, X)
  else:
    raise TypeError('Unrecognized model type: %s' % model)


def standardize_binary_probabilities(probs):
  """Converts binary probability labels to standard shape and form.

  This utility accepts label probabilities in multiple formats, and standardizes
  them into a single format.

  Args:
    probs: An np.array where the first dimension is the number of examples. It
      can be one-dimensional or two-dimensional, and may contain either only the
      probabilities for the positive label, or both the negative and positive
      labels.

  Returns:
    An np.array of shape (num_examples,) containing only the probabilities for
    the positive label.
  """
  if len(probs.shape) > 1 and probs.shape[1] > 1:
    probs = probs[:, 1]
  return probs.flatten()


def ROC_curve(model, X, y, PR_curve=False, title=''):
  """Displays an ROC curve or a Precision-Recall curve."""
  scores = get_model_probabilities(model, X)
  scores = standardize_binary_probabilities(scores)
  y = standardize_binary_probabilities(y)

  if PR_curve:
    curve_type = 'Precision-Recall'
    total_metric = metrics.average_precision_score(y, scores)
    total_metric_name = 'AP'
    y_values, x_values, _ = metrics.precision_recall_curve(y, scores)
    x_name = 'Recall'
    y_name = 'Precision'
  else:
    curve_type = 'ROC'
    total_metric = metrics.roc_auc_score(y, scores)
    total_metric_name = 'AUC'
    x_values, y_values, _ = metrics.roc_curve(y, scores)
    x_name = 'False Positive rate'
    y_name = 'True Positive rate'

  step_kwargs = (
      {'step': 'post'}
      if 'step' in fixes.signature(plt.fill_between).parameters
      else {}
  )
  plt.step(x_values, y_values, color='b', alpha=0.2, where='post')
  plt.fill_between(x_values, y_values, alpha=0.2, color='b', **step_kwargs)
  plt.xlabel(x_name)
  plt.ylabel(y_name)
  plt.ylim([0.0, 1.0])
  plt.xlim([0.0, 1.0])
  plt.title(
      '{title:s} {curve_type:s} curve: {metric_name:s}={metric_value:0.2f}'
      .format(
          title=title,
          curve_type=curve_type,
          metric_name=total_metric_name,
          metric_value=total_metric,
      )
  )
  plt.show()

  return total_metric


def discretize_continuous_features(X, num_classes):
  """Discretize features that have more than num_classes values.

  Args:
    X: A features np.array of shape (num_examples, num_features).
    num_classes: The maximum number of classes to discretize each feature into.

  Returns:
    A discretized version of the feature array, where each feature has at most
    num_classes different values.
  """
  succeeded = False
  while not succeeded:
    class_nums = [
        min(num_classes, len(np.unique(X[:, i]))) for i in range(X.shape[1])
    ]
    class_nums = [max(x, 2) for x in class_nums]
    discretizer = preprocessing.KBinsDiscretizer(
        n_bins=class_nums, encode='ordinal', strategy='quantile'
    )
    discretizer.fit(X)
    try:
      discretized = discretizer.transform(X)
      succeeded = True
    except ValueError:
      # Sometimes fails because the bins end up too close. In this case, reduce
      # by 1 the number of classes.
      num_classes -= 1
      print(
          'Warning: Discretization failed, reducing num_classes to %d'
          % num_classes
      )
  return discretized


def features_mutual_information(X, y, num_classes=5):
  """Calculates the mutual information between each feature and the label."""
  X = np.array(X)
  y = np.array(y)
  assert len(X.shape) == 2
  X = discretize_continuous_features(X, num_classes)

  return np.array(
      [
          metrics.mutual_info_score(y, X[:, feature_idx])
          for feature_idx in range(X.shape[1])
      ]
  )


def xr_log_loss(true_labels, predicted, dim=None, eps=1e-12):
  """Calculates the logloss between two xr.DataArrays.

  Leverages the broadcasting behavior of xarray to automatically broadcast
  predicted and true_label along the right dimensions.

  Args:
    true_labels: An xarray.DataArray with boolean labels.
    predicted: Numpy numeric or xr.DataArray of the predicted probablity of
      observing a True label.
    dim: Dimension name, or list of names, to calculate the average on. If None
      (default), average is calculated over all dimensions.
    eps: Small number to clip probabilities close to zero or unity.

  Returns:
    An xr.DataArray with the log-loss between the predicted and true labels.
  """
  if isinstance(predicted, (int, float)) or predicted.size == 1:
    predicted = xr.full_like(true_labels, predicted, dtype='d')

  true_labels, predicted = xr.broadcast(true_labels, predicted)
  predicted = predicted.clip(eps, 1 - eps)
  score = xr.where(true_labels, -np.log(predicted), -np.log(1 - predicted))
  return score.mean(dim)
