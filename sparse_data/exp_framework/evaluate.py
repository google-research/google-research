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

"""Evaluates predictions.

Computes a variety of metrics (accuracy, auc, log loss, confusion matrix).
"""

import numpy as np
from sklearn import metrics
from sklearn import preprocessing


def classification_metrics(y_train, y_train_proba, y_test, y_test_proba,
                           labels):
  """Generate classification metrics.

  Args:
    y_train: Training labels.
    y_train_proba: Predictions from an estimator with probs on train data.
    y_test: Testing labels.
    y_test_proba: Predictions from an estimator with probs on test data.
    labels: Possible labels.

  Returns:
    dict: dictionary with train/test auc, accuracy and loss.
  """

  return {
      'micro_auc': auc(y_test, y_test_proba, labels, average='micro'),
      'macro_auc': auc(y_test, y_test_proba, labels, average='macro'),
      'train_loss': loss(y_train, y_train_proba, labels),
      'test_loss': loss(y_test, y_test_proba, labels),
      'train_acc': accuracy(y_train, np.argmax(y_train_proba, axis=1)),
      'test_acc': accuracy(y_test, np.argmax(y_test_proba, axis=1))
  }


def regression_metrics(y_train, y_train_pred, y_test, y_test_pred):
  """Generate regression metrics.

  Args:
    y_train: Training labels.
    y_train_pred: Predictions from an estimator with on train data.
    y_test: Testing labels.
    y_test_pred: Predictions from an estimator with probs on test data.

  Returns:
    dict: dictionary with train/test mse.
  """

  return {
      'train_mse': mean_squared_error(y_train, y_train_pred),
      'test_mse': mean_squared_error(y_test, y_test_pred)
  }


def auc(y_test, y_proba, labels, average='macro'):
  """Computes the ROC AUC.

  Args:
    y_test: np.array 1-D array of true class labels
    y_proba: np.array (num_sample, num_feature) array of probabilities over
      class labels
    labels: [int] list of all possible class labels
    average: string averaging method for computing multiclass AUC; values =
      'macro', 'micro'

  Returns:
    auc: float
      ROC AUC value
  """
  if len(y_test.shape) < 2:
    enc = preprocessing.LabelBinarizer()
    enc.fit(labels)
    y_test_binary = enc.transform(y_test)
  else:
    y_test_binary = y_test

  if len(labels) <= 2:
    y_proba = y_proba[:, 1]

  return metrics.roc_auc_score(y_test_binary, y_proba, average=average)


def loss(y_test, y_proba, labels):
  """Computes log loss.

  Args:
    y_test: np.array 1-D array of true class labels
    y_proba: np.array (num_sample, num_feature) array of probabilities over
      class labels
    labels: [int] list of all possible class labels

  Returns:
    loss: float
      log loss score
  """
  return metrics.log_loss(y_test, y_proba, labels=labels)


def accuracy(y_test, y_pred):
  """Computes the accuracy score.

  Args:
    y_test: np.array 1-D array of true class labels
    y_pred: np.array 1-D array of predicted class labels

  Returns:
    accuracy: float
      accuracy score
  """
  return metrics.accuracy_score(y_test, y_pred)


def mean_squared_error(y_test, y_pred):
  """Computes the mean squared error (MSE).

  Args:
    y_test: np.array 1-D array of true class labels
    y_pred: np.array 1-D array of predicted class labels
  Returns:
    mean squared error: float.
  """
  return metrics.mean_squared_error(y_test, y_pred)


def confusion_matrix(y_test, y_pred, labels):
  """Computes the confusion matrix.

  Args:
    y_test: np.array 1-D array of true class labels
    y_pred: np.array 1-D array of predicted class labels
    labels: [int] list of all possible class labels

  Returns:
    confusion: np.array
      confusion matrix
  """
  return metrics.confusion_matrix(y_test, y_pred, labels=labels)
