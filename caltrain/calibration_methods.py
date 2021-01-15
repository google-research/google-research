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

"""Calibration methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np
from scipy.optimize import minimize
from six.moves import range
import sklearn.isotonic
import sklearn.metrics

from caltrain import utils


class CalibrationMethod(abc.ABC):
  """General interface for specifying post-hoc calibration methods."""

  @abc.abstractmethod
  def fit(self, logits, one_hot_labels):
    """Fit the calibration method.

    Args:
        logits: raw (non-normalized) predictions that a classification model
          generates, shape=(num_examples, num_classes)
        one_hot_labels: one-hot-encoding of true labels, shape=(num_examples,
          num_classes)
    """
    pass

  @abc.abstractmethod
  def predict(self, logits):
    """Predict new calibrated softmax probabilities.

    Args:
        logits: raw (non-normalized) predictions that a classification model
          generates, shape=(num_examples, num_classes)

    Returns:
        calibrated softmax probabilities, shape=(num_samples, num_classes)
    """
    pass


class IsotonicRegression(CalibrationMethod):
  """Isotonic regression calibration method.

  Learns piece-wise constant function to perform post-hoc calibration by
  minimizing the square loss between softmax outputs and true class labels.
  """

  def __init__(self, num_classes=10):
    self.num_classes = num_classes
    self.ir_per_class = [
        sklearn.isotonic.IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds='clip')
        for _ in range(num_classes)
    ]

  def fit(self, logits, one_hot_labels):
    """Fit the isotonic regression model.

    Args:
        logits: raw (non-normalized) predictions that a classification model
          generates, shape=(num_examples, num_classes)
        one_hot_labels: one-hot-encoding of true labels, shape=(num_examples,
          num_classes)
    """
    assert logits.shape[1] == self.num_classes
    assert logits.shape == one_hot_labels.shape

    softmax_probabilities = utils.to_softmax(logits)
    for i in range(self.num_classes):
      self.ir_per_class[i].fit(softmax_probabilities[:, i], one_hot_labels[:,
                                                                           i])

  def predict(self, logits):
    """Predict new softmax probabilities from logit scores.

    Uses linear interpolation in underlying scikit learn call.

    Args:
        logits: raw (non-normalized) predictions that a classification model
          generates, shape=(num_examples, num_classes)

    Returns:
        calibrated softmax probabilities, shape = (num_examples, num_classes)
    """
    assert logits.shape[1] == self.num_classes
    input_probabilities = utils.to_softmax(logits)
    new_probabilities = np.ones(np.shape(input_probabilities))
    for i in range(self.num_classes):
      new_probabilities[:, i] = self.ir_per_class[i].predict(
          input_probabilities[:, i])
    # normalize each row of the probability vector if in multiclass setting
    if self.num_classes > 1:
      row_sums = np.sum(new_probabilities, axis=1)
      return new_probabilities / row_sums[:, np.newaxis]
    else:
      return new_probabilities


class TemperatureScaling(CalibrationMethod):
  """Temperature scaling calibration method."""

  def __init__(self, temperature=1, max_iterations=50, solver='BFGS'):
    """Initialize class.

    Args:
        temperature (float): starting temperature, default 1
        max_iterations (int): maximum iterations done by optimizer
        solver (string): type of optimization method to use
    """
    self.temperature = temperature
    self.max_iterations = max_iterations
    self.solver = solver

  def _loss(self, t, logits, one_hot_labels):
    """Calculates the cross-entropy loss.

    Args:
        t: temperature
        logits: raw (non-normalized) predictions that a classification model
          generates, shape=(num_examples, num_classes)
        one_hot_labels: one-hot-encoding of true labels, shape=(num_examples,
          num_classes)

    Returns:
        value of log loss
    """
    scaled_softmax_probabilities = self.predict(logits, t)
    return sklearn.metrics.log_loss(
        y_true=one_hot_labels, y_pred=scaled_softmax_probabilities)

  def fit(self, logits, one_hot_labels):
    """Trains the model and finds optimal temperature.

    Args:
        logits: raw (non-normalized) predictions that a classification model
          generates, shape=(num_examples, num_classes)
        one_hot_labels: one-hot-encoding of true labels, shape=(num_examples,
          num_classes)

    Returns:
        the results of optimizer after minimizing is finished.
    """
    print(('Initial loss: {:0.4f}'.format(
        self._loss(self.temperature, logits, one_hot_labels))))
    opt = minimize(
        self._loss,
        x0=self.temperature,
        args=(logits, one_hot_labels),
        options={'maxiter': self.max_iterations},
        method=self.solver)
    self.temperature = opt.x[0]

    print(('Final loss: {:0.4f}'.format(
        self._loss(self.temperature, logits, one_hot_labels))))
    print(('Temperature: {:0.2f}'.format(self.temperature)))

    return opt

  def predict(self, logits, temperature=None):
    """Scales logits based on the temperature, returns calibrated probabilities.

    Args:
        logits: raw (non-normalized) predictions that a classification model
          generates, shape=(num_examples, num_classes)
        temperature: temperature to scale logits by

    Returns:
        calibrated softmax probabilities shape=(num_samples, num_classes)
    """

    if not temperature:
      return utils.to_softmax(logits / self.temperature)
    else:
      return utils.to_softmax(logits / temperature)
