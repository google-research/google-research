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

"""Error metrics for evaluation of algorithms.

The metrics provided include thresholded root mean square relative error
(RMSRE_tau) and root mean square error (RMSE).
"""

import abc
import numpy as np
import pandas as pd


class ErrorMetric(abc.ABC):
  """Base class for error metrics."""

  @abc.abstractmethod
  def error(self, bias, variance,
            true_value):
    """Compute the error of a series of random variables estimating values.

    Args:
      bias: the bias of each random variable
      variance: the variance of each random variable
      true_value: the values being estimated

    Returns:
      The error of each estimate, where different error metrics are used
      in different instantiations of this abstract class.
    """

  @abc.abstractmethod
  def avg_error(self, errors):
    """Compute the average of the given error values.

    The type of average (e.g. arithmetic mean or root mean square) is
    instantiated in subclasses and depends on which error metric is used.

    Args:
      errors: a list or series of errors of individual estimates

    Returns: the average error
    """


class L2Metric(ErrorMetric):
  """Base class for L2-type error metrics with root mean square averaging."""

  def avg_error(self, errors):
    return np.linalg.norm(errors) / (len(errors)**0.5)


class RMSRETauMetric(L2Metric):
  """Thresholded root mean square relative error metric."""

  def __init__(self, tau):
    self.tau = tau

  def error(self, bias, variance,
            true_value):
    return ((bias**2 + variance) /
            true_value.clip(lower=self.tau)**2)**0.5


class RMSEMetric(L2Metric):
  """Root mean square error metric."""

  def error(self, bias, variance,
            true_value):
    return (bias**2 + variance)**0.5
