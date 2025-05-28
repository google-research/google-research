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

"""Module for constructing sklearn GaussianProcess regressor model."""
import os
from typing import Any, Dict

import joblib
import numpy as np
from sklearn import gaussian_process

from al_for_fep.models import makita_model


def _tanimoto_similarity(a, b):
  """Computes the Tanimoto similarity for all pairs.

  Args:
    a: Numpy array with shape [batch_size_a, num_features].
    b: Numpy array with shape [batch_size_b, num_features].

  Returns:
    Numpy array with shape [batch_size_a, batch_size_b].
  """
  aa = np.sum(a, axis=1, keepdims=True)
  bb = np.sum(b, axis=1, keepdims=True)
  ab = np.matmul(a, b.T)
  return np.true_divide(ab, aa + bb.T - ab)


class TanimotoKernel(gaussian_process.kernels.NormalizedKernelMixin,
                     gaussian_process.kernels.StationaryKernelMixin,
                     gaussian_process.kernels.Kernel):
  """Custom Gaussian process kernel that computes Tanimoto similarity."""

  def __init__(self):
    """Initializer."""
    pass  # Does nothing; this is required by get_params().

  def __call__(self, X, Y=None, eval_gradient=False):  # pylint: disable=invalid-name
    """Computes the pairwise Tanimoto similarity.

    Args:
      X: Numpy array with shape [batch_size_a, num_features].
      Y: Numpy array with shape [batch_size_b, num_features]. If None, X is
        used.
      eval_gradient: Whether to compute the gradient.

    Returns:
      Numpy array with shape [batch_size_a, batch_size_b].

    Raises:
      NotImplementedError: If eval_gradient is True.
    """
    if eval_gradient:
      raise NotImplementedError
    if Y is None:
      Y = X
    return _tanimoto_similarity(X, Y)


class SklearnGaussianProcessModel(makita_model.MakitaModel):
  """Sklearn Gaussian Process regression model for Makita concentrated pipelines."""

  def __init__(self, model_hparams,
               tuning_parameters):
    """Initialize model.

    Args:
      model_hparams: Dictionary of extra string hyperparameter names specific to
        each model type to the parameter values.
      tuning_parameters: Not used in this model. Dictionary of string
        hyperparameter names to ranges of values to use in a grid search for
        hyperparameter optimization.
    """
    super().__init__(model_hparams, tuning_parameters)
    self._model = gaussian_process.GaussianProcessRegressor(
        kernel=TanimotoKernel(), **self._model_hparams)

  def save(self, save_dir):
    """Saves the model to the supplied location.

    Args:
      save_dir: Location to save the model to.
    """
    with open(os.path.join(save_dir, 'model.joblib'), 'wb') as model_out:
      joblib.dump(self._model, model_out)

  def get_model(self,):
    """Method to access the protected model.

    Returns:
      The protected model.
    """
    return self._model
