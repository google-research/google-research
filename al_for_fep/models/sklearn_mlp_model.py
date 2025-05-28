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
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn import neural_network

from al_for_fep.models import makita_model


class SklearnMLPModel(makita_model.MakitaModel):
  """Sklearn MultiLayer Perceptron regression model for Makita concentrated pipelines."""

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
    self._model = neural_network.MLPRegressor(**self._model_hparams)

  def compile(self, compile_params):
    """Compiles the model with the supplied parameters.

    Not used for linear regression in sklearn.

    Args:
      compile_params: Dictionary of extra string hyperparameter names specific
        to each model type to the parameter values to be used for compiling.

    Returns:
      Self.
    """
    return self

  def fit(self, training_data,
          fit_params):
    """Fits the model with the supplied data.

    Args:
      training_data: The data to be fit. Sklearn RF expects training_data to be
        a tuple of np.ndarray's of shapes (n_samples, n_features) and
        (n_samples) corresponding to inputs and outputs of the training data.
      fit_params: Dictionary of additional parameters to be used during fitting.
        Not used for Sklearn Linear Model.

    Returns:
      Self.
    """
    self._model.fit(training_data[0], training_data[1])
    return self

  def predict(self, testing_data,
              predict_params):
    """Makes predictions on the supplied data with the model.

    Args:
      testing_data: The data on which to perform inference. Sklearn RF expects
        testing_data to be a np.ndarray of shape (n_samples, n_features)
        corresponding to inputs of the testing data.
      predict_params: Dictionary of additional parameters to be used during
        inference. Not used for Sklearn Linear Model.

    Returns:
      An array of predictions.
    """
    return self._model.predict(testing_data)

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
