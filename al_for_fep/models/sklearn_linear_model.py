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

"""Module for constructing sklearn linear regressor model."""
import os
from typing import Any, Dict

import joblib
from sklearn import linear_model

from al_for_fep.models import makita_model


class SklearnLinearModel(makita_model.MakitaModel):
  """Sklearn linear regression model for Makita concentrated pipelines."""

  def __init__(self, model_hparams,
               tuning_parameters):
    """Initialize model.

    Args:
      model_hparams: Not used. Dictionary of extra string hyperparameter names
        specific to each model type to the parameter values.
      tuning_parameters: Not used. Dictionary of string hyperparameter names to
        ranges of values to use in a grid search for hyperparameter
        optimization.
    """
    super().__init__(model_hparams, tuning_parameters)
    self._model = linear_model.LinearRegression()

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
