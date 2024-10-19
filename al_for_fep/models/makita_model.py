# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Base class for models for concentrated Makita pipelines."""
import abc
from typing import Any, Dict


class MakitaModel(abc.ABC):
  """Base class for models for concentrated Makita pipelines."""

  def __init__(self, model_hparams,
               tuning_parameters):
    """Initialize model.

    Args:
      model_hparams: Dictionary of extra string hyperparameter names specific to
        each model type to the parameter values.
      tuning_parameters: Dictionary of string hyperparameter names to ranges of
        values to use in a grid search for hyperparameter optimization.
    """
    super().__init__()
    self._model_hparams = model_hparams
    self._tuning_parameters = tuning_parameters
    self._model = None

  @abc.abstractmethod
  def save(self, save_dir):
    """Saves the model to the supplied location.

    Args:
      save_dir: Base directory to save the model in.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_model(self,):
    """Method to access the protected model.

    Returns:
      The protected model.
    """
    return self._model
