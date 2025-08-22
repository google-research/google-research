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

"""Definition of experiment hyperparameters."""

from typing import Any

from causal_evaluation import types
import numpy as np

CROSS_VAL_PARAMS = {
    types.ModelType.LOGISTIC_REGRESSION: [{'penalty': [None]}],
    types.ModelType.GRADIENT_BOOSTING: [{
        'max_leaf_nodes': [10, 25, 50],
    }],
    types.ModelType.RIDGE: [
        {'penalty': ['l2'], 'C': list(10.0 ** np.arange(-2, 3))},
        {'penalty': [None]},
    ],
}

DEFAULT_PARAMS = {
    types.ModelType.LOGISTIC_REGRESSION: {'penalty': 'none'},
    types.ModelType.RIDGE: {'penalty': 'l2'},
}


def get_classifier_default_hparams(
    model_type = 'logistic',
    cross_val = False,
):
  """Returns default hyperparameters for different model types.

  Arguments:
    model_type: A string name of a member of types.ModelType.
    cross_val: If False, returns a dict of hyperparameter values. If True,
      returns a dict or list of dicts with lists of parameters as values,
      matching the 'param_grid' input of sklearn.model_selection.GridSearchCV.

  Returns:
    If the cross_val argument is False, returns a dict containing default
    hyperparameters.
    If cross_val is True, returns a dict or list of dicts containing cross
    validation grid parameters, consistent with the 'param_grid' input of
    sklearn.model_selection.GridSearchCV.
  """
  if isinstance(model_type, str):
    model_type = types.ModelType(model_type)

  param_map = CROSS_VAL_PARAMS if cross_val else DEFAULT_PARAMS

  return param_map[model_type] if model_type in param_map else {}
