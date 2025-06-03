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

"""Classifiers and sklearn pipelines for training."""

from causal_evaluation import types
import sklearn
import sklearn.base
import sklearn.calibration
import sklearn.ensemble
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline


MODELS = {
    types.ModelType.LOGISTIC_REGRESSION: (
        sklearn.linear_model.LogisticRegression
    ),
    types.ModelType.RIDGE: sklearn.linear_model.LogisticRegression,
    types.ModelType.GRADIENT_BOOSTING: (
        sklearn.ensemble.HistGradientBoostingClassifier
    ),
}


def get_classifier(
    model_type = 'logistic', **model_kwargs
):
  """Return a Scikit-learn model or pipeline.

  Arguments:
    model_type: A string name of a member of types.ModelType.
    **model_kwargs: Additional keyword arguments passed to the model
      constructor.

  Returns:
    A sklearn estimator.
  """
  if isinstance(model_type, str):
    model_type = types.ModelType(model_type)

  model = MODELS[model_type](**model_kwargs)

  return model
