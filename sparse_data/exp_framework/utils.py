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

"""Broadly useful functions."""

import itertools
import numpy as np
from sparse_data.exp_framework.evaluate import classification_metrics
from sparse_data.exp_framework.evaluate import regression_metrics


def generic_pipeline(estimator,
                     x_train,
                     y_train,
                     x_test,
                     y_test,
                     problem='classification'):
  """Pipeline for training and testing an estimator.

  Args:
    estimator: Estimator that has fit and predict method.
    x_train: Training features.
    y_train: Training labels.
    x_test: Testing features.
    y_test: Testing labels.
    problem: Type of problem - classification or regression.

  Returns:
    estimator: Trained estimator.
    metrics: training and testing metrics.
  """
  assert problem in ['classification', 'regression']

  estimator.fit(x_train, y_train)

  if problem == 'regression':
    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)
    metrics = regression_metrics(y_train, y_train_pred, y_test, y_test_pred)
  else:
    y_train_proba = estimator.predict_proba(x_train)
    y_test_proba = estimator.predict_proba(x_test)
    labels = list(set(y_train))
    metrics = classification_metrics(y_train, y_train_proba, y_test,
                                     y_test_proba, labels)

  return estimator, metrics


def generate_param_configs(param_grid, num_iteration, seed=1):
  """Generate a list of parameter configurations from grid of parameter values.

  Uses exhaustive grid or random search depending on the size of the space
  of the search parameter values, and the number of searches specified.

  Args:
    param_grid: {str: [?]} dictionary of parameters and all possible values
    num_iteration: number of iterations (searches)
    seed: int seed value for reproducibility in random sampling

  Returns:
    out: [{str: ?}]
      a list of parameter configurations represented as dictionaries
  """
  rng = np.random.RandomState()
  rng.seed(seed)

  out = []
  num_param_config = np.prod([len(v) for v in param_grid.values()])
  if num_param_config <= num_iteration:  # exhaustive grid
    for values in itertools.product(*param_grid.values()):
      out.append({k: v for k, v in zip(param_grid.keys(), values)})
    assert len(out) <= num_iteration
  else:  # random
    for _ in range(num_iteration):
      out.append({k: v[rng.randint(len(v))] for k, v in param_grid.items()})
    assert len(out) == num_iteration
  return out
