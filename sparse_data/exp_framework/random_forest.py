# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Train and evaluate random forest classifiers.

Train and evaluate random forest classification models using scikit-learn.
Perform parameter tuning with grid search.
"""

from sklearn import ensemble

from sparse_data.exp_framework.utils import generic_pipeline


def pipeline(x_train,
             y_train,
             x_test,
             y_test,
             param_dict=None,
             problem='classification'):
  """Trains and evaluates a random forest classifier.

  Args:
    x_train: np.array or scipy.sparse.*matrix array of features of training data
    y_train: np.array 1-D array of class labels of training data
    x_test: np.array or scipy.sparse.*matrix array of features of test data
    y_test: np.array 1-D array of class labels of the test data
    param_dict: {string: ?} dictionary of parameters and their values
    problem: string type of learning problem; values = 'classification',
      'regression'

  Returns:
    model: sklearn.ensemble.RandomForestClassifier
      trained random forest model
    metrics: {str: float}
      dictionary of metric scores
  """
  assert problem in ['classification', 'regression']

  if param_dict is None:
    param_dict = {}

  if problem == 'regression':
    model = ensemble.RandomForestRegressor(**param_dict)
  else:
    model = ensemble.RandomForestClassifier(**param_dict)

  return generic_pipeline(
      model, x_train, y_train, x_test, y_test, problem=problem)
