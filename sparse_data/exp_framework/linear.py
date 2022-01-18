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

"""Train and evaluate logistic regression classifiers.

Train and evaluate logistic regression classification models using scikit-learn.
Perform parameter tuning with grid search.
"""

from sklearn import linear_model
from sklearn import preprocessing

from sparse_data.exp_framework.utils import generic_pipeline


def choose_linear_model(problem, penalty=None):
  """Choose a linear model based on the learning problem and parameters.

  Args:
    problem: string type of learning problem; values = 'classification',
      'regression'
    penalty: string type of regularization

  Returns:
    init_method: initialization method for sklearn.linear_model.*

  Raises:
    ValueError: if value of `penalty` is unknown
  """
  assert problem in ['classification', 'regression']

  if problem == 'classification':
    return linear_model.LogisticRegression
  elif penalty == 'l1':
    return linear_model.Lasso
  elif penalty == 'l2':
    return linear_model.Ridge
  else:
    raise ValueError('Unknown penalty for linear model: {}'.format(penalty))


def pipeline(x_train,
             y_train,
             x_test,
             y_test,
             param_dict=None,
             problem='classification'):
  """Trains and evaluates a logistic regression classifier.

  Args:
    x_train: np.array or scipy.sparse.*matrix array of features of training data
    y_train: np.array 1-D array of class labels of training data
    x_test: np.array or scipy.sparse.*matrix array of features of test data
    y_test: np.array 1-D array of class labels of the test data
    param_dict: {string: ?} dictionary of parameters and their values
    problem: string type of learning problem; values = 'classification',
      'regression'

  Returns:
    model: sklearn.linear_model.*
      trained linear model
    metrics: {str: float}
      dictionary of metric scores
  """
  assert problem in ['classification', 'regression']

  if param_dict is None:
    param_dict = {}

  if problem == 'classification':
    scaler = preprocessing.MaxAbsScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

  if 'penalty' in param_dict and problem == 'regression':
    penalty = param_dict.pop('penalty')
  elif 'penalty' in param_dict:
    penalty = param_dict['penalty']
  else:
    penalty = 'l2'  # default to l2

  model_init = choose_linear_model(problem, penalty)
  model = model_init(**param_dict)

  return generic_pipeline(
      model, x_train, y_train, x_test, y_test, problem=problem)
