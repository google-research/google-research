# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Trains and evaluates gradient boosted decision tree (GBDT) classifiers.

Trains and evaluates GBDT models using XGBoost. Visualizes tree structures.
"""

import os
from absl import logging
import matplotlib.pyplot as plt
import xgboost as xgb
from sparse_data.exp_framework.utils import generic_pipeline

FILES_PATH = '/trees/out'


def get_objective(is_binary):
  """Gets classifier objective string which depends on if the problem is binary.

  Args:
    is_binary: boolean whether classification task is binary

  Returns:
    objective: string
  """
  if is_binary:
    return 'binary:logistic'
  else:
    return 'multi:softprob'


def pipeline(x_train,
             y_train,
             x_test,
             y_test,
             param_dict=None,
             problem='classification'):
  """Runs a pipeline to train and evaluate GBDT classifiers.

  Args:
    x_train: np.array or scipy.sparse.*matrix array of features of training data
    y_train: np.array 1-D array of class labels of training data
    x_test: np.array or scipy.sparse.*matrix array of features of test data
    y_test: np.array 1-D array of class labels of the test data
    param_dict: {string: ?} dictionary of parameters and their values
    problem: string type of learning problem; values = 'classification',
      'regression'

  Returns:
    model: xgb.Booster
      trained XGBoost gradient boosted trees model
    metrics: {str: float}
      dictionary of metric scores
  """
  assert problem in ['classification', 'regression']

  if param_dict is None:
    param_dict = {}

  if problem == 'regression':
    model = xgb.XGBRegressor(**param_dict)
  else:
    is_binary = max(y_train) + 1 == 2
    if 'objective' not in param_dict:
      param_dict['objective'] = get_objective(is_binary)
    model = xgb.XGBClassifier(**param_dict)

  return generic_pipeline(
      model, x_train, y_train, x_test, y_test, problem=problem)


def plot_tree(model, directory, num_tree=10):
  """Creates and saves a plot of the trees in a gradient boosted tree model.

  Args:
    model: xgb.Booster trained XGBoost gradient boosted trees model
    directory: string directory of save location
    num_tree: number of trees to plot
  """
  base_path = '{}/{}'.format(FILES_PATH, directory)
  os.makedirs(base_path)

  for tree_idx in range(num_tree):
    xgb.plot_tree(model, num_trees=tree_idx)
    fig = plt.gcf()
    fig.set_size_inches(120, 120)

    path = '{}/tree-{}.png'.format(base_path, tree_idx)
    fig.savefig(path)

  logging.info('Saved plots to: %s \n', base_path)
  plt.close('all')
