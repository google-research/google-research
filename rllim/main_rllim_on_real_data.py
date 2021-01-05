# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""RL-LIM Experiments on real datasets.

Understanding Black-box Model Predictions using RL-LIM.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import lightgbm
import numpy as np
import pandas as pd
from sklearn import linear_model

from rllim import data_loading
from rllim import rllim
from rllim import rllim_metrics


def main(args):
  """Main function of RL-LIM for synthetic data experiments.

  Args:
    args: data_name, train_no, probe_no, test_no,
          seed, hyperparameters, network parameters
  """

  # Problem specification
  problem = args.problem

  # The ratio between training and probe datasets
  train_rate = args.train_rate
  probe_rate = args.probe_rate

  dict_rate = {'train': train_rate, 'probe': probe_rate}

  # Random seed
  seed = args.seed

  # Network parameters
  parameters = dict()
  parameters['hidden_dim'] = args.hidden_dim
  parameters['iterations'] = args.iterations
  parameters['num_layers'] = args.num_layers
  parameters['batch_size'] = args.batch_size
  parameters['batch_size_inner'] = args.batch_size_inner
  parameters['lambda'] = args.hyper_lambda

  # Checkpoint file name
  checkpoint_file_name = args.checkpoint_file_name

  # Number of sample explanations
  n_exp = args.n_exp

  # Loads data
  data_loading.load_facebook_data(dict_rate, seed)

  print('Finished data loading.')

  # Preprocesses data
  # Normalization methods: either 'minmax' or 'standard'
  normalization = args.normalization

  # Extracts features and labels & normalizes features
  x_train, y_train, x_probe, _, x_test, y_test, col_names = \
  data_loading.preprocess_data(normalization,
                               'train.csv', 'probe.csv', 'test.csv')

  print('Finished data preprocess.')

  # Trains black-box model
  # Initializes black-box model
  if problem == 'regression':
    bb_model = lightgbm.LGBMRegressor()
  elif problem == 'classification':
    bb_model = lightgbm.LGBMClassifier()

  # Trains black-box model
  bb_model = bb_model.fit(x_train, y_train)

  print('Finished black-box model training.')

  # Constructs auxiliary datasets
  if problem == 'regression':
    y_train_hat = bb_model.predict(x_train)
    y_probe_hat = bb_model.predict(x_probe)
  elif problem == 'classification':
    y_train_hat = bb_model.predict_proba(x_train)[:, 1]
    y_probe_hat = bb_model.predict_proba(x_probe)[:, 1]

  print('Finished auxiliary dataset construction.')

  # Trains interpretable baseline
  # Defines baseline
  baseline = linear_model.Ridge(alpha=1)

  # Trains baseline model
  baseline.fit(x_train, y_train_hat)

  print('Finished interpretable baseline training.')

  # Trains instance-wise weight estimator
  # Defines locally interpretable model
  interp_model = linear_model.Ridge(alpha=1)

  # Initializes RL-LIM
  rllim_class = rllim.Rllim(x_train, y_train_hat, x_probe, y_probe_hat,
                            parameters, interp_model,
                            baseline, checkpoint_file_name)

  # Trains RL-LIM
  rllim_class.rllim_train()

  print('Finished instance-wise weight estimator training.')

  # Interpretable inference
  # Trains locally interpretable models and output
  # instance-wise explanations (test_coef) and
  # interpretable predictions (test_y_fit)
  test_y_fit, test_coef = rllim_class.rllim_interpreter(x_train, y_train_hat,
                                                        x_test, interp_model)

  print('Finished instance-wise predictions and local explanations.')

  # Overall performance
  mae = rllim_metrics.overall_performance_metrics(y_test, test_y_fit,
                                                  metric='mae')
  print('Overall performance of RL-LIM in terms of MAE: ' +
        str(np.round(mae, 4)))

  # Black-box model predictions
  y_test_hat = bb_model.predict(x_test)

  # Fidelity in terms of MAE
  mae = rllim_metrics.fidelity_metrics(y_test_hat, test_y_fit, metric='mae')
  print('Fidelity of RL-LIM in terms of MAE: ' + str(np.round(mae, 4)))

  # Fidelity in terms of R2 Score
  r2 = rllim_metrics.fidelity_metrics(y_test_hat, test_y_fit, metric='r2')
  print('Fidelity of RL-LIM in terms of R2 Score: ' + str(np.round(r2, 4)))

  # Instance-wise explanations
  # Local explanations of n_exp samples
  local_explanations = test_coef[:n_exp, :]

  final_col_names = np.concatenate((np.asarray(['intercept']), col_names),
                                   axis=0)
  pd.DataFrame(data=local_explanations, index=range(n_exp),
               columns=final_col_names)


if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--problem',
      help='regression or classification',
      default='regression',
      type=str)
  parser.add_argument(
      '--normalization',
      help='minmax or standard',
      default='minmax',
      type=str)
  parser.add_argument(
      '--train_rate',
      help='rate of training samples',
      default=0.9,
      type=float)
  parser.add_argument(
      '--probe_rate',
      help='rate of probe samples',
      default=0.1,
      type=float)
  parser.add_argument(
      '--seed',
      help='random seed',
      default=0,
      type=int)
  parser.add_argument(
      '--hyper_lambda',
      help='main hyper-parameter of RL-LIM (lambda)',
      default=1.0,
      type=float)
  parser.add_argument(
      '--hidden_dim',
      help='dimensions of hidden states',
      default=100,
      type=int)
  parser.add_argument(
      '--num_layers',
      help='number of network layers',
      default=5,
      type=int)
  parser.add_argument(
      '--iterations',
      help='number of iterations',
      default=2000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='number of batch size for RL',
      default=5000,
      type=int)
  parser.add_argument(
      '--batch_size_inner',
      help='number of batch size for inner iterations',
      default=10,
      type=int)
  parser.add_argument(
      '--n_exp',
      help='the number of sample explanations',
      default=5,
      type=int)
  parser.add_argument(
      '--checkpoint_file_name',
      help='file name for saving and loading the trained model',
      default='./tmp/model.ckpt',
      type=str)

  args_in = parser.parse_args()

  # Calls main function
  main(args_in)
