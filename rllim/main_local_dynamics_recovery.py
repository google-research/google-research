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

"""RL-LIM Experiments on three synthetic datasets.

Recovering local dynamics using RL-LIM with Synthetic datasets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
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

  # Inputs
  data_name = args.data_name

  # The number of training, probe and testing samples
  train_no = args.train_no
  probe_no = args.probe_no
  test_no = args.test_no
  dim_no = args.dim_no

  dict_no = {'train': train_no, 'probe': probe_no, 'test': test_no,
             'dim': dim_no}

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

  # Loads data
  x_train, y_train, x_probe, y_probe, x_test, y_test, c_test = \
      data_loading.load_synthetic_data(data_name, dict_no, seed)

  print('Finish ' + str(data_name) + ' data loading')

  # Trains interpretable baseline
  # Defins baseline
  baseline = linear_model.Ridge(alpha=1)

  # Trains interpretable baseline model
  baseline.fit(x_train, y_train)

  print('Finished interpretable baseline training.')

  # Trains instance-wise weight estimator
  # Defines locally interpretable model
  interp_model = linear_model.Ridge(alpha=1)

  # Initializes RL-LIM
  rllim_class = rllim.Rllim(x_train, y_train, x_probe, y_probe,
                            parameters, interp_model,
                            baseline, checkpoint_file_name)

  # Trains RL-LIM
  rllim_class.rllim_train()

  print('Finished instance-wise weight estimator training.')

  # Interpretable inference
  # Trains locally interpretable models and output
  # instance-wise explanations (test_coef)
  # and interpretable predictions (test_y_fit)
  test_y_fit, test_coef = \
      rllim_class.rllim_interpreter(x_train, y_train, x_test, interp_model)

  print('Finished interpretable predictions and local explanations.')

  # Fidelity
  mae = rllim_metrics.fidelity_metrics(y_test, test_y_fit, metric='mae')
  print('fidelity of RL-LIM in terms of MAE: ' + str(np.round(mae, 4)))

  # Absolute Weight Differences (AWD) between ground truth local dynamics and
  # estimated local dynamics by RL-LIM
  awd = rllim_metrics.awd_metric(c_test, test_coef)
  print('AWD of RL-LIM: ' + str(np.round(awd, 4)))

  # Fidelity plot
  rllim_metrics.plot_result(x_test, data_name, y_test, test_y_fit,
                            c_test, test_coef,
                            metric='mae', criteria='Fidelity')

  # AWD plot
  rllim_metrics.plot_result(x_test, data_name, y_test, test_y_fit,
                            c_test, test_coef,
                            metric='mae', criteria='AWD')


if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--data_name',
      help='Synthetic data name (Syn1 or Syn2 or Syn3)',
      default='Syn1',
      type=str)
  parser.add_argument(
      '--train_no',
      help='number of training samples',
      default=1000,
      type=int)
  parser.add_argument(
      '--probe_no',
      help='number of probe samples',
      default=100,
      type=int)
  parser.add_argument(
      '--test_no',
      help='number of testing samples',
      default=1000,
      type=int)
  parser.add_argument(
      '--dim_no',
      help='number of feature dimensions',
      default=11,
      type=int)
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
      default=900,
      type=int)
  parser.add_argument(
      '--batch_size_inner',
      help='number of batch size for inner iterations',
      default=10,
      type=int)
  parser.add_argument(
      '--checkpoint_file_name',
      help='file name for saving and loading the trained model',
      default='./tmp/model.ckpt',
      type=str)

  args_in = parser.parse_args()

  # Calls main function
  main(args_in)
