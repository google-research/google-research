# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""RL-LIM Experiments on thres synthetic datasets.
"""

# Necessary functions and packages call
from __future__ import print_function
import argparse

from data_loading import synthetic_data_loading

import numpy as np

from rllim import rllim
from utils import metrics


def main(args):
  """Main function of RL-LIM for synthetic data experiments.

  Args:
    args: data_name, data_no, seed, hyperparams, network parameters

  Returns:
    awd: Absolute Weight Difference
  """

  # Inputs
  data_name = args.data_name
  data_no = args.data_no
  seed = args.seed
  hyperparam = args.hyperparam

  # Network parameters
  parameters = dict()
  parameters['hidden_dim'] = args.hidden_dim
  parameters['iterations'] = args.iterations
  parameters['layer_number'] = args.layer_number
  parameters['batch_size'] = args.batch_size
  parameters['batch_size_small'] = args.batch_size_small

  # Data loading
  train_x, train_y_hat, test_x, test_y_hat, test_c, test_idx = \
      synthetic_data_loading(data_name, data_no, seed)

  print('Finish ' + str(data_name) + ' data loading')

  # Fits RL-LIM
  np.random.seed(seed)
  idx = np.random.permutation(len(train_y_hat))
  train_idx = idx[:int(0.9*len(train_y_hat))]
  valid_idx = idx[int(0.9*len(train_y_hat)):]

  valid_x = train_x[valid_idx, :]
  valid_y_hat = train_y_hat[valid_idx]

  train_x = train_x[train_idx, :]
  train_y_hat = train_y_hat[train_idx]

  test_y_fit, test_coef = rllim(train_x, train_y_hat,
                                valid_x, valid_y_hat,
                                test_x, parameters, hyperparam)

  # Performance evaluation
  _, awd = metrics(test_y_hat, test_y_fit, test_coef, test_c, test_idx)

  print('AWD' + str(np.round(awd, 4)))

  return awd


#%%
if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--data_name',
      help='Synthetic data name (Syn1 or Syn2 or Syn3)',
      default='Syn1',
      type=str)
  parser.add_argument(
      '--data_no',
      help='number of training and testing samples',
      default=1000,
      type=int)
  parser.add_argument(
      '--seed',
      help='random seed',
      default=0,
      type=int)
  parser.add_argument(
      '--hyperparam',
      help='hyper-parameter lambda',
      default=1.0,
      type=float)
  parser.add_argument(
      '--hidden_dim',
      help='dimensions of hidden states',
      default=100,
      type=int)
  parser.add_argument(
      '--layer_number',
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
      '--batch_size_small',
      help='number of batch size for inner iterations',
      default=10,
      type=int)

  args_in = parser.parse_args()

  # Calls main function
  awd_out = main(args_in)
