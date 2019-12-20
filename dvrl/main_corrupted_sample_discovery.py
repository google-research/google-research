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

"""Main experiment for corrupted sample discovery and robust learning.

Main experiment of corrupted sample discovery and robust learning applications
using "Data Valuation using Reinforcement Learning (DVRL)"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import lightgbm
import numpy as np
from sklearn import linear_model
import tensorflow.compat.v1 as tf

from dvrl import data_loading
from dvrl import dvrl
from dvrl import dvrl_metrics


def main(args):
  """Main function of DVRL for corrupted sample discovery experiment.

  Args:
    args: data_name, train_no, valid_no, noise_rate,
          normalization, network parameters
  """
  # Data loading and sample corruption
  data_name = args.data_name

  # The number of training and validation samples
  dict_no = dict()
  dict_no['train'] = args.train_no
  dict_no['valid'] = args.valid_no

  # Additional noise ratio
  noise_rate = args.noise_rate

  # Checkpoint file name
  checkpoint_file_name = args.checkpoint_file_name

  # Data loading and label corruption
  noise_idx = data_loading.load_tabular_data(data_name, dict_no, noise_rate)
  # noise_idx: ground truth noisy label indices

  print('Finished data loading.')

  # Data preprocessing
  # Normalization methods: 'minmax' or 'standard'
  normalization = args.normalization

  # Extracts features and labels. Then, normalizes features
  x_train, y_train, x_valid, y_valid, x_test, y_test, _ = \
  data_loading.preprocess_data(normalization, 'train.csv',
                               'valid.csv', 'test.csv')

  print('Finished data preprocess.')

  # Run DVRL
  # Resets the graph
  tf.reset_default_graph()

  # Network parameters
  parameters = dict()
  parameters['hidden_dim'] = args.hidden_dim
  parameters['comb_dim'] = args.comb_dim
  parameters['activation'] = tf.nn.relu
  parameters['iterations'] = args.iterations
  parameters['layer_number'] = args.layer_number
  parameters['batch_size'] = args.batch_size
  parameters['learning_rate'] = args.learning_rate

  # In this example, we consider a classification problem and we use Logistic
  # Regression as the predictor model.
  problem = 'classification'
  pred_model = linear_model.LogisticRegression(solver='lbfgs')

  # Flags for using stochastic gradient descent / pre-trained model
  flags = {'sgd': False, 'pretrain': False}

  # Initalizes DVRL
  dvrl_class = dvrl.Dvrl(x_train, y_train, x_valid, y_valid,
                         problem, pred_model, parameters,
                         checkpoint_file_name, flags)

  # Trains DVRL
  dvrl_class.train_dvrl('auc')

  print('Finished dvrl training.')

  # Outputs
  # Data valuation
  dve_out = dvrl_class.data_valuator(x_train, y_train)

  print('Finished date valuation.')

  # Evaluations
  # Evaluation model
  eval_model = lightgbm.LGBMClassifier()

  # 1. Robust learning (DVRL-weighted learning)
  robust_perf = dvrl_metrics.learn_with_dvrl(dve_out, eval_model,
                                             x_train, y_train, x_valid, y_valid,
                                             x_test, y_test, 'accuracy')

  print('DVRL-weighted learning performance: ' + str(np.round(robust_perf, 4)))

  # 2. Performance after removing high/low values
  _ = dvrl_metrics.remove_high_low(dve_out, eval_model, x_train, y_train,
                                   x_valid, y_valid, x_test, y_test,
                                   'accuracy', plot=True)

  # 3. Corrupted sample discovery
  # If noise_rate is positive value.
  if noise_rate > 0:
    # Evaluates corrupted_sample_discovery
    # and plot corrupted sample discovery results
    _ = dvrl_metrics.discover_corrupted_sample(dve_out,
                                               noise_idx, noise_rate, plot=True)


if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--data_name',
      choices=['adult', 'blog'],
      help='data name (adult or blog)',
      default='adult',
      type=str)
  parser.add_argument(
      '--normalization',
      choices=['minmax', 'standard'],
      help='data normalization method',
      default='minmax',
      type=str)
  parser.add_argument(
      '--train_no',
      help='number of training samples',
      default=1000,
      type=int)
  parser.add_argument(
      '--valid_no',
      help='number of validation samples',
      default=400,
      type=int)
  parser.add_argument(
      '--noise_rate',
      help='label corruption ratio',
      default=0.2,
      type=float)
  parser.add_argument(
      '--hidden_dim',
      help='dimensions of hidden states',
      default=100,
      type=int)
  parser.add_argument(
      '--comb_dim',
      help='dimensions of hidden states after combinding with prediction diff',
      default=10,
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
      default=2000,
      type=int)
  parser.add_argument(
      '--learning_rate',
      help='learning rates for RL',
      default=0.01,
      type=float)
  parser.add_argument(
      '--checkpoint_file_name',
      help='file name for saving and loading the trained model',
      default='./tmp/model.ckpt',
      type=str)

  args_in = parser.parse_args()

  # Calls main function
  main(args_in)
