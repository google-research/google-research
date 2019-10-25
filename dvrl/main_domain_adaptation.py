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

"""Main experiment for domain adaptation.

Main experiment of a domain adaptation application
using "Data Valuation using Reinforcement Learning (DVRL)"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import lightgbm
import numpy as np
import tensorflow as tf

from dvrl import data_loading
from dvrl import dvrl
from dvrl import dvrl_metrics


def main(args):
  """Main function of DVRL for domain adaptation experiment.

  Args:
    args: train_no, valid_no,
          normalization, network parameters
  """

  # Data loading
  # The number of training and validation samples
  dict_no = dict()
  dict_no['source'] = args.train_no
  dict_no['valid'] = args.valid_no

  # Setting and target store type
  setting = 'train-on-rest'
  target_store_type = 'B'

  # Network parameters
  parameters = dict()
  parameters['hidden_dim'] = args.hidden_dim
  parameters['comb_dim'] = args.comb_dim
  parameters['iterations'] = args.iterations
  parameters['activation'] = tf.nn.tanh
  parameters['layer_number'] = args.layer_number
  parameters['batch_size'] = args.batch_size
  parameters['learning_rate'] = args.learning_rate

  # Checkpoint file name
  checkpoint_file_name = args.checkpoint_file_name

  # Data loading
  data_loading.load_rossmann_data(dict_no, setting, target_store_type)

  print('Finished data loading.')

  # Data preprocessing
  # Normalization methods: 'minmax' or 'standard'
  normalization = args.normalization

  # Extracts features and labels. Then, normalizes features
  x_source, y_source, x_valid, y_valid, x_target, y_target, _ = \
      data_loading.preprocess_data(normalization,
                                   'source.csv', 'valid.csv', 'target.csv')

  print('Finished data preprocess.')

  # Run DVRL
  # Resets the graph
  tf.reset_default_graph()

  problem = 'regression'
  # Predictor model definition
  pred_model = lightgbm.LGBMRegressor()

  # Flags for using stochastic gradient descent / pre-trained model
  flags = {'sgd': False, 'pretrain': False}

  # Initializes DVRL
  dvrl_class = dvrl.Dvrl(x_source, y_source, x_valid, y_valid,
                         problem, pred_model, parameters,
                         checkpoint_file_name, flags)

  # Trains DVRL
  dvrl_class.train_dvrl('rmspe')

  print('Finished dvrl training.')

  # Outputs
  # Data valuation
  dve_out = dvrl_class.data_valuator(x_source, y_source)

  print('Finished date valuation.')

  # Evaluations
  # Evaluation model
  eval_model = lightgbm.LGBMRegressor()

  # DVRL-weighted learning
  dvrl_perf = dvrl_metrics.learn_with_dvrl(dve_out, eval_model,
                                           x_source, y_source,
                                           x_valid, y_valid,
                                           x_target, y_target, 'rmspe')

  # Baseline prediction performance (treat all training samples equally)
  base_perf = dvrl_metrics.learn_with_baseline(eval_model,
                                               x_source, y_source,
                                               x_target, y_target, 'rmspe')

  print('Finish evaluation.')
  print('DVRL learning performance: ' + str(np.round(dvrl_perf, 4)))
  print('Baseline performance: ' + str(np.round(base_perf, 4)))


if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--normalization',
      choices=['minmax', 'standard'],
      help='data normalization method',
      default='minmax',
      type=str)
  parser.add_argument(
      '--train_no',
      help='number of training samples',
      default=667027,
      type=int)
  parser.add_argument(
      '--valid_no',
      help='number of validation samples',
      default=8443,
      type=int)
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
      default=1000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='number of batch size for RL',
      default=50000,
      type=int)
  parser.add_argument(
      '--learning_rate',
      help='learning rates for RL',
      default=0.001,
      type=float)
  parser.add_argument(
      '--checkpoint_file_name',
      help='file name for saving and loading the trained model',
      default='./tmp/model.ckpt',
      type=str)

  args_in = parser.parse_args()

  # Calls main function
  main(args_in)
