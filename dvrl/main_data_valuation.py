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

"""Main experiment for data valuation.

Main experiment of a data valuation application
using "Data Valuation using Reinforcement Learning (DVRL)"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import keras
import lightgbm
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from dvrl import data_loading
from dvrl import dvrl
from dvrl import dvrl_metrics


def main(args):
  """Main function of DVRL for data valuation experiment.

  Args:
    args: data_name, train_no, valid_no,
          normalization, network parameters, number of examples
  """
  # Data loading and sample corruption
  data_name = args.data_name

  # The number of training and validation samples
  dict_no = dict()
  dict_no['train'] = args.train_no
  dict_no['valid'] = args.valid_no

  # Network parameters
  parameters = dict()
  parameters['hidden_dim'] = args.hidden_dim
  parameters['comb_dim'] = args.comb_dim
  parameters['iterations'] = args.iterations
  parameters['activation'] = tf.nn.relu
  parameters['inner_iterations'] = args.inner_iterations
  parameters['layer_number'] = args.layer_number
  parameters['learning_rate'] = args.learning_rate
  parameters['batch_size'] = args.batch_size
  parameters['batch_size_predictor'] = args.batch_size_predictor

  # The number of examples
  n_exp = args.n_exp

  # Checkpoint file name
  checkpoint_file_name = args.checkpoint_file_name

  # Data loading
  _ = data_loading.load_tabular_data(data_name, dict_no, 0.0)

  print('Finished data loading.')

  # Data preprocessing
  # Normalization methods: 'minmax' or 'standard'
  normalization = args.normalization

  # Extracts features and labels. Then, normalizes features
  x_train, y_train, x_valid, y_valid, x_test, y_test, col_names = \
  data_loading.preprocess_data(normalization, 'train.csv',
                               'valid.csv', 'test.csv')

  print('Finished data preprocess.')

  # Run DVRL
  # Resets the graph
  tf.reset_default_graph()
  keras.backend.clear_session()

  # Here, we assume a classification problem and we assume a predictor model
  # in the form of a simple multi-layer perceptron.
  problem = 'classification'
  # Predictive model define
  pred_model = keras.models.Sequential()
  pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
                                    activation='relu'))
  pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
                                    activation='relu'))
  pred_model.add(keras.layers.Dense(2, activation='softmax'))
  pred_model.compile(optimizer='adam', loss='categorical_crossentropy',
                     metrics=['accuracy'])

  # Flags for using stochastic gradient descent / pre-trained model
  flags = {'sgd': True, 'pretrain': False}

  # Initializes DVRL
  dvrl_class = dvrl.Dvrl(x_train, y_train, x_valid, y_valid,
                         problem, pred_model, parameters,
                         checkpoint_file_name, flags)

  # Trains DVRL
  dvrl_class.train_dvrl('auc')

  print('Finished dvrl training.')

  # Outputs
  # Data valuation
  dve_out = dvrl_class.data_valuator(x_train, y_train)

  print('Finished data valuation.')

  # Evaluations
  # 1. Data valuation
  # Data valuation
  sorted_idx = np.argsort(-dve_out)
  sorted_x_train = x_train[sorted_idx]

  # Indices of top n high valued samples
  print('Indices of top ' + str(n_exp) + ' high valued samples: '
        + str(sorted_idx[:n_exp]))
  print(pd.DataFrame(data=sorted_x_train[:n_exp, :], index=range(n_exp),
                     columns=col_names).head())

  # Indices of top n low valued samples
  print('Indices of top ' + str(n_exp) + ' low valued samples: '
        + str(sorted_idx[-n_exp:]))
  print(pd.DataFrame(data=sorted_x_train[-n_exp:, :], index=range(n_exp),
                     columns=col_names).head())

  # 2. Performance after removing high/low values
  # Here, as the evaluation model, we use LightGBM.
  eval_model = lightgbm.LGBMClassifier()

  # Performance after removing high/low values
  _ = dvrl_metrics.remove_high_low(dve_out, eval_model, x_train, y_train,
                                   x_valid, y_valid, x_test, y_test,
                                   'accuracy', plot=True)


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
      '--inner_iterations',
      help='number of iterations',
      default=100,
      type=int)
  parser.add_argument(
      '--batch_size_predictor',
      help='number of batch size for predictor',
      default=256,
      type=int)
  parser.add_argument(
      '--n_exp',
      help='number of examples',
      default=5,
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
