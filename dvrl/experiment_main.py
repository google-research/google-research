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

"""Main experiments of DVRL on Adult and Blog datasets.

(A) Standard supervised learning experiment
- Set noise_rate = 0.0
- Output: Remove high/low values performances  (Figure 2)

(B) Corrupted sample discovery & Remove high/low values experiment
- Set noise_rate = 0.2
- Output:
    (1) Noise label discovery (Figure 4 and 9)
    (2) Remove high/low values performances (Figure 3 and 8)

Inputs:
- noise_amount: 0.0 or 0.2
- data_name: adult, blog

Outputs:
- Corrupted sample discovery TPR with various fractions of data inspections
- Prediction performance after removing high/low value samples
"""

# Necessary packages and function call
from __future__ import print_function
import argparse
import warnings

# Data Loading
from data_loading import tabular_data_loading

# DVRL
from dvrl import dvrl

# Metrics
from metrics import noise_label_discovery
from metrics import remove_high_low

import numpy as np

# Label corruption
from utils import label_corruption

warnings.filterwarnings('ignore')


#%% main iterations
def main(args):
  """Main function for corrupted sample discovery.

  Args:
    args: data_name, dict_no, noise_rate, parameters

  Returns:
    noise_discovery_performance
    remove_high_low_performance
  """

  # Parameters
  data_name = args.data_name
  noise_rate = args.noise_rate

  dict_no = dict()
  dict_no['train'] = args.train_no
  dict_no['valid'] = args.valid_no

  parameters = dict()
  parameters['hidden_dim'] = args.hidden_dim
  parameters['iterations'] = args.iterations
  parameters['layer_number'] = args.layer_number
  parameters['batch_size'] = args.batch_size

  # Data loading
  print('Data name: ' + data_name)

  x_train, y_train, x_valid, y_valid, x_test, y_test = \
      tabular_data_loading(data_name, dict_no)

  # Add noise
  y_train, _, noise_idx = label_corruption(y_train, noise_rate)

  print('Data loaded... Noise added...')

  # Apply DVRL
  dve_out = dvrl(x_train, y_train, x_valid, y_valid, parameters, 'log_loss')

  print('Finish date valuation...')

  # Evaluations
  # (1) Noise label discovery
  noise_discovery_performance = np.zeros([20, 2])
  if noise_rate > 0:
    noise_discovery_performance = noise_label_discovery(dve_out, noise_idx)

  print('Finish corrupted data discovery metric computation...')
  print('Results of Noisy sample discovery: ')
  print(noise_discovery_performance)

  # (2) Remove high/low value data
  remove_high_low_performance = remove_high_low(dve_out, x_train, y_train,
                                                x_test, y_test)

  print('Finish removing high/low value data metric computation...')
  print('Results of removing high/low value data experiments: ')
  print(remove_high_low_performance)

  return noise_discovery_performance, remove_high_low_performance


if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--data_name',
      help='input_data_name (adult or blog)',
      default='adult',
      type=str)
  parser.add_argument(
      '--train_no',
      help='the number of training samples',
      default=1000,
      type=int)
  parser.add_argument(
      '--valid_no',
      help='the number of validation samples',
      default=400,
      type=int)
  parser.add_argument(
      '--noise_rate',
      help='additional noise ratio',
      default=0.2,
      type=float)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions',
      default=100,
      type=int)
  parser.add_argument(
      '--layer_number',
      help='number of layers',
      default=5,
      type=int)
  parser.add_argument(
      '--iterations',
      help='number of iterations for RL',
      default=5000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='number of mini-batch for RL',
      default=2000,
      type=int)

  args_in = parser.parse_args()

  # Calls main function
  noise_discovery_perf, remove_high_low_perf = main(args_in)
