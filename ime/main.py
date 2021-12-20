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

"""Main experiments for interpretable mixture of experts on time series and tabular data."""
import argparse
import os
import pathlib

from exp.exp_baseline import ExpBaseline
from exp.exp_baseline_tabular import ExpBaselineTabular
from exp.exp_IME import ExpIME
from exp.exp_IME_tabular import ExpIMETabular
import pandas as pd
import torch
from utils.tools import seed_torch

# Setting a seed for reproducibility
seed_torch(42)

parser = argparse.ArgumentParser(description='Long Sequences Forecasting')

parser.add_argument(
    '--model',
    type=str,
    required=True,
    default='LSTM',
    help='model of experiment, options: [LSTM, Linear, ARNet, IME_WW,IME_BW]')

# Data specific parameters
parser.add_argument(
    '--data', type=str, required=True, default='ECL', help='data')
parser.add_argument(
    '--output_dir', type=str, default='./outputs/', help='output directory')
parser.add_argument(
    '--root_path',
    type=str,
    default='./data/',
    help='root path of the data file')
parser.add_argument(
    '--features',
    type=str,
    default='S',
    help='forecasting task, options:[S, MS]; S:univariate predict univariate, MS:multivariate predict univariate'
)
parser.add_argument(
    '--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument(
    '--num_ts',
    type=int,
    default=50,
    help='Number Of Independent Time Series to Model')
parser.add_argument(
    '--scale', type=bool, default=True, help='Scaling the dataset')
parser.add_argument(
    '--plot',
    type=bool,
    default=True,
    help='Plotting Different losses for debugging purposes')
parser.add_argument(
    '--plot_dir', type=str, default='./Graphs/', help='output directory')

# Experiment specific parameters
parser.add_argument(
    '--checkpoints',
    type=str,
    default='./checkpoints/',
    help='location of model checkpoints')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument(
    '--use_multi_gpu',
    action='store_true',
    help='use multiple gpus',
    default=False)
parser.add_argument(
    '--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# Model specific parameters
parser.add_argument(
    '--train_epochs', type=int, default=500, help='train epochs')
parser.add_argument(
    '--batch_size',
    type=int,
    default=512,
    help='batch size of train input data')
parser.add_argument(
    '--patience', type=int, default=10, help='early stopping patience')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.0001,
    help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument(
    '--seq_len', type=int, default=168, help='input sequence length')
parser.add_argument(
    '--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--input_dim', type=int, default=7, help='input size')
parser.add_argument('--output_dim', type=int, default=7, help='output size')
parser.add_argument(
    '--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--layers', type=int, default=2, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument(
    '--depth', type=int, default=5, help='depth of the decision tree')

# IME Specific parameters
parser.add_argument(
    '--learning_rate_gate',
    type=float,
    default=0.001,
    help='optimizer learning rate for gate network')
parser.add_argument(
    '--accuracy_hp',
    type=float,
    default=1,
    help='Accuracy Loss Hyperparameter for IME')
parser.add_argument(
    '--diversity_hp',
    type=float,
    default=1,
    help='Diversity Loss Hyperparameter for IME')
parser.add_argument(
    '--utilization_hp',
    type=float,
    default=1,
    help='Utilization Loss Hyperparameter for IME')
parser.add_argument(
    '--smoothness_hp',
    type=float,
    default=1,
    help='Smoothness Loss Hyperparameter for IME')
parser.add_argument(
    '--gate_hp', type=float, default=1, help='Gate Loss Hyperparameter for IME')
parser.add_argument(
    '--num_experts', type=int, default=3, help='number of experts')
parser.add_argument(
    '--expert_type',
    type=str,
    default='Linear',
    help='Expert arcitecture used by IME')
parser.add_argument(
    '--freeze',
    type=bool,
    default=True,
    help='Freezing experts after and training the gate module ')
parser.add_argument(
    '--noise',
    type=bool,
    default=False,
    help='Adds noise to expert weights and input samples during training')
args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
  args.devices = args.devices.replace(' ', '')
  device_ids = args.devices.split(',')
  args.device_ids = [int(id_) for id_ in device_ids]
  args.gpu = args.device_ids[0]

data_parser = {
    'ECL': {
        'root_path': './data/ECL/',
        'S': [1, 1],
        'MS': [1, 1]
    },
    'M5': {
        'root_path': './data/M5/',
        'S': [1, 1],
        'MS': [18, 1]
    },
    'Rossmann': {
        'root_path': './data/Rossmann/',
        'S': [30, 1],
        'MS': [30, 1]
    },
}

if args.data in data_parser.keys():
  data_info = data_parser[args.data]
  args.root_path = data_info['root_path']
  args.input_dim, args.output_dim = data_info[args.features]

print('Args in experiment:')
print(args)

if args.data == 'Rossmann':
  if (args.model == 'IME_WW' or args.model == 'IME_BW'):
    Exp = ExpIMETabular
  else:
    Exp = ExpBaselineTabular
else:
  if (args.model == 'IME_WW' or args.model == 'IME_BW'):
    Exp = ExpIME
  else:
    Exp = ExpBaseline

# Loop for running the experiments
for ii in range(args.itr):
  if args.data == 'Rossmann':
    # setting record of experiments
    if args.model == 'Linear':
      setting = '{}_{}_lr{}_{}_{}'.format(args.model, args.data,
                                          args.learning_rate, args.des, ii)
    elif args.model == 'MLP':
      setting = '{}_{}_lr{}_dmodel{}_layers{}_{}_{}'.format(
          args.model, args.data, args.learning_rate, args.d_model, args.layers,
          args.des, ii)
    elif args.model == 'MLP':
      setting = '{}_{}_lr{}_dmodel{}_layers{}_{}_{}'.format(
          args.model, args.data, args.learning_rate, args.d_model, args.layers,
          args.des, ii)
    elif args.model == 'SDT':
      setting = '{}_{}_lr{}_depth{}_{}_{}'.format(args.model, args.data,
                                                  args.learning_rate,
                                                  args.depth, args.des, ii)

    elif (args.model == 'IME_WW' or args.model == 'IME_BW'):
      setting = '{}_{}_numExperts{}_expertType{}_lr{}_lrGate{}_accHp{}_divHp{}_utilHp{}_smoothHp{}_gateHp{}_freeze{}_noise{}_{}_{}'.format(
          args.model, args.data, args.num_experts, args.expert_type,
          args.learning_rate, args.learning_rate_gate, args.accuracy_hp,
          args.diversity_hp, args.utilization_hp, args.smoothness_hp,
          args.gate_hp, args.freeze, args.noise, args.des, ii)
    else:
      raise NotImplementedError

  else:
    # setting record of experiments
    if (args.model == ' ARNet' or args.model == 'Linear'):
      setting = '{}_{}_ft{}_sl{}_pl{}_lr{}_{}_{}'.format(
          args.model, args.data, args.features, args.seq_len, args.pred_len,
          args.learning_rate, args.des, ii)
    elif args.model == 'LSTM':
      setting = '{}_{}_ft{}_sl{}_pl{}_lr{}_dmodel{}_layers{}_{}_{}'.format(
          args.model, args.data, args.features, args.seq_len, args.pred_len,
          args.learning_rate, args.d_model, args.layers, args.des, ii)
    elif (args.model == 'IME_WW' or args.model == 'IME_BW'):
      setting = '{}_{}_ft{}_sl{}_pl{}_numExperts{}_expertType{}_lr{}_lrGate{}_accHp{}_divHp{}_utilHp{}_smoothHp{}_gateHp{}_freeze{}_noise{}_{}_{}'.format(
          args.model, args.data, args.features, args.seq_len, args.pred_len,
          args.num_experts, args.expert_type, args.learning_rate,
          args.learning_rate_gate, args.accuracy_hp, args.diversity_hp,
          args.utilization_hp, args.smoothness_hp, args.gate_hp, args.freeze,
          args.noise, args.des, ii)
    else:
      raise NotImplementedError

  exp = Exp(args)  # set experiments
  print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
  exp.train(setting)

  print(
      '>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
  mae, mse, rmse, mape, mspe, upper_loss, orcale_acc = exp.predict(
      setting, True)

  # Saving results in an file for each mode
  df = pd.DataFrame({
      'setting': [setting],
      'mae': [mae],
      'mse': [mse],
      'rmse': [rmse],
      'mape': [mape],
      'mspe': [mspe],
      'upper_bound': [upper_loss],
      'orcale_acc': [orcale_acc]
  })

  check_folder = os.path.isdir(args.output_dir)

  # If folder doesn't exist, then create it.
  if not check_folder:
    os.makedirs(args.output_dir)

  RESULT = pathlib.Path(args.output_dir + args.model + '.csv')
  if RESULT.is_file():
    df.to_csv(RESULT, mode='a', header=False, index=False)
  else:
    df.to_csv(RESULT, index=False)
