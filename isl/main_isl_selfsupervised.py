# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""This file contains code for running ISL on self supervised learning dataset.

The goal is to discover the DAG of the data.
"""

import argparse
import json
import os

import isl_module as isl
import metrics
import numpy as np
import torch
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# pylint: disable=invalid-name
# pylint: disable=f-string-without-interpolation
# pylint: disable=unused-variable
# pylint: disable=redefined-outer-name
def squared_loss(output, target):
  n = target.shape[0]
  loss = 0.5 / n * np.sum((output - target)**2)
  return loss


def main(args):
  torch.set_default_dtype(torch.double)
  np.set_printoptions(precision=3)

  # load data
  X = utils.load_data(dataset_name=args.dataset_name, num_envs=args.num_envs)

  # load vertex labels
  vertex_label = utils.load_vertex_label(dataset_name=args.dataset_name)

  # the initial potential causal parents for each variable
  if args.dataset_name == 'Insurance':
    Init_Relation = {
        'PropCost': ['ThisCarCost', 'OtherCarCost'],  # 0
        'GoodStudent': ['Age', 'SocioEcon'],  # 1
        'Age': ['GoodStudent', 'RiskAversion'],  # 2
        'SocioEcon': ['RiskAversion', 'MakeModel', 'HomeBase'],  # 3
        'RiskAversion': [
            'SocioEcon', 'DrivQuality', 'DrivingSkill', 'HomeBase'
        ],  # 4
        'VehicleYear': [
            'SocioEcon', 'MakeModel', 'Antilock', 'CarValue', 'Airbag'
        ],  # 5
        'ThisCarDam': ['RuggedAuto', 'Accident', 'ThisCarCost'],  # 6
        'RuggedAuto': ['VehicleYear', 'MakeModel', 'Antilock',
                       'Cushioning'],  # 7
        'Accident': ['ThisCarDam', 'RuggedAuto'],  # 8
        'MakeModel': ['SocioEcon', 'VehicleYear', 'RuggedAuto',
                      'CarValue'],  # 9
        'DrivQuality': ['RiskAversion', 'DrivingSkill'],  # 10
        'Mileage': ['MakeModel', 'CarValue'],  # 11
        'Antilock': ['VehicleYear', 'MakeModel'],  # 12
        'DrivingSkill': ['Age', 'DrivQuality'],  # 13
        'SeniorTrain': ['Age', 'RiskAversion'],  # 14
        'ThisCarCost': ['RuggedAuto', 'CarValue'],  # 15
        'Theft': [
            'ThisCarDam', 'MakeModel', 'CarValue', 'HomeBase', 'AntiTheft'
        ],  # 16
        'CarValue': ['VehicleYear', 'MakeModel'],  # 17
        'HomeBase': ['SocioEcon', 'RiskAversion'],  # 18
        'AntiTheft': ['SocioEcon', 'RiskAversion'],  # 19
        'OtherCarCost': ['RuggedAuto', 'Accident'],  # 20
        'OtherCar': ['SocioEcon'],  # 21
        'MedCost': ['Age', 'Accident', 'Cushioning'],  # 22
        'Cushioning': ['RuggedAuto', 'Airbag'],  # 23
        'Airbag': ['VehicleYear'],  # 24
        'ILiCost': ['Accident'],  # 25
        'DrivHist': ['RiskAversion', 'DrivingSkill'],  # 26
    }
  elif args.dataset_name == 'Sachs':
    Init_Relation = {
        'Raf': ['Mek'],
        'Mek': ['Raf'],
        'Plcg': ['PIP2'],
        'PIP2': ['Plcg', 'PIP3'],
        'PIP3': ['Plcg', 'PIP2'],
        'Erk': ['Akt', 'Mek', 'PKA'],
        'Akt': ['PKA', 'Erk'],
        'PKA': ['Akt'],
        'PKC': ['P38'],
        'P38': ['PKA', 'PKC'],
        'Jnk': ['PKC'],
    }
  Init_DAG = np.zeros((len(vertex_label), len(vertex_label)))
  for x in vertex_label:
    x_index = vertex_label.index(x)
    for y in Init_Relation[x]:  # for each causal parent of x
      y_index = vertex_label.index(y)
      Init_DAG[y_index, x_index] = 1

  n, d = X.shape
  model = isl.notearsmlp_self_supervised(
      dims=[d, args.hidden, 1], bias=True, Init_DAG=Init_DAG)
  model.to(device)

  # To use a different feature as label, change y_index to column index of the
  # feature.
  w_est_origin = isl.notears_nonlinear(
      model, X, lambda1=args.lambda1, lambda2=args.lambda2,
      w_threshold=0)  # keep origin w_est

  # tune W for the best shd score
  g, w_true = utils.load_w_true(args.dataset_name)

  wthresh_w_shd_dict = utils.rank_W(
      w_est_origin,
      10,
      90,
      w_true,
      w_threshold_low=0,
      w_threshold_high=10,
      step_size=0.01)
  w_est = None
  w_threshold = None
  for threshold, w_element in wthresh_w_shd_dict.items():
    if utils.is_dag(w_element[0]):
      w_est = w_element[0]
      w_threshold = threshold
      break

  exp = (
      args.Output_path +
      'notear_mlp_sachs_norm={}wthreshold={}lambda1={}lambda2={}'.format(
          args.normalization, w_threshold, args.lambda1, args.lambda2))

  os.makedirs(args.Output_path, exist_ok=True)

  np.savetxt(exp + '_w_est_origin.csv', w_est_origin, delimiter=',')
  np.savetxt(exp + '_w_est.csv', w_est, delimiter=',')
  utils.save_dag(w_est, exp + '_w_est', vertex_label=vertex_label)

  acc = metrics.count_accuracy(w_true, w_est != 0)
  print(acc)

  y = model(torch.from_numpy(X))
  y = y.cpu().detach().numpy()
  mse = squared_loss(y[:, 0], X[:, 0])
  print('mse:', mse)
  acc['mse'] = mse

  with open(f'{exp}_metrics.json', 'w+') as f:
    json.dump(acc, f)


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(description='Run NOTEARS algorithm')
  parser.add_argument(
      '--dataset_name',
      type=str,
      default='BH',
      help='BH (for boston housing), Insurance, Sachs')
  parser.add_argument(
      '--X_path', type=str, help='n by p data matrix in csv format')
  parser.add_argument(
      '--y_index',
      type=int,
      default=0,
      help='use feature y-index as y for prediction')
  parser.add_argument(
      '--hidden', type=int, default=10, help='Number of hidden units')
  parser.add_argument(
      '--lambda1', type=float, default=0.01, help='L1 regularization parameter')
  parser.add_argument(
      '--lambda2', type=float, default=0.01, help='L2 regularization parameter')
  parser.add_argument(
      '--w_path',
      type=str,
      default='w_est.csv',
      help='p by p weighted adjacency matrix of estimated DAG in csv format')
  parser.add_argument(
      '--Output_path',
      type=str,
      default='./output/sparseInit_Ins/',
      help='output path')
  parser.add_argument(
      '--w_threshold', type=float, default=0.5, help='i < threshold no edge')
  parser.add_argument(
      '--Notear_activation', type=str, default='relu', help='relu, sigmoid')
  parser.add_argument(
      '--normalization',
      type=str,
      default='standard',
      help='use normalization preprocess standard or minmax')
  parser.add_argument('--datasource', type=str, default='raw', help='raw, IRM')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  main(args)
