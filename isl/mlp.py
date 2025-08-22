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

"""A simple MLP used as baseline for comparision.

It's also used for making predictions based on causal parents discovered by ISL.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import utils


# pylint: disable=invalid-name
# pylint: disable=f-string-without-interpolation
# pylint: disable=unused-variable
# pylint: disable=redefined-outer-name
def mean_squared_loss(n, target, output):
  """Returns MSE loss."""
  return 0.5 / n * np.sum((output - target)**2)


class MLP(torch.nn.Module):
  """MLP class."""

  def __init__(self, n_feature, n_hidden, n_output, activation='ReLU'):
    super(MLP, self).__init__()
    self.hidden_layer = torch.nn.Linear(n_feature, n_hidden)
    self.predict_layer = torch.nn.Linear(n_hidden, n_output)

    if activation == 'ReLU':
      self.seq_layer = torch.nn.model = torch.nn.Sequential(
          torch.nn.Linear(n_feature, n_hidden),
          torch.nn.ReLU(),
      )
    elif activation == 'Sigmoid':
      self.seq_layer = torch.nn.model = torch.nn.Sequential(
          torch.nn.Linear(n_feature, n_hidden * 2),
          torch.nn.Sigmoid(),
          torch.nn.Linear(n_hidden * 2, n_hidden),
          torch.nn.Sigmoid(),
      )

  def forward(self, x):
    hidden_result = self.seq_layer(x)
    predict_result = self.predict_layer(hidden_result)
    return predict_result


class CustomDataset(Dataset):
  """Custom dataset class."""

  def __init__(self, x, y, transform=None, target_transform=None):
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    x = self.x[idx]
    y = self.y[idx]
    return x, y


def train(net):
  """Training function."""

  net.train()
  loss_all = 0
  for data in train_loader:
    optimizer.zero_grad()
    output = net(data[0])
    output = torch.squeeze(output)
    loss = loss_func(output, data[1])
    loss.backward()
    loss_all += loss.item()
    optimizer.step()
  return loss_all / len(train_dataset)


def test(net, loader):
  net.eval()
  loss_all = 0
  for data in loader:
    output = net(data[0])
    loss = loss_func(output, data[1])
    loss_all += loss.item()
  return loss_all / len(loader.dataset)


def parse_args():
  """Parses arguments."""

  parser = argparse.ArgumentParser(description='Run NOTEARS algorithm')
  parser.add_argument(
      '--hidden', type=int, default=10, help='Number of hidden units')
  parser.add_argument(
      '--lambda1', type=float, default=0.01, help='L1 regularization parameter')
  parser.add_argument(
      '--lambda2', type=float, default=0.01, help='L2 regularization parameter')
  parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
  parser.add_argument(
      '--Output_path', type=str, default='./output/', help='output path')
  parser.add_argument(
      '--num_epoch', type=int, default=300, help='number of training epoches')
  parser.add_argument(
      '--activation', type=str, default='relu', help='relu, sigmoid')
  parser.add_argument(
      '--data_source', type=str, default='BH', help='BH, insurance')
  parser.add_argument('--normalization', type=str, default='standard', help='')
  parser.add_argument('--batch', type=int, default=256, help='')
  parser.add_argument('--causal', type=bool, default=False, help='')
  parser.add_argument('--y_index', type=int, default=0, help='')
  parser.add_argument('--vars', type=str, default=None, help='')
  parser.add_argument('--num_xy', type=int, default=3, help='')

  args = parser.parse_args()
  return args


if __name__ == '__main__':
  torch.set_default_dtype(torch.double)
  args = parse_args()

  if args.data_source == 'BH':
    X_train = np.loadtxt(
        './data/BH_train_test_trainparam/standard_BH_train.csv', delimiter=',')
    X_test = np.loadtxt(
        'data/BH_train_test_trainparam/standard_BH_test.csv', delimiter=',')
    activation = 'ReLU'

  elif args.data_source == 'Insurance':
    X_train = np.loadtxt(
        './data/Insurance/IRM/standard_Insurance_train.csv', delimiter=',')
    X_test = np.loadtxt(
        'data/Insurance/IRM/standard_Insurance_test.csv', delimiter=',')
    vertex_label = [
        'PropCost', 'GoodStudent', 'Age', 'SocioEcon', 'RiskAversion',
        'VehicleYear', 'ThisCarDam', 'RuggedAuto', 'Accident', 'MakeModel',
        'DrivQuality', 'Mileage', 'Antilock', 'DrivingSkill', 'SeniorTrain',
        'ThisCarCost', 'Theft', 'CarValue', 'HomeBase', 'AntiTheft',
        'OtherCarCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag',
        'ILiCost', 'DrivHist'
    ]
    activation = 'ReLU'

  elif args.data_source == 'synthetic':
    X_train = np.loadtxt(
        f'./output/synthetic/binary/{args.vars}/prob0123/X_train.csv',
        delimiter=',')
    X_test = np.loadtxt(
        f'./output/synthetic/binary/{args.vars}/prob0123/X_test_ID.csv',
        delimiter=',')
    activation = 'Sigmoid'

  input_feature_dim = X_train.shape[1] - 1
  output_feature_dim = 1

  if args.y_index != 0:
    utils.swap(X_train, 0, args.y_index)
    utils.swap(X_test, 0, args.y_index)

  if args.causal:
    causal_parents = X_train[:, 1:args.num_xy]

    x_train = torch.Tensor(causal_parents)
    y_train = torch.Tensor(X_train[:, 0])
    x_test = torch.Tensor(X_test[:, 1:args.num_xy])

    y_test = torch.Tensor(X_test[:, 0])

    input_feature_dim = x_train.shape[1] if len(x_train.shape) > 1 else 1

  else:
    x_train = torch.Tensor(X_train[:, 1:])
    y_train = torch.Tensor(X_train[:, 0])
    x_test = torch.Tensor(X_test[:, 1:])
    y_test = torch.Tensor(X_test[:, 0])
    input_feature_dim = x_train.shape[1]

  train_dataset = CustomDataset(x_train, y_train)
  test_dataset = CustomDataset(x_test, y_test)

  test_loader = DataLoader(test_dataset, batch_size=64)
  train_loader = DataLoader(train_dataset, batch_size=64)

  net = MLP(
      n_feature=input_feature_dim,
      n_hidden=args.hidden,
      n_output=output_feature_dim,
      activation=activation)

  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

  loss_func = torch.nn.MSELoss(reduction='mean')

  for epoch in range(1, args.num_epoch):
    loss = train(net)
    train_loss = test(net, train_loader)
    test_loss = test(net, test_loader)
    if epoch % 10 == 0:
      print('Epoch: {:03d}, Loss: {:.5f}, train_loss: {:.5f}, test_loss: {:.5f}'
            .format(epoch, loss, train_loss, test_loss))

  net.eval()
  output = net(x_test)
  print('y_index:', args.y_index)
  print(
      mean_squared_loss(x_test.shape[0], np.array(y_test.detach().cpu()),
                        np.array(output[:, 0].detach().cpu())))

  if args.vars:
    X_OOD_test = np.loadtxt(
        f'./output/synthetic/binary/{args.vars}/prob0123/X_test_OOD.csv',
        delimiter=',')
    print('ood_test:')
    if not args.causal:
      output = net(torch.tensor(X_OOD_test[:, 1:]))
    else:
      output = net(torch.tensor(X_OOD_test[:, 1:args.num_xy]))

    print(
        mean_squared_loss(X_OOD_test.shape[0], X_OOD_test[:, 0],
                          np.array(output[:, 0].detach())))
