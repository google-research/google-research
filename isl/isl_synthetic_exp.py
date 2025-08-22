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

"""This file contains ISL training and evaluation on synthetic data."""

import argparse
import json
import os

import data_gen_syn
import isl_module as isl
import metrices
import mlp as MLP
import nonlinear_castle
import nonlinear_gpu as nonlinear
import numpy as np
from scipy.special import expit as sigmoid
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

metrics = dict()


# pylint: disable=invalid-name
# pylint: disable=f-string-without-interpolation
# pylint: disable=unused-variable
# pylint: disable=redefined-outer-name
# pylint: disable=unexpected-keyword-arg
# pylint: disable=eval-used
# pylint: disable=no-value-for-parameter
def mean_squared_loss(n, target, output):
  """Generates synthetic data (in different envs) depending on data_source."""
  loss = 0.5 / n * np.sum((output - target)**2)
  return loss


def gen_synthetic_env(args,
                      data_source,
                      dim,
                      num_env,
                      random,
                      ISL_param=None,
                      castle_param=None,
                      noise_type='None'):
  """Generates synthetic data (in different envs) depending on data_source."""

  if data_source == 'binary':
    vertex_label = ['Y', 'X1', 'X2', 'S1']
    return np.array(
        data_gen_syn.bi_classify_env_jointSampled(
            dim, num_env, 1.0, random,
            ISL_param['probs'])), vertex_label, utils.customized_graph(
                data_source)

  elif data_source == 'ISL':
    num_xy = ISL_param['num_xy']
    num_s = ISL_param['num_s']
    vertex_label = list()
    vertex_label.append('Y')
    vertex_label += [f'X{i+1}' for i in range(num_xy - 1)]
    vertex_label += [f'S{i+1}' for i in range(num_s)]
    if ISL_param['train']:
      return data_gen_syn.gen_ISL_env(
          num_xy, num_s, dim, 3,
          probs=ISL_param['probs']), vertex_label, utils.customized_graph(
              data_source, vertex_label=vertex_label)
    else:
      return data_gen_syn.gen_ISL_env(
          num_xy, num_s, dim, 3,
          probs=ISL_param['test_probs']), vertex_label, utils.customized_graph(
              data_source, vertex_label=vertex_label)

  elif data_source == 'ISL_counter':
    num_xy = ISL_param['num_xy']
    num_s = ISL_param['num_s']
    vertex_label = list()
    vertex_label.append('Y')
    vertex_label += [f'X{i+1}' for i in range(num_xy - 1)]
    vertex_label += [f'S{i+1}' for i in range(num_s)]
    return data_gen_syn.gen_ISL_simple(
        num_env, dim, ISL_param), vertex_label, utils.customized_graph(
            data_source, vertex_label=vertex_label)

  elif data_source == 'xyz_complex':
    vertex_label = ['z', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1', 'y2', 'y3']

    return np.array(data_gen_syn.gen_xyz_complex_env(
        dim)), vertex_label, utils.customized_graph(data_source)

  elif data_source == 'castle_complex':
    vertex_label = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
    return data_gen_syn.gen_castle_env(
        dim, castle_param,
        num_env), vertex_label, utils.customized_graph(data_source)

  elif data_source == 'binary_ISL':
    num_xy = ISL_param['num_xy']
    num_s = ISL_param['num_s']
    vertex_label = list()
    vertex_label.append('Y')
    vertex_label += [f'X{i+1}' for i in range(num_xy - 1)]
    vertex_label += [f'S{i+1}' for i in range(num_s)]
    if ISL_param['train']:
      return data_gen_syn.binary_ISL(
          args,
          dim,
          num_xy,
          num_s,
          3,
          probs=ISL_param['probs'],
          sem_type=noise_type), vertex_label, utils.customized_graph(
              data_source, vertex_label=vertex_label)
    else:
      return data_gen_syn.binary_ISL(
          args,
          dim,
          num_xy,
          num_s,
          3,
          probs=ISL_param['test_probs'],
          sem_type=noise_type), vertex_label, utils.customized_graph(
              data_source, vertex_label=vertex_label)

  elif data_source == 'random':
    return

  else:
    ValueError('invalid data source')


def train(X, b_true, vertex_label, model_type, args):
  """Trains the model."""

  n_envs, n, d = X.shape
  os.makedirs(args.Output_path, exist_ok=True)
  if model_type == 'notear-mlp':
    X = np.vstack(X)
    model = nonlinear.NotearsMLP(dims=[d, args.hidden, 1], bias=True)
    w_est_origin = nonlinear.notears_nonlinear(
        model, X, lambda1=args.lambda1, lambda2=args.lambda2, w_threshold=0)

    wthresh_w_shd_dict = utils.rank_W(
        w_est_origin,
        2,
        10,
        b_true,
        w_threshold_low=0,
        w_threshold_high=5,
        step_size=0.02)
    w_est = None
    w_threshold = None
    for threshold, w_element in wthresh_w_shd_dict.items():
      if utils.is_dag(w_element[0]):
        w_est = w_element[0]
        w_threshold = threshold
        break

    exp = f'notear_mlp_syn_{args.synthetic_source}_W-thresh{round(w_threshold, 3)}'

    # save the learned W matrix
    np.savetxt(
        args.Output_path + f'{exp}_West.csv', w_est, fmt='%.3f', delimiter=',')
    np.savetxt(
        args.Output_path + f'{exp}_WOrigin.csv',
        w_est_origin,
        fmt='%.3f',
        delimiter=',')

    # save the learned DAG
    utils.save_dag(
        w_est, args.Output_path + f'{exp}_DAG', vertex_label=vertex_label)

    # count the accuracy of Y
    b_true_Y = np.zeros_like(b_true)
    b_true_Y[:, 0] = b_true[:, 0]
    w_est_Y = np.zeros_like(w_est)
    w_est_Y[:, 0] = w_est[:, 0]
    acc = metrices.count_accuracy(b_true_Y, w_est_Y != 0)
    metrics[f'{exp}_train_acc'] = acc

    y = model(torch.from_numpy(X))
    y = y.cpu().detach().numpy()

    mse = mean_squared_loss(y.shape[0], y[:, 0], X[:, 0])
    print('mse:', mse)
    metrics[f'{model_type}_train_MSE'] = mse

  elif model_type == 'notear-castle':
    X = np.vstack(X)
    model = nonlinear_castle.NotearsMLP(dims=[d, args.hidden, 1], bias=True)

    # To use a different feature as label, change y_index to column
    # index of the feature.
    w_est_origin = nonlinear_castle.notears_nonlinear(
        model,
        X,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        w_threshold=args.w_threshold)
    wthresh_w_shd_dict = utils.rank_W(
        w_est_origin,
        2,
        10,
        b_true,
        w_threshold_low=0,
        w_threshold_high=5,
        step_size=0.02)
    w_est = None
    w_threshold = None
    for threshold, w_element in wthresh_w_shd_dict.items():
      if utils.is_dag(w_element[0]):
        w_est = w_element[0]
        w_threshold = threshold
        break

    exp = f'notear_castle_syn_{args.synthetic_source}_W-thresh-{round(w_threshold, 3)}'

    # save the learned W matrix
    np.savetxt(
        args.Output_path + f'{exp}_West.csv', w_est, fmt='%.2f', delimiter=',')
    np.savetxt(
        args.Output_path + f'{exp}_WOrigin.csv',
        w_est_origin,
        fmt='%.3f',
        delimiter=',')

    # save the learned DAG
    utils.save_dag(
        w_est, args.Output_path + f'{exp}_DAG', vertex_label=vertex_label)

    # estimate the accuracy of Y
    b_true_Y = np.zeros_like(b_true)
    b_true_Y[:, 0] = b_true[:, 0]
    w_est_Y = np.zeros_like(w_est)
    w_est_Y[:, 0] = w_est[:, 0]
    acc = metrices.count_accuracy(b_true_Y, w_est_Y != 0)
    metrics[f'{exp}_train_acc'] = acc

    y = model(torch.from_numpy(X))
    y = y.cpu().detach().numpy()

    mse = mean_squared_loss(y.shape[0], y[:, 0], X[:, 0])
    print('mse:', mse)
    metrics[f'{model_type}_train_MSE'] = mse

  elif model_type == 'ISL':
    model = isl.isl_module(
        n_envs=n_envs,
        Y_dims=[d, args.hidden, 1],
        dims=[d, args.hidden, 1],
        bias=True)
    model.to(device)

    # To use a different feature as label, change y_index to column index of
    # the feature.
    _, w_est_origin_envs = isl.notears_nonlinear(
        model,
        X,
        y_index=0,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda1_Y=args.lambda1_Y,
        lambda2_Y_fc1=args.lambda2_Y_fc1,
        lambda2_Y_fc2=args.lambda2_Y_fc2,
        beta=args.beta,
        w_threshold=0)

    for i in range(len(w_est_origin_envs)):

      wthresh_w_shd_dict = utils.rank_W(
          w_est_origin_envs[i],
          2,
          30,
          b_true,
          w_threshold_low=0,
          w_threshold_high=5,
          step_size=0.02)
      w_est = None
      w_threshold = None
      for threshold, w_element in wthresh_w_shd_dict.items():
        if utils.is_dag(w_element[0]):
          w_est = w_element[0]
          w_threshold = threshold
          break

      exp = f'notear_ISL_syn_{args.synthetic_source}_W-thresh{round(w_threshold, 3)}'

      # save the leared W matrix
      np.savetxt(
          args.Output_path + f'{exp}_West_{i}.csv',
          w_est,
          fmt='%.3f',
          delimiter=',')
      np.savetxt(
          args.Output_path + f'{exp}_WOrigin_env_{i}.csv',
          w_est_origin_envs[i],
          fmt='%.3f',
          delimiter=',')

      # save the learned DAG
      utils.save_dag(
          w_est, args.Output_path + f'{exp}_DAG_{i}', vertex_label=vertex_label)

      # count the accuracy of Y
      b_true_Y = np.zeros_like(b_true)
      b_true_Y[:, 0] = b_true[:, 0]
      w_est_Y = np.zeros_like(w_est)
      w_est_Y[:, 0] = w_est[:, 0]
      acc = metrices.count_accuracy(b_true_Y, w_est_Y != 0)
      print(acc)
      metrics[f'{exp}_train_env_{i}_acc'] = acc
      y = model.test(X)

      mse = mean_squared_loss(y.shape[0] * y.shape[1], y,
                              X[:, :, 0][:, :, np.newaxis])
      print('mse:', mse)
      metrics[f'{exp}_train_env_{i}_mse'] = mse

  else:
    ValueError('Invalid model type')

  return model, w_est


def test(model, X, model_type, test_type, counter=False):
  """Test functions."""

  if model_type == 'notear-mlp':
    X = np.vstack(X)
    y = model(torch.from_numpy(X))
    y = y.cpu().detach().numpy()
    mse = mean_squared_loss(y.shape[0], y[:, 0], X[:, 0])
  elif model_type == 'notear-castle':
    X = np.vstack(X)
    y = model(torch.from_numpy(X))
    y = y.cpu().detach().numpy()
    mse = mean_squared_loss(y.shape[0], y[:, 0], X[:, 0])
  elif model_type == 'ISL':
    y = model.test(X)
    mse = mean_squared_loss(y.shape[0] * y.shape[1], y, X[:, :, 0][:, :,
                                                                   np.newaxis])
  if not counter:
    if test_type == 'ID':
      metrics[f'{model_type}_testID_MSE'] = mse
    elif test_type == 'OOD':
      metrics[f'{model_type}_testOOD_MSE'] = mse
  else:
    if test_type == 'ID':
      metrics[f'{model_type}_counter_testID_MSE'] = mse
    elif test_type == 'OOD':
      metrics[f'{model_type}_counter_testOOD_MSE'] = mse

  return mse


def exp(args, model_type, vertex_label, b_true, X_train, X_test_ID, X_test_OOD):
  """Experiment function."""

  torch.set_default_dtype(torch.double)
  np.set_printoptions(precision=3)

  model, w_est = train(
      X_train,
      vertex_label=vertex_label,
      b_true=b_true,
      model_type=model_type,
      args=args)

  # ID test
  test(
      model,
      X_test_ID,
      vertex_label=vertex_label,
      b_true=b_true,
      model_type=model_type,
      args=args,
      test_type='ID')

  # OOD test
  test(
      model,
      X_test_OOD,
      vertex_label=vertex_label,
      b_true=b_true,
      model_type=model_type,
      args=args,
      test_type='OOD')

  return model, w_est


def find_causal_parents(W):
  """Finds causal parents."""
  causal_parents = []
  for i in range(W.shape[0]):
    if W[i][0] > 0:
      causal_parents.append(i)

  return causal_parents


def combined_exp(args):
  """Combines experiments."""

  args.probs = [eval(prob) for prob in args.probs.split(',')]
  args.test_probs = np.random.uniform(low=0.0, high=1.0, size=3)
  ISL_param = dict()
  ISL_param['num_xy'] = args.num_xy
  ISL_param['num_s'] = args.num_s
  ISL_param['train'] = True
  ISL_param['probs'] = args.probs
  ISL_param['test_probs'] = args.test_probs

  # train data generation
  X_train, vertex_label, b_true = gen_synthetic_env(
      args,
      args.synthetic_source,
      dim=1000,
      num_env=3,
      random=False,
      ISL_param=ISL_param,
      castle_param=[args.probs, args.probs],
      noise_type=args.noise_type)

  # ID test data generation
  X_test_ID, _, _ = gen_synthetic_env(
      args,
      args.synthetic_source,
      dim=200,
      num_env=3,
      random=False,
      ISL_param=ISL_param,
      noise_type=args.noise_type)

  # OOD test data generation
  ISL_param['train'] = False
  X_test_OOD, _, _ = gen_synthetic_env(
      args,
      args.synthetic_source,
      dim=200,
      num_env=3,
      random=True,
      ISL_param=ISL_param,
      noise_type=args.noise_type)
  args.synthetic_source += 'trainProb' + str(args.probs)
  args.synthetic_source += 'testProb' + str(args.test_probs)
  notear_model, _ = exp(args, 'notear-mlp', vertex_label, b_true, X_train,
                        X_test_ID, X_test_OOD)

  notear_ISL_model, w_est = exp(args, 'ISL', vertex_label, b_true, X_train,
                                X_test_ID, X_test_OOD)
  causal_parents = find_causal_parents(w_est)
  # passing the causal parents of X to a MLP to predict
  X_train = np.vstack(X_train)
  X_test_ID = np.vstack(X_test_ID)
  X_test_OOD = np.vstack(X_test_OOD)
  model_ID, ISL_id_train_mse, ISL_id_test_mse = mlp(
      X_train[:, 1:args.num_xy],
      X_train[:, 0],
      X_test_ID[:, 1:args.num_xy],
      X_test_ID[:, 0],
      epoches=200)
  model_OOD, ISL_ood_train_mse, ISL_ood_test_mse = mlp(
      X_train[:, 1:args.num_xy],
      X_train[:, 0],
      X_test_OOD[:, 1:args.num_xy],
      X_test_OOD[:, 0],
      epoches=200)
  metrics['ISL_id_train_mse'] = ISL_id_train_mse
  metrics['ISL_id_test_mse'] = ISL_id_test_mse
  metrics['ISL_ood_train_mse'] = ISL_ood_train_mse
  metrics['ISL_ood_test_mse'] = ISL_ood_test_mse

  with open(args.Output_path + f'{args.synthetic_source}_metrics.json',
            'w+') as f:
    json.dump(metrics, f, indent=4)

  np.savetxt(f'{args.Output_path}X_train.csv', X_train, delimiter=',')
  np.savetxt(f'{args.Output_path}X_test_ID.csv', X_test_ID, delimiter=',')
  np.savetxt(f'{args.Output_path}X_test_OOD.csv', X_test_OOD, delimiter=',')
  torch.save(notear_model, f'{args.Output_path}notear.pt')
  # torch.save(notear_castle_model, f'{args.Output_path}notear_castle.pt')
  torch.save(notear_ISL_model, f'{args.Output_path}ISL_notear.pt')

  return


def counter_exp(args):
  """Experiments with a counter."""

  torch.set_default_dtype(torch.double)
  X_train = np.loadtxt(f'{args.Output_path}X_train.csv', delimiter=',')
  X_test_ID = np.loadtxt(f'{args.Output_path}X_test_ID.csv', delimiter=',')
  X_test_OOD = np.loadtxt(f'{args.Output_path}X_test_OOD.csv', delimiter=',')

  model_ID = MLP.MLP(
      n_feature=2, n_hidden=100, n_output=1, activation='Sigmoid')
  model_ID.load_state_dict(torch.load(f'{args.Output_path}_induced_mlp.pt'))
  notear_model = torch.load(f'{args.Output_path}notear.pt')

  counter = dict()
  model_ID.eval()

  data = X_train.copy()
  for counter_index in range(1, args.num_xy + args.num_s, 1):

    # counterfactual on trainining data
    data[:, counter_index] = np.random.normal(
        loc=5, scale=10, size=data.shape[0])
    # make prediction on the modified data
    data[:, 0] = sigmoid(np.sum(data[:, 1:args.num_xy], axis=1))

    ISL_pred = model_ID(torch.tensor(data[:, 1:args.num_xy]))
    notear_model.eval()
    notear_pred = notear_model(torch.tensor(data))
    ISL_mse = mean_squared_loss(data.shape[0], data[:, 0],
                                np.array(ISL_pred.detach().cpu())[:, 0])
    notear_mse = mean_squared_loss(data.shape[0], data[:, 0],
                                   np.array(notear_pred.detach().cpu())[:, 0])
    print(f'counteron{counter_index}_ISL_pred', ISL_mse)
    print(f'counteron{counter_index}notear_pred_train', notear_mse)

    counter[f'counteron{counter_index}_ISL_pred'] = ISL_mse
    counter[f'counteron{counter_index}notear_pred_train'] = notear_mse

  with open(f'{args.Output_path}counter_u=5s=0_mse.json', 'w+') as f:
    json.dump(metrics, f, indent=4)


def test_mlp(args):
  """Tests MLP."""

  torch.set_default_dtype(torch.double)
  X_train = np.loadtxt(f'{args.Output_path}X_train.csv', delimiter=',')
  X_test_ID = np.loadtxt(f'{args.Output_path}X_test_ID.csv', delimiter=',')
  X_test_OOD = np.loadtxt(f'{args.Output_path}X_test_OOD.csv', delimiter=',')
  notear_model = torch.load(f'{args.Output_path}notear.pt')

  model, ISL_id_train_mse, ISL_id_test_mse = mlp(
      X_train[:, 1:args.num_xy],
      X_train[:, 0],
      X_test_ID[:, 1:args.num_xy],
      X_test_ID[:, 0],
      epoches=200)
  # model_OOD, ISL_ood_train_mse, ISL_ood_test_mse = mlp(
  #  X_train[:,1:args.num_xy], X_train[:,0], X_test_OOD[:, 1:args.num_xy],
  #  X_test_OOD[:,0], epoches=200)
  torch.save(model.state_dict(), f'{args.Output_path}_induced_mlp.pt')
  model = MLP.MLP(
      n_feature=args.num_xy - 1, n_hidden=100, n_output=1, activation='Sigmoid')

  model.load_state_dict(torch.load(f'{args.Output_path}_induced_mlp.pt'))
  for data in [X_test_ID, X_test_OOD]:
    ISL_pred = model(torch.tensor(data[:, 1:args.num_xy]))
    notear_model.eval()
    notear_pred = notear_model(torch.tensor(data))
    ISL_mse = mean_squared_loss(data.shape[0], data[:, 0],
                                np.array(ISL_pred.detach().cpu())[:, 0])
    notear_mse = mean_squared_loss(data.shape[0], data[:, 0],
                                   np.array(notear_pred.detach().cpu())[:, 0])
    print(f'_ISL_pred', ISL_mse)
    print(f'_notear_pred_train', notear_mse)


def mlp(X_train,
        Y_train,
        X_test,
        Y_test,
        epoches=100,
        hidden=100,
        lr=0.01,
        batch_size=64):
  """MLP training."""

  input_feature_dim = X_train.shape[1]
  output_feature_dim = 1
  x_train = torch.tensor(X_train)
  y_train = torch.tensor(Y_train)
  x_test = torch.tensor(X_test)
  y_test = torch.tensor(Y_test)
  train_dataset = MLP.CustomDataset(
      torch.tensor(X_train), torch.tensor(Y_train))
  test_dataset = MLP.CustomDataset(torch.tensor(X_test), torch.tensor(Y_test))
  test_loader = DataLoader(test_dataset, batch_size=batch_size)
  train_loader = DataLoader(train_dataset, batch_size=batch_size)

  model = MLP.MLP(
      n_feature=input_feature_dim,
      n_hidden=hidden,
      n_output=output_feature_dim,
      activation='Sigmoid')

  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
  loss_func = torch.nn.MSELoss()

  for epoch in range(1, epoches):
    loss = MLP.train(model, epoch, train_dataset, train_loader, optimizer,
                     loss_func)
    train_loss = MLP.test(model, train_loader, loss_func)
    test_loss = MLP.test(model, test_loader, loss_func)
    if epoch % 10 == 0:
      print('Epoch: {:03d}, Loss: {:.5f}, train_loss: {:.5f}, test_loss: {:.5f}'
            .format(epoch, loss, train_loss, test_loss))

  model.eval()
  output = model(x_test)
  mse = mean_squared_loss(x_test.shape[0], np.array(y_test.detach()),
                          np.array(output[:, 0].detach()))
  train_output = model(x_train)
  train_mse = mean_squared_loss(x_train.shape[0], np.array(y_train.detach()),
                                np.array(train_output[:, 0].detach()))
  print(mse)
  return model, train_mse, mse


def main(args):

  torch.set_default_dtype(torch.double)
  np.set_printoptions(precision=3)

  X, vertex_label, b_true = gen_synthetic_env(args.synthetic_source, dim=200)
  n_envs, n, d = X.shape
  model = isl.isl_module(
      n_envs=n_envs,
      Y_dims=[d, args.hidden, args.hidden, 1],
      dims=[d, args.hidden, 1],
      bias=True)
  model.to(device)
  # To use a different feature as label, change y_index to column
  # index of the feature
  y_pred_loss, w_est_envs = isl.notears_nonlinear(
      model,
      X,
      y_index=0,
      lambda1=args.lambda1,
      lambda2=args.lambda2,
      w_threshold=args.w_threshold)

  exp = f'notear_castle_ISL_syn_{args.synthetic_source}'
  metrics = dict()
  for i in range(len(w_est_envs)):

    # save the learned W matrix
    np.savetxt(
        args.Output_path + f'{exp}_West_{i}.csv', w_est_envs[i], delimiter=',')
    # save the learned DAG
    utils.save_dag(
        w_est_envs[i],
        args.Output_path + f'{exp}_DAG_{i}',
        vertex_label=vertex_label)
    # count the accuracy
    acc = metrices.count_accuracy(b_true, w_est_envs[i] != 0)
    print(acc)
    metrics[f'env_{i}'] = acc

  y = model.test(X)

  def squared_loss(output, target):
    n = target.shape[0] * target.shape[1]
    loss = 0.5 / n * np.sum((output - target)**2)
    return loss

  mse = squared_loss(y, X[:, :, 0][:, :, np.newaxis])
  print('mse:', mse)
  metrics['mse'] = mse
  with open(args.Output_path + f'{exp}_metrics.json', 'w+') as f:
    json.dump(acc, f)


def parse_args():
  """Parses arguments."""

  parser = argparse.ArgumentParser(description='Run NOTEARS algorithm')
  parser.add_argument(
      '--hidden', type=int, default=10, help='Number of hidden units')
  parser.add_argument(
      '--lambda1', type=float, default=0.01, help='L1 regularization parameter')
  parser.add_argument(
      '--lambda2', type=float, default=0.01, help='L2 regularization parameter')
  parser.add_argument(
      '--beta', type=float, default=1, help='weight of the y prediction')
  parser.add_argument(
      '--w_path',
      type=str,
      default='w_est.csv',
      help='p by p weighted adjacency matrix of estimated DAG in csv format')
  parser.add_argument(
      '--Output_path',
      type=str,
      default='./output/synthetic/motivition_1_neg/',
      help='output path')
  parser.add_argument(
      '--w_threshold', type=float, default=0.5, help='i < threshold no edge')
  parser.add_argument(
      '--Notear_activation', type=str, default='relu', help='relu, sigmoid')
  parser.add_argument(
      '--synthetic_source',
      type=str,
      default='binary',
      help='binary, xyz_complex, castle_complex, random')
  parser.add_argument(
      '--probs',
      type=str,
      default='0.1,0.2,0.8',
      help='the prob distribution for spurious correlation ')
  parser.add_argument(
      '--test_probs',
      type=str,
      default='0.1,0.1,0.1',
      help='the prob distribution for spurious correlation')
  parser.add_argument(
      '--perturb',
      type=str,
      default='X1',
      help='the variabel name to change its range')
  parser.add_argument(
      '--num_xy', type=int, default=3, help='the number of xy variables')
  parser.add_argument(
      '--num_s',
      type=int,
      default=1,
      help='the number of spurious correlation variables')
  parser.add_argument(
      '--noise_type',
      type=str,
      default='uniform',
      help='the number of spurious correlation variables')
  parser.add_argument(
      '--lambda1_Y',
      type=float,
      default=0.001,
      help='L1 regularization parameter on Y')
  parser.add_argument(
      '--lambda2_Y_fc1',
      type=float,
      default=0.001,
      help='L2 regularization parameter on the first layer of Y')
  parser.add_argument(
      '--lambda2_Y_fc2',
      type=float,
      default=0.001,
      help='L2 regularization parameter on the second layer of Y')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  combined_exp(args)
  if args.synthetic_source == 'ISL':
    combined_exp(args)
  else:
    combined_exp(args)
    counter_exp(args)
