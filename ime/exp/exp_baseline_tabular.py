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

"""Baseline models for tabular data."""
import os
import time
import warnings

from data.data_loader import Rossmann
from exp.exp_basic import  ExpBasic
import matplotlib.pyplot as plt
from models.linear import Linear
from models.mlp import MLP
from models.sdt import SDT
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from utils.metrics import Metric as metric
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')


class ExpBaselineTabular(ExpBasic):
  """Baseline experiments for tabular data."""

  def _get_dataset(self):
    """Create dataset based on data name in the parser.

    Returns:
          Data: An instant of the dataset created
    """

    if self.args.data == 'Rossmann':
      data = Rossmann(self.args.root_path, self.args.scale)
    else:
      raise NotImplementedError

    if self.args.scale:
      self.label_scaler = data.label_scaler
    return data

  def _build_model(self):
    """Function that creates a model instance based on the model name.

      Here we only support MLP, Linear and SDT.

    Returns:
        model: An instance of the model.
    """
    if self.args.model == 'MLP':
      model = MLP(self.args.input_dim, self.args.output_dim, self.args.d_model,
                  self.args.layers).float()
    elif self.args.model == 'Linear':
      model = Linear(
          self.args.output_dim,
          self.args.input_dim,
      ).float()

    elif self.args.model == 'SDT':
      model = SDT(
          self.args.input_dim,
          self.args.output_dim,
          depth=self.args.depth,
          device=self.device).float()
    else:
      raise NotImplementedError

    # if multiple GPU are to be used parralize model
    if self.args.use_multi_gpu and self.args.use_gpu:
      model = nn.DataParallel(model, device_ids=self.args.device_ids)

    return model

  def _get_data(self, flag):
    """Function that creats a dataloader basd on flag.

    Args:
      flag: Flag indicating if we should return training/validation/testing
        dataloader

    Returns:
      data_loader: Dataloader for the required dataset.
    """
    args = self.args
    if flag == 'test':
      shuffle_flag = False
      drop_last = True
      batch_size = args.batch_size
      data_set = TensorDataset(
          torch.Tensor(self.data.test_x), torch.Tensor(self.data.test_y))
    elif flag == 'pred':
      shuffle_flag = False
      drop_last = False
      batch_size = args.batch_size
      data_set = TensorDataset(
          torch.Tensor(self.data.test_x), torch.Tensor(self.data.test_y))
    elif flag == 'val':
      shuffle_flag = False
      drop_last = False
      batch_size = args.batch_size
      data_set = TensorDataset(
          torch.Tensor(self.data.valid_x), torch.Tensor(self.data.valid_y))
    else:
      shuffle_flag = True
      drop_last = True
      batch_size = args.batch_size
      data_set = TensorDataset(
          torch.Tensor(self.data.train_x), torch.Tensor(self.data.train_y))

    print('Data for', flag, 'dataset size', len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_loader

  def _select_optimizer(self):
    """Function that returns the optimizer based on learning rate.

    Returns:
        model_optim: model optimizer
    """
    model_optim = optim.Adam(
        self.model.parameters(), lr=self.args.learning_rate)
    return model_optim

  def vali(self, vali_loader, criterion):
    """Validation Function.

    Args:
        vali_loader: Validation dataloader
        criterion: criterion used in for loss function

    Returns:
        total_loss: average loss
    """
    self.model.eval()
    total_loss = []
    for (batch_x, batch_y) in vali_loader:
      pred, true = self._process_one_batch(batch_x, batch_y, validation=True)
      loss = criterion(pred.detach().cpu(), true.detach().cpu())
      total_loss.append(loss)
    total_loss = np.average(total_loss)
    self.model.train()
    return total_loss

  def train(self, setting):
    """Training Function.

    Args:
        setting: Name used to save the model

    Returns:
        model: Trained model
    """

    # Load different datasets
    train_loader = self._get_data(flag='train')
    vali_loader = self._get_data(flag='val')
    test_loader = self._get_data(flag='test')

    path = os.path.join(self.args.checkpoints, setting)
    if not os.path.exists(path):
      os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    # Setting optimizer and loss functions
    model_optim = self._select_optimizer()
    criterion = nn.MSELoss()

    all_training_loss = []
    all_validation_loss = []

    # Training Loop
    for epoch in range(self.args.train_epochs):
      iter_count = 0
      train_loss = []

      self.model.train()
      epoch_time = time.time()
      for i, (batch_x, batch_y) in enumerate(train_loader):
        iter_count += 1
        model_optim.zero_grad()
        if self.model_type == 'SDT':
          (pred, panelty), true = self._process_one_batch(batch_x, batch_y)
          loss = criterion(pred, true) + panelty
        else:
          pred, true = self._process_one_batch(batch_x, batch_y)
          loss = criterion(pred, true)
        train_loss.append(loss.item())

        if (i + 1) % 100 == 0:
          print('\titers: {0}/{1}, epoch: {2} | loss: {3:.7f}'.format(
              i + 1, train_steps, epoch + 1, loss.item()))
          speed = (time.time() - time_now) / iter_count
          left_time = speed * (
              (self.args.train_epochs - epoch) * train_steps - i)
          print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(
              speed, left_time))
          iter_count = 0
          time_now = time.time()

        loss.backward()
        model_optim.step()

      print('Epoch: {} cost time: {}'.format(epoch + 1,
                                             time.time() - epoch_time))
      train_loss = np.average(train_loss)
      all_training_loss.append(train_loss)
      vali_loss = self.vali(vali_loader, criterion)
      all_validation_loss.append(vali_loss)
      test_loss = self.vali(test_loader, criterion)

      print(
          'Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}'
          .format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
      early_stopping(vali_loss, self.model, path)

      # Plotting train and validation loss
      if ((epoch + 1) % 5 == 0 and self.args.plot):
        check_folder = os.path.isdir(self.args.plot_dir)

        # If folder doesn't exist, then create it.
        if not check_folder:
          os.makedirs(self.args.plot_dir)

        plt.figure()
        plt.plot(all_training_loss, label='train loss')
        plt.plot(all_validation_loss, label='Val loss')
        plt.legend()
        plt.savefig(self.args.plot_dir + setting + '.png')
        plt.show()
        plt.close()

      # If ran out of patience stop training
      if early_stopping.early_stop:
        if self.args.plot:
          plt.figure()
          plt.plot(all_training_loss, label='train loss')
          plt.plot(all_validation_loss, label='Val loss')
          plt.legend()
          plt.savefig(self.args.plot_dir + setting + '.png')
          plt.show()
        print('Early stopping')
        break
    best_model_path = path + '/' + 'checkpoint.pth'
    self.model.load_state_dict(torch.load(best_model_path))

    return self.model

  def predict(self, setting, load=False):
    """Prediction Function.

    Args:
      setting: Name used to be used for prediction
      load: whether to load best model

    Returns:
      mae: Mean absolute error
      mse: Mean squared error
      rmse: Root mean squared error
      mape: Mean absolute percentage error
      mspe: Mean squared percentage error
    """

    # Create prediction dataset
    pred_loader = self._get_data(flag='pred')

    # Load best model saved in the checkpoint folder
    if load:
      path = os.path.join(self.args.checkpoints, setting)
      best_model_path = path + '/' + 'checkpoint.pth'
      self.model.load_state_dict(torch.load(best_model_path))

    # Get model predictions
    self.model.eval()

    for i, (batch_x, batch_y) in enumerate(pred_loader):
      pred, true = self._process_one_batch(batch_x, batch_y, validation=True)
      if i == 0:
        preds = pred.detach().cpu().numpy()
        trues = true.detach().cpu().numpy()
      else:
        preds = np.concatenate((preds, pred.detach().cpu().numpy()), axis=0)
        trues = np.concatenate((trues, true.detach().cpu().numpy()), axis=0)

    if self.args.scale:
      # Transform dataset back to orignal form
      preds = self.label_scaler.inverse_transform(preds.reshape(-1, 1))
      trues = self.label_scaler.inverse_transform(trues.reshape(-1, 1))

    # save predictions made by model
    folder_path = './results/' + setting + '/'
    check_folder = os.path.isdir(folder_path)
    if not check_folder:
      os.makedirs(folder_path)
    np.save(folder_path + 'real_prediction.npy', preds)

    # Evaluate the model preformance
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))

    return mae, mse, rmse, mape, mspe, 0, 0

  def _process_one_batch(self, batch_x, batch_y, validation=False):
    """Function to process batch and send it to model and get output.

    Args:
       batch_x: batch input
       batch_y: batch target
       validation: flag to determine if this process is done for training or
         testing

    Returns:
        outputs: model outputs
        batch_y: batch target
    """

    batch_x = batch_x.float().to(self.device)
    batch_y = batch_y.float().to(self.device)

    if self.model_type == 'SDT':
      if not validation:
        outputs, panelty = self.model(batch_x, is_training_data=True)
        return (outputs, panelty), batch_y
      else:
        outputs = self.model(batch_x, is_training_data=False)
        return outputs, batch_y
    else:
      outputs = self.model(batch_x)
      batch_y = batch_y.squeeze()
      return outputs, batch_y
