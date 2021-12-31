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

"""Interpretable mixture of experts for time series data."""
import os
import warnings

from data.data_loader import ECL
from exp.exp_basic import  ExpBasic
import matplotlib.pyplot as plt
from models.ime import IME
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from utils.metrics import Metric as metric
from utils.tools import add_gaussian_noise
from utils.tools import EarlyStopping

warnings.filterwarnings("ignore")
eps = 1e-7


class ExpIME(ExpBasic):
  """Experiments for Interpretable mixture of experts for time series data."""

  def __init__(self, args):
    super(ExpIME, self).__init__(args)
    self.n_forecasts = args.pred_len
    self.n_lags = args.seq_len
    self.num_experts = args.num_experts
    self.error_scaler = StandardScaler()

  def _get_dataset(self):
    """Function creates dataset based on data name in the parsers.

    Returns:
         data: An instant of the dataset created
    """

    if self.args.data == "ECL":
      data = ECL(self.args.root_path, self.args.seq_len, self.args.pred_len,
                 self.args.features, self.args.scale, self.args.num_ts)
    else:
      raise NotImplementedError
    return data

  def _build_model(self):
    """Function that creates a model instance based on the model name.

        Here we only support LSTM, Linear and  ARNet.

    Returns:
          model: An instance of the model.
    """

    if self.args.model == "IME_WW":
      model = IME(
          num_experts=self.args.num_experts,
          n_forecasts=self.args.pred_len,
          n_lags=self.args.seq_len * self.args.input_dim,
          input_features=self.args.input_dim,
          gate_type="Linear",
          expert_type=self.args.expert_type,
          dropout=self.args.dropout,
          device=self.device)
    else:

      model = IME(
          num_experts=self.args.num_experts,
          n_forecasts=self.args.pred_len,
          n_lags=self.args.seq_len * self.args.input_dim,
          input_features=self.args.input_dim,
          gate_type="LSTM",
          expert_type=self.args.expert_type,
          d_model=self.args.d_model,
          layers=self.args.layers,
          dropout=self.args.dropout,
          device=self.device)
    # if multiple GPU are to be used parallelize model

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
    # Here we initialize matrix that will store last past error
    # and selection probabilty for each expert by the gate
    if flag == "test":
      shuffle_flag = False
      drop_last = True
      batch_size = self.args.batch_size
      data_set = TensorDataset(
          torch.Tensor(self.data.test_x), torch.Tensor(self.data.test_index),
          torch.Tensor(self.data.test_y))
      self.past_test_error = torch.zeros(
          (len(self.data.test_x), self.num_experts),
          requires_grad=False).to(self.device)
      self.gate_weights_test = torch.ones(
          (len(self.data.test_x), self.num_experts), requires_grad=False).to(
              self.device) * 1 / self.num_experts

    elif flag == "pred":
      shuffle_flag = False
      drop_last = False
      batch_size = self.args.batch_size
      data_set = TensorDataset(
          torch.Tensor(self.data.test_x), torch.Tensor(self.data.test_index),
          torch.Tensor(self.data.test_y))
      self.past_test_error = torch.zeros(
          (len(self.data.test_x), self.num_experts),
          requires_grad=False).to(self.device)
      self.past_train_error = torch.zeros(
          (len(self.data.train_x), self.num_experts),
          requires_grad=False).to(self.device)

    elif flag == "val":
      shuffle_flag = False
      drop_last = False
      batch_size = self.args.batch_size
      data_set = TensorDataset(
          torch.Tensor(self.data.valid_x), torch.Tensor(self.data.valid_index),
          torch.Tensor(self.data.valid_y))
      self.past_val_error = torch.zeros(
          (len(self.data.valid_x), self.num_experts),
          requires_grad=False).to(self.device)
      self.gate_weights_val = torch.ones(
          (len(self.data.valid_x), self.num_experts), requires_grad=False).to(
              self.device) * 1 / self.num_experts
    else:
      shuffle_flag = False
      drop_last = True
      batch_size = self.args.batch_size
      data_set = TensorDataset(
          torch.Tensor(self.data.train_x), torch.Tensor(self.data.train_index),
          torch.Tensor(self.data.train_y))
      self.past_train_error = torch.zeros(
          (len(self.data.train_x), self.num_experts),
          requires_grad=False).to(self.device)
      self.gate_weights_train = torch.ones(
          (len(self.data.train_x), self.num_experts), requires_grad=False).to(
              self.device) * 1 / self.num_experts

    print("Data for", flag, "dataset size", len(data_set))

    # Fitting past error matrix
    self.error_scaler.fit(
        self.past_train_error.detach().cpu().numpy().flatten().reshape(-1, 1))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=self.args.num_workers,
        drop_last=drop_last)

    if flag == "train":
      data_loader_shuffled = DataLoader(
          data_set,
          batch_size=batch_size,
          shuffle=True,
          num_workers=self.args.num_workers,
          drop_last=drop_last)

      return data_loader, data_loader_shuffled
    else:
      return data_loader

  def expert_utilization_loss(self, batch_expert_utilization, k=1):
    """Function calculates the overall expert utilization loss.

    Args:
      batch_expert_utilization: the utilization of each expert in a given batch
      k: a constant

    Returns:
       expert_utilization_loss= 1/num_experts * sum_i=1^1
                                                 [num_experts(e^-kU_i-e^-k)]
    """
    expert_utilization_loss = torch.Tensor([0]).to(self.device)
    batch_expert_utilization = batch_expert_utilization.squeeze()
    for y in range(self.num_experts):
      expert_utilization_loss += (
          torch.exp(-k * batch_expert_utilization[y]) -
          torch.exp(-torch.Tensor([k])).to(self.device))
    return expert_utilization_loss / self.num_experts

  def accuracy_loss(self, pred, true, weights):
    """Function to calculate mixture of expert loss.

       The loss is  described here:
       https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf

    Args:
          pred: a tensor containing prediction made by each expert
          true: a tensor containing the ground truth value
          weights: a tensor containing the weight assigned to each expert by the
            gating module

    Returns:
          loss: mixture of expert accuracy loss
    """

    batch_size = pred.shape[0]
    output = torch.zeros((batch_size, self.num_experts))
    for y in range(self.num_experts):
      output[:, y] = weights[:, y, 0] * (1 / (self.num_experts)) * torch.exp(
          -0.5 *
          torch.mean(torch.pow(true.squeeze(-1) - pred[:, :, y], 2), dim=1))

    loss = -torch.log(output.sum(dim=1))
    loss = torch.mean(loss)
    return loss

  def diversity_loss(self, pred, pred_noisy, temperature=0.2):
    """Function calculates the diversity loss between different experts.

        The loss is adapted from https://arxiv.org/abs/2002.05709

    Args:
      pred: a tensor containing prediction made by each expert
      pred_noisy: a tensor containing noisy prediction made by each expert
      temperature: temperature used to adjust strength of the exponential

    Returns:
        loss_dv: diversity loss.
    """

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_dv = 0
    for x in range(self.num_experts):
      denominator = 0
      for y in range(self.num_experts):
        if x != y:
          sim_x_y = torch.mean(cos(pred[:, :, x], pred[:, :, y]))
          denominator += torch.exp(sim_x_y / temperature)
        else:
          sim_x_x = torch.mean(cos(pred[:, :, x], pred_noisy[:, :, y]))
          numerator = torch.exp(sim_x_x / temperature)
      loss_dv += -torch.log(numerator / denominator)
    # If loss is negative set to zero
    if loss_dv < 0:
      loss_dv = torch.tensor([0.0], requires_grad=True).to(self.device)

    return loss_dv

  def model_assignment_error(self, pred, true):
    """Calculates the error made by each expert.

    Args:
        pred: a tensor containing prediction made by each expert
        true: a tensor containing the ground truth value

    Returns:
          masked_error: error made by each expert for the first time step
          error: error made by each expert over the entire forecasting horizon
    """

    batch_size = pred.shape[0]
    masked_pred = pred[:, 0, :].clone().unsqueeze(1)
    masked_true = true[:, 0].clone().unsqueeze(1)
    error = torch.zeros((batch_size, self.num_experts)).to(self.device)
    masked_error = torch.zeros((batch_size, self.num_experts)).to(self.device)
    for y in range(self.num_experts):
      masked_error[:, y] = torch.mean(
          torch.pow(masked_true.squeeze(-1) - masked_pred[:, :, y], 2), dim=1)
      error[:, y] = torch.mean(
          torch.pow(true.squeeze(-1) - pred[:, :, y], 2), dim=1)
    return masked_error, error

  def get_gate_assignment_weights(self, index, flag):
    """Functions returns the assignement weights of the previous time series.

    Args:
        index: a tensor containing index of each sample in the batch
        flag: a string indicating dataset type

    Returns:
          selected_values: a tensor containing the weights of each index
          previous time step
    """

    batch_size = index.shape[0]
    index = index.to(self.device)
    selected_values = torch.zeros(
        (batch_size, self.num_experts)).to(self.device)

    if 0 in index:
      for i in range(batch_size):
        if index[i] != 0:
          if flag == "test":
            selected_values[i] = self.gate_weights_test[(index[i] - 1).long()]
          elif flag == "train":
            selected_values[i] = self.gate_weights_train[(index[i] - 1).long()]
          elif flag == "val":
            selected_values[i] = self.gate_weights_val[(index[i] - 1).long()]
        else:
          # if index is zero i.e no previous index set the weights to be
          # uniformly distributed
          uniform = [i / self.num_experts for i in range(self.num_experts)]
          selected_values[i] = torch.Tensor(uniform)
    else:
      if flag == "test":
        selected_values = self.gate_weights_test[index.long() - 1, :]
      elif flag == "train":
        selected_values = self.gate_weights_train[index.long() - 1, :]
      elif flag == "val":
        selected_values = self.gate_weights_val[index.long() - 1, :]

    return selected_values

  def set_gate_assignment_weights(self, weights, index, flag):
    """Functions that saves assignment weights made by the gate for experts.

    Args:
        weights: a tensor containing the assignment weights for each expert
        index: a tensor containing index of each sample in the batch
        flag: a string indicating dataset type
    """
    if flag == "test":
      self.gate_weights_test[index.long(), :] = weights
    elif flag == "train":
      self.gate_weights_train[index.long(), :] = weights
    elif flag == "val":
      self.gate_weights_val[index.long(), :] = weights
    return

  def get_past_errors(self, index, flag):
    """Functions returns the error made by experts in the previous time series.

      given an index.

    Args:
      index: a tensor containing index of each sample in the batch
      flag: a string indicating dataset type

    Returns:
        selected_values: a tensor containing the error made by experts in
                        previous time step
    """

    batch_size = index.shape[0]

    if 0 in index:
      past_errors = torch.zeros((batch_size, self.num_experts)).to(self.device)
      for i in range(batch_size):
        if index[i] != 0:
          if flag == "test":
            past_errors[i] = self.past_test_error[(index[i] - 1).long()]
          elif flag == "train":
            past_errors[i] = self.past_train_error[(index[i] - 1).long()]
          elif flag == "val":
            past_errors[i] = self.past_val_error[(index[i] - 1).long()]
    else:
      if flag == "test":
        past_errors = self.past_test_error[index.long() - 1, :]
      elif flag == "train":
        past_errors = self.past_train_error[index.long() - 1, :]
      elif flag == "val":
        past_errors = self.past_val_error[index.long() - 1, :]

    return past_errors

  def set_past_errors(self, error, index, flag):
    """Functions that updates error made by each expert.

    Args:
          error: a tensor containing the error made by each expert
          index: a tensor containing index of each sample in the batch
          flag: a string indicating dataset type
    """

    if flag == "test":
      self.past_test_error[index.long(), :] = error
    elif flag == "train":
      self.past_train_error[index.long(), :] = error
    elif flag == "val":
      self.past_val_error[index.long(), :] = error
    return

  def _select_optimizer(self):
    """Function that returns the optimizers based on learning rate.

    Returns:
        model_optim: model optimizer
        gate_optim: gate network optimizer

    """

    model_optim = optim.Adam([{
        "params": self.model.experts.parameters()
    }, {
        "params": self.model.gate_network.parameters(),
        "lr": self.args.learning_rate_gate
    }],
                             lr=self.args.learning_rate)

    # SGD is used as gate optimizer as it does not contain weight decay
    gate_optim = optim.SGD(
        self.model.gate_network.parameters(), lr=self.args.learning_rate_gate)

    return model_optim, gate_optim

  def _select_criterion(self):
    """Function that returns the loss functions.

    Returns:
          criterion_mse: criterion used in for MSE loss function
          criterion_kl: criterion used in for KL divergance loss function
    """

    criterion_mse = nn.MSELoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")

    return criterion_mse, criterion_kl

  def get_upper_bound_accuracy(self,
                               test_loader,
                               hide_print=False,
                               flag="train",
                               plot=False,
                               setting=None):
    """Function that calculates upper bound, oracle accuracy and mse.

    Args:
      test_loader: data loader.
      hide_print: a boolean to indicate whether to print each expert accuracy.
      flag: a string indicating dataset type.
      plot:  a boolean to indicate whether plot oracle assignment over time and
        expert error.
      setting: a string containing model name.

    Returns:
      mse: mean squared error for entire IME.
      upper_bound: best possible mse had the gate network choosen the best
      expert.
      oracle_acc: Oracle accuracy showing the percentage of times the gate
      network chooses the best expert.


    """
    self.model.eval()
    upper_bound = 0
    expert = [0 for i in range(self.num_experts)]
    expert_error = [[] for i in range(self.num_experts)]
    for i, (batch_x, index, batch_y) in enumerate(test_loader):

      # get the error made by experts in each previous time step
      past_val_errors = self.get_past_errors(index, flag)
      # get the output of the model (Argmax weights * expert),
      # each expert output and weights assigned to each expert
      output, expert_output, weights, true = self._process_one_batch(
          batch_x, batch_y, past_errors=past_val_errors, validation=True)

      # get the error made by each expert
      masked_error, error = self.model_assignment_error(expert_output, true)

      # update the error matrix with new errors note that we update with
      # only first time step error to prevent information leakage
      self.set_past_errors(masked_error.detach(), index, flag)

      # the best accuracy would be the one that gives minimum error.
      _, actual_best = torch.min(error, 1)
      # the choosen is the expert with higest weight by gate network
      _, choosen = torch.max(weights, 1)

      upper_bound += torch.mean(torch.min(error, 1)[0])

      for y in range(self.num_experts):
        expert[y] += torch.mean(error[:, y]).item()
        expert_error[y].append(torch.mean(error[:, y]).item())

      # or the entire dataset get predictions and labels
      if i == 0:
        preds = output.detach().cpu().numpy()
        trues = true.detach().cpu().numpy()
      else:
        preds = np.concatenate((preds, output.detach().cpu().numpy()), axis=0)
        trues = np.concatenate((trues, true.detach().cpu().numpy()), axis=0)

      # for the entire dataset get best expert (with minimum error)
      # and choosen expert by gate network
      if i == 0:
        actual_best_expert = actual_best.detach().cpu().numpy()
        choosen_expert = choosen.squeeze(-1).detach().cpu().numpy()
      else:
        actual_best_expert = np.concatenate(
            (actual_best_expert, actual_best.detach().cpu().numpy()), axis=0)
        choosen_expert = np.concatenate(
            (choosen_expert, choosen.squeeze(-1).detach().cpu().numpy()),
            axis=0)
    # Oracle accuracy calcuates the percentage of times the gate network
    # chooses the best expert
    oracle_acc = accuracy_score(actual_best_expert, choosen_expert)

    upper_bound = upper_bound.item() / len(test_loader)

    # update past error matrix
    if flag == "test":
      self.past_test_error = torch.Tensor(
          self.error_scaler.transform(
              self.past_test_error.detach().cpu().numpy().flatten().reshape(
                  -1, 1))).reshape(-1, self.num_experts).to(self.device)
    elif flag == "train":
      self.error_scaler.fit(
          self.past_train_error.detach().cpu().numpy().flatten().reshape(-1, 1))
      self.past_train_error = torch.Tensor(
          self.error_scaler.transform(
              self.past_train_error.detach().cpu().numpy().flatten().reshape(
                  -1, 1))).reshape(-1, self.num_experts).to(self.device)
    else:
      self.past_val_error = torch.Tensor(
          self.error_scaler.transform(
              self.past_val_error.detach().cpu().numpy().flatten().reshape(
                  -1, 1))).reshape(-1, self.num_experts).to(self.device)

    # get accuracy metrics
    _, mse, _, _, _ = metric(preds, trues)

    if not hide_print:
      print(flag)
      print("oracle accuracy: {0:.7f}".format(oracle_acc))
      print("Upper bound: {0:.7f}".format(upper_bound))
      for y in range(self.num_experts):
        print("Expert {0}:  {1:.7f}".format(y, expert[y] / len(test_loader)))
      print("MSE: {0:.7f}".format(mse))
    print()

    if plot:
      plt.figure()
      plt.plot(actual_best_expert)
      plt.title("oracle assignment")
      plt.legend()
      plt.savefig("./Graphs/" + setting + "_oracle_assignment_" + flag + ".png")
      plt.show()
      plt.close()
      plt.figure()

      _, axs = plt.subplots(self.num_experts, 1)

      for y in range(self.num_experts):
        axs[y].plot(expert_error[y])
        axs[y].set_xlabel("expert" + str(y))
      plt.savefig("./Graphs/" + setting + "_expert_error_" + flag + ".png")
      plt.show()
      plt.close()

    return mse, upper_bound, oracle_acc

  def vali(self, vali_loader, flag="val"):
    """Validation Function.

    Args:
      vali_loader: Validation dataloader.
      flag: a string indicating dataset type.

    Returns:
      vali_loss: overall validation loss.
      avg_acc_loss: average accuracy loss.
      avg_utilization_loss: average utilization loss.
      avg_smoothness_loss: average smoothness loss.
      avg_diversity_loss: average diversity loss.

    """

    self.model.eval()
    vali_loss = []
    avg_acc_loss = 0
    avg_utilization_loss = 0
    avg_diversity_loss = 0
    avg_smoothness_loss = 0
    avg_gate_loss = 0
    criterion_mse, criterion_kl = self._select_criterion()

    for (batch_x, index, batch_y) in vali_loader:

      past_val_errors = self.get_past_errors(index, flag)

      pred, true, weights, reg_out = self._process_one_batch(
          batch_x, batch_y, past_errors=past_val_errors)

      batch_size = batch_x.shape[0]

      # Calcuate accuracy loss
      accuracy_loss = self.accuracy_loss(pred, true, weights)
      avg_acc_loss += self.args.accuracy_hp * accuracy_loss.item()

      # Calcuate gate loss
      gate_loss = criterion_mse(reg_out, true)
      avg_gate_loss += self.args.gate_hp * gate_loss.item()

      # Calcuate utilization loss
      if self.args.utilization_hp != 0:
        batch_expert_utilization = torch.sum(
            weights.squeeze(-1), dim=0) / batch_size
        expert_utilization_loss = self.expert_utilization_loss(
            batch_expert_utilization)
        avg_utilization_loss += self.args.utilization_hp * expert_utilization_loss.item(
        )
      else:
        expert_utilization_loss = 0

      # Calcuate smoothness loss
      if self.args.smoothness_hp != 0:
        previous_weight = self.get_gate_assignment_weights(index, "train")
        smoothness_loss = criterion_kl(
            torch.log(weights.squeeze(-1) + eps), previous_weight)
        self.set_gate_assignment_weights(
            weights.squeeze(-1).detach(), index, "train")
        avg_smoothness_loss += self.args.smoothness_hp * smoothness_loss.item()
      else:
        smoothness_loss = 0

      # Calcuate diversity loss
      if self.args.diversity_hp != 0:
        batch_x_noisy = add_gaussian_noise(batch_x)
        pred_noisy, _, _, _ = self._process_one_batch(
            batch_x_noisy, batch_y, past_errors=past_val_errors)
        diversity_loss = self.diversity_loss(pred, pred_noisy)
        avg_diversity_loss += self.args.diversity_hp * diversity_loss.item()
      else:
        diversity_loss = 0

      # Get expert error
      masked_error, _ = self.model_assignment_error(pred, true)
      self.set_past_errors(masked_error.detach(), index, "train")

      loss = (
          self.args.accuracy_hp * accuracy_loss +
          self.args.gate_hp * gate_loss +
          self.args.utilization_hp * expert_utilization_loss +
          self.args.smoothness_hp * smoothness_loss +
          self.args.diversity_hp * diversity_loss)
      vali_loss.append(loss.item())

    vali_loss = np.average(vali_loss)

    # update past error matrix
    if flag == "val":
      self.past_val_error = torch.Tensor(
          self.error_scaler.transform(
              self.past_val_error.detach().cpu().numpy().flatten().reshape(
                  -1, 1))).reshape(-1, self.num_experts).to(self.device)
    else:
      self.error_scaler.fit(
          self.past_train_error.detach().cpu().numpy().flatten().reshape(-1, 1))
      self.past_train_error = torch.Tensor(
          self.error_scaler.transform(
              self.past_train_error.detach().cpu().numpy().flatten().reshape(
                  -1, 1))).reshape(-1, self.num_experts).to(self.device)

    avg_acc_loss = avg_acc_loss / len(vali_loader)
    avg_utilization_loss = avg_utilization_loss / len(vali_loader)
    avg_smoothness_loss = avg_smoothness_loss / len(vali_loader)
    avg_gate_loss = avg_gate_loss / len(vali_loader)
    avg_diversity_loss = avg_diversity_loss / len(vali_loader)

    print(
        flag,
        ":\tAcc Loss: {0:.7f} | Utilization Loss: {1:.7f} | Smoothness Loss: {2:.7f}  | Diversity Loss {3:.7f} | Gate loss {4:.7f} "
        .format(avg_acc_loss, avg_utilization_loss, avg_smoothness_loss,
                avg_diversity_loss, avg_gate_loss))

    return (vali_loss, avg_acc_loss, avg_utilization_loss, avg_smoothness_loss,
            avg_diversity_loss)

  def train(self, setting):
    """Training Function.

    Args:
      setting: Name used to save the model.

    Returns:
       model: Trained model.
    """
    # Load different datasets
    train_loader, train_loader_shuffled = self._get_data(flag="train")
    vali_loader = self._get_data(flag="val")
    test_loader = self._get_data(flag="test")

    path = os.path.join(self.args.checkpoints, setting)
    if not os.path.exists(path):
      os.makedirs(path)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    # Setting optimizer and loss functions
    model_optim, gate_optim = self._select_optimizer()
    criterion, criterion_kl = self._select_criterion()

    loss_train = []
    acc_loss_train = []
    utilization_loss_train = []
    smoothness_loss_train = []
    diversity_loss_train = []
    loss_val = []
    acc_loss_val = []
    utilization_loss_val = []
    smoothness_loss_val = []
    diversity_loss_val = []

    mse_train_ = []
    mse_test_ = []
    mse_val_ = []
    upper_bound_train_ = []
    upper_bound_test_ = []
    upper_bound_val_ = []
    oracle_acc_test_ = []
    oracle_acc_train_ = []
    oracle_acc_val_ = []

    # Getting the intial mse, oracle accuracy and upper bound
    (mse_train, upper_bound_train,
     oracle_acc_train) = self.get_upper_bound_accuracy(
         train_loader, flag="train")
    (mse_test, upper_bound_test,
     oracle_acc_test) = self.get_upper_bound_accuracy(
         test_loader, flag="test")
    (mse_val, upper_bound_val, oracle_acc_val) = self.get_upper_bound_accuracy(
        vali_loader, flag="val")

    mse_train_.append(mse_train)
    mse_test_.append(mse_test)
    mse_val_.append(mse_val)

    upper_bound_train_.append(upper_bound_train)
    upper_bound_test_.append(upper_bound_test)
    upper_bound_val_.append(upper_bound_val)

    oracle_acc_train_.append(oracle_acc_train)
    oracle_acc_test_.append(oracle_acc_test)
    oracle_acc_val_.append(oracle_acc_val)

    # Training loop
    for epoch in range(self.args.train_epochs):

      self.model.train()
      loss_all = 0

      # Add noise to the weights of the expert this promotes diversity
      if self.args.noise:
        with torch.no_grad():
          for param in self.model.experts.parameters():
            param.add_(torch.randn(param.size()).to(self.device) * 0.01)

      for i, (batch_x, index, batch_y) in enumerate(train_loader_shuffled):
        # get past error made by experts
        past_errors = self.get_past_errors(index, "train")

        model_optim.zero_grad()
        pred, true, weights, reg_out = self._process_one_batch(
            batch_x, batch_y, past_errors=past_errors)

        batch_size = pred.shape[0]

        # Calcuate accuracy loss
        accuracy_loss = self.accuracy_loss(pred, true, weights)

        # Calcuate gate loss
        gate_loss = criterion(reg_out, true)

        # Calcuate utilization loss
        if self.args.utilization_hp != 0:
          batch_expert_utilization = torch.sum(
              weights.squeeze(-1), dim=0) / batch_size
          expert_utilization_loss = self.expert_utilization_loss(
              batch_expert_utilization)
        else:
          expert_utilization_loss = 0

        # Calcuate smoothness loss
        if self.args.smoothness_hp != 0:
          previous_weight = self.get_gate_assignment_weights(index, "train")
          smoothness_loss = criterion_kl(
              torch.log(weights.squeeze(-1) + eps), previous_weight)
          self.set_gate_assignment_weights(
              weights.squeeze(-1).detach(), index, "train")
        else:
          smoothness_loss = 0

        # Calcuate diversity loss
        if self.args.diversity_hp != 0:
          batch_x_noisy = add_gaussian_noise(batch_x)
          pred_noisy, _, _, _ = self._process_one_batch(
              batch_x_noisy, batch_y, past_errors=past_errors)
          diversity_loss = self.diversity_loss(pred, pred_noisy)
          # avg_diversity_loss += self.args.diversity_hp * diversity_loss.item()
        else:
          diversity_loss = 0

        # set expert error
        masked_error, _ = self.model_assignment_error(pred, true)
        self.set_past_errors(masked_error.detach(), index, "train")

        loss = (
            self.args.accuracy_hp * accuracy_loss +
            self.args.gate_hp * gate_loss +
            self.args.utilization_hp * expert_utilization_loss +
            self.args.smoothness_hp * smoothness_loss +
            self.args.diversity_hp * diversity_loss)

        loss.backward()
        model_optim.step()
        loss_all += loss.item()

        if (i + 1) % 50 == 0:
          print("\tOne iters: {0}/{1}, epoch: {2} | loss: {3:.7f}".format(
              i + 1, len(train_loader), epoch + 1, loss_all / i))
      # update past error matrix
      self.error_scaler.fit(
          self.past_train_error.detach().cpu().numpy().flatten().reshape(-1, 1))
      self.past_train_error = torch.Tensor(
          self.error_scaler.transform(
              self.past_train_error.detach().cpu().numpy().flatten().reshape(
                  -1, 1))).reshape(-1, self.num_experts).to(self.device)

      # Getting different losses
      (train_loss, acc_train, utilization_train, smoothness_train,
       diversity_train) = self.vali(train_loader, "train")

      loss_train.append(train_loss)
      acc_loss_train.append(acc_train)
      utilization_loss_train.append(utilization_train)
      smoothness_loss_train.append(smoothness_train)
      diversity_loss_train.append(diversity_train)

      (val_loss, acc_val, utilization_val, smoothness_val,
       diversity_val) = self.vali(
           vali_loader, flag="val")
      loss_val.append(val_loss)
      acc_loss_val.append(acc_val)
      utilization_loss_val.append(utilization_val)
      smoothness_loss_val.append(smoothness_val)
      diversity_loss_val.append(diversity_val)

      # getting mse, oracle accuracy and upper bound
      (mse_train, upper_bound_train,
       oracle_acc_train) = self.get_upper_bound_accuracy(
           train_loader, flag="train")
      (mse_test, upper_bound_test,
       oracle_acc_test) = self.get_upper_bound_accuracy(
           test_loader, flag="test")
      (mse_val, upper_bound_val,
       oracle_acc_val) = self.get_upper_bound_accuracy(
           vali_loader, flag="val")

      mse_train_.append(mse_train)
      mse_test_.append(mse_test)
      mse_val_.append(mse_val)

      upper_bound_train_.append(upper_bound_train)
      upper_bound_test_.append(upper_bound_test)
      upper_bound_val_.append(upper_bound_val)

      oracle_acc_train_.append(oracle_acc_train)
      oracle_acc_test_.append(oracle_acc_test)
      oracle_acc_val_.append(oracle_acc_test)

      # early stopping depends on the validation accuarcy loss
      early_stopping(acc_val, self.model, path)

      print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}"
            .format(epoch + 1, train_steps, train_loss, val_loss))

      if ((epoch + 1) % 10 == 0 and self.args.plot):

        self.plot_all(setting, mse_test_, mse_train_, mse_val_,
                      upper_bound_test_, upper_bound_train_, upper_bound_val_,
                      oracle_acc_test_, oracle_acc_train_, oracle_acc_val_,
                      loss_train, loss_val, acc_loss_train, acc_loss_val,
                      utilization_loss_train, utilization_loss_val,
                      smoothness_loss_train, smoothness_loss_val,
                      diversity_loss_train, diversity_loss_val)

      # when training runs out of patience
      if early_stopping.early_stop:
        break

    # if freezing experts and tunning gate network
    if (self.args.freeze and oracle_acc_train != 1):

      # load the best model
      best_model_path = path + "/" + "checkpoint.pth"
      self.model.load_state_dict(torch.load(best_model_path))

      # set past errors to zero
      self.past_train_error[:, :] = 0
      self.past_test_error[:, :] = 0
      self.past_val_error[:, :] = 0

      # get validation accuracy on the best model
      (val_loss, acc_val, utilization_val, smoothness_val,
       diversity_val) = self.vali(
           vali_loader, flag="val")

      # reseting and adjusting early_stopping
      early_stopping.val_loss_min = acc_val
      early_stopping.counter = 0
      early_stopping.early_stop = False

      for e in range(self.args.train_epochs):
        self.model.train()
        loss_all = 0
        for i, (batch_x, index, batch_y) in enumerate(train_loader_shuffled):
          past_errors = self.get_past_errors(index, "train")
          gate_optim.zero_grad()

          pred, true, weights, reg_out = self._process_one_batch(
              batch_x, batch_y, past_errors=past_errors)

          accuracy_loss = self.accuracy_loss(pred, true, weights)
          # set expert error
          masked_error, _ = self.model_assignment_error(pred, true)
          self.set_past_errors(masked_error.detach(), index, "train")

          loss = accuracy_loss
          loss.backward()

          # clear the expert gradients since we want them frozen
          self.model.experts.zero_grad()

          gate_optim.step()
          loss_all += loss.item()

          if (i + 1) % 50 == 0:
            print(
                "\tFreeze iters: {0}/{1}, epoch: {2}  sub epoch {4} | loss: {3:.7f}"
                .format(i + 1, len(train_loader), epoch + 1, loss_all / i, e))

        # update past error matrix
        self.error_scaler.fit(
            self.past_train_error.detach().cpu().numpy().flatten().reshape(
                -1, 1))
        self.past_train_error = torch.Tensor(
            self.error_scaler.transform(
                self.past_train_error.detach().cpu().numpy().flatten().reshape(
                    -1, 1))).reshape(-1, self.num_experts).to(self.device)

        # Getting different losses
        (train_loss, acc_train, utilization_train, smoothness_train,
         diversity_train) = self.vali(
             train_loader, flag="train")

        loss_train.append(train_loss)
        acc_loss_train.append(acc_train)
        utilization_loss_train.append(utilization_train)
        smoothness_loss_train.append(smoothness_train)
        diversity_loss_train.append(diversity_train)

        (val_loss, acc_val, utilization_val, smoothness_val,
         diversity_val) = self.vali(
             vali_loader, flag="val")
        loss_val.append(val_loss)
        acc_loss_val.append(acc_val)
        utilization_loss_val.append(utilization_val)
        smoothness_loss_val.append(smoothness_val)
        diversity_loss_val.append(diversity_val)

        # getting mse, oracle accuracy and upper bound
        (mse_train, upper_bound_train,
         oracle_acc_train) = self.get_upper_bound_accuracy(
             train_loader, flag="train")
        (mse_test, upper_bound_test,
         oracle_acc_test) = self.get_upper_bound_accuracy(
             test_loader, flag="test")
        (mse_val, upper_bound_val,
         oracle_acc_val) = self.get_upper_bound_accuracy(
             vali_loader, flag="val")

        mse_train_.append(mse_train)
        mse_test_.append(mse_test)
        mse_val_.append(mse_val)

        upper_bound_train_.append(upper_bound_train)
        upper_bound_test_.append(upper_bound_test)
        upper_bound_val_.append(upper_bound_val)

        oracle_acc_train_.append(oracle_acc_train)
        oracle_acc_test_.append(oracle_acc_test)
        oracle_acc_val_.append(oracle_acc_test)

        # early stopping depends on the validation accuarcy loss
        early_stopping(acc_val, self.model, path)

        print(
            "Frozen Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}"
            .format(e + 1, train_steps, train_loss, val_loss))

        if early_stopping.early_stop:
          print("Early stopping")
          break

    if self.args.plot:
      self.plot_all(setting, mse_test_, mse_train_, mse_val_, upper_bound_test_,
                    upper_bound_train_, upper_bound_val_, oracle_acc_test_,
                    oracle_acc_train_, oracle_acc_val_, loss_train, loss_val,
                    acc_loss_train, acc_loss_val, utilization_loss_train,
                    utilization_loss_val, smoothness_loss_train,
                    smoothness_loss_val, diversity_loss_train,
                    diversity_loss_val)
    # Load the best model on the validation dataset
    best_model_path = path + "/" + "checkpoint.pth"
    self.model.load_state_dict(torch.load(best_model_path))

    return self.model

  def plot_all(self, setting, mse_test_, mse_train_, mse_val_,
               upper_bound_test_, upper_bound_train_, upper_bound_val_,
               oracle_acc_test_, oracle_acc_train_, oracle_acc_val_, loss_train,
               loss_val, acc_loss_train, acc_loss_val, utilization_loss_train,
               utilization_loss_val, smoothness_loss_train, smoothness_loss_val,
               diversity_loss_train, diversity_loss_val):
    """Plotting Function.

    Args:
      setting: Name used to save the model.
      mse_test_: test mean square error.
      mse_train_: train mean square error.
      mse_val_: validation mean square error.
      upper_bound_test_: test upper bound.
      upper_bound_train_: test upper bound.
      upper_bound_val_: validation upper bound.
      oracle_acc_test_: test oracle accuracy.
      oracle_acc_train_: train oracle accuracy.
      oracle_acc_val_: validation oracle accuracy.
      loss_train: train  overall loss.
      loss_val: validation overall loss.
      acc_loss_train: train accuracy loss.
      acc_loss_val: validation accuracy loss.
      utilization_loss_train: train utilization loss.
      utilization_loss_val: validation utilization loss.
      smoothness_loss_train: train smoothness loss.
      smoothness_loss_val: validation smoothness loss.
      diversity_loss_train: train diversity loss.
      diversity_loss_val: validation diversity loss.
    """
    plt.figure()
    plt.plot(mse_test_, label="test")
    plt.plot(mse_train_, label="train")
    plt.plot(mse_val_, label="val")
    plt.plot(upper_bound_test_, label="test_upperbound")
    plt.plot(upper_bound_train_, label="train_upperbound")
    plt.plot(upper_bound_val_, label="val_upperbound")
    plt.title("MSE")
    plt.legend()
    plt.savefig("./Graphs/" + setting + "_model_mse.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(oracle_acc_test_, label="test")
    plt.plot(oracle_acc_train_, label="train")
    plt.plot(oracle_acc_val_, label="val")
    plt.ylim(0, 1)
    plt.title("Oracle Accuracy")
    plt.legend()
    plt.savefig("./Graphs/" + setting + "_oracle_acc.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(loss_train, label="train")
    plt.plot(loss_val, label="val")
    plt.title("Overall Loss")
    plt.legend()
    plt.savefig("./Graphs/" + setting + "_loss.png")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(acc_loss_train, label="train")
    plt.plot(acc_loss_val, label="val")
    plt.title("Accuracy Loss")
    plt.legend()
    plt.savefig("./Graphs/" + setting + "_loss.png")
    plt.show()
    plt.close()

    if self.args.utilization_hp != 0:
      plt.figure()
      plt.plot(utilization_loss_train, label="train")
      plt.plot(utilization_loss_val, label="val")
      plt.title("Utilization Loss")
      plt.legend()
      plt.savefig("./Graphs/" + setting + "_utilization_loss.png")
      plt.show()
      plt.close()
    if self.args.smoothness_hp != 0:
      plt.figure()
      plt.plot(smoothness_loss_train, label="train")
      plt.plot(smoothness_loss_val, label="val")
      plt.title("Smoothness Loss")
      plt.legend()
      plt.savefig("./Graphs/" + setting + "_smoothness_loss.png")
      plt.show()
      plt.close()
    if self.args.diversity_hp != 0:
      plt.figure()
      plt.plot(diversity_loss_train, label="train")
      plt.plot(diversity_loss_val, label="val")
      plt.title("Diversity Loss")
      plt.legend()
      plt.savefig("./Graphs/" + setting + "_diversity_loss.png")
      plt.show()
      plt.close()

  def predict(self, setting, load=False):
    """Prediction Function.

    Args:
      setting: Name used to be used for prediction.
      load: whether to load best model.

    Returns:
      mae: Mean absolute error.
      mse: Mean squared error.
      rmse: Root mean squared error.
      mape: Mean absolute percentage error.
      mspe: Mean squared percentage error.
    """

    # Create prediction dataset
    pred_loader = self._get_data(flag="pred")

    # Load best model saved in the checkpoint folder
    if load:
      path = os.path.join(self.args.checkpoints, setting)
      best_model_path = path + "/" + "checkpoint.pth"
      self.model.load_state_dict(torch.load(best_model_path))

    # Get model predictions
    self.model.eval()

    for i, (batch_x, index, batch_y) in enumerate(pred_loader):

      # get the error made by experts in each previous time step
      past_val_errors = self.get_past_errors(index, "test")
      # get the output of the model (Argmax weights * expert),
      # each expert output and weights assigned to each expert
      output, expert_output, weights, true = self._process_one_batch(
          batch_x, batch_y, past_errors=past_val_errors, validation=True)

      # the choosen is the expert with higest weight by gate network
      _, choosen = torch.max(weights, 1)

      # get the error made by each expert
      masked_error, _ = self.model_assignment_error(expert_output, true)

      # update the error matrix with new errors note that we update with
      # only first time step error to prevent information leakage
      masked_error = torch.Tensor(
          self.error_scaler.transform(
              masked_error.detach().cpu().numpy().flatten().reshape(
                  -1, 1))).reshape(-1, self.num_experts).to(self.device)
      self.set_past_errors(masked_error.detach(), index, "test")

      if i == 0:
        preds = output.detach().cpu().numpy()
        trues = true.detach().cpu().numpy()
        assignment = choosen.detach().cpu().numpy()
      else:
        preds = np.concatenate((preds, output.detach().cpu().numpy()), axis=0)
        trues = np.concatenate((trues, true.detach().cpu().numpy()), axis=0)
        assignment = np.concatenate(
            (assignment, choosen.detach().cpu().numpy()), axis=0)

    if self.args.plot:
      assignment = np.array(assignment).squeeze()
      plt.figure()
      plt.plot(assignment)
      plt.title("assignment over time")
      plt.legend()
      plt.savefig("./Graphs/" + setting + "_Final_assignment.png")
      plt.show()
      plt.close()

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    # save predictions made by model
    folder_path = "./results/" + setting + "/"
    check_folder = os.path.isdir(folder_path)
    if not check_folder:
      os.makedirs(folder_path)
    np.save(folder_path + "real_prediction.npy", preds)

    # Evaluate the model performance
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print("mse:{}, mae:{}, rmse:{}".format(mse, mae, rmse))

    _, upper_bound, oracle_acc = self.get_upper_bound_accuracy(
        pred_loader, flag="test", plot=self.args.plot, setting=setting)

    return mae, mse, rmse, mape, mspe, upper_bound, oracle_acc

  def _process_one_batch(self, batch_x, batch_y, past_errors, validation=False):
    """Function to process batch and send it to model forward or predict function.

    Args:
       batch_x: batch input
       batch_y: batch target
       past_errors: past error made by the experts
       validation: flag to determine if this process is done for training or
         testing

    Returns:
        outputs: model outputs during inference
        all_expert_output: output produced by each expert
        weights: weight assignment for each expert
        batch_y: batch target
    """

    # Reshape input for IME_WW
    if self.model_type == "IME_WW":
      batch_size, _, _ = batch_x.shape
      batch_x = batch_x.reshape(batch_size, -1)
    batch_x = batch_x.float().to(self.device)

    batch_y = batch_y.float()
    batch_y = batch_y[:, -self.args.pred_len:, 0].to(self.device).squeeze(-1)

    if validation:
      output, all_expert_output, weights = self.model.predict(
          batch_x, past_errors)
      return output, all_expert_output, weights, batch_y
    else:
      all_expert_output, weights, reg_out = self.model(
          batch_x, batch_y, past_errors, noise=self.args.noise)
      return all_expert_output, batch_y, weights, reg_out
