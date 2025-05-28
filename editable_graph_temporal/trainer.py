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

# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Defines the trainer class for training the GATRNN model."""
import os
import numpy as np
import torch
from tqdm import tqdm

from editable_graph_temporal import data
from editable_graph_temporal.model import gat_model


class Trainer:
  """Implements the overall training process."""

  def __init__(self, args):
    """Instantiates the trainer class.

    Args:
      args: python argparse.ArgumentParser class, containing configuration
        arguments for data, model, and training hyperparameters.
    """
    dataset = np.load(args.data_path)
    time_data, gt_adj = dataset["x"], dataset["adj"]
    self.gt_adj_torch = torch.from_numpy(gt_adj).float()

    _, num_nodes, num_attrs = time_data.shape
    args.num_nodes = num_nodes
    args.input_dim, args.output_dim = num_attrs, num_attrs
    adj_mx = self.gt_adj_torch

    self.args = args

    self.datamodule = data.DataModule(time_data, args)
    self.datamodule.setup()
    self.train_loader = self.datamodule.train_dataloader()
    self.val_loader = self.datamodule.val_dataloader()
    self.test_loader = self.datamodule.test_dataloader()

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gat_model.GATRNN(adj_mx, args)
    self.model = model.to(self.device)

  def _train_epoch(self, optim):
    """Runs one epoch of training.

    Args:
      optim: torch.optim class, the pytorch optimizer used to optimize the
        model.

    Returns:
      (average_loss, average_time_series_forecast_loss,
        average_regularization_loss).
    """
    self.model.train()
    total_loss, total_ts_loss, total_reg_loss = 0.0, 0.0, 0.0
    batch_idx = 0
    for batch in tqdm(self.train_loader):
      input_batch, y = batch
      input_batch, y = input_batch.to(self.device), y.to(self.device)
      output_batch, prob = self.model(input_batch)
      ts_loss = self.model.loss(output_batch, y)
      total_ts_loss += ts_loss.detach().cpu().item()

      reg_loss = self._reg_loss(prob)
      total_reg_loss += reg_loss.detach().cpu().item()

      loss = ts_loss + self.args.reg_co * reg_loss

      optim.zero_grad()
      loss.backward()
      optim.step()
      total_loss += loss.detach().cpu().item()
      batch_idx += 1

    return total_loss / (batch_idx + 1), total_ts_loss / (batch_idx +
                                                          1), total_reg_loss / (
                                                              batch_idx + 1)

  def _reg_loss(self, prob):
    """Computes and returns the regularization loss.

    Args:
      prob: predicted relation type probability for each two-node pair, with
        shape (self.num_nodes*self.num_nodes, self.num_relation_types).

    Returns:
      regularization loss tensor.
    """
    if prob is None or self.args.reg_loss == "none":
      return torch.tensor(0.0, device=self.device)
    if self.args.reg_loss == "uniform":
      kl_div = prob * torch.log(prob + 1e-16)
      return kl_div.sum() / (self.args.num_nodes * self.args.num_nodes)
    if self.args.reg_loss == "zero":
      kl_zero = -torch.log(prob[:, 0] + 1e-16)
      return kl_zero.sum() / (self.args.num_nodes * self.args.num_nodes)

  def train(self):
    """Running the overall training process."""

    if not os.path.isdir(os.path.join(self.args.save_dir, "checkpoints")):
      os.mkdir(os.path.join(self.args.save_dir, "checkpoints"))
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr=self.args.learning_rate,
        eps=1.0e-3)

    with open(os.path.join(self.args.save_dir, "train.txt"), "a") as f:
      f.write(
          "hidden dim {}, num layers {}, dropout {}, lr {}, bs {}, lagging {}, horizon {}, temp {}, reg_co {}\n"
          .format(self.args.hidden_dim, self.args.num_layers, self.args.dropout,
                  self.args.learning_rate, self.args.batch_size,
                  self.args.input_len, self.args.output_len,
                  self.args.temperature, self.args.reg_co))

    with open(os.path.join(self.args.save_dir, "val_test.txt"), "a") as f:
      f.write(
          "hidden dim {}, num layers {}, dropout {}, lr {}, bs {}, lagging {}, horizon {}, temp {}, reg_co {}\n"
          .format(self.args.hidden_dim, self.args.num_layers, self.args.dropout,
                  self.args.learning_rate, self.args.batch_size,
                  self.args.input_len, self.args.output_len,
                  self.args.temperature, self.args.reg_co))

    best_epoch, best_avg_val_loss, best_test_mae = 0, 100, None
    for epoch in range(1, self.args.num_epochs + 1):
      avg_loss = self._train_epoch(optim)

      with open(os.path.join(self.args.save_dir, "train.txt"), "a") as f:
        f.write("Epoch {}, average training loss: {}\n".format(epoch, avg_loss))

      avg_val_loss, avg_val_ts_loss, avg_val_reg_loss = self.valid()
      test_results = self.test()

      with open(os.path.join(self.args.save_dir, "val_test.txt"), "a") as f:
        if self.args.adj_type == "learned":
          f.write(
              "Epoch {}, average validation loss: {}, time series loss: {}, regularization loss: {}\n"
              .format(epoch, avg_val_loss, avg_val_ts_loss, avg_val_reg_loss))
          f.write("testing mae: {}, adj accuracy: {}, sparsity: {}\n".format(
              test_results[0], test_results[1], test_results[2]))
        else:
          f.write(
              "Epoch {}, average validation loss: {}, time series loss: {}, regularization loss: {}, testing mae: {}\n"
              .format(epoch, avg_val_loss, avg_val_ts_loss, avg_val_reg_loss,
                      test_results[0]))

      if avg_val_ts_loss < best_avg_val_loss:
        best_epoch = epoch
        best_avg_val_loss = avg_val_ts_loss
        best_test_mae = test_results[0]

      if self.args.save_model:
        torch.save(
            self.model.state_dict(),
            os.path.join(self.args.save_dir, "checkpoints",
                         "epoch_{}.ckpt".format(epoch)))

    with open(os.path.join(self.args.save_dir, "val_test.txt"), "a") as f:
      f.write(
          "Best epoch {}, average validation time series loss: {}, testing mae: {}\n"
          .format(best_epoch, best_avg_val_loss, best_test_mae))

  @torch.no_grad()
  def valid(self):
    """Computes and returns the validation loss.

    Returns:
      (average_loss, average_time_series_forecast_loss,
        average_regularization_loss).
    """
    self.model.eval()
    total_loss, total_ts_loss, total_reg_loss = 0.0, 0.0, 0.0
    batch_idx = 0
    for batch in tqdm(self.val_loader):
      input_batch, y = batch
      input_batch, y = input_batch.to(self.device), y.to(self.device)
      output_batch, prob = self.model(input_batch)
      ts_loss = self.model.loss(output_batch, y)
      total_ts_loss += ts_loss.detach().cpu().item()

      reg_loss = self._reg_loss(prob)
      total_reg_loss += reg_loss.detach().cpu().item()

      loss = ts_loss + self.args.reg_co * reg_loss
      total_loss += loss.detach().cpu().item()

      batch_idx += 1

    return total_loss / (batch_idx + 1), total_ts_loss / (batch_idx +
                                                          1), total_reg_loss / (
                                                              batch_idx + 1)

  @torch.no_grad()
  def test(self):
    """Does testing on the test dataset.

    Returns:
      (mean absolute error of time series forecast,
        relational graph edge prediction accuracy,
        sparsity ratio of the predicted graph).
    """
    self.model.eval()
    preds, ys = [], []
    for batch in tqdm(self.test_loader):
      input_batch, y = batch
      input_batch, y = input_batch.to(self.device), y.to(self.device)
      output_batch, _ = self.model(input_batch)
      output_batch, y = output_batch.detach().cpu(), y.detach().cpu()
      preds.append(self.datamodule.transform.inverse_transform(output_batch))
      ys.append(self.datamodule.transform.inverse_transform(y))

    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)
    mae = (preds - ys).abs().mean()

    if self.args.adj_type == "learned":
      adj_pred, _ = self.model.pred_adj()
      adj_pred = adj_pred.detach().cpu()
      adj_gt = self.gt_adj_torch if self.gt_adj_torch.dim(
      ) == 3 else self.gt_adj_torch.unsqueeze(2)
      num_nodes = self.args.num_nodes
      adj_acc = ((adj_gt == adj_pred).sum() - num_nodes * adj_gt.shape[2]) / (
          adj_gt.shape[0] * adj_gt.shape[1] * adj_gt.shape[2] -
          num_nodes * adj_gt.shape[2])
      sparsity_ratio = adj_pred.sum() / (
          num_nodes * num_nodes * adj_gt.shape[2] - num_nodes * adj_gt.shape[2])
      return mae, adj_acc, sparsity_ratio

    return mae, None, None
