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
"""Defines the editer class for editing the trained GATRNN model."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from editable_graph_temporal import data
from editable_graph_temporal import model
from editable_graph_temporal import optim


class Editer:
  """Implements the editing process."""

  def __init__(self, args):
    """Instantiates the editer class.

    Args:
      args: python argparse.ArgumentParser class, containing configuration
        arguments for data, model, and editing hyperparameters.
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
    self.train_iter = iter(self.datamodule.train_dataloader())
    self.val_loader = self.datamodule.val_dataloader()
    self.test_loader = self.datamodule.test_dataloader()

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gat_model = model.GATRNN(adj_mx, args)
    gat_model.load_state_dict(torch.load(args.model_path))
    self.model = gat_model.to(self.device)

  @torch.no_grad()
  def _get_error_edges_new_labels(self):
    """Gets error edges set and edge label for editing the model.

    Returns:
      (error edges set,
        edge prediction of old model,
        edge label for model editing,
        edge weights for cross-entropy loss in model editing).
    """
    self.model.eval()
    old_adj_pred, _ = self.model.pred_adj()
    old_adj_pred = old_adj_pred.detach().cpu()
    adj_gt = self.gt_adj_torch if self.gt_adj_torch.dim(
    ) == 3 else self.gt_adj_torch.unsqueeze(2)

    error_edge_idxs = []
    error_edge_gts = []
    no_con_adj_gt = 1.0 - adj_gt.sum(dim=2)
    no_con_adj_pred = 1.0 - old_adj_pred.sum(dim=2)
    error_edge_idxs.append(
        torch.nonzero(
            torch.logical_and(no_con_adj_gt != no_con_adj_pred,
                              no_con_adj_gt == 1.0)))
    error_edge_gts.append(torch.tensor([0]))
    for i in range(self.args.num_relation_types - 1):
      error_edge_idxs.append(
          torch.nonzero(
              torch.logical_and(adj_gt[:, :, i] != old_adj_pred[:, :, i],
                                adj_gt[:, :, i] == 1.0)))
      error_edge_gts.append(torch.tensor([i + 1]))
    num_nodes = self.args.num_nodes
    adj_label = torch.cat((no_con_adj_pred.unsqueeze(dim=2), old_adj_pred),
                          dim=-1).argmax(dim=2).reshape(num_nodes * num_nodes)

    false_pos_idxs, false_neg_idxs = error_edge_idxs[0], error_edge_idxs[1]
    edit_edge_idxs = torch.cat((false_pos_idxs, false_neg_idxs), dim=0)
    edit_edge_idxs_one_dim1 = false_pos_idxs[:,
                                             0] * num_nodes + false_pos_idxs[:,
                                                                             1]
    adj_label[edit_edge_idxs_one_dim1] = 0
    edit_edge_idxs_one_dim2 = false_neg_idxs[:,
                                             0] * num_nodes + false_neg_idxs[:,
                                                                             1]
    adj_label[edit_edge_idxs_one_dim2] = 1

    weights = torch.ones(
        size=(len(adj_label),), device=self.device) / (
            num_nodes * num_nodes - num_nodes - len(edit_edge_idxs))
    weights[edit_edge_idxs_one_dim1] = 1.0 / (
        len(edit_edge_idxs_one_dim1) + len(edit_edge_idxs_one_dim2))
    weights[edit_edge_idxs_one_dim2] = 1.0 / (
        len(edit_edge_idxs_one_dim1) + len(edit_edge_idxs_one_dim2))
    ignore_idxs = torch.arange(num_nodes) * num_nodes + torch.arange(num_nodes)
    weights[ignore_idxs] = 0.0

    return edit_edge_idxs, old_adj_pred, adj_label, weights

  def _train_iter(self, optims, adj_label, weights):
    """Runs one epoch of training.

    Args:
      optims: list of the pytorch optimizers used to optimize different parts of
        the model.
      adj_label: supervision labels for all two-node pairs, with shape
        (self.args.num_nodes * self.args.num_nodes).
      weights: weights for all two-node pairs when computing corss-entropy loss,
        with shape (self.args.num_nodes * self.args.num_nodes).
    """
    self.model.train()

    all_edge_recs_embs = torch.matmul(self.model.fc_graph_rec,
                                      self.model.global_embs)
    all_edge_sends_embs = torch.matmul(self.model.fc_graph_send,
                                       self.model.global_embs)
    x = torch.cat([all_edge_sends_embs, all_edge_recs_embs], dim=1)
    x = torch.relu(self.model.fc_out(x))
    x = self.model.fc_cat(x)
    all_edge_loss = (F.cross_entropy(x, adj_label, reduction="none") *
                     weights).sum()

    try:
      input_batch, y = next(self.train_iter)
    except StopIteration:
      self.train_iter = iter(self.datamodule.train_dataloader())
      input_batch, y = next(self.train_iter)
    input_batch, y = input_batch.to(self.device), y.to(self.device)
    output_batch, _ = self.model(input_batch)
    ts_loss = self.model.loss(output_batch, y)

    loss = ts_loss + self.args.edge_co * all_edge_loss
    for o in optims:
      o.zero_grad()
    loss.backward()
    for o in optims:
      o.step()

  def train(self):
    """Running the overall editing process."""

    if not os.path.isdir(os.path.join(self.args.save_dir, "checkpoints")):
      os.mkdir(os.path.join(self.args.save_dir, "checkpoints"))

    lr1, lr2, lr3 = self.args.learning_rates
    if self.args.use_cons_optim:
      cons_thre1, cons_thre2, cons_thre3 = self.args.cons_thres
      global_emb_optim = optim.AdamCons([self.model.global_embs],
                                        lr=lr1,
                                        eps=1.0e-3,
                                        cons_thre=cons_thre1)
      graph_module_optim = optim.AdamCons(
          list(self.model.fc_out.parameters()) +
          list(self.model.fc_cat.parameters()),
          lr=lr2,
          eps=1.0e-3,
          cons_thre=cons_thre2)
      ts_module_optim = optim.AdamCons(
          list(self.model.encoder.parameters()) +
          list(self.model.decoder.parameters()),
          lr=lr3,
          eps=1.0e-3,
          cons_thre=cons_thre3)
    else:
      global_emb_optim = Adam([self.model.global_embs], lr=lr1, eps=1.0e-3)
      graph_module_optim = Adam(
          list(self.model.fc_out.parameters()) +
          list(self.model.fc_cat.parameters()),
          lr=lr2,
          eps=1.0e-3)
      ts_module_optim = Adam(
          list(self.model.encoder.parameters()) +
          list(self.model.decoder.parameters()),
          lr=lr3,
          eps=1.0e-3)
    optims = [global_emb_optim, graph_module_optim, ts_module_optim]

    edit_edge_idxs, old_adj_pred, adj_label, weights = self._get_error_edges_new_labels(
    )
    adj_label = adj_label.to(self.device)

    for iter_num in range(self.args.num_max_fine_tune_iters):
      self._train_iter(optims, adj_label, weights)

      if (iter_num + 1) % 100 == 0:
        test_results = self.test(edit_edge_idxs, old_adj_pred, adj_label)

        with open(os.path.join(self.args.save_dir, "val_test.txt"), "w") as f:
          f.write(
              "testing mae: {}, adj accuracy: {}, sparsity: {}, preserve: {}, edit edge accuracy: {}\n"
              .format(test_results[0], test_results[1], test_results[2],
                      test_results[3], test_results[4]))

        if self.args.save_model:
          torch.save(
              self.model.state_dict(),
              os.path.join(self.args.save_dir, "checkpoints",
                           "iter_{}.ckpt".format(iter_num)))

  @torch.no_grad()
  def test(self, edit_edge_idxs, old_adj_pred, adj_label):
    """Does testing on the test dataset.

    Args:
      edit_edge_idxs: error edge indexs, with shape (num_error_edges, 2).
      old_adj_pred: predicted adjacency matrix by the old model, with shape
        (self.args.num_nodes, self.args.num_nodes,
        self.args.num_relation_types-1).
      adj_label: adjacency matrix label for model editing, with shape
        (self.args.num_nodes * self.args.num_nodes).

    Returns:
      (mean absolute error of time series forecast,
        relational graph edge prediction accuracy,
        sparsity ratio of the predicted graph,
        prediction preservation ratio of edges not aiming to be edited,
        prediction accuracy of edges aiming to be edited).
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

    preserve_num = 0
    for j in range(self.args.num_relation_types - 1):
      preserve = (adj_pred[:, :, j] == old_adj_pred[:, :, j]).float()
      preserve[edit_edge_idxs[:, 0], edit_edge_idxs[:, 1]] = False
      preserve_num += preserve.sum() - num_nodes
    num_old_edges = num_nodes * num_nodes * adj_pred.shape[
        2] - num_nodes * adj_pred.shape[2] - len(
            edit_edge_idxs) * adj_pred.shape[2]
    preserve_ratio = preserve_num / num_old_edges

    all_edge_recs_embs = torch.matmul(self.model.fc_graph_rec,
                                      self.model.global_embs)
    all_edge_sends_embs = torch.matmul(self.model.fc_graph_send,
                                       self.model.global_embs)
    x = torch.cat([all_edge_sends_embs, all_edge_recs_embs], dim=1)
    x = torch.relu(self.model.fc_out(x))
    x = self.model.fc_cat(x)
    edit_edge_idxs_one_dim = edit_edge_idxs[:,
                                            0] * num_nodes + edit_edge_idxs[:,
                                                                            1]
    edit_edge_acc = (x[edit_edge_idxs_one_dim].argmax(dim=-1)
                     == adj_label[edit_edge_idxs_one_dim]
                    ).sum() / len(edit_edge_idxs_one_dim)

    return mae, adj_acc, sparsity_ratio, preserve_ratio, edit_edge_acc
