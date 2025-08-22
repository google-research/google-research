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

"""Training and evalution functions for one epoch."""

import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group["lr"]


def train_epoch_koopman(train_loader,
                        model,
                        loss_fun,
                        optimizer,
                        regularize_rank=False):
  """Train the KNF model for one epoch.

  Args:
    train_loader: the dataloader of the training set
    model: KNF model
    loss_fun: loss function
    optimizer: Adam
    regularize_rank: whether to regularize rank

  Returns:
    RMSE on the training set

  """
  train_loss = []
  for inps, tgts in train_loader:
    if len(inps.shape) > 2:
      inps = inps.to(device)
      tgts = tgts.to(device)
    else:
      inps = inps.unsqueeze(-1).to(device)
      tgts = tgts.unsqueeze(-1).to(device)
    if regularize_rank:
      (_, [norm_outs, norm_tgts], [norm_recons, norm_inp_preds, norm_inps],
       [enc_embeds, pred_embeds], rank_regularizer) = model(inps, tgts)
    else:
      (_, [norm_outs,
           norm_tgts], [norm_recons, norm_inp_preds,
                        norm_inps], [enc_embeds,
                                     pred_embeds]) = model(inps, tgts)

    loss = loss_fun(norm_outs,
                    norm_tgts) + loss_fun(norm_recons, norm_inps) + loss_fun(
                        norm_inp_preds, norm_inps[:, 1:]) + loss_fun(
                            enc_embeds, pred_embeds)

    if regularize_rank:
      loss += rank_regularizer

    train_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
    optimizer.step()

  return np.sqrt(np.mean(train_loss))


def eval_epoch_koopman(eval_loader, model, loss_fun, regularize_rank=False):
  """Evaluate the KNF model on the validation set.

  Args:
    eval_loader: the dataloader of the validation/test set
    model: KNF model
    loss_fun: MSE loss
    regularize_rank: whether to regularize rank.
  Returns:
    RMSE, predictions and ground truth on the evalution set

  """
  eval_loss = []
  all_preds = []
  all_trues = []
  for inps, tgts in eval_loader:
    if len(inps.shape) > 2:
      inps = inps.to(device)
      tgts = tgts.to(device)
    else:
      inps = inps.unsqueeze(-1).to(device)
      tgts = tgts.unsqueeze(-1).to(device)

    if regularize_rank:
      (denorm_outs, [norm_outs,
                     norm_tgts], [norm_recons, norm_inp_preds, norm_inps],
       [enc_embeds, pred_embeds], rank_regularizer) = model(inps, tgts)
    else:
      denorm_outs, [norm_outs,
                    norm_tgts], [norm_recons, norm_inp_preds,
                                 norm_inps], [enc_embeds,
                                              pred_embeds] = model(inps, tgts)

    loss = loss_fun(norm_outs,
                    norm_tgts) + loss_fun(norm_recons, norm_inps) + loss_fun(
                        norm_inp_preds, norm_inps[:, 1:]) + loss_fun(
                            enc_embeds, pred_embeds)

    if regularize_rank:
      loss += rank_regularizer

    eval_loss.append(loss.item())
    all_preds.append(denorm_outs.cpu().data.numpy())
    all_trues.append(tgts.cpu().data.numpy())
  return np.sqrt(np.mean(eval_loss)), np.concatenate(
      all_preds, axis=0), np.concatenate(
          all_trues, axis=0)
