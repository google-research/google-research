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

"""Subset selection."""

import torch
from torch.utils import data
from gradient_coresets_replay.utils.gradmatchstrategy import OMPGradMatchStrategy

NAMES = {
    OMPGradMatchStrategy.name: OMPGradMatchStrategy,
}
nn = torch.nn


class CustomDataset(data.Dataset):
  """Custom Dataset class."""

  def __init__(self, x, y, logits, weights):
    self.x = x
    self.y = y
    self.logits = logits
    self.weights = weights

  def __len__(self):
    return len(self.y)

  def __getitem__(self, index):
    img, target, logit, weight = (
        self.x[index],
        self.y[index],
        self.logits[index],
        self.weights[index],
    )
    return img, target, logit, weight


class CustomLoss(nn.Module):
  """inner loss function for selection strategies."""

  def __init__(self, alpha=0.1, beta=0.5):
    super(CustomLoss, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    self.mse_loss = nn.MSELoss(reduction='none')

  def forward(self, outputs, targets, logits, weights=None):
    loss = self.beta * self.ce_loss(outputs, targets) + self.alpha * torch.mean(
        self.mse_loss(outputs, logits), 1
    )
    return loss * weights


class CustomOuterLoss(nn.Module):
  """outer loss function for selection strategies."""

  def __init__(self):
    super(CustomOuterLoss, self).__init__()
    self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    # self.mse_loss = nn.MSELoss(reduction='none')

  def forward(self, outputs, targets, weights=None):
    loss = self.ce_loss(outputs, targets)
    if weights is None:
      return loss
    return weights * loss


def select_subset(
    model,
    model_params,
    examples,
    labels,
    logits,
    weights,
    num_classes,
    subset_size,
    strategy_name,
    transform,
    device,
    args,
):
  """selects and returns subset using weighted gradient matching strategy."""

  examples = torch.stack([transform(ee.cpu()) for ee in examples])
  trainset = CustomDataset(examples, labels, logits, weights)
  train_loader = data.DataLoader(trainset, batch_size=32, shuffle=False)

  valset = CustomDataset(examples, labels, logits, weights)
  val_loader = data.DataLoader(valset, batch_size=32, shuffle=False)
  if strategy_name in ['gradmatch']:
    inner_loss = CustomLoss(args.alpha, args.beta)
    outer_loss = CustomOuterLoss()
  else:
    raise ValueError('selection strategy not implemented.')

  return NAMES[strategy_name](
      model,
      train_loader,
      val_loader,
      num_classes,
      inner_loss,
      outer_loss,
      device,
  ).select(subset_size, model_params)
