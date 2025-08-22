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

"""Base class for data subset selection."""

import torch


class DataSelectionStrategy(object):
  """Base class for data subset selection strategy."""

  def __init__(
      self,
      model,
      trainloader,
      valloader,
      num_classes,
      inner_loss,
      outer_loss,
      device,
  ):
    """Constructer method."""

    self.model = model
    self.trainloader = trainloader
    self.valloader = valloader
    self.n_trn = len(trainloader.sampler)
    self.n_val = len(valloader.sampler)
    self.grads_per_elem = None
    self.val_grads_per_elem = None
    self.num_selected = 0
    self.num_classes = num_classes
    self.trn_lbls = None
    self.val_lbls = None
    self.inner_loss = inner_loss
    self.outer_loss = outer_loss
    self.device = device

  def select(self, budget, model_params):
    pass

  def get_budget_per_class(self, labels, total_budget, keep_ratio=True):
    """calculates and returns budget for each class for the buffer according to class ratio."""

    unique_labels, label_counts = torch.unique(labels, return_counts=True)
    sort_indices = torch.sort(label_counts).indices
    unique_labels, label_counts = (
        unique_labels[sort_indices],
        label_counts[sort_indices],
    )
    ratio = {
        i: j / len(labels)
        for i, j in zip(unique_labels.cpu().numpy(), label_counts.cpu().numpy())
    }
    k = len(unique_labels)
    budget_per_class = {i.item(): 0 for i in unique_labels}

    if keep_ratio:
      for i in unique_labels:
        budget_per_class[i.item()] = max(1, int(total_budget * ratio[i.item()]))
      return budget_per_class

    for label, label_count in zip(unique_labels, label_counts):
      average_budget = total_budget // k
      k -= 1
      if label_count <= average_budget:
        budget_per_class[label.item()] = label_count
        total_budget = total_budget - label_count
      else:
        budget_per_class[label.item()] = average_budget
        total_budget = total_budget - average_budget
    return budget_per_class

  def get_labels(self, valid=False):
    """returns data labels/classes."""

    for batch_idx, (_, targets, _, _) in enumerate(self.trainloader):
      if batch_idx == 0:
        self.trn_lbls = targets.view(-1, 1)
      else:
        self.trn_lbls = torch.cat((self.trn_lbls, targets.view(-1, 1)), dim=0)
    self.trn_lbls = self.trn_lbls.view(-1)

    if valid:
      for batch_idx, (_, targets, _, _) in enumerate(self.valloader):
        if batch_idx == 0:
          self.val_lbls = targets.view(-1, 1)
        else:
          self.val_lbls = torch.cat((self.val_lbls, targets.view(-1, 1)), dim=0)
      self.val_lbls = self.val_lbls.view(-1)

  def compute_gradients(self, valid=False, batch=False, per_class=False):
    """computes model gradients for input examples."""

    if per_class:
      emb_dim = self.model.get_embedding_dim()
      for batch_idx, (inputs, targets, logits, weights) in enumerate(
          self.pctrainloader
      ):
        inputs, targets, logits, weights = (
            inputs.to(self.device),
            targets.to(self.device, non_blocking=True),
            logits.to(self.device),
            weights.to(self.device),
        )
        if batch_idx == 0:
          out, l1 = self.model(inputs, last=True, freeze=False)
          loss = self.inner_loss(out, targets, logits, weights).sum()
          l0_grads = torch.autograd.grad(loss, out)[0]
          if self.linear_layer:
            l0_expand = torch.repeat_interleave(l0_grads, emb_dim, dim=1)
            l1_grads = l0_expand * l1.repeat(1, self.num_classes)
          if batch:
            l0_grads = l0_grads.mean(dim=0).view(1, -1)
            if self.linear_layer:
              l1_grads = l1_grads.mean(dim=0).view(1, -1)
        else:
          out, l1 = self.model(inputs, last=True, freeze=False)
          loss = self.inner_loss(out, targets, logits, weights).sum()
          batch_l0_grads = torch.autograd.grad(loss, out)[0]
          if self.linear_layer:
            batch_l0_expand = torch.repeat_interleave(
                batch_l0_grads, emb_dim, dim=1
            )
            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
          if batch:
            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
            if self.linear_layer:
              batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
          l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
          if self.linear_layer:
            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

      if self.linear_layer:
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
      else:
        self.grads_per_elem = l0_grads
      torch.cuda.empty_cache()
      if valid:
        for batch_idx, (inputs, targets, logits, weights) in enumerate(
            self.pcvalloader
        ):
          inputs, targets, logits, weights = (
              inputs.to(self.device),
              targets.to(self.device, non_blocking=True),
              logits.to(self.device),
              weights.to(self.device),
          )
          if batch_idx == 0:
            out, l1 = self.model(inputs, last=True, freeze=False)
            loss = self.outer_loss(out, targets, weights).sum()
            l0_grads = torch.autograd.grad(loss, out)[0]
            if self.linear_layer:
              l0_expand = torch.repeat_interleave(l0_grads, emb_dim, dim=1)
              l1_grads = l0_expand * l1.repeat(1, self.num_classes)
            if batch:
              l0_grads = l0_grads.mean(dim=0).view(1, -1)
              if self.linear_layer:
                l1_grads = l1_grads.mean(dim=0).view(1, -1)
          else:
            out, l1 = self.model(inputs, last=True, freeze=False)
            loss = self.outer_loss(out, targets, weights).sum()
            batch_l0_grads = torch.autograd.grad(loss, out)[0]
            if self.linear_layer:
              batch_l0_expand = torch.repeat_interleave(
                  batch_l0_grads, emb_dim, dim=1
              )
              batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
            if batch:
              batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
              if self.linear_layer:
                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
            if self.linear_layer:
              l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
        torch.cuda.empty_cache()
        if self.linear_layer:
          self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        else:
          self.val_grads_per_elem = l0_grads
    else:
      emb_dim = self.model.get_embedding_dim()
      for batch_idx, (inputs, targets, logits, weights) in enumerate(
          self.trainloader
      ):
        inputs, targets, logits, weights = (
            inputs.to(self.device),
            targets.to(self.device, non_blocking=True),
            logits.to(self.device),
            weights.to(self.device),
        )
        if batch_idx == 0:
          out, l1 = self.model(inputs, last=True, freeze=False)
          loss = self.inner_loss(out, targets, logits, weights).sum()
          l0_grads = torch.autograd.grad(loss, out)[0]
          if self.linear_layer:
            l0_expand = torch.repeat_interleave(l0_grads, emb_dim, dim=1)
            l1_grads = l0_expand * l1.repeat(1, self.num_classes)
          if batch:
            l0_grads = l0_grads.mean(dim=0).view(1, -1)
            if self.linear_layer:
              l1_grads = l1_grads.mean(dim=0).view(1, -1)
        else:
          out, l1 = self.model(inputs, last=True, freeze=False)
          loss = self.inner_loss(out, targets, logits, weights).sum()
          batch_l0_grads = torch.autograd.grad(loss, out)[0]
          if self.linear_layer:
            batch_l0_expand = torch.repeat_interleave(
                batch_l0_grads, emb_dim, dim=1
            )
            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

          if batch:
            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
            if self.linear_layer:
              batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
          l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
          if self.linear_layer:
            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

      torch.cuda.empty_cache()

      if self.linear_layer:
        self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
      else:
        self.grads_per_elem = l0_grads
      if valid:
        for batch_idx, (inputs, targets, logits, weights) in enumerate(
            self.valloader
        ):
          inputs, targets, logits, weights = (
              inputs.to(self.device),
              targets.to(self.device, non_blocking=True),
              logits.to(self.device),
              weights.to(self.device),
          )
          if batch_idx == 0:
            out, l1 = self.model(inputs, last=True, freeze=False)
            loss = self.outer_loss(out, targets, weights).sum()
            l0_grads = torch.autograd.grad(loss, out)[0]
            if self.linear_layer:
              l0_expand = torch.repeat_interleave(l0_grads, emb_dim, dim=1)
              l1_grads = l0_expand * l1.repeat(1, self.num_classes)
            if batch:
              l0_grads = l0_grads.mean(dim=0).view(1, -1)
              if self.linear_layer:
                l1_grads = l1_grads.mean(dim=0).view(1, -1)
          else:
            out, l1 = self.model(inputs, last=True, freeze=False)
            loss = self.outer_loss(out, targets, weights).sum()
            batch_l0_grads = torch.autograd.grad(loss, out)[0]
            if self.linear_layer:
              batch_l0_expand = torch.repeat_interleave(
                  batch_l0_grads, emb_dim, dim=1
              )
              batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

            if batch:
              batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
              if self.linear_layer:
                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
            if self.linear_layer:
              l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
        torch.cuda.empty_cache()
        if self.linear_layer:
          self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        else:
          self.val_grads_per_elem = l0_grads

  def update_model(self, model_params):
    """Updates the models parameters."""

    self.model.load_state_dict(model_params)
