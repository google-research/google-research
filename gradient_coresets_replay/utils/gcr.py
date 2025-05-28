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

"""Gradient Coresets."""

import torch
from torch.nn import functional as F
from gradient_coresets_replay.utils.continual_model import ContinualModel
from gradient_coresets_replay.utils.coreset import Coreset


class GCR(ContinualModel):
  """Gradient Coresets Buffer and Trainer Class."""

  name = 'gcr'
  compatibility = ['class-il', 'domain-il', 'task-il', 'general-continual']

  def __init__(
      self, backbone, loss, args, transform, barlow_transform, num_classes=10
  ):
    super(GCR, self).__init__(backbone, loss, args, transform, barlow_transform)
    self.buffer = Coreset(
        num_classes,
        self.args.buffer_size,
        self.args.reservoir_size,
        self.device,
        self.args,
    )

  def observe(self, inputs, labels, not_aug_inputs):
    self.opt.zero_grad()
    outputs = self.net(inputs)
    loss = self.loss(outputs, labels)

    if not self.buffer.is_empty():
      if self.args.alpha != 0:
        buf_inputs, _, buf_logits, buf_weights = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform
        )
        buf_outputs = self.net(buf_inputs)
        loss_alpha = (
            self.args.alpha
            * F.mse_loss(buf_outputs, buf_logits, reduction='none').mean(axis=1)
            * buf_weights
            / buf_weights.sum()
        )
        loss += loss_alpha.sum()
      if self.args.beta != 0:
        buf_inputs, buf_labels, _, buf_weights = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform
        )
        buf_outputs = self.net(buf_inputs)
        loss_beta = (
            self.args.beta
            * self.loss(buf_outputs, buf_labels, reduction='none')
            * buf_weights
            / buf_weights.sum()
        )
        loss += loss_beta.sum()
      if self.args.gamma != 0:
        x1, x2, l, buf_weights = self.buffer.get_barlow_data(
            barlow_transform=self.barlow_transform,
            size=self.args.minibatch_size,
        )
        loss += self.args.gamma * self.net.contrastive_forward(
            x1, x2, l, buf_weights
        )

    loss.backward()
    self.opt.step()

    self.buffer.add_reservoir_data(
        examples=not_aug_inputs,
        labels=labels,
        logits=outputs.data,
        weights=torch.ones(len(labels)).to(self.device),
    )

    return loss.item()
