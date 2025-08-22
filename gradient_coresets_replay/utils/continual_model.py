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

"""Continual learning base model."""

from continual_learning_rishabh.utils.conf import get_device
import torch
from torch.optim import SGD

nn = torch.nn


class ContinualModel(nn.Module):
  """Continual learning base model."""

  name = None
  compatibility = []

  def __init__(
      self,
      backbone,
      loss,
      args,
      transform,
      barlow_transform=None,
  ):
    super().__init__()

    self.net = backbone
    self.loss = loss
    self.args = args
    self.transform = transform
    self.opt = SGD(self.net.parameters(), lr=self.args.lr)
    self.device = get_device()
    self.barlow_transform = barlow_transform

  def forward(self, x):
    """Computes a forward pass.

    Args:
      x: batch of inputs

    Returns:
      the result of the computation
    """
    return self.net(x)

  def observe(
      self,
      inputs,
      labels,
      not_aug_inputs,
  ):
    """Compute a training step over a given batch of examples.

    Args:
      inputs: batch of examples
      labels: ground-truth labels
      not_aug_inputs: batch of examples without applying augmentations

    Returns: the value of the loss function
    """
