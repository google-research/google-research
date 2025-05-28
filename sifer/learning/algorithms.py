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

"""Implements subgroup robustness algorithms."""

from collections.abc import Iterable
from typing import Any, Optional

import torch
import transformers

from sifer.learning import optimizers
from sifer.models import networks


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class Algorithm(torch.nn.Module):
  """A subclass of Algorithm implements a subgroup robustness algorithm.

  Subclasses should implement the following: - _init_model() - _compute_loss() -
  update() - predict()
  """

  def __init__(
      self,
      data_type,
      input_shape,
      num_classes,
      num_attributes,
      num_examples,
      hparams,
      grp_sizes = None,
  ):
    super().__init__()
    self.hparams = hparams
    self.data_type = data_type
    self.num_classes = num_classes
    self.num_attributes = num_attributes
    self.num_examples = num_examples
    self.loss = None
    self.optimizer: torch.optim.Optimizer
    self.lr_scheduler: Optional[torch.scheduler.Scheduler]

  def _init_model(self):
    """Initializes model, optimizer and loss functions."""
    raise NotImplementedError

  def _compute_loss(
      self,
      index,
      x,
      y,
      attribute_idx,
      step,
  ):
    raise NotImplementedError

  def update(self, minibatch, step):
    """Perform one update step."""
    raise NotImplementedError

  def predict(self, x):
    """Predict outputs."""
    raise NotImplementedError

  def return_groups(
      self, y, attribute_idx
  ):
    """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup."""
    idx_g, idx_samples = [], []
    all_groups = y * self.num_attributes + attribute_idx

    for group_num in all_groups.unique():
      idx_g.append(group_num)
      idx_samples.append(all_groups == group_num)

    return zip(idx_g, idx_samples)

  def return_attributes(
      self, all_attributes
  ):
    """Given a list of attributes, return indexes of samples belonging to each attribute."""
    idx_a, idx_samples = [], []

    for attribute in all_attributes.unique():
      idx_a.append(attribute)
      idx_samples.append(all_attributes == attribute)

    return zip(idx_a, idx_samples)


class ERM(Algorithm):
  """Empirical Risk Minimization (ERM)."""

  def __init__(
      self,
      data_type,
      input_shape,
      num_classes,
      num_attributes,
      num_examples,
      hparams,
      grp_sizes = None,
  ):
    super().__init__(
        data_type,
        input_shape,
        num_classes,
        num_attributes,
        num_examples,
        hparams,
        grp_sizes,
    )

    self.featurizer = networks.Featurizer(data_type, input_shape, self.hparams)
    self.classifier = networks.Classifier(
        self.featurizer.n_outputs,
        num_classes,
        self.hparams["nonlinear_classifier"],
    )
    self.network = torch.nn.Sequential(self.featurizer, self.classifier)
    self.register_buffer(
        "q", torch.ones(self.num_classes * self.num_attributes).cuda()
    )
    self._init_model()

  def _init_model(self):
    self.clip_grad = (
        self.data_type == "text" and self.hparams["optimizer"] == "adamw"
    )

    if self.data_type in ["images", "tabular"]:
      self.optimizer = optimizers.get_optimizers["sgd"](
          self.network, self.hparams["lr"], self.hparams["weight_decay"]
      )
      self.lr_scheduler = None
      self.loss = torch.nn.CrossEntropyLoss(reduction="none")
    elif self.data_type == "text":
      self.network.zero_grad()
      self.optimizer = optimizers.get_optimizers[self.hparams["optimizer"]](
          self.network, self.hparams["lr"], self.hparams["weight_decay"]
      )
      self.lr_scheduler = transformers.get_scheduler(
          "linear",
          optimizer=self.optimizer,
          num_warmup_steps=0,
          num_training_steps=self.hparams["steps"],
      )
      self.loss = torch.nn.CrossEntropyLoss(reduction="none")
    else:
      raise NotImplementedError(f"{self.data_type} not supported.")

  def _compute_loss(
      self,
      index,
      x,
      y,
      attribute_idx,
      step,
  ):
    return self.loss(self.predict(x), y).mean()

  def update(self, minibatch, step):
    all_i, all_x, all_y, all_a = minibatch
    loss = self._compute_loss(all_i, all_x, all_y, all_a, step)

    self.optimizer.zero_grad()
    loss.backward()
    if self.clip_grad:
      torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
    self.optimizer.step()

    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

    if self.data_type == "text":
      self.network.zero_grad()

    return {"loss": loss.item()}

  def predict(self, x):
    return self.network(x)


class SiFeR(ERM):
  """SiFeR Algorithm.

  Overcoming Simplicity Bias in Deep Networks using a Feature Sieve, ICML'23
  """

  def __init__(
      self,
      data_type,
      input_shape,
      num_classes,
      num_attributes,
      num_examples,
      hparams,
      grp_sizes = None,
  ):
    super().__init__(
        data_type,
        input_shape,
        num_classes,
        num_attributes,
        num_examples,
        hparams,
        grp_sizes,
    )
    self.aux_network = networks.AuxFeaturizer(
        self.featurizer,
        self.classifier,
        num_classes,
        hparams,
        aux_pos=hparams.aux_pos,
    )
    self.__init_model()

  def __init_model(self):
    self.clip_grad = (
        self.data_type == "text" and self.hparams["optimizer"] == "adamw"
    )
    if self.data_type in ["images", "tabular"]:
      self.optimizer = optimizers.get_optimizers["sgd"](
          self.aux_network.network,
          self.hparams.lr,
          self.hparams.weight_decay,
      )
      self.aux_opt = optimizers.get_optimizers["sgd"](
          self.aux_network.aux_layer,
          self.hparams.aux_lr,
          self.hparams.aux_weight_decay,
      )
      params_list = [
          self.aux_network.featurizer.network.conv1.parameters(),
          self.aux_network.featurizer.network.bn1.parameters(),
          self.aux_network.featurizer.network.layer1.parameters(),
          self.aux_network.featurizer.network.layer2.parameters(),
          self.aux_network.featurizer.network.layer3.parameters(),
          self.aux_network.featurizer.network.layer4.parameters(),
      ]
      forget_params_list = []
      for i in params_list[: 2 + self.hparams.aux_pos]:
        forget_params_list.extend(i)
      self.forget_opt = torch.optim.SGD(
          forget_params_list, self.hparams.forget_lr, 0.0
      )
      self.lr_scheduler = None
      self.loss = torch.nn.CrossEntropyLoss(reduction="none")
    else:
      raise NotImplementedError

  def _compute_loss(
      self,
      index,
      x,
      y,
      attribute_idx,
      step,
  ):
    return self.loss(self.predict(x), y).mean()

  def _compute_aux_loss(self, x, y):
    return self.loss(self.aux_predict(x), y).mean()

  def _compute_forget_loss(self, x):
    aux = self.aux_predict(x)
    return self.loss(
        aux, torch.ones_like(aux) * (1.0 / self.num_classes)
    ).mean()

  def update(self, minibatch, step):
    all_i, all_x, all_y, all_a = minibatch

    aux_loss = self._compute_aux_loss(all_x, all_y)
    self.aux_opt.zero_grad()
    aux_loss.backward()
    self.aux_opt.step()

    if step % self.hparams.forget_after_iters == 0:
      forget_loss = self._compute_forget_loss(all_x)
      self.forget_opt.zero_grad()
      forget_loss.backward()
      self.forget_opt.step()

    loss = self._compute_loss(all_i, all_x, all_y, all_a, step)

    self.optimizer.zero_grad()
    loss.backward()
    if self.clip_grad:
      torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
    self.optimizer.step()

    # if self.lr_scheduler is not None:
    #   self.lr_scheduler.step()

    if self.data_type == "text":
      self.network.zero_grad()

    return {"loss": loss.item()}

  def predict(self, x):
    return self.network(x)

  def aux_predict(self, x):
    return self.aux_network(x)[0]
