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

"""A basic MoCo based representation encoder, with CT backbone."""

import copy
from typing import Union

import numpy as np
import omegaconf
from src.data import RealDatasetCollection
from src.data import SyntheticDatasetCollection
from src.models.rep_est.CT_utils.encoder import CTEncoder
from src.models.rep_est.MoCo import MoCov3Encoder
import torch

deepcopy = copy.deepcopy
DictConfig = omegaconf.DictConfig


class CTMoCoEncoder(MoCov3Encoder):
  """Causal transformer moco encoder."""

  def __init__(
      self,
      args,
      dataset_collection,
      autoregressive = None,
      has_vitals = None,
      bce_weights = None,
      **kwargs,
  ):
    super().__init__(
        args, dataset_collection, autoregressive, has_vitals, bce_weights
    )

  def _build_base_encoder(self, sub_args):
    return CTEncoder(
        input_size=self.input_size,
        dim_treatments=self.dim_treatments,
        dim_vitals=self.dim_vitals,
        dim_outcome=self.dim_outcome,
        dim_static_features=self.dim_static_features,
        has_vitals=self.has_vitals,
        sub_args=sub_args,
    )

  def _init_specific(self, sub_args):
    super()._init_specific(sub_args)
    self.use_comp_contrast = sub_args.use_comp_contrast

  def _encode(
      self,
      net,
      batch,
      active_entries,
      return_flatten=False,
      return_comp_reps=False,
  ):
    # only keep encodings at active_entries=1 steps
    enc = net(batch, return_comp_reps=return_comp_reps)
    if return_comp_reps:
      enc, comp_reps = enc
      rep_list = [enc] + comp_reps
      if return_flatten:
        return [x[active_entries.squeeze(-1) == 1] for x in rep_list]
      else:
        return rep_list
    else:
      if return_flatten:
        return enc[active_entries.squeeze(-1) == 1]
      else:
        return enc

  def _compose_batch(self, batch):
    name2dims = {}
    curr_dim = 0
    vitals_or_prev_outputs = []
    if self.has_vitals:
      vitals_or_prev_outputs.append(batch['vitals'])
      name2dims['vitals'] = (curr_dim, curr_dim + batch['vitals'].shape[-1])
      curr_dim += batch['vitals'].shape[-1]
    if self.autoregressive:
      vitals_or_prev_outputs.append(batch['prev_outputs'])
      name2dims['prev_outputs'] = (
          curr_dim,
          curr_dim + batch['prev_outputs'].shape[-1],
      )
      curr_dim += batch['prev_outputs'].shape[-1]
    vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)

    prev_treatments = batch['prev_treatments']
    name2dims['prev_treatments'] = (
        curr_dim,
        curr_dim + prev_treatments.shape[-1],
    )
    curr_dim += prev_treatments.shape[-1]

    x = torch.cat((vitals_or_prev_outputs, prev_treatments), dim=-1)
    return x, name2dims

  def augment(self, batch):
    x, name2dims = self._compose_batch(batch)
    aug_x = self._jitter(self._shift(self._scale(x)))
    aug_batch = {}
    for k, (d_start, d_end) in name2dims.items():
      aug_batch[k] = aug_x[Ellipsis, d_start:d_end]
    aug_batch['static_features'] = batch['static_features']
    aug_batch['active_entries'] = batch['active_entries']
    return aug_batch

  def forward(self, batch):
    # skip augmentation for debugging
    active_entries = batch['active_entries']
    batch1, batch2 = self.augment(batch), self.augment(batch)
    q1s = [
        self.predictor(x)
        for x in self._encode(
            self.base_encoder,
            batch1,
            active_entries,
            return_flatten=True,
            return_comp_reps=True,
        )
    ]
    q2s = [
        self.predictor(x)
        for x in self._encode(
            self.base_encoder,
            batch2,
            active_entries,
            return_flatten=True,
            return_comp_reps=True,
        )
    ]
    q1, q1_comp_reps = q1s[0], q1s[1:]
    q2, q2_comp_reps = q2s[0], q2s[1:]

    with torch.no_grad():
      self._update_momentum_encoder(self.momentum)
      k1s = self._encode(
          self.momentum_encoder,
          batch1,
          active_entries,
          return_flatten=True,
          return_comp_reps=True,
      )
      k2s = self._encode(
          self.momentum_encoder,
          batch2,
          active_entries,
          return_flatten=True,
          return_comp_reps=True,
      )
      k1, k1_comp_reps = k1s[0], k1s[1:]
      k2, k2_comp_reps = k2s[0], k2s[1:]

    loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

    if self.use_comp_contrast:
      loss_comp = 0
      for q1_comp, k2_comp in zip(q1_comp_reps, k2_comp_reps):
        loss_comp += self.contrastive_loss(q1_comp, k2_comp)
      for q2_comp, k1_comp in zip(q2_comp_reps, k1_comp_reps):
        loss_comp += self.contrastive_loss(q2_comp, k1_comp)
      loss = loss + loss_comp * 1.0 / len(q1_comp_reps)

    return loss

  def encode(self, batch):
    active_entries = batch['active_entries']
    return self._encode(self.base_encoder, batch, active_entries)

  def optimizer_step(
      self,
      *args,
      epoch = None,
      batch_idx = None,
      optimizer=None,
      optimizer_idx = None,
      **kwargs,
  ):
    super().optimizer_step(
        epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs
    )
