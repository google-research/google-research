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

"""A basic MoCo based representation encoder."""

from typing import Union

import numpy as np
import omegaconf
from src.data import RealDatasetCollection
from src.data import SyntheticDatasetCollection
from src.models.rep_est.rep_est import EstHead
from src.models.rep_est.rep_est import EstHeadAutoreg
from src.models.rep_est.rep_est import RepEncoder
import torch
from torch import nn

DictConfig = omegaconf.DictConfig
TransformerEncoder = nn.TransformerEncoder
TransformerEncoderLayer = nn.TransformerEncoderLayer


class BaseTransformerEncoder(nn.Module):
  """Base transformer encoder."""

  def __init__(self, input_size, d_model, n_head, dropout, num_layers):
    super().__init__()
    self.input_size = input_size
    self.d_model = d_model
    self.n_head = n_head
    self.dropout = dropout
    self.num_layers = num_layers

    self.input_enc = nn.Linear(self.input_size, d_model)
    tf_net_layer = TransformerEncoderLayer(
        d_model=self.d_model,
        nhead=self.n_head,
        dim_feedforward=4 * self.d_model,
        dropout=self.dropout,
        activation='relu',
        batch_first=True,
    )
    self.net = TransformerEncoder(tf_net_layer, num_layers=self.num_layers)

  def forward(self, x, active_entries):
    # only use causal mask to encode
    x_input = self.input_enc(x)
    active_entries = active_entries.squeeze(-1)  # [B, L]
    mask = active_entries.unsqueeze(1).expand(
        -1, active_entries.shape[1], -1
    )  # [B, L, L]
    mask = torch.tril(mask) == 0
    mask = torch.repeat_interleave(mask, self.n_head, dim=0)
    x_enc = self.net(x_input, mask=mask)
    return x_enc


class MoCov3Encoder(RepEncoder):
  """Mocov3 encoder."""

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
    return BaseTransformerEncoder(
        input_size=self.input_size,
        d_model=sub_args.d_model,
        n_head=sub_args.n_head,
        dropout=sub_args.dropout,
        num_layers=sub_args.num_layers,
    )

  def _init_specific(self, sub_args):
    self.d_model = sub_args.d_model
    self.base_encoder = self._build_base_encoder(sub_args)
    self.momentum_encoder = self._build_base_encoder(sub_args)
    self.momentum = sub_args.momentum
    self.temperature = sub_args.temperature

    self.predictor = nn.Sequential(
        nn.Linear(sub_args.d_model, 2 * sub_args.d_model),
        nn.ReLU(),
        nn.Linear(2 * sub_args.d_model, sub_args.d_model),
    )

    for param_b, param_m in zip(
        self.base_encoder.parameters(), self.momentum_encoder.parameters()
    ):
      param_m.data.copy_(param_b.data)  # initialize
      param_m.requires_grad = False  # not update by gradient

    self.p = sub_args.aug_prob if hasattr(sub_args, 'aug_prob') else 0.5
    self.sigma = sub_args.aug_sigma if hasattr(sub_args, 'aug_sigma') else 0.5

  @torch.no_grad()
  def _update_momentum_encoder(self, m):
    """Momentum update of the momentum encoder."""
    for param_b, param_m in zip(
        self.base_encoder.parameters(), self.momentum_encoder.parameters()
    ):
      param_m.data = param_m.data * m + param_b.data * (1.0 - m)

  def contrastive_loss(self, q, k):
    # normalize
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    # Einstein sum is more intuitive
    logits = torch.einsum('nc,mc->nm', [q, k]) / self.temperature
    bign = logits.shape[0]  # batch size per GPU
    labels = torch.arange(bign, dtype=torch.long).to(logits.device)
    return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temperature)

  def _jitter(self, x):
    if np.random.rand() > self.p:
      return x
    return x + (torch.randn(x.shape) * self.sigma).to(x.device)

  def _scale(self, x):
    if np.random.rand() > self.p:
      return x
    return x * (torch.randn(x.size(-1)) * self.sigma + 1).to(x.device)

  def _shift(self, x):
    if np.random.rand() > self.p:
      return x
    return x + (torch.randn(x.size(-1)) * self.sigma).to(x.device)

  def augment(self, x):
    return self._jitter(self._shift(self._scale(x)))

  def _encode(self, net, x, active_entries, return_flatten=False):
    # only keep encodings at active_entries=1 steps
    enc = net(x, active_entries)
    if return_flatten:
      return enc[active_entries.squeeze(-1) == 1]
    else:
      return enc

  def forward(self, batch):
    vitals_or_prev_outputs = []
    if self.has_vitals:
      vitals_or_prev_outputs.append(batch['vitals'])
    if self.autoregressive:
      vitals_or_prev_outputs.append(batch['prev_outputs'])
    vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
    static_features = batch['static_features']
    static_features_t = static_features.unsqueeze(1).expand(
        -1, vitals_or_prev_outputs.size(1), -1
    )
    prev_treatments = batch['prev_treatments']
    active_entries = batch['active_entries']

    x = torch.cat(
        (vitals_or_prev_outputs, prev_treatments, static_features_t), dim=-1
    )
    # skip augmentation for debugging
    x1, x2 = self.augment(x), self.augment(x)
    q1 = self.predictor(
        self._encode(self.base_encoder, x1, active_entries, return_flatten=True)
    )
    q2 = self.predictor(
        self._encode(self.base_encoder, x2, active_entries, return_flatten=True)
    )

    with torch.no_grad():
      self._update_momentum_encoder(self.momentum)
      k1 = self._encode(
          self.momentum_encoder, x1, active_entries, return_flatten=True
      )
      k2 = self._encode(
          self.momentum_encoder, x2, active_entries, return_flatten=True
      )

    loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
    return loss

  def encode(self, batch):
    vitals_or_prev_outputs = []
    if self.has_vitals:
      vitals_or_prev_outputs.append(batch['vitals'])
    if self.autoregressive:
      vitals_or_prev_outputs.append(
          batch['prev_outputs']
      )
    vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
    static_features = batch['static_features']
    static_features_t = static_features.unsqueeze(1).expand(
        -1, vitals_or_prev_outputs.size(1), -1
    )
    prev_treatments = batch['prev_treatments']
    active_entries = batch['active_entries']

    x = torch.cat(
        (vitals_or_prev_outputs, prev_treatments, static_features_t), dim=-1
    )
    return self._encode(self.base_encoder, x, active_entries)

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


class TCNEstHead(EstHead):
  """TCN estimation head."""

  def __init__(
      self,
      args,
      rep_encoder,
      dataset_collection,
      autoregressive = None,
      has_vitals = None,
      bce_weights = None,
      prefix = '',
      **kwargs,
  ):
    super().__init__(
        args,
        rep_encoder,
        dataset_collection,
        autoregressive,
        has_vitals,
        bce_weights,
        prefix=prefix,
    )

  def _init_specific(self, sub_args):
    super()._init_specific(sub_args)
    self.prev_treatment_encoder = nn.Conv1d(
        self.dim_treatments,
        sub_args.emb_dim,
        kernel_size=self.output_horizon,
        padding=self.output_horizon - 1,
    )
    self.treatment_encoder = nn.Linear(self.dim_treatments, sub_args.emb_dim)
    self.predict_net = nn.Sequential(
        nn.Linear(sub_args.emb_dim, sub_args.hidden_dim),
        nn.ReLU(),
        nn.Linear(sub_args.hidden_dim, self.dim_outcome),
    )

  def forward(self, batch, return_rep=False):
    vitals_or_prev_outputs = []
    if self.has_vitals:
      vitals_or_prev_outputs.append(batch['vitals'])
    if self.autoregressive:
      vitals_or_prev_outputs.append(
          batch['prev_outputs']
      )
    prev_treatments = batch['prev_treatments']
    current_treatments = batch['current_treatments']

    x_enc = self.rep_encoder.encode(batch)  # [B, T, D]

    # unroll prev_treatments to [B, T, output_horizon, D]
    unrolled_prev_treatments = self._unroll_horizon(
        prev_treatments, self.output_horizon
    )
    unrolled_prev_treatments[:, :, 0, :] = 0
    batch_size, step_num, output_horizon, dim = unrolled_prev_treatments.shape
    # [B x T, D, output_horizon]
    unrolled_prev_treatments = unrolled_prev_treatments.reshape(
        -1, output_horizon, dim
    ).permute(0, 2, 1)
    # [B x T, D, output_horizon]
    unrolled_prev_tr_enc = self.prev_treatment_encoder(
        unrolled_prev_treatments
    )[:, :, :output_horizon]
    enc_dim = unrolled_prev_tr_enc.shape[1]
    unrolled_prev_tr_enc = unrolled_prev_tr_enc.permute(0, 2, 1).reshape(
        batch_size, step_num, output_horizon, enc_dim
    )

    # unroll treatments to [B, T, output_horizon, D]
    unrolled_current_treatments = self._unroll_horizon(
        current_treatments, self.output_horizon
    )
    unrolled_current_tr_enc = self.treatment_encoder(
        unrolled_current_treatments
    )

    input_enc = (
        x_enc.unsqueeze(2) + unrolled_prev_tr_enc + unrolled_current_tr_enc
    )
    pred = self.predict_net(input_enc)

    if return_rep:
      return pred, x_enc
    else:
      return pred


class TCNMultiEstHead(EstHead):
  """TCN multiple estimation head."""

  def __init__(
      self,
      args,
      rep_encoder,
      dataset_collection,
      autoregressive = None,
      has_vitals = None,
      bce_weights = None,
      prefix = '',
      **kwargs,
  ):
    super().__init__(
        args,
        rep_encoder,
        dataset_collection,
        autoregressive,
        has_vitals,
        bce_weights,
        prefix=prefix,
    )

  def _init_specific(self, sub_args):
    super()._init_specific(sub_args)
    self.prev_treatment_encoder = nn.Conv1d(
        self.dim_treatments,
        sub_args.emb_dim,
        kernel_size=self.output_horizon,
        padding=self.output_horizon - 1,
    )
    self.treatment_encoder = nn.Linear(self.dim_treatments, sub_args.emb_dim)

    self.predict_nets = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(sub_args.emb_dim, sub_args.hidden_dim),
                nn.ReLU(),
                nn.Linear(sub_args.hidden_dim, self.dim_outcome),
            )
            for _ in range(self.output_horizon)
        ]
    )

  def forward(self, batch, return_rep=False):
    vitals_or_prev_outputs = []
    if self.has_vitals:
      vitals_or_prev_outputs.append(batch['vitals'])
    if self.autoregressive:
      vitals_or_prev_outputs.append(
          batch['prev_outputs']
      )
    prev_treatments = batch['prev_treatments']
    current_treatments = batch['current_treatments']

    x_enc = self.rep_encoder.encode(batch)  # [B, T, D]

    # unroll prev_treatments to [B, T, output_horizon, D]
    unrolled_prev_treatments = self._unroll_horizon(
        prev_treatments, self.output_horizon
    )
    unrolled_prev_treatments[:, :, 0, :] = 0
    batch_size, step_num, output_horizon, dim = unrolled_prev_treatments.shape
    # [B x T, D, output_horizon]
    unrolled_prev_treatments = unrolled_prev_treatments.reshape(
        -1, output_horizon, dim
    ).permute(0, 2, 1)
    # [B x T, D, output_horizon]
    unrolled_prev_tr_enc = self.prev_treatment_encoder(
        unrolled_prev_treatments
    )[:, :, :output_horizon]
    enc_dim = unrolled_prev_tr_enc.shape[1]
    unrolled_prev_tr_enc = unrolled_prev_tr_enc.permute(0, 2, 1).reshape(
        batch_size, step_num, output_horizon, enc_dim
    )

    # unroll treatments to [B, T, output_horizon, D]
    unrolled_current_treatments = self._unroll_horizon(
        current_treatments, self.output_horizon
    )
    unrolled_current_tr_enc = self.treatment_encoder(
        unrolled_current_treatments
    )

    input_enc = (
        x_enc.unsqueeze(2) + unrolled_prev_tr_enc + unrolled_current_tr_enc
    )
    preds = []
    for h in range(self.output_horizon):
      preds.append(self.predict_nets[h](input_enc[:, :, h]))
    pred = torch.stack(preds, dim=2)

    if return_rep:
      return pred, x_enc
    else:
      return pred


class TCNEstHeadAutoreg(EstHeadAutoreg):
  """TCN estimation head autoregression."""

  def __init__(
      self,
      args,
      rep_encoder,
      dataset_collection,
      autoregressive = None,
      has_vitals = None,
      bce_weights = None,
      prefix = '',
      **kwargs,
  ):
    super().__init__(
        args,
        rep_encoder,
        dataset_collection,
        autoregressive,
        has_vitals,
        bce_weights,
        prefix=prefix,
    )

  def _init_specific(self, sub_args):
    super()._init_specific(sub_args)
    self.prev_treatment_encoder = nn.Conv1d(
        self.dim_treatments,
        sub_args.emb_dim,
        kernel_size=self.output_horizon,
        padding=self.output_horizon - 1,
    )
    self.prev_output_encoder = nn.Conv1d(
        self.dim_outcome,
        sub_args.emb_dim,
        kernel_size=self.output_horizon,
        padding=self.output_horizon - 1,
    )
    self.treatment_encoder = nn.Linear(self.dim_treatments, sub_args.emb_dim)
    self.predict_net = nn.Sequential(
        nn.Linear(sub_args.emb_dim, sub_args.hidden_dim),
        nn.ReLU(),
        nn.Linear(sub_args.hidden_dim, self.dim_outcome),
    )

  def _1dconv_unrolled(self, unrolled, conv1d_layer):
    batch_size, step_num, output_horizon, dim = unrolled.shape
    # [B x T, D, output_horizon]
    unrolled = unrolled.reshape(-1, output_horizon, dim).permute(0, 2, 1)
    # [B x T, D, output_horizon]
    unrolled_enc = conv1d_layer(unrolled)[:, :, :output_horizon]
    enc_dim = unrolled_enc.shape[1]
    unrolled_enc = unrolled_enc.permute(0, 2, 1).reshape(
        batch_size, step_num, output_horizon, enc_dim
    )
    return unrolled_enc

  def forward(self, batch, return_rep=False, one_step=False):
    prev_treatments = batch['prev_treatments']
    current_treatments = batch['current_treatments']

    x_enc = self.rep_encoder.encode(batch)  # [B, T, D]

    # unroll prev_treatments to [B, T, output_horizon, D]
    unrolled_prev_treatments = self._unroll_horizon(
        prev_treatments, self.output_horizon
    )
    batch_size, step_num, output_horizon, _ = unrolled_prev_treatments.shape
    unrolled_prev_treatments[:, :, 0, :] = 0
    unrolled_prev_tr_enc = self._1dconv_unrolled(
        unrolled_prev_treatments, self.prev_treatment_encoder
    )

    # unroll treatments to [B, T, output_horizon, D]
    unrolled_current_treatments = self._unroll_horizon(
        current_treatments, self.output_horizon
    )
    unrolled_current_tr_enc = self.treatment_encoder(
        unrolled_current_treatments
    )
    partial_input_enc = (
        x_enc.unsqueeze(2) + unrolled_prev_tr_enc + unrolled_current_tr_enc
    )

    # unroll prev_outputs to [B, T, output_horizon, D]
    pred = []
    unrolled_prev_pred_outputs = torch.zeros(
        (batch_size, step_num, output_horizon, self.dim_outcome)
    ).to(x_enc.device)

    tmax = 1 if one_step else self.output_horizon
    for t in range(tmax):
      unrolled_prev_pred_output_enc = self._1dconv_unrolled(
          unrolled_prev_pred_outputs, self.prev_output_encoder
      )
      input_enc_t = partial_input_enc + unrolled_prev_pred_output_enc
      pred_t = self.predict_net(input_enc_t)[:, :, t]
      pred.append(pred_t)
      if t < tmax - 1:
        unrolled_prev_pred_outputs[:, :, t + 1] = pred_t.detach()
    pred = torch.stack(pred, dim=2)

    if return_rep:
      return pred, x_enc
    else:
      return pred
