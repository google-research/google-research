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

"""Causal transformer."""

import gzip
import logging
import os
import pickle as pkl
from typing import Dict
from typing import Union
import numpy as np
import omegaconf
from src.data import RealDatasetCollection
from src.data import SyntheticDatasetCollection
from src.models.edct import EDCT
from src.models.utils import BRTreatmentOutcomeHead
from src.models.utils import BRTreatmentOutcomeTLearnerHead
from src.models.utils_transformer import TransformerMultiInputBlock
import torch
from torch import nn
from torch_ema import ExponentialMovingAverage

DictConfig = omegaconf.DictConfig
MissingMandatoryValue = omegaconf.errors.MissingMandatoryValue
DataLoader = torch.utils.data.DataLoader
Dataset = torch.utils.data.Dataset
logger = logging.getLogger(__name__)


class CT(EDCT):
  """Pytorch-Lightning implementation of Causal Transformer (CT)."""

  model_type = 'multi'  # multi-input model
  possible_model_types = {'multi'}

  def __init__(
      self,
      args,
      dataset_collection = None,
      autoregressive = None,
      has_vitals = None,
      projection_horizon = None,
      bce_weights = None,
      **kwargs,
  ):
    super().__init__(
        args, dataset_collection, autoregressive, has_vitals, bce_weights
    )

    if self.dataset_collection is not None:
      self.projection_horizon = self.dataset_collection.projection_horizon
    else:
      self.projection_horizon = projection_horizon

    # Used in hparam tuning
    self.input_size = max(
        self.dim_treatments,
        self.dim_static_features,
        self.dim_vitals,
        self.dim_outcome,
    )
    logger.info('%s', f'Max input size of {self.model_type}: {self.input_size}')
    assert self.autoregressive  # prev_outcomes are obligatory

    self.basic_block_cls = TransformerMultiInputBlock
    self._init_specific(args.model.multi)
    self.save_hyperparameters(args)

    self.save_prepared_data = args.exp.save_prepared_data
    self.save_prepared_data_times = 0
    if self.save_prepared_data:
      self.saved_prepared_data_filename = (
          f'{args.model.name}-{self.model_type}-{args.dataset.coeff}-{args.dataset.seed}'
          + '-t{save_prepared_data_times}.pkl'
      )

  def _init_specific(self, sub_args):
    try:
      super(CT, self)._init_specific(sub_args)

      if (
          self.seq_hidden_units is None
          or self.br_size is None
          or self.fc_hidden_units is None
          or self.dropout_rate is None
      ):
        raise MissingMandatoryValue()

      self.treatments_input_transformation = nn.Linear(
          self.dim_treatments, self.seq_hidden_units
      )
      self.vitals_input_transformation = (
          nn.Linear(self.dim_vitals, self.seq_hidden_units)
          if self.has_vitals
          else None
      )
      self.vitals_input_transformation = (
          nn.Linear(self.dim_vitals, self.seq_hidden_units)
          if self.has_vitals
          else None
      )
      self.outputs_input_transformation = nn.Linear(
          self.dim_outcome, self.seq_hidden_units
      )
      self.static_input_transformation = nn.Linear(
          self.dim_static_features, self.seq_hidden_units
      )

      self.n_inputs = (
          3 if self.has_vitals else 2
      )  # prev_outcomes and prev_treatments

      self.transformer_blocks = nn.ModuleList(
          [
              self.basic_block_cls(
                  self.seq_hidden_units,
                  self.num_heads,
                  self.head_size,
                  self.seq_hidden_units * 4,
                  self.dropout_rate,
                  self.dropout_rate if sub_args.attn_dropout else 0.0,
                  self_positional_encoding_k=self.self_positional_encoding_k,
                  self_positional_encoding_v=self.self_positional_encoding_v,
                  n_inputs=self.n_inputs,
                  disable_cross_attention=sub_args.disable_cross_attention,
                  isolate_subnetwork=sub_args.isolate_subnetwork,
              )
              for _ in range(self.num_layer)
          ]
      )

      if sub_args.outcome_learner == 'slearner':
        self.br_treatment_outcome_head = BRTreatmentOutcomeHead(
            self.seq_hidden_units,
            self.br_size,
            self.fc_hidden_units,
            self.dim_treatments,
            self.dim_outcome,
            self.alpha,
            self.update_alpha,
            self.balancing,
            alpha_prev_treat=self.alpha_prev_treat,
            update_alpha_prev_treat=self.update_alpha_prev_treat,
            alpha_age=self.alpha_age,
            update_alpha_age=self.update_alpha_age,
        )
      elif sub_args.outcome_learner == 'tlearner':
        self.br_treatment_outcome_head = BRTreatmentOutcomeTLearnerHead(
            self.seq_hidden_units,
            self.br_size,
            self.fc_hidden_units,
            self.dim_treatments,
            self.dim_outcome,
            self.alpha,
            self.update_alpha,
            self.balancing,
            alpha_prev_treat=self.alpha_prev_treat,
            update_alpha_prev_treat=self.update_alpha_prev_treat,
            alpha_age=self.alpha_age,
            update_alpha_age=self.update_alpha_age,
            is_one_hot_treatment=sub_args.is_one_hot_treatment,
        )

      # self.last_layer_norm = LayerNorm(self.seq_hidden_units)

      self.prefix_tuning = sub_args.prefix_tuning
      if self.prefix_tuning:
        self.prefix_length = sub_args.prefix_length
        self.prefix_dim = sub_args.prefix_dim
        self.prefix_mid_dim = sub_args.prefix_mid_dim
        self.register_parameter(
            name='prefix_params',
            param=nn.Parameter(
                torch.randn(self.prefix_length, self.prefix_dim)
            ),
        )
        total_attn_block_num = sum(
            [len(tb.attention_block_name2idx) for tb in self.transformer_blocks]
        )
        self.prefix_control_trans = nn.Sequential(
            nn.Linear(self.prefix_dim, self.prefix_mid_dim),
            nn.Tanh(),
            nn.Linear(
                self.prefix_mid_dim,
                total_attn_block_num * 2 * self.num_heads * self.head_size,
            ),
        )
    except MissingMandatoryValue:
      logger.warning(
          '%s',
          f'{self.model_type} not fully initialised - some mandatory args are'
          " missing! (It's ok, if one will perform hyperparameters search"
          ' afterward).',
      )

  def configure_optimizers(self):
    if not self.prefix_tuning:
      return super().configure_optimizers()
    else:
      treatment_head_params = [
          'br_treatment_outcome_head.' + s
          for s in self.br_treatment_outcome_head.treatment_head_params
      ]
      treatment_head_params_update = []
      for k in dict(self.named_parameters()):
        for param in treatment_head_params:
          if k.startswith(param):
            treatment_head_params_update.append(k)
      treatment_head_params = treatment_head_params_update
      non_treatment_head_params = [
          k
          for k in dict(self.named_parameters())
          if (k not in treatment_head_params) and not k.startswith('prefix_')
      ]

      treatment_head_params = [
          (k, v)
          for k, v in dict(self.named_parameters()).items()
          if k in treatment_head_params
      ]
      non_treatment_head_params = [
          (k, v)
          for k, v in dict(self.named_parameters()).items()
          if k in non_treatment_head_params
      ]

      if self.hparams.exp.weights_ema:
        self.ema_treatment = ExponentialMovingAverage(
            [par[1] for par in treatment_head_params],
            decay=self.hparams.exp.beta,
        )
        self.ema_non_treatment = ExponentialMovingAverage(
            [par[1] for par in non_treatment_head_params],
            decay=self.hparams.exp.beta,
        )

      prefix_parameters = [
          (k, v)
          for k, v in dict(self.named_parameters()).items()
          if k.startswith('prefix_')
      ]
      prefix_optimizer = self._get_optimizer(prefix_parameters)
      if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
        return self._get_lr_schedulers(prefix_optimizer)
      return prefix_optimizer

  def prepare_data(self):
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.processed_data_multi
    ):
      self.dataset_collection.process_data_multi()
    if self.bce_weights is None and self.hparams.exp.bce_weight:
      self._calculate_bce_weights()

    if self.save_prepared_data:
      save_path = os.path.join(
          os.getcwd(),
          self.saved_prepared_data_filename.format(
              save_prepared_data_times=self.save_prepared_data_times
          ),
      )
      self.save_prepared_data_times += 1
      # if not os.path.exists(save_path):
      logger.info('%s', f'Saving prepared data to {save_path}...')
      with gzip.open(save_path, 'wb') as f:
        pkl.dump(self.dataset_collection, f)

  def forward(self, batch, detach_treatment=False):
    fixed_split = (
        batch['future_past_split'] if 'future_past_split' in batch else None
    )

    if (
        self.training
        and self.hparams.model.multi.augment_with_masked_vitals
        and self.has_vitals
    ):
      # Augmenting original batch with vitals-masked copy
      assert fixed_split is None  # Only for training data
      fixed_split = torch.empty((2 * len(batch['active_entries']),)).type_as(
          batch['active_entries']
      )
      for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
        fixed_split[i] = seq_len  # Original batch
        fixed_split[len(batch['active_entries']) + i] = torch.randint(
            0, int(seq_len) + 1, (1,)
        ).item()  # Augmented batch

      for k, v in batch.items():
        batch[k] = torch.cat((v, v), dim=0)

    prev_treatments = batch['prev_treatments']
    vitals = batch['vitals'] if self.has_vitals else None
    prev_outputs = batch['prev_outputs']
    static_features = batch['static_features']
    curr_treatments = batch['current_treatments']
    active_entries = batch['active_entries']

    br = self.build_br(
        prev_treatments,
        vitals,
        prev_outputs,
        static_features,
        active_entries,
        fixed_split,
    )
    treatment_pred = self.br_treatment_outcome_head.build_treatment(
        br, detach_treatment
    )
    outcome_pred = self.br_treatment_outcome_head.build_outcome(
        br, curr_treatments
    )

    if self.alpha_prev_treat > 0 or self.alpha_age > 0:
      domain_label_pred = self.br_treatment_outcome_head.build_domain_label(
          br, detach_treatment
      )
      treatment_pred = [treatment_pred, domain_label_pred]

    return treatment_pred, outcome_pred, br

  def build_br(
      self,
      prev_treatments,
      vitals,
      prev_outputs,
      static_features,
      active_entries,
      fixed_split,
  ):
    active_entries_treat_outcomes = torch.clone(active_entries)
    active_entries_vitals = torch.clone(active_entries)

    if (
        fixed_split is not None and self.has_vitals
    ):  # Test sequence data / Train augmented data
      for i in range(len(active_entries)):
        # Masking vitals in range [fixed_split: ]
        active_entries_vitals[i, int(fixed_split[i]) :, :] = 0.0
        vitals[i, int(fixed_split[i]) :] = 0.0

    x_t = self.treatments_input_transformation(prev_treatments)
    x_o = self.outputs_input_transformation(prev_outputs)
    x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None
    x_s = self.static_input_transformation(
        static_features.unsqueeze(1)
    )  # .expand(-1, x_t.size(1), -1)

    prefix_embs = None
    prefix_secs = None
    if self.prefix_tuning:
      prefix_embs = self.prefix_control_trans(self.prefix_params)
      # (prefix_len, total_attn_block_num * 2 * self.num_heads * self.head_size)
      prefix_embs = prefix_embs.reshape(
          self.prefix_length, -1, 2, self.num_heads, self.head_size
      ).permute(1, 2, 0, 3, 4)
      # (total_attn_block_num, 2, prefix_len, self.num_heads * self.head_size)
      prefix_secs = [0]
      for block in self.transformer_blocks:
        prefix_secs.append(
            prefix_secs[-1] + len(block.attention_block_name2idx)
        )

    # if active_encoder_br is None and encoder_r is None:  # Only self-attention
    for block_i, block in enumerate(self.transformer_blocks):
      if self.self_positional_encoding is not None:
        x_t = x_t + self.self_positional_encoding(x_t)
        x_o = x_o + self.self_positional_encoding(x_o)
        x_v = (
            x_v + self.self_positional_encoding(x_v)
            if self.has_vitals
            else None
        )

      if self.prefix_tuning:
        prefix_list = prefix_embs[
            prefix_secs[block_i] : prefix_secs[block_i + 1]
        ]
      else:
        prefix_list = None

      if self.has_vitals:
        x_t, x_o, x_v = block(
            (x_t, x_o, x_v),
            x_s,
            active_entries_treat_outcomes,
            active_entries_vitals,
            prefix_list=prefix_list,
        )
      else:
        x_t, x_o = block(
            (x_t, x_o),
            x_s,
            active_entries_treat_outcomes,
            prefix_list=prefix_list,
        )

    if not self.has_vitals:
      x = (x_o + x_t) / 2
    else:
      if fixed_split is not None:  # Test seq data
        x = torch.empty_like(x_o)
        for i in range(len(active_entries)):
          # Masking vitals in range [fixed_split: ]
          x[i, : int(fixed_split[i])] = (
              x_o[i, : int(fixed_split[i])]
              + x_t[i, : int(fixed_split[i])]
              + x_v[i, : int(fixed_split[i])]
          ) / 3
          x[i, int(fixed_split[i]) :] = (
              x_o[i, int(fixed_split[i]) :] + x_t[i, int(fixed_split[i]) :]
          ) / 2
      else:  # Train data always has vitals
        x = (x_o + x_t + x_v) / 3

    output = self.output_dropout(x)
    br = self.br_treatment_outcome_head.build_br(output)
    return br

  def get_autoregressive_predictions(self, dataset):
    logger.info('%s', f'Autoregressive Prediction for {dataset.subset_name}.')

    predicted_outputs = np.zeros((
        len(dataset),
        self.hparams.dataset.projection_horizon,
        self.dim_outcome,
    ))

    for t in range(self.hparams.dataset.projection_horizon + 1):
      logger.info('%s', f't = {t + 1}')
      outputs_scaled = self.get_predictions(dataset)

      for i in range(len(dataset)):
        split = int(dataset.data['future_past_split'][i])
        if t < self.hparams.dataset.projection_horizon:
          dataset.data['prev_outputs'][i, split + t, :] = outputs_scaled[
              i, split - 1 + t, :
          ]
        if t > 0:
          predicted_outputs[i, t - 1, :] = outputs_scaled[i, split - 1 + t, :]

    return predicted_outputs

  def get_autoregressive_predictions_generator(self, dataset):
    logger.info('%s', f'Autoregressive Prediction for {dataset.subset_name}.')
    # Creating Dataloader
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    self.to('cuda:0')
    self.eval()
    with torch.no_grad():
      for batch_idx, batch in enumerate(data_loader):
        for k in batch:
          batch[k] = batch[k].to('cuda:0')
        predicted_outputs = np.zeros((
            len(batch['prev_outputs']),
            self.hparams.dataset.projection_horizon,
            self.dim_outcome,
        ))
        for t in range(self.hparams.dataset.projection_horizon):
          outputs_scaled, _ = self.predict_step(batch, batch_idx)
          for i in range(len(batch['future_past_split'])):
            split = int(batch['future_past_split'][i])
            if t < self.hparams.dataset.projection_horizon:
              batch['prev_outputs'][i, split + t, :] = outputs_scaled[
                  i, split - 1 + t, :
              ]
            if t > 0:
              predicted_outputs[i, t - 1, :] = (
                  outputs_scaled[i, split - 1 + t, :].cpu().numpy()
              )
        yield predicted_outputs, None, None

  def get_autoregressive_causal_representations(
      self, dataset
  ):
    logger.info(
        '%s',
        f'Autoregressive Causal Representations for {dataset.subset_name}.',
    )

    predicted_causal_reps = None
    for t in range(self.hparams.dataset.projection_horizon + 1):
      logger.info('%s', f't = {t + 1}')
      outputs_scaled = self.get_predictions(dataset)
      causal_reps = self.get_causal_representations(dataset)
      if predicted_causal_reps is None:
        predicted_causal_reps = {}
        for k in causal_reps:
          predicted_causal_reps[k] = np.zeros((
              len(dataset),
              self.hparams.dataset.projection_horizon,
              causal_reps[k].shape[-1],
          ))

      for i in range(len(dataset)):
        split = int(dataset.data['future_past_split'][i])
        if t < self.hparams.dataset.projection_horizon:
          dataset.data['prev_outputs'][i, split + t, :] = outputs_scaled[
              i, split - 1 + t, :
          ]
        if t > 0:
          for k in causal_reps:
            predicted_causal_reps[k][i, t - 1, :] = causal_reps[k][
                i, split - 1 + t, :
            ]

    return predicted_causal_reps

  def visualize(self, dataset, index=0, artifacts_path=None):
    fig_keys = [
        'self_attention_o',
        'self_attention_t',
        'cross_attention_ot',
        'cross_attention_to',
    ]
    if self.has_vitals:
      fig_keys += [
          'cross_attention_vo',
          'cross_attention_ov',
          'cross_attention_vt',
          'cross_attention_tv',
          'self_attention_v',
      ]
    self._visualize(fig_keys, dataset, index, artifacts_path)
