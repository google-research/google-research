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

"""CRN."""

import gzip
import logging
import os
import pickle as pkl
from typing import Union

import numpy as np
import omegaconf
from src.data import RealDatasetCollection
from src.data import SyntheticDatasetCollection
from src.models.time_varying_model import BRCausalModel
from src.models.utils import BRTreatmentOutcomeHead
from src.models.utils_lstm import VariationalLSTM
import torch

DictConfig = omegaconf.DictConfig
MissingMandatoryValue = omegaconf.errors.MissingMandatoryValue
logger = logging.getLogger(__name__)


class CRN(BRCausalModel):
  """Pytorch-Lightning implementation of Counterfactual Recurrent Network (CRN).

  (https://arxiv.org/abs/2002.04083,
  https://github.com/ioanabica/Counterfactual-Recurrent-Network)
  """

  model_type = None  # Will be defined in subclasses
  possible_model_types = {'encoder', 'decoder'}

  def __init__(
      self,
      args,
      dataset_collection = None,
      autoregressive = None,
      has_vitals = None,
      bce_weights = None,
      **kwargs,
  ):
    super().__init__(
        args, dataset_collection, autoregressive, has_vitals, bce_weights
    )

  def _init_specific(self, sub_args):
    # Encoder/decoder-specific parameters
    try:
      self.br_size = sub_args.br_size  # balanced representation size
      self.seq_hidden_units = sub_args.seq_hidden_units
      self.fc_hidden_units = sub_args.fc_hidden_units
      self.dropout_rate = sub_args.dropout_rate
      self.num_layer = sub_args.num_layer

      # Pytorch model init
      if (
          self.seq_hidden_units is None
          or self.br_size is None
          or self.fc_hidden_units is None
          or self.dropout_rate is None
      ):
        raise MissingMandatoryValue()

      self.lstm = VariationalLSTM(
          self.input_size,
          self.seq_hidden_units,
          self.num_layer,
          self.dropout_rate,
      )

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

    except MissingMandatoryValue:
      logger.warning(
          '%s',
          f'{self.model_type} not fully initialised - some mandatory args are'
          " missing! (It's ok, if one will perform hyperparameters search"
          ' afterward).',
      )

  @staticmethod
  def set_hparams(
      model_args, new_args, input_size, model_type
  ):
    sub_args = model_args[model_type]
    sub_args.optimizer.learning_rate = new_args['learning_rate']
    sub_args.batch_size = new_args['batch_size']
    if 'seq_hidden_units' in new_args:  # Only relevant for encoder
      sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
    sub_args.br_size = int(input_size * new_args['br_size'])
    sub_args.fc_hidden_units = int(
        sub_args.br_size * new_args['fc_hidden_units']
    )
    sub_args.dropout_rate = new_args['dropout_rate']
    sub_args.num_layer = new_args['num_layer']

  def build_br(
      self,
      prev_treatments,
      vitals_or_prev_outputs,
      static_features,
      init_states=None,
  ):
    x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)
    x = torch.cat(
        (x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1
    )
    x = self.lstm(x, init_states=init_states)
    br = self.br_treatment_outcome_head.build_br(x)
    return br


class CRNEncoder(CRN):
  """CRN encoder."""

  model_type = 'encoder'

  def __init__(
      self,
      args,
      dataset_collection = None,
      autoregressive = None,
      has_vitals = None,
      bce_weights = None,
      **kwargs,
  ):
    super().__init__(
        args, dataset_collection, autoregressive, has_vitals, bce_weights
    )
    self.input_size = self.dim_treatments + self.dim_static_features
    self.input_size += self.dim_vitals if self.has_vitals else 0
    self.input_size += self.dim_outcome if self.autoregressive else 0
    logger.info('%s', f'Input size of {self.model_type}: {self.input_size}')

    self._init_specific(args.model.encoder)
    self.save_hyperparameters(args)

    self.save_prepared_data = args.exp.save_prepared_data
    self.save_prepared_data_times = 0
    if self.save_prepared_data:
      self.saved_prepared_data_filename = (
          f'{args.model.name}-{self.model_type}-{args.dataset.coeff}-{args.dataset.seed}'
          + '-t{save_prepared_data_times}.pkl'
      )

  def prepare_data(self):
    # Datasets normalisation etc.
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.processed_data_encoder
    ):
      self.dataset_collection.process_data_encoder()
    if self.bce_weights is None and self.hparams.exp.bce_weight:
      self._calculate_bce_weights()

    if self.save_prepared_data:
      # save_path = os.path.join(os.getcwd(), self.saved_prepared_data_filename)
      # # if not os.path.exists(save_path):
      # if True:
      #     logger.info(f'Saving prepared data to {save_path}...')
      #     with gzip.open(save_path, 'wb') as f:
      #         pkl.dump(self.dataset_collection, f)
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
    prev_treatments = batch['prev_treatments']
    vitals_or_prev_outputs = []
    if self.has_vitals:
      vitals_or_prev_outputs.append(batch['vitals'])
    if self.autoregressive:
      vitals_or_prev_outputs.append(batch['prev_outputs'])
    vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)

    static_features = batch['static_features']
    curr_treatments = batch['current_treatments']
    init_states = None  # None for encoder

    br = self.build_br(
        prev_treatments, vitals_or_prev_outputs, static_features, init_states
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


class CRNDecoder(CRN):
  """CRN decoder."""

  model_type = 'decoder'

  def __init__(
      self,
      args,
      encoder = None,
      dataset_collection = None,
      encoder_r_size = None,
      autoregressive = None,
      has_vitals = None,
      bce_weights = None,
      **kwargs,
  ):
    super().__init__(
        args, dataset_collection, autoregressive, has_vitals, bce_weights
    )

    self.input_size = (
        self.dim_treatments + self.dim_static_features + self.dim_outcome
    )
    logger.info('%s', f'Input size of {self.model_type}: {self.input_size}')

    self.encoder = encoder
    args.model.decoder.seq_hidden_units = (
        self.encoder.br_size if encoder is not None else encoder_r_size
    )
    self._init_specific(args.model.decoder)
    self.save_hyperparameters(args)

    self.save_prepared_data = args.exp.save_prepared_data
    self.save_prepared_data_times = 0
    if self.save_prepared_data:
      self.saved_prepared_data_filename = (
          f'{args.model.name}-{self.model_type}-{args.dataset.coeff}-{args.dataset.seed}'
          + '-t{save_prepared_data_times}.pkl'
      )

  def prepare_data(self):
    # Datasets normalisation etc.
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.processed_data_decoder
    ):
      self.dataset_collection.process_data_decoder(self.encoder)
    if self.bce_weights is None and self.hparams.exp.bce_weight:
      self._calculate_bce_weights()

    if self.save_prepared_data:
      # save_path = os.path.join(os.getcwd(), self.saved_prepared_data_filename)
      # # if not os.path.exists(save_path):
      # if True:
      #     logger.info(f'Saving prepared data to {save_path}...')
      #     with gzip.open(save_path, 'wb') as f:
      #         pkl.dump(self.dataset_collection, f)
      save_path = os.path.join(
          os.getcwd(),
          self.saved_prepared_data_filename.format(
              save_prepared_data_times=self.save_prepared_data_times
          ),
      )
      self.save_prepared_data_times += 1

      logger.info('%s', f'Saving prepared data to {save_path}...')
      with gzip.open(save_path, 'wb') as f:
        pkl.dump(self.dataset_collection, f)

  def forward(self, batch, detach_treatment=False):
    prev_treatments = batch['prev_treatments']
    prev_outputs = batch['prev_outputs']
    static_features = batch['static_features']
    curr_treatments = batch['current_treatments']
    init_states = batch['init_state']

    br = self.build_br(
        prev_treatments, prev_outputs, static_features, init_states
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
