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

"""RMSN."""

import gzip
import logging
import os
import pickle as pkl
from typing import Union

import numpy as np
import omegaconf
from src.data import RealDatasetCollection
from src.data import SyntheticDatasetCollection
from src.models.time_varying_model import TimeVaryingCausalModel
from src.models.utils import clip_normalize_stabilized_weights
from src.models.utils_lstm import VariationalLSTM
import torch
from torch import nn
import torch.nn.functional as F


DictConfig = omegaconf.DictConfig
MissingMandatoryValue = omegaconf.errors.MissingMandatoryValue
DataLoader = torch.utils.data.DataLoader
Dataset = torch.utils.data.Dataset
logger = logging.getLogger(__name__)


class RMSN(TimeVaryingCausalModel):
  """Pytorch-Lightning implementation.

  Recurrent Marginal Structural Networks (RMSNs).
  """

  model_type = None  # Will be defined in subclasses
  possible_model_types = {
      'encoder',
      'decoder',
      'propensity_treatment',
      'propensity_history',
  }
  tuning_criterion = None

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

  def _init_specific(self, sub_args, encoder_r_size = None):
    # Encoder/decoder-specific parameters
    try:
      self.seq_hidden_units = sub_args.seq_hidden_units
      self.dropout_rate = sub_args.dropout_rate
      self.num_layer = sub_args.num_layer

      # Pytorch model init
      if self.seq_hidden_units is None or self.dropout_rate is None:
        raise MissingMandatoryValue()

      if self.model_type == 'decoder':
        self.memory_adapter = nn.Linear(encoder_r_size, self.seq_hidden_units)

      self.lstm = VariationalLSTM(
          self.input_size,
          self.seq_hidden_units,
          self.num_layer,
          self.dropout_rate,
      )
      self.output_layer = nn.Linear(self.seq_hidden_units, self.output_size)

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
    """Used for hyperparameter tuning and model reinitialisation."""
    sub_args = model_args[model_type]
    sub_args.optimizer.learning_rate = new_args['learning_rate']
    sub_args.batch_size = new_args['batch_size']
    sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
    sub_args.dropout_rate = new_args['dropout_rate']
    sub_args.num_layer = new_args['num_layer']
    sub_args.max_grad_norm = new_args['max_grad_norm']

  def get_propensity_scores(self, dataset):
    logger.info('%s', f'Propensity scores for {dataset.subset_name}.')
    if (
        self.model_type == 'propensity_treatment'
        or self.model_type == 'propensity_history'
    ):
      data_loader = DataLoader(
          dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
      )
      propensity_scores = torch.cat(self.trainer.predict(self, data_loader))
    else:
      raise NotImplementedError()
    return propensity_scores.numpy()


class RMSNPropensityNetworkTreatment(RMSN):
  """Propensity network."""

  model_type = 'propensity_treatment'
  tuning_criterion = 'bce'

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
    self.input_size = self.dim_treatments
    logger.info('%s', f'Input size of {self.model_type}: {self.input_size}')

    self.output_size = self.dim_treatments

    self._init_specific(args.model.propensity_treatment)
    self.save_hyperparameters(args)

    self.save_prepared_data = args.exp.save_prepared_data
    if self.save_prepared_data:
      self.saved_prepared_data_filename = f'{args.model.name}-{self.model_type}-{args.dataset.coeff}-{args.dataset.seed}.pkl'

  def prepare_data(self):
    # Datasets normalisation etc.
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.processed_data_encoder
    ):
      assert (
          self.hparams.dataset.treatment_mode == 'multilabel'
      )  # Only binary multilabel regime possible
      self.dataset_collection.process_data_encoder()
    if self.bce_weights is None and self.hparams.exp.bce_weight:
      self._calculate_bce_weights()

    if self.save_prepared_data:
      save_path = os.path.join(os.getcwd(), self.saved_prepared_data_filename)
      # if not os.path.exists(save_path):
      logger.info('%s', f'Saving prepared data to {save_path}...')
      with gzip.open(save_path, 'wb') as f:
        pkl.dump(self.dataset_collection, f)

  def forward(self, batch):
    prev_treatments = batch['prev_treatments']
    x = self.lstm(prev_treatments, init_states=None)
    treatment_pred = self.output_layer(x)
    return treatment_pred

  def training_step(self, batch, batch_ind):
    _ = batch_ind
    treatment_pred = self(batch)
    bce_loss = self.bce_loss(
        treatment_pred, batch['current_treatments'].double(), kind='predict'
    )
    bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch[
        'active_entries'
    ].sum()
    self.log(
        f'{self.model_type}_bce_loss',
        bce_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    return bce_loss

  def _eval_step(self, batch, batch_ind, subset_name):
    _ = batch_ind
    treatment_pred = self(batch)
    bce_loss = self.bce_loss(
        treatment_pred, batch['current_treatments'].double(), kind='predict'
    )
    bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch[
        'active_entries'
    ].sum()

    self.log(
        f'{self.model_type}_{subset_name}_bce_loss',
        bce_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    # validation metric
    if subset_name == self.val_dataloader().dataset.subset_name:
      self.log(
          '{}-val_metric'.format(self.model_type),
          bce_loss,
          on_epoch=True,
          on_step=False,
          sync_dist=True,
      )

  def validation_step(self, batch, batch_ind, **kwargs):
    _ = kwargs
    subset_name = self.val_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def test_step(self, batch, batch_ind, **kwargs):
    _ = kwargs
    subset_name = self.test_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def predict_step(self, batch, batch_ind, dataset_idx=None):
    (
        _,
        _,
    ) = (
        batch_ind,
        dataset_idx,
    )
    return F.sigmoid(self(batch)).cpu()


class RMSNPropensityNetworkHistory(RMSN):
  """Propensity network history module."""

  model_type = 'propensity_history'
  tuning_criterion = 'bce'

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

    self.output_size = self.dim_treatments

    self._init_specific(args.model.propensity_history)
    self.save_hyperparameters(args)

    self.save_prepared_data = args.exp.save_prepared_data
    if self.save_prepared_data:
      self.saved_prepared_data_filename = f'{args.model.name}-{self.model_type}-{args.dataset.coeff}-{args.dataset.seed}.pkl'

  def prepare_data(self):
    # Datasets normalisation etc.
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.processed_data_encoder
    ):
      assert (
          self.hparams.dataset.treatment_mode == 'multilabel'
      )  # Only binary multilabel regime possible
      self.dataset_collection.process_data_encoder()
    if self.bce_weights is None and self.hparams.exp.bce_weight:
      self._calculate_bce_weights()

    if self.save_prepared_data:
      save_path = os.path.join(os.getcwd(), self.saved_prepared_data_filename)
      # if not os.path.exists(save_path):
      logger.info('%s', f'Saving prepared data to {save_path}...')
      with gzip.open(save_path, 'wb') as f:
        pkl.dump(self.dataset_collection, f)

  def forward(self, batch, detach_treatment=False):
    _ = detach_treatment
    prev_treatments = batch['prev_treatments']
    vitals_or_prev_outputs = []
    if self.has_vitals:
      vitals_or_prev_outputs.append(batch['vitals'])
    if self.autoregressive:
      vitals_or_prev_outputs.append(batch['prev_outputs'])
    vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
    static_features = batch['static_features']

    x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)
    x = torch.cat(
        (x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1
    )
    x = self.lstm(x, init_states=None)
    treatment_pred = self.output_layer(x)
    return treatment_pred

  def training_step(self, batch, batch_ind):
    _ = batch_ind
    treatment_pred = self(batch)
    bce_loss = self.bce_loss(
        treatment_pred, batch['current_treatments'].double(), kind='predict'
    )
    bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch[
        'active_entries'
    ].sum()
    self.log(
        f'{self.model_type}_bce_loss',
        bce_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    return bce_loss

  def _eval_step(self, batch, batch_ind, subset_name):
    _ = batch_ind
    treatment_pred = self(batch)
    bce_loss = self.bce_loss(
        treatment_pred, batch['current_treatments'].double(), kind='predict'
    )
    bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch[
        'active_entries'
    ].sum()

    self.log(
        f'{self.model_type}_{subset_name}_bce_loss',
        bce_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    # validation metric
    if subset_name == self.val_dataloader().dataset.subset_name:
      self.log(
          '{}-val_metric'.format(self.model_type),
          bce_loss,
          on_epoch=True,
          on_step=False,
          sync_dist=True,
      )

  def validation_step(self, batch, batch_ind, **kwargs):
    _ = kwargs
    subset_name = self.val_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def test_step(self, batch, batch_ind, **kwargs):
    _ = kwargs
    subset_name = self.test_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def predict_step(self, batch, batch_ind, dataset_idx=None):
    _, _ = batch_ind, dataset_idx
    return F.sigmoid(self(batch)).cpu()


class RMSNEncoder(RMSN):
  """Encoder module."""

  model_type = 'encoder'
  tuning_criterion = 'rmse'

  def __init__(
      self,
      args,
      propensity_treatment = None,
      propensity_history = None,
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

    self.output_size = self.dim_outcome

    self.propensity_treatment = propensity_treatment
    self.propensity_history = propensity_history

    self._init_specific(args.model.encoder)
    self.save_hyperparameters(args)

    self.save_prepared_data = args.exp.save_prepared_data
    if self.save_prepared_data:
      self.saved_prepared_data_filename = f'{args.model.name}-{self.model_type}-{args.dataset.coeff}-{args.dataset.seed}.pkl'

  def prepare_data(self):
    # Datasets normalisation etc.
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.processed_data_encoder
    ):
      self.dataset_collection.process_data_encoder()
    if (
        self.dataset_collection is not None
        and 'sw_tilde_enc' not in self.dataset_collection.train_f.data
    ):
      self.dataset_collection.process_propensity_train_f(
          self.propensity_treatment, self.propensity_history
      )
      self.dataset_collection.train_f.data['sw_tilde_enc'] = (
          clip_normalize_stabilized_weights(
              self.dataset_collection.train_f.data['stabilized_weights'],
              self.dataset_collection.train_f.data['active_entries'],
              multiple_horizons=False,
          )
      )

    if self.bce_weights is None and self.hparams.exp.bce_weight:
      self._calculate_bce_weights()

    if self.save_prepared_data:
      save_path = os.path.join(os.getcwd(), self.saved_prepared_data_filename)
      logger.info('%s', f'Saving prepared data to {save_path}...')
      with gzip.open(save_path, 'wb') as f:
        pkl.dump(self.dataset_collection, f)

  def forward(self, batch, detach_treatment=False):
    _ = detach_treatment
    vitals_or_prev_outputs = []
    if self.has_vitals:
      vitals_or_prev_outputs.append(batch['vitals'])
    if self.autoregressive:
      vitals_or_prev_outputs.append(batch['prev_outputs'])
    vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
    static_features = batch['static_features']
    curr_treatments = batch['current_treatments']

    x = torch.cat((vitals_or_prev_outputs, curr_treatments), dim=-1)
    x = torch.cat(
        (x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1
    )
    r = self.lstm(x, init_states=None)
    outcome_pred = self.output_layer(r)
    return outcome_pred, r

  def training_step(self, batch, batch_ind):
    _ = batch_ind
    outcome_pred, _ = self(batch)
    mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
    weighted_mse_loss = mse_loss * batch['sw_tilde_enc'].unsqueeze(-1)
    weighted_mse_loss = (
        batch['active_entries'] * weighted_mse_loss
    ).sum() / batch['active_entries'].sum()
    self.log(
        f'{self.model_type}_weighted_mse_loss',
        weighted_mse_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    return weighted_mse_loss

  def _eval_step(self, batch, batch_ind, subset_name):
    _ = batch_ind
    outcome_pred, _ = self(batch)
    mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
    mse_loss = (batch['active_entries'] * mse_loss).sum() / batch[
        'active_entries'
    ].sum()
    self.log(
        f'{self.model_type}_{subset_name}_mse_loss',
        mse_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    if subset_name == self.val_dataloader().dataset.subset_name:
      self.log('{}-val_metric'.format(self.model_type), mse_loss)

  def validation_step(self, batch, batch_ind, **kwargs):
    _ = kwargs
    subset_name = self.val_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def test_step(self, batch, batch_ind, **kwargs):
    _ = kwargs
    subset_name = self.test_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def predict_step(self, batch, batch_ind, dataset_idx=None):
    _, _ = batch_ind, dataset_idx
    outcome_pred, r = self(batch)
    return outcome_pred.cpu(), r.cpu()

  def get_representations(self, dataset):
    logger.info('%s', f'Representations inference for {dataset.subset_name}.')
    # Creating Dataloader
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    _, r = [
        torch.cat(arrs)
        for arrs in zip(*self.trainer.predict(self, data_loader))
    ]
    return r.numpy()

  def get_predictions(self, dataset):
    logger.info('%s', f'Predictions for {dataset.subset_name}.')
    # Creating Dataloader
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    outcome_pred, _ = [
        torch.cat(arrs)
        for arrs in zip(*self.trainer.predict(self, data_loader))
    ]
    return outcome_pred.numpy()


class RMSNDecoder(RMSN):
  """Decoder module."""

  model_type = 'decoder'
  tuning_criterion = 'rmse'

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

    self.output_size = self.dim_outcome

    self.encoder = encoder
    encoder_r_size = (
        self.encoder.seq_hidden_units if encoder is not None else encoder_r_size
    )

    self._init_specific(args.model.decoder, encoder_r_size=encoder_r_size)
    self.save_hyperparameters(args)

    self.save_prepared_data = args.exp.save_prepared_data
    if self.save_prepared_data:
      self.saved_prepared_data_filename = f'{args.model.name}-{self.model_type}-{args.dataset.coeff}-{args.dataset.seed}.pkl'

  def prepare_data(self):
    # Datasets normalisation etc.
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.processed_data_decoder
    ):
      self.dataset_collection.process_data_decoder(self.encoder)
    if (
        self.dataset_collection is not None
        and 'sw_tilde_dec' not in self.dataset_collection.train_f.data
    ):
      self.dataset_collection.train_f.data['stabilized_weights'] = np.cumprod(
          self.dataset_collection.train_f.data['stabilized_weights'], axis=-1
      )[:, 1:]
      self.dataset_collection.train_f.data['sw_tilde_dec'] = (
          clip_normalize_stabilized_weights(
              self.dataset_collection.train_f.data['stabilized_weights'],
              self.dataset_collection.train_f.data['active_entries'],
              multiple_horizons=True,
          )
      )
    if self.bce_weights is None and self.hparams.exp.bce_weight:
      self._calculate_bce_weights()

    if self.save_prepared_data:
      save_path = os.path.join(os.getcwd(), self.saved_prepared_data_filename)
      logger.info('%s', f'Saving prepared data to {save_path}...')
      with gzip.open(save_path, 'wb') as f:
        pkl.dump(self.dataset_collection, f)

  def forward(self, batch, detach_treatment=False):
    _ = detach_treatment
    curr_treatments = batch['current_treatments']
    prev_outputs = batch['prev_outputs']
    static_features = batch['static_features']
    init_states = batch['init_state']

    x = torch.cat((curr_treatments, prev_outputs), dim=-1)
    x = torch.cat(
        (x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1
    )
    x = self.lstm(x, init_states=self.memory_adapter(init_states))
    outcome_pred = self.output_layer(x)
    return outcome_pred

  def training_step(self, batch, batch_ind):
    _ = batch_ind
    outcome_pred = self(batch)
    mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
    weighted_mse_loss = mse_loss * batch['sw_tilde_dec'].unsqueeze(-1)
    weighted_mse_loss = (
        batch['active_entries'] * weighted_mse_loss
    ).sum() / batch['active_entries'].sum()
    self.log(
        f'{self.model_type}_weighted_mse_loss',
        weighted_mse_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    return weighted_mse_loss

  def _eval_step(self, batch, batch_ind, subset_name):
    _ = batch_ind
    outcome_pred = self(batch)
    mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
    mse_loss = (batch['active_entries'] * mse_loss).sum() / batch[
        'active_entries'
    ].sum()
    self.log(
        f'{self.model_type}_{subset_name}_mse_loss',
        mse_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    if subset_name == self.val_dataloader().dataset.subset_name:
      self.log('{}-val_metric'.format(self.model_type), mse_loss)

  def validation_step(self, batch, batch_ind, **kwargs):
    _ = kwargs
    subset_name = self.val_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def test_step(self, batch, batch_ind, **kwargs):
    _ = kwargs
    subset_name = self.test_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def predict_step(self, batch, batch_ind, dataset_idx=None):
    _, _ = batch_ind, dataset_idx
    return self(batch).cpu()

  def get_predictions(self, dataset):
    logger.info('%s', f'Predictions for {dataset.subset_name}.')
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    outcome_pred = torch.cat(self.trainer.predict(self, data_loader))
    return outcome_pred.numpy()
