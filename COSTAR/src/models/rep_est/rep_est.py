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

"""Representation and estimators."""

import logging
import os
from typing import List, Union
import numpy as np
import omegaconf
from src.data import RealDatasetCollection
from src.data import SyntheticDatasetCollection
from src.models.time_varying_model import TimeVaryingCausalModel
import torch
import torch.nn.functional as F
import tqdm

DictConfig = omegaconf.DictConfig
DataLoader = torch.utils.data.DataLoader
Dataset = torch.utils.data.Dataset
tqdm = tqdm.tqdm
logger = logging.getLogger(__name__)


class RepEncoder(TimeVaryingCausalModel):
  """Representation encoder."""

  model_type = 'rep_encoder'
  possible_model_types = {'rep_encoder'}

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
    self.input_size = self.dim_treatments + self.dim_static_features
    self.input_size += self.dim_vitals if self.has_vitals else 0
    self.input_size += self.dim_outcome if self.autoregressive else 0
    logger.info('%s', f'Input size of {self.model_type}: {self.input_size}')

    self._init_specific(args.model.rep_encoder)
    self.save_hyperparameters(args)

  def _init_specific(self, sub_args):
    raise NotImplementedError()

  def prepare_data(self):
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.process_data_rep_est
    ):
      self.dataset_collection.process_data_rep_est()

  def configure_optimizers(self):
    optimizer = self._get_optimizer(list(self.named_parameters()))
    if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
      return self._get_lr_schedulers(optimizer)
    return optimizer

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

  def training_step(self, batch, batch_ind):
    loss = self(batch)
    return loss

  def _eval_step(self, batch, batch_ind, subset_name):
    loss = self.training_step(batch, batch_ind)
    self.log(
        f'{self.model_type}_{subset_name}_loss',
        loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    if subset_name == self.val_dataloader().dataset.subset_name:
      self.log('{}-val_metric'.format(self.model_type), loss)

  def validation_step(self, batch, batch_ind, **kwargs):
    subset_name = self.val_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def test_step(self, batch, batch_ind, **kwargs):
    subset_name = self.test_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def predict_step(self, batch, batch_ind, dataset_idx=None):
    return self.encode(batch)

  def get_representations(self, dataset):
    logger.info('%s', f'Collecting representations for {dataset.subset_name}.')
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    self.eval()

    reps = []
    current_treatments, outcomes = [], []
    active_entries = []
    with torch.no_grad():
      for batch in tqdm(data_loader, total=len(data_loader), desc='rep'):
        batch_rep = self.encode(batch)
        batch_current_treatments = batch['current_treatments'].detach().cpu()
        batch_outcomes = batch['outputs'].detach().cpu()
        batch_active_entries = batch['active_entries'].detach().cpu()

        reps.append(batch_rep.detach().cpu())
        current_treatments.append(batch_current_treatments)
        outcomes.append(batch_outcomes)
        active_entries.append(batch_active_entries)

    reps = torch.cat(reps, dim=0).numpy()
    current_treatments = torch.cat(current_treatments, dim=0).numpy()
    outcomes = torch.cat(outcomes, dim=0).numpy()
    active_entries = torch.cat(active_entries, dim=0).numpy()
    return {
        'reps': reps,
        'current_treatments': current_treatments,
        'outcomes': outcomes,
        'active_entries': active_entries,
    }


class EstHead(TimeVaryingCausalModel):
  """Estimation head."""

  model_type = 'est_head'
  possible_model_types = {'est_head'}

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
        args, dataset_collection, autoregressive, has_vitals, bce_weights
    )

    self.rep_encoder = rep_encoder
    self.projection_horizon = args.dataset.projection_horizon
    self.output_horizon = args.dataset.projection_horizon + 1
    self.prefix = prefix

    self._init_specific(args.model.est_head)
    self.save_hyperparameters(args)

  def _init_specific(self, sub_args):
    self.step_mse_loss_weights_type = sub_args.step_mse_loss_weights_type
    if self.step_mse_loss_weights_type == 'avg':
      step_mse_loss_weights = torch.ones(self.output_horizon)
    elif self.step_mse_loss_weights_type == '1_step_only':
      step_mse_loss_weights = torch.zeros(self.output_horizon)
      step_mse_loss_weights[0] = 1.0
    elif self.step_mse_loss_weights_type == 'square_inverse':
      step_mse_loss_weights = 1.0 / torch.tensor(
          [i**2 for i in range(1, self.output_horizon + 1)]
      )
    elif self.step_mse_loss_weights_type == 'inverse':
      step_mse_loss_weights = 1.0 / torch.tensor(
          list(range(1, self.output_horizon + 1))
      )
    else:
      raise NotImplementedError(
          'step_mse_loss_weights_type: {} is not implemented'.format(
              self.step_mse_loss_weights_type
          )
      )
    step_mse_loss_weights *= self.output_horizon / step_mse_loss_weights.sum()
    self.register_buffer('step_mse_loss_weights', step_mse_loss_weights)

  def prepare_data(self):
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.process_data_rep_est
    ):
      self.dataset_collection.process_data_rep_est()

  def configure_optimizers(self):
    optimizer = self._get_optimizer(list(self.named_parameters()))
    if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
      return self._get_lr_schedulers(optimizer)
    return optimizer

  def _unroll_horizon(self, x, horizon):
    # input: [B, T, D]
    # output: [B, T, horizon, D]
    unrolled = []
    total_len = x.shape[1]
    for h in range(horizon):
      unrolled_h = x[:, h:, :]
      if unrolled_h.shape[1] < total_len:
        pad_num = total_len - unrolled_h.shape[1]
        unrolled_h = torch.cat(
            [
                unrolled_h,
                torch.zeros(unrolled_h.shape[0], pad_num, unrolled_h.shape[2])
                .to(unrolled_h.dtype)
                .to(unrolled_h.device),
            ],
            dim=1,
        )
      unrolled.append(unrolled_h)
    return torch.stack(unrolled, dim=2)

  def _calc_mse_loss(self, outcome_pred, outputs, active_entries):
    unrolled_outputs = self._unroll_horizon(outputs, self.output_horizon)
    unrolled_active_entries = self._unroll_horizon(
        active_entries, self.output_horizon
    )

    mse_loss = F.mse_loss(
        outcome_pred, unrolled_outputs, reduce=False
    )  # [B, T, output_horizon, D]
    mse_loss = torch.einsum(
        'bthd,h->bthd', mse_loss, self.step_mse_loss_weights
    )

    # Masking for shorter sequences
    # Attention!
    # Averaging across all the active entries (= sequence masks) for full batch
    mse_loss = (
        unrolled_active_entries * mse_loss
    ).sum() / unrolled_active_entries.sum()

    return mse_loss

  def training_step(self, batch, batch_ind):
    outcome_pred = self(batch)  # [B, T, output_horizon, D]
    mse_loss = self._calc_mse_loss(
        outcome_pred, batch['outputs'], batch['active_entries']
    )

    loss = mse_loss

    self.log(
        f'{self.model_type}_train_loss',
        loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    self.log(
        f'{self.model_type}_train_mse_loss',
        mse_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )

    return loss

  def _eval_step(self, batch, batch_ind, subset_name):
    outcome_pred = self(batch)  # [B, T, output_horizon, D]
    mse_loss = self._calc_mse_loss(
        outcome_pred, batch['outputs'], batch['active_entries']
    )

    loss = mse_loss

    self.log(
        f'{self.model_type}_{subset_name}_loss',
        loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    self.log(
        f'{self.model_type}_{subset_name}_mse_loss',
        mse_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )

    # validation metric
    if subset_name == self.val_dataloader().dataset.subset_name:
      self.log(
          '{}-{}-val_metric'.format(self.prefix, self.model_type),
          mse_loss,
          on_epoch=True,
          on_step=False,
          sync_dist=True,
      )

  def validation_step(self, batch, batch_ind, **kwargs):
    subset_name = self.val_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def test_step(self, batch, batch_ind, **kwargs):
    subset_name = self.test_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def predict_step(self, batch, batch_idx, dataset_idx=None):
    outcome_pred = self(batch)
    return outcome_pred.cpu()

  def get_predictions(self, dataset):
    logger.info('%s', f'Predictions for {dataset.subset_name}.')
    # Creating Dataloader
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    outcome_pred = torch.cat(self.trainer.predict(self, data_loader))
    return outcome_pred.numpy()

  def get_representations(self, dataset):
    logger.info('%s', f'Collecting representations for {dataset.subset_name}.')
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    self.eval()

    outcome_preds, reps = [], []
    current_treatments, outcomes = [], []
    active_entries = []
    with torch.no_grad():
      for batch in tqdm(data_loader, total=len(data_loader), desc='rep'):
        batch_outcome_pred, batch_rep = self(batch, return_rep=True)
        batch_current_treatments = batch['current_treatments'].detach().cpu()
        batch_outcomes = batch['outputs'].detach().cpu()
        batch_active_entries = batch['active_entries'].detach().cpu()

        outcome_preds.append(batch_outcome_pred.detach().cpu())
        reps.append(batch_rep.detach().cpu())
        current_treatments.append(batch_current_treatments)
        outcomes.append(batch_outcomes)
        active_entries.append(batch_active_entries)

    outcome_preds = torch.cat(outcome_preds, dim=0).numpy()
    reps = torch.cat(reps, dim=0).numpy()
    current_treatments = torch.cat(current_treatments, dim=0).numpy()
    outcomes = torch.cat(outcomes, dim=0).numpy()
    active_entries = torch.cat(active_entries, dim=0).numpy()
    return {
        'reps': reps,
        'current_treatments': current_treatments,
        'outcome_preds': outcome_preds,
        'outcomes': outcomes,
        'active_entries': active_entries,
    }

  def get_nonautoregressive_predictions(self, dataset):
    logger.info(
        '%s', f'Non-autoregressive Prediction for {dataset.subset_name}.'
    )
    predicted_outputs = self.get_predictions(
        dataset
    )  # [B, T, H, D], including active_entries = 0
    return predicted_outputs

  def get_normalised_1_step_rmse_syn(
      self, dataset, datasets_mc = None, prefix=None
  ):
    logger.info(
        '%s',
        f'RMSE calculation for {dataset.subset_name}, 1-step counterfactual.',
    )

    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse
    outputs_scaled = self.get_nonautoregressive_predictions(
        dataset if datasets_mc is None else datasets_mc
    )
    outputs_scaled = outputs_scaled[:, :, 0, :]  # only keep 1-step

    data_to_save = {}

    if unscale:
      output_stds, output_means = (
          dataset.scaling_params['output_stds'],
          dataset.scaling_params['output_means'],
      )
      outputs_unscaled = outputs_scaled * output_stds + output_means
      real_unscaled_outputs = (
          dataset.data['outputs'] * output_stds + output_means
      )
      outputs_calc, real_outputs_calc = outputs_unscaled, real_unscaled_outputs
    else:
      outputs_calc, real_outputs_calc = outputs_scaled, dataset.data['outputs']

    if dataset.subset_name == 'test':
      data_to_save.update({
          'means': outputs_calc,
          'output': real_outputs_calc,
          'active_entries': dataset.data['active_entries'],
      })
      pred_file = 'predictions.npz'
      if prefix:
        pred_file = f'{prefix}_predictions.npz'
      np.savez_compressed(os.path.join(os.getcwd(), pred_file), **data_to_save)

    # Only considering last active entry with actual counterfactuals
    num_samples, _, output_dim = dataset.data['active_entries'].shape
    last_entries = dataset.data['active_entries'] - np.concatenate(
        [
            dataset.data['active_entries'][:, 1:, :],
            np.zeros((num_samples, 1, output_dim)),
        ],
        axis=1,
    )
    mse_last = ((outputs_calc - real_outputs_calc) ** 2) * last_entries
    mse_last = mse_last.sum() / last_entries.sum()
    rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

    if percentage:
      rmse_normalised_last *= 100.0

    return rmse_normalised_last

  def get_normalised_n_step_rmses_syn(
      self, dataset, datasets_mc = None, prefix=None
  ):
    logger.info(
        '%s',
        f'RMSE calculation for {dataset.subset_name}, n-step counterfactual.',
    )

    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse
    outputs_scaled = self.get_nonautoregressive_predictions(
        dataset if datasets_mc is None else datasets_mc
    )

    data_to_save = {}

    if unscale:
      output_stds, output_means = (
          dataset.scaling_params['output_stds'],
          dataset.scaling_params['output_means'],
      )
      outputs_unscaled = outputs_scaled * output_stds + output_means
      real_unscaled_outputs = (
          dataset.data['outputs'] * output_stds + output_means
      )
      outputs_calc, real_outputs_calc = outputs_unscaled, real_unscaled_outputs
    else:
      outputs_calc, real_outputs_calc = outputs_scaled, dataset.data['outputs']

    if dataset.subset_name == 'test':
      data_to_save.update({
          'means': outputs_calc,
          'output': real_outputs_calc,
          'active_entries': dataset.data['active_entries'],
      })
      pred_file = 'predictions.npz'
      if prefix:
        pred_file = f'{prefix}_predictions.npz'
      np.savez_compressed(os.path.join(os.getcwd(), pred_file), **data_to_save)

    # Only considering last active entry with actual counterfactuals
    factual_seq_lengths = (
        dataset.data['sequence_lengths'].astype(int)
        - self.projection_horizon
        - 1
    )
    outputs_calc = outputs_calc[
        np.arange(len(outputs_calc)), factual_seq_lengths
    ]
    outputs_calc = outputs_calc[:, 1:]  # remove 1-step part, [B, H, D]
    real_outputs_calc_ms = []
    active_entries_ms = []
    for i in range(len(real_outputs_calc)):
      real_outputs_calc_ms.append(
          real_outputs_calc[
              i,
              factual_seq_lengths[i]
              + 1 : factual_seq_lengths[i]
              + 1
              + self.projection_horizon,
          ]
      )
      active_entries_ms.append(
          dataset.data['active_entries'][
              i,
              factual_seq_lengths[i]
              + 1 : factual_seq_lengths[i]
              + 1
              + self.projection_horizon,
          ]
      )
    real_outputs_calc_ms = np.stack(real_outputs_calc_ms, axis=0)  # [B, H, D]
    active_entries_ms = np.stack(active_entries_ms, axis=0)
    mse_last = ((outputs_calc - real_outputs_calc_ms) ** 2) * active_entries_ms
    mse_last = mse_last.sum(axis=-1).sum(axis=0) / active_entries_ms.sum(
        axis=-1
    ).sum(axis=0)
    rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

    if percentage:
      rmse_normalised_last *= 100.0

    return rmse_normalised_last

  def get_normalised_n_step_rmses_real(
      self, dataset, datasets_mc = None, prefix=None
  ):
    logger.info(
        '%s',
        f'RMSE calculation for {dataset.subset_name}, n-step counterfactual.',
    )

    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse
    outputs_scaled = self.get_nonautoregressive_predictions(
        dataset if datasets_mc is None else datasets_mc
    )

    data_to_save = {}

    if unscale:
      output_stds, output_means = (
          dataset.scaling_params['output_stds'],
          dataset.scaling_params['output_means'],
      )
      outputs_unscaled = outputs_scaled * output_stds + output_means
      real_unscaled_outputs = (
          dataset.data['outputs'] * output_stds + output_means
      )
      outputs_calc, real_outputs_calc = outputs_unscaled, real_unscaled_outputs
    else:
      outputs_calc, real_outputs_calc = outputs_scaled, dataset.data['outputs']

    if dataset.subset_name == 'test':
      data_to_save.update({
          'means': outputs_calc,
          'output': real_outputs_calc,
          'active_entries': dataset.data['active_entries'],
      })
      pred_file = 'predictions.npz'
      if prefix:
        pred_file = f'{prefix}_predictions.npz'
      np.savez_compressed(os.path.join(os.getcwd(), pred_file), **data_to_save)

    # explode real_outputs_calc
    horizon_rmses = []
    for horizon in range(self.output_horizon):
      outputs_calc_h = outputs_calc[:, :, horizon]
      real_outputs_calc_h = real_outputs_calc[:, horizon:, :]
      active_entries_h = dataset.data['active_entries'][:, horizon:, :]
      if real_outputs_calc_h.shape[1] < outputs_calc_h.shape[1]:
        pad_num = outputs_calc_h.shape[1] - real_outputs_calc_h.shape[1]
        real_outputs_calc_h = np.concatenate(
            [
                real_outputs_calc_h,
                np.zeros(
                    (
                        real_outputs_calc_h.shape[0],
                        pad_num,
                        real_outputs_calc_h.shape[2],
                    ),
                    dtype=real_outputs_calc_h.dtype,
                ),
            ],
            axis=1,
        )
        active_entries_h = np.concatenate(
            [
                active_entries_h,
                np.zeros(
                    (
                        active_entries_h.shape[0],
                        pad_num,
                        active_entries_h.shape[2],
                    ),
                    dtype=active_entries_h.dtype,
                ),
            ],
            axis=1,
        )
      mse = ((outputs_calc_h - real_outputs_calc_h) ** 2) * active_entries_h
      mse = mse.sum() / active_entries_h.sum()
      rmse_normalised = np.sqrt(mse) / dataset.norm_const
      horizon_rmses.append(rmse_normalised)

    horizon_rmses = np.array(horizon_rmses)
    if percentage:
      horizon_rmses *= 100.0

    return horizon_rmses


class EstHeadAutoreg(TimeVaryingCausalModel):
  """Estimation head autoregressive."""

  model_type = 'est_head'
  possible_model_types = {'est_head'}

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
        args, dataset_collection, autoregressive, has_vitals, bce_weights
    )

    self.rep_encoder = rep_encoder
    self.projection_horizon = args.dataset.projection_horizon
    self.output_horizon = args.dataset.projection_horizon + 1
    self.prefix = prefix

    self._init_specific(args.model.est_head)
    self.save_hyperparameters(args)

  def _init_specific(self, sub_args):
    pass

  def prepare_data(self):
    if (
        self.dataset_collection is not None
        and not self.dataset_collection.process_data_rep_est
    ):
      self.dataset_collection.process_data_rep_est()

  def configure_optimizers(self):
    optimizer = self._get_optimizer(list(self.named_parameters()))
    if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
      return self._get_lr_schedulers(optimizer)
    return optimizer

  def _unroll_horizon(self, x, horizon):
    # input: [B, T, D]
    # output: [B, T, horizon, D]
    unrolled = []
    total_len = x.shape[1]
    for h in range(horizon):
      unrolled_h = x[:, h:, :]
      if unrolled_h.shape[1] < total_len:
        pad_num = total_len - unrolled_h.shape[1]
        unrolled_h = torch.cat(
            [
                unrolled_h,
                torch.zeros(unrolled_h.shape[0], pad_num, unrolled_h.shape[2])
                .to(unrolled_h.dtype)
                .to(unrolled_h.device),
            ],
            dim=1,
        )
      unrolled.append(unrolled_h)
    return torch.stack(unrolled, dim=2)

  def _calc_mse_loss(self, outcome_pred, outputs, active_entries):
    # only use one step
    unrolled_outputs = outputs.unsqueeze(2)
    unrolled_active_entries = active_entries.unsqueeze(2)

    mse_loss = F.mse_loss(
        outcome_pred, unrolled_outputs, reduce=False
    )  # [B, T, output_horizon, D]

    # Masking for shorter sequences
    # Attention!
    # Averaging across all the active entries (= sequence masks) for full batch
    mse_loss = (
        unrolled_active_entries * mse_loss
    ).sum() / unrolled_active_entries.sum()

    return mse_loss

  def training_step(self, batch, batch_ind):
    outcome_pred = self(batch, one_step=True)  # [B, T, output_horizon, D]
    mse_loss = self._calc_mse_loss(
        outcome_pred, batch['outputs'], batch['active_entries']
    )

    loss = mse_loss

    self.log(
        f'{self.model_type}_train_loss',
        loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    self.log(
        f'{self.model_type}_train_mse_loss',
        mse_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )

    return loss

  def _eval_step(self, batch, batch_ind, subset_name):
    outcome_pred = self(batch, one_step=True)  # [B, T, output_horizon, D]
    mse_loss = self._calc_mse_loss(
        outcome_pred, batch['outputs'], batch['active_entries']
    )

    loss = mse_loss

    self.log(
        f'{self.model_type}_{subset_name}_loss',
        loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    self.log(
        f'{self.model_type}_{subset_name}_mse_loss',
        mse_loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )

    # validation metric
    if subset_name == self.val_dataloader().dataset.subset_name:
      self.log(
          '{}-{}-val_metric'.format(self.prefix, self.model_type),
          mse_loss,
          on_epoch=True,
          on_step=False,
          sync_dist=True,
      )

  def validation_step(self, batch, batch_ind, **kwargs):
    subset_name = self.val_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def test_step(self, batch, batch_ind, **kwargs):
    subset_name = self.test_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name)

  def predict_step(self, batch, batch_idx, dataset_idx=None):
    outcome_pred = self(batch)
    return outcome_pred.cpu()

  def get_predictions(self, dataset, one_step=False):
    logger.info('%s', f'Predictions for {dataset.subset_name}.')
    # Creating Dataloader
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    if one_step:
      self.to('cuda:0')
      self.eval()
      with torch.no_grad():
        outcome_pred = []
        for batch in tqdm(
            data_loader, total=len(data_loader), desc='1-step pred'
        ):
          for k in batch:
            batch[k] = batch[k].to('cuda:0')
          outcome_pred.append(self(batch, one_step=True).detach().cpu())
        outcome_pred = torch.cat(outcome_pred)
    else:
      outcome_pred = torch.cat(self.trainer.predict(self, data_loader))
    return outcome_pred.cpu().numpy()

  def get_representations(self, dataset):
    logger.info('%s', f'Collecting representations for {dataset.subset_name}.')
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    self.to('cuda:0')
    self.eval()

    outcome_preds, reps = [], []
    current_treatments, outcomes = [], []
    active_entries = []
    with torch.no_grad():
      for batch in tqdm(data_loader, total=len(data_loader), desc='rep'):
        batch_outcome_pred, batch_rep = self(batch, return_rep=True)
        batch_current_treatments = batch['current_treatments'].detach().cpu()
        batch_outcomes = batch['outputs'].detach().cpu()
        batch_active_entries = batch['active_entries'].detach().cpu()

        outcome_preds.append(batch_outcome_pred.detach().cpu())
        reps.append(batch_rep.detach().cpu())
        current_treatments.append(batch_current_treatments)
        outcomes.append(batch_outcomes)
        active_entries.append(batch_active_entries)

    outcome_preds = torch.cat(outcome_preds, dim=0).numpy()
    reps = torch.cat(reps, dim=0).numpy()
    current_treatments = torch.cat(current_treatments, dim=0).numpy()
    outcomes = torch.cat(outcomes, dim=0).numpy()
    active_entries = torch.cat(active_entries, dim=0).numpy()
    return {
        'reps': reps,
        'current_treatments': current_treatments,
        'outcome_preds': outcome_preds,
        'outcomes': outcomes,
        'active_entries': active_entries,
    }

  def get_autoregressive_predictions(
      self, dataset, one_step=False
  ):
    logger.info('%s', f'Autoregressive Prediction for {dataset.subset_name}.')
    predicted_outputs = self.get_predictions(
        dataset, one_step=one_step
    )  # [B, T, H, D], including active_entries = 0
    return predicted_outputs

  def get_normalised_1_step_rmse_syn(
      self, dataset, datasets_mc = None, prefix=None
  ):
    logger.info(
        '%s',
        f'RMSE calculation for {dataset.subset_name}, 1-step counterfactual.',
    )

    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse
    outputs_scaled = self.get_autoregressive_predictions(
        dataset if datasets_mc is None else datasets_mc, one_step=True
    )
    outputs_scaled = outputs_scaled[:, :, 0, :]  # only keep 1-step

    data_to_save = {}

    if unscale:
      output_stds, output_means = (
          dataset.scaling_params['output_stds'],
          dataset.scaling_params['output_means'],
      )
      outputs_unscaled = outputs_scaled * output_stds + output_means
      real_unscaled_outputs = (
          dataset.data['outputs'] * output_stds + output_means
      )
      outputs_calc, real_outputs_calc = outputs_unscaled, real_unscaled_outputs
    else:
      outputs_calc, real_outputs_calc = outputs_scaled, dataset.data['outputs']

    if dataset.subset_name == 'test':
      data_to_save.update({
          'means': outputs_calc,
          'output': real_outputs_calc,
          'active_entries': dataset.data['active_entries'],
      })
      pred_file = 'predictions.npz'
      if prefix:
        pred_file = f'{prefix}_predictions.npz'
      np.savez_compressed(os.path.join(os.getcwd(), pred_file), **data_to_save)

    # Only considering last active entry with actual counterfactuals
    num_samples, _, output_dim = dataset.data['active_entries'].shape
    last_entries = dataset.data['active_entries'] - np.concatenate(
        [
            dataset.data['active_entries'][:, 1:, :],
            np.zeros((num_samples, 1, output_dim)),
        ],
        axis=1,
    )
    mse_last = ((outputs_calc - real_outputs_calc) ** 2) * last_entries
    mse_last = mse_last.sum() / last_entries.sum()
    rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

    if percentage:
      rmse_normalised_last *= 100.0

    return rmse_normalised_last

  def get_normalised_n_step_rmses_syn(
      self, dataset, datasets_mc = None, prefix=None
  ):
    logger.info(
        '%s',
        f'RMSE calculation for {dataset.subset_name}, n-step counterfactual.',
    )

    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse
    outputs_scaled = self.get_autoregressive_predictions(
        dataset if datasets_mc is None else datasets_mc
    )

    data_to_save = {}

    if unscale:
      output_stds, output_means = (
          dataset.scaling_params['output_stds'],
          dataset.scaling_params['output_means'],
      )
      outputs_unscaled = outputs_scaled * output_stds + output_means
      real_unscaled_outputs = (
          dataset.data['outputs'] * output_stds + output_means
      )
      outputs_calc, real_outputs_calc = outputs_unscaled, real_unscaled_outputs
    else:
      outputs_calc, real_outputs_calc = outputs_scaled, dataset.data['outputs']

    if dataset.subset_name == 'test':
      data_to_save.update({
          'means': outputs_calc,
          'output': real_outputs_calc,
          'active_entries': dataset.data['active_entries'],
      })
      pred_file = 'predictions.npz'
      if prefix:
        pred_file = f'{prefix}_predictions.npz'
      np.savez_compressed(os.path.join(os.getcwd(), pred_file), **data_to_save)

    # Only considering last active entry with actual counterfactuals
    factual_seq_lengths = (
        dataset.data['sequence_lengths'].astype(int)
        - self.projection_horizon
        - 1
    )
    outputs_calc = outputs_calc[
        np.arange(len(outputs_calc)), factual_seq_lengths
    ]
    outputs_calc = outputs_calc[:, 1:]  # remove 1-step part, [B, H, D]
    real_outputs_calc_ms = []
    active_entries_ms = []
    for i in range(len(real_outputs_calc)):
      real_outputs_calc_ms.append(
          real_outputs_calc[
              i,
              factual_seq_lengths[i]
              + 1 : factual_seq_lengths[i]
              + 1
              + self.projection_horizon,
          ]
      )
      active_entries_ms.append(
          dataset.data['active_entries'][
              i,
              factual_seq_lengths[i]
              + 1 : factual_seq_lengths[i]
              + 1
              + self.projection_horizon,
          ]
      )
    real_outputs_calc_ms = np.stack(real_outputs_calc_ms, axis=0)  # [B, H, D]
    active_entries_ms = np.stack(active_entries_ms, axis=0)
    mse_last = ((outputs_calc - real_outputs_calc_ms) ** 2) * active_entries_ms
    mse_last = mse_last.sum(axis=-1).sum(axis=0) / active_entries_ms.sum(
        axis=-1
    ).sum(axis=0)
    rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

    if percentage:
      rmse_normalised_last *= 100.0

    return rmse_normalised_last

  def get_normalised_n_step_rmses_real(
      self, dataset, datasets_mc = None, prefix=None
  ):
    logger.info(
        '%s',
        f'RMSE calculation for {dataset.subset_name}, n-step counterfactual.',
    )

    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse
    outputs_scaled = self.get_autoregressive_predictions(
        dataset if datasets_mc is None else datasets_mc
    )

    data_to_save = {}

    if unscale:
      output_stds, output_means = (
          dataset.scaling_params['output_stds'],
          dataset.scaling_params['output_means'],
      )
      outputs_unscaled = outputs_scaled * output_stds + output_means
      real_unscaled_outputs = (
          dataset.data['outputs'] * output_stds + output_means
      )
      outputs_calc, real_outputs_calc = outputs_unscaled, real_unscaled_outputs
    else:
      outputs_calc, real_outputs_calc = outputs_scaled, dataset.data['outputs']

    if dataset.subset_name == 'test':
      data_to_save.update({
          'means': outputs_calc,
          'output': real_outputs_calc,
          'active_entries': dataset.data['active_entries'],
      })
      pred_file = 'predictions.npz'
      if prefix:
        pred_file = f'{prefix}_predictions.npz'
      np.savez_compressed(os.path.join(os.getcwd(), pred_file), **data_to_save)

    # explode real_outputs_calc
    horizon_rmses = []
    for horizon in range(self.output_horizon):
      outputs_calc_h = outputs_calc[:, :, horizon]
      real_outputs_calc_h = real_outputs_calc[:, horizon:, :]
      active_entries_h = dataset.data['active_entries'][:, horizon:, :]
      if real_outputs_calc_h.shape[1] < outputs_calc_h.shape[1]:
        pad_num = outputs_calc_h.shape[1] - real_outputs_calc_h.shape[1]
        real_outputs_calc_h = np.concatenate(
            [
                real_outputs_calc_h,
                np.zeros(
                    (
                        real_outputs_calc_h.shape[0],
                        pad_num,
                        real_outputs_calc_h.shape[2],
                    ),
                    dtype=real_outputs_calc_h.dtype,
                ),
            ],
            axis=1,
        )
        active_entries_h = np.concatenate(
            [
                active_entries_h,
                np.zeros(
                    (
                        active_entries_h.shape[0],
                        pad_num,
                        active_entries_h.shape[2],
                    ),
                    dtype=active_entries_h.dtype,
                ),
            ],
            axis=1,
        )
      mse = ((outputs_calc_h - real_outputs_calc_h) ** 2) * active_entries_h
      mse = mse.sum() / active_entries_h.sum()
      rmse_normalised = np.sqrt(mse) / dataset.norm_const
      horizon_rmses.append(rmse_normalised)

    horizon_rmses = np.array(horizon_rmses)
    if percentage:
      horizon_rmses *= 100.0

    return horizon_rmses
