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

"""Base class of all time varying models."""

import ast
import copy
import functools
import json
import logging
import multiprocessing as mp
import os
import pathlib
from typing import Dict
from typing import List
from typing import Union
from captum.attr import IntegratedGradients
import numpy as np
import omegaconf
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import ray
from ray import tune
from src.data import RealDatasetCollection
from src.data import SyntheticDatasetCollection
from src.models.utils import AlphaRise
from src.models.utils import bce
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
import tqdm

deepcopy = copy.deepcopy
partial = functools.partial
Path = pathlib.Path
DictConfig = omegaconf.DictConfig
OmegaConf = omegaconf.OmegaConf
ray_constants = ray.ray_constants
DataLoader = torch.utils.data.DataLoader
Dataset = torch.utils.data.Dataset
tqdm = tqdm.tqdm

LIBROOT = str(Path(__file__).parent.parent.parent.absolute())

logger = logging.getLogger(__name__)
ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD = 10**8  # ~ 100Mb

NUM_PHY_CPU_CORES = mp.cpu_count() // 2


def train_eval_factual(
    args,
    train_f,
    val_f,
    orig_hparams,
    input_size,
    model_cls,
    tuning_criterion='rmse',
    **kwargs,
):
  """Globally defined method, used for ray tuning."""
  OmegaConf.register_new_resolver('sum', lambda x, y: x + y, replace=True)
  new_params = deepcopy(orig_hparams)
  model_cls.set_hparams(
      new_params.model, args, input_size, model_cls.model_type
  )
  if model_cls.model_type == 'decoder':
    # Passing encoder takes too much memory
    encoder_r_size = (
        new_params.model.encoder.br_size
        if 'br_size' in new_params.model.encoder
        else new_params.model.encoder.seq_hidden_units
    )  # Using either br_size or Memory adapter
    model = model_cls(new_params, encoder_r_size=encoder_r_size, **kwargs).to(
        torch.get_default_dtype()
    )
  else:
    model = model_cls(new_params, **kwargs).to(torch.get_default_dtype())

  if hasattr(model_cls, 'collate_fn_augment_with_masked_vitals'):
    train_loader = DataLoader(
        train_f,
        shuffle=True,
        batch_size=new_params.model[model_cls.model_type]['batch_size'],
        drop_last=True,
        collate_fn=partial(
            model_cls.collate_fn_augment_with_masked_vitals,
            has_vitals=model.has_vitals,
            autoregressive=model.autoregressive,
        ),
    )
  else:
    train_loader = DataLoader(
        train_f,
        shuffle=True,
        batch_size=new_params.model[model_cls.model_type]['batch_size'],
        drop_last=True,
    )
  val_loader = DataLoader(
      val_f,
      shuffle=False,
      batch_size=new_params.model[model_cls.model_type]['batch_size'],
      drop_last=True,
  )

  if (
      'max_grad_norm' in new_params.model[model_cls.model_type]
      and new_params.model[model_cls.model_type]['max_grad_norm'] is not None
  ):
    trainer = Trainer(
        gpus=ast.literal_eval(str(new_params.exp.gpus))[:1],
        logger=None,
        max_epochs=new_params.exp.max_epochs,
        progress_bar_refresh_rate=0,
        gradient_clip_val=new_params.model[model_cls.model_type][
            'max_grad_norm'
        ],
        callbacks=[AlphaRise(rate=new_params.exp.alpha_rate)],
    )
  else:
    trainer = Trainer(
        gpus=ast.literal_eval(str(new_params.exp.gpus))[:1],
        logger=None,
        max_epochs=new_params.exp.max_epochs,
        progress_bar_refresh_rate=0,
        callbacks=[AlphaRise(rate=new_params.exp.alpha_rate)],
    )
  trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

  if tuning_criterion == 'rmse':
    val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(val_f)
    tune.report(val_rmse_orig=val_rmse_orig, val_rmse_all=val_rmse_all)
  elif tuning_criterion == 'bce':
    val_bce_orig, val_bce_all = model.get_masked_bce(val_f)
    tune.report(val_bce_orig=val_bce_orig, val_bce_all=val_bce_all)
  else:
    raise NotImplementedError()


class TimeVaryingCausalModel(LightningModule):
  """Abstract class for models, estimating counterfactual outcomes over time."""

  model_type = None  # Will be defined in subclasses
  possible_model_types = None  # Will be defined in subclasses
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
    super().__init__()
    self.dataset_collection = dataset_collection
    if dataset_collection is not None:
      self.autoregressive = self.dataset_collection.autoregressive
      self.has_vitals = self.dataset_collection.has_vitals
      self.bce_weights = None  # Will be calculated, when calling preparing data
      if hasattr(dataset_collection.train_f, 'feature_names'):
        self.feature_names = dataset_collection.train_f.feature_names
      else:
        self.feature_names = {}
    else:
      self.autoregressive = autoregressive
      self.has_vitals = has_vitals
      self.bce_weights = bce_weights
      print(self.bce_weights)

    # General datasets parameters
    self.dim_treatments = args.model.dim_treatments
    self.dim_vitals = args.model.dim_vitals
    self.dim_static_features = args.model.dim_static_features
    self.dim_outcome = args.model.dim_outcomes

    self.input_size = None  # Will be defined in subclasses

    self.save_hyperparameters(args)  # Will be logged to mlflow

  def _get_optimizer(self, param_optimizer):
    no_decay = ['bias', 'layer_norm']
    sub_args = self.hparams.model[self.model_type]
    optimizer_grouped_parameters = [
        {
            'params': [
                p
                for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': sub_args['optimizer']['weight_decay'],
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    lr = sub_args['optimizer']['learning_rate']
    optimizer_cls = sub_args['optimizer']['optimizer_cls']
    if optimizer_cls.lower() == 'adamw':
      optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    elif optimizer_cls.lower() == 'adam':
      optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optimizer_cls.lower() == 'sgd':
      optimizer = optim.SGD(
          optimizer_grouped_parameters,
          lr=lr,
          momentum=sub_args['optimizer']['momentum'],
      )
    else:
      raise NotImplementedError()

    return optimizer

  def _construct_lr_scheduler(self, optimizer):
    if type(self.hparams.model[self.model_type]['optimizer']['lr_scheduler']):
      return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif (
        'lr_scheduler_params'
        in self.hparams.model[self.model_type]['optimizer']
    ):
      sub_args = self.hparams.model[self.model_type]['optimizer'][
          'lr_scheduler_params'
      ]
      scheduler_name = sub_args['name']
      if scheduler_name == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sub_args['milestones'],
            gamma=sub_args['gamma'],
        )
      else:
        raise NotImplementedError(f'Scheduler {scheduler_name} not implemented')
      return scheduler
    else:
      return None

  def _get_lr_schedulers(self, optimizer):
    if not isinstance(optimizer, list):
      lr_scheduler = self._construct_lr_scheduler(optimizer)
      return [optimizer], [lr_scheduler]
    else:
      lr_schedulers = []
      for opt in optimizer:
        lr_schedulers.append(self._construct_lr_scheduler(opt))
      return optimizer, lr_schedulers

  def configure_optimizers(self):
    optimizer = self._get_optimizer(list(self.named_parameters()))
    if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
      return self._get_lr_schedulers(optimizer)
    elif (
        'lr_scheduler_params'
        in self.hparams.model[self.model_type]['optimizer']
    ):
      return self._get_lr_schedulers(optimizer)
    return optimizer

  def train_dataloader(self):
    sub_args = self.hparams.model[self.model_type]
    drop_last = True
    # drop_last = False
    if self.hparams.dataset['few_shot_sample_num'] > 0:
      drop_last = False
    return DataLoader(
        self.dataset_collection.train_f,
        shuffle=True,
        batch_size=sub_args['batch_size'],
        drop_last=drop_last,
    )

  def val_dataloader(self):
    return DataLoader(
        self.dataset_collection.val_f,
        batch_size=self.hparams.dataset.val_batch_size,
        shuffle=False,
        generator=torch.Generator(),
    )

  def get_predictions(self, dataset):
    raise NotImplementedError()

  def get_propensity_scores(self, dataset):
    raise NotImplementedError()

  def get_representations(self, dataset):
    raise NotImplementedError()

  def get_causal_representations(self, dataset):
    raise NotImplementedError()

  def get_autoregressive_predictions(self, dataset):
    logger.info('%s', f'Autoregressive Prediction for {dataset.subset_name}.')
    if self.model_type == 'decoder':  # CRNDecoder / EDCTDecoder / RMSN Decoder
      predicted_outputs = np.zeros((
          len(dataset),
          self.hparams.dataset.projection_horizon,
          self.dim_outcome,
      ))
      for t in range(self.hparams.dataset.projection_horizon):
        logger.info('%s', f't = {t + 2}')

        outputs_scaled = self.get_predictions(dataset)
        predicted_outputs[:, t] = outputs_scaled[:, t]

        if t < (self.hparams.dataset.projection_horizon - 1):
          dataset.data['prev_outputs'][:, t + 1, :] = outputs_scaled[:, t, :]
    else:
      raise NotImplementedError()

    return predicted_outputs

  def get_autoregressive_causal_representations(
      self, dataset
  ):
    logger.info(
        '%s',
        f'Autoregressive Causal Representations for {dataset.subset_name}.',
    )
    if self.model_type == 'decoder':  # CRNDecoder / EDCTDecoder / RMSN Decoder
      predicted_causal_reps = []
      for t in range(self.hparams.dataset.projection_horizon):
        logger.info('%s', f't = {t + 2}')

        outputs_scaled = self.get_predictions(dataset)
        causal_reps = self.get_causal_representations(dataset)
        for k in causal_reps:
          causal_reps[k] = causal_reps[k][:, t]
        predicted_causal_reps.append(causal_reps)

        if t < (self.hparams.dataset.projection_horizon - 1):
          dataset.data['prev_outputs'][:, t + 1, :] = outputs_scaled[:, t, :]
    else:
      raise NotImplementedError()

    predicted_causal_reps_lon = {}
    for k in predicted_causal_reps[0]:
      predicted_causal_reps_lon[k] = np.stack(
          [pcr[k] for pcr in predicted_causal_reps], axis=1
      )
    return predicted_causal_reps_lon

  def get_masked_bce(self, dataset):
    logger.info('%s', f'BCE calculation for {dataset.subset_name}.')
    treatment_pred = torch.tensor(self.get_propensity_scores(dataset))
    current_treatments = torch.tensor(dataset.data['current_treatments'])

    bce_arr = (
        (self.bce_loss(treatment_pred, current_treatments, kind='predict'))
        .unsqueeze(-1)
        .numpy()
    )
    bce_arr = bce_arr * dataset.data['active_entries']

    # Calculation like in original paper
    # (Masked-Averaging over datapoints (& outputs)
    # and then non-masked time axis)
    bce_orig = bce_arr.sum(0).sum(-1) / dataset.data['active_entries'].sum(
        0
    ).sum(-1)
    bce_orig = bce_orig.mean()

    # Masked averaging over all dimensions at once
    bce_all = bce_arr.sum() / dataset.data['active_entries'].sum()

    return bce_orig, bce_all

  def get_normalised_masked_rmse(
      self, dataset, one_step_counterfactual=False
  ):
    logger.info('%s', f'RMSE calculation for {dataset.subset_name}.')
    outputs_scaled = self.get_predictions(dataset)
    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse

    data_to_save = {}
    if self.hparams.exp.save_representations and dataset.subset_name == 'test':
      data_to_save.update(self.get_causal_representations(dataset))

    outputs_unscaled = None
    if unscale:
      output_stds, output_means = (
          dataset.scaling_params['output_stds'],
          dataset.scaling_params['output_means'],
      )
      outputs_unscaled = outputs_scaled * output_stds + output_means

      # Batch-wise masked-MSE calculation is tricky,
      # thus calculating for full dataset at once
      mse = (
          (outputs_unscaled - dataset.data['unscaled_outputs']) ** 2
      ) * dataset.data['active_entries']

      if dataset.subset_name == 'test':
        data_to_save.update({
            'means': outputs_unscaled,
            'output': dataset.data['unscaled_outputs'],
            'active_entries': dataset.data['active_entries'],
        })
        np.savez_compressed(
            os.path.join(os.getcwd(), 'predictions_encoder.npz'), **data_to_save
        )

    else:
      # Batch-wise masked-MSE calculation is tricky,
      # thus calculating for full dataset at once
      mse = ((outputs_scaled - dataset.data['outputs']) ** 2) * dataset.data[
          'active_entries'
      ]
      if dataset.subset_name == 'test':
        data_to_save.update({
            'means': outputs_scaled,
            'output': dataset.data['outputs'],
            'active_entries': dataset.data['active_entries'],
        })
        np.savez_compressed(
            os.path.join(os.getcwd(), 'predictions_encoder.npz'), **data_to_save
        )

    # Calculation like in original paper
    # (Masked-Averaging over datapoints (& outputs)
    # and then non-masked time axis)
    mse_orig = mse.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(
        -1
    )
    mse_orig = mse_orig.mean()
    rmse_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

    # Masked averaging over all dimensions at once
    mse_all = mse.sum() / dataset.data['active_entries'].sum()
    rmse_normalised_all = np.sqrt(mse_all) / dataset.norm_const

    if percentage:
      rmse_normalised_orig *= 100.0
      rmse_normalised_all *= 100.0

    if one_step_counterfactual:
      # Only considering last active entry with actual counterfactuals
      num_samples, _, output_dim = dataset.data['active_entries'].shape
      last_entries = dataset.data['active_entries'] - np.concatenate(
          [
              dataset.data['active_entries'][:, 1:, :],
              np.zeros((num_samples, 1, output_dim)),
          ],
          axis=1,
      )
      if unscale:
        mse_last = (
            (outputs_unscaled - dataset.data['unscaled_outputs']) ** 2
        ) * last_entries
      else:
        mse_last = (
            (outputs_scaled - dataset.data['outputs']) ** 2
        ) * last_entries

      mse_last = mse_last.sum() / last_entries.sum()
      rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

      if percentage:
        rmse_normalised_last *= 100.0

      return rmse_normalised_orig, rmse_normalised_all, rmse_normalised_last

    return rmse_normalised_orig, rmse_normalised_all

  def get_normalised_n_step_rmses(
      self, dataset, datasets_mc = None
  ):
    logger.info('%s', f'RMSE calculation for {dataset.subset_name}.')
    assert (
        self.model_type == 'decoder'
        or self.model_type == 'multi'
        or self.model_type == 'g_net'
        or self.model_type == 'msm_regressor'
    )
    assert hasattr(dataset, 'data_processed_seq')

    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse
    outputs_scaled = self.get_autoregressive_predictions(
        dataset if datasets_mc is None else datasets_mc
    )

    data_to_save = {}
    if self.hparams.exp.save_representations and dataset.subset_name == 'test':
      data_to_save.update(
          self.get_autoregressive_causal_representations(dataset)
      )

    if unscale:
      output_stds, output_means = (
          dataset.scaling_params['output_stds'],
          dataset.scaling_params['output_means'],
      )
      outputs_unscaled = outputs_scaled * output_stds + output_means

      mse = (
          (outputs_unscaled - dataset.data_processed_seq['unscaled_outputs'])
          ** 2
      ) * dataset.data_processed_seq['active_entries']

      if dataset.subset_name == 'test':
        data_to_save.update({
            'means': outputs_unscaled,
            'output': dataset.data_processed_seq['unscaled_outputs'],
            'active_entries': dataset.data_processed_seq['active_entries'],
        })
        np.savez_compressed(
            os.path.join(os.getcwd(), 'predictions.npz'), **data_to_save
        )

    else:
      mse = (
          (outputs_scaled - dataset.data_processed_seq['outputs']) ** 2
      ) * dataset.data_processed_seq['active_entries']
      if dataset.subset_name == 'test':
        data_to_save.update({
            'means': outputs_scaled,
            'output': dataset.data_processed_seq['outputs'],
            'active_entries': dataset.data_processed_seq['active_entries'],
        })
        np.savez_compressed(
            os.path.join(os.getcwd(), 'predictions.npz'), **data_to_save
        )

    nan_idx = np.unique(
        np.where(np.isnan(dataset.data_processed_seq['outputs']))[0]
    )
    not_nan = np.array(
        [i for i in range(outputs_scaled.shape[0]) if i not in nan_idx]
    )

    # Calculation like in original paper
    # (Masked-Averaging over datapoints (& outputs)
    # and then non-masked time axis)
    mse_orig = mse[not_nan].sum(0).sum(-1) / dataset.data_processed_seq[
        'active_entries'
    ][not_nan].sum(0).sum(-1)
    rmses_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

    if percentage:
      rmses_normalised_orig *= 100.0

    return rmses_normalised_orig

  def get_feature_importance(self, dataset):
    pass

  @staticmethod
  def set_hparams(
      model_args, new_args, input_size, model_type
  ):
    raise NotImplementedError()

  def finetune(self, resources_per_trial):
    """Hyperparameter tuning with ray[tune]."""
    self.prepare_data()
    sub_args = self.hparams.model[self.model_type]
    logger.info(
        '%s',
        'Running hyperparameters selection with'
        f' {sub_args["tune_range"]} trials',
    )
    ray.init(
        num_gpus=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')),
        num_cpus=NUM_PHY_CPU_CORES - 1,
        _redis_max_memory=ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD,
        runtime_env={
            'env_vars': {
                'PYTHONPATH': LIBROOT,
                'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
            }
        },
    )

    hparams_grid = {
        k: tune.choice(v) for k, v in sub_args['hparams_grid'].items()
    }
    analysis = tune.run(
        tune.with_parameters(
            train_eval_factual,
            input_size=self.input_size,
            model_cls=self.__class__,
            tuning_criterion=self.tuning_criterion,
            train_f=deepcopy(self.dataset_collection.train_f),
            val_f=deepcopy(self.dataset_collection.val_f),
            orig_hparams=self.hparams,
            autoregressive=self.autoregressive,
            has_vitals=self.has_vitals,
            bce_weights=self.bce_weights,
            projection_horizon=self.projection_horizon
            if hasattr(self, 'projection_horizon')
            else None,
        ),
        resources_per_trial=resources_per_trial,
        metric=f'val_{self.tuning_criterion}_all',
        mode='min',
        config=hparams_grid,
        num_samples=sub_args['tune_range'],
        name=f'{self.__class__.__name__}{self.model_type}',
        max_failures=3,
        local_dir=os.path.join(os.getcwd(), 'results'),
    )
    ray.shutdown()

    logger.info('%s', f'Best hyperparameters found: {analysis.best_config}.')
    logger.info('Resetting current hyperparameters to best values.')

    with open(
        os.path.join(
            os.getcwd(), '{}_best_hparams.json'.format(self.model_type)
        ),
        'w',
    ) as f:
      json.dump(analysis.best_config, f)

    self.set_hparams(
        self.hparams.model,
        analysis.best_config,
        self.input_size,
        self.model_type,
    )

    self.__init__(
        self.hparams,
        dataset_collection=self.dataset_collection,
        encoder=self.encoder if hasattr(self, 'encoder') else None,
        propensity_treatment=self.propensity_treatment
        if hasattr(self, 'propensity_treatment')
        else None,
        propensity_history=self.propensity_history
        if hasattr(self, 'propensity_history')
        else None,
    )
    return self

  def visualize(self, dataset, index=0, artifacts_path=None):
    pass

  def bce_loss(self, treatment_pred, current_treatments, kind='predict'):
    mode = self.hparams.dataset.treatment_mode
    bce_weights = (
        torch.tensor(self.bce_weights).type_as(current_treatments)
        if self.hparams.exp.bce_weight
        else None
    )

    if kind == 'predict':
      bce_loss = bce(treatment_pred, current_treatments, mode, bce_weights)
    elif kind == 'confuse':
      uniform_treatments = torch.ones_like(current_treatments)
      if mode == 'multiclass':
        uniform_treatments *= 1 / current_treatments.shape[-1]
      elif mode == 'multilabel':
        uniform_treatments *= 0.5
      bce_loss = bce(treatment_pred, uniform_treatments, mode)
    else:
      raise NotImplementedError()
    return bce_loss

  def mse_loss(self, domain_label_pred, domain_label, kind='predict'):
    if kind == 'predict':
      mse_loss = nn.MSELoss(reduce=False)(domain_label_pred, domain_label)
    elif kind == 'confuse':
      mse_loss = -nn.MSELoss(reduce=False)(domain_label_pred, domain_label)
    else:
      raise NotImplementedError()
    return mse_loss.mean(dim=-1)

  def on_fit_start(
      self,
  ):  # Issue with logging not yet existing parameters in MlFlow
    if self.trainer.logger is not None:
      self.trainer.logger.filter_submodels = list(
          self.possible_model_types - {self.model_type}
      )

  def on_fit_end(
      self,
  ):  # Issue with logging not yet existing parameters in MlFlow
    if self.trainer.logger is not None:
      self.trainer.logger.filter_submodels = list(self.possible_model_types)

  def on_save_checkpoint(self, checkpoint):
    # add ema to checkpoint if applicable
    if self.hparams.exp.weights_ema:
      checkpoint['ema_state_dict'] = {
          'ema_treatment': self.ema_treatment.state_dict(),
          'ema_non_treatment': self.ema_non_treatment.state_dict(),
      }


class BRCausalModel(TimeVaryingCausalModel):
  """Abstract class for models.

  Estimating counterfactual outcomes over time with balanced representations
  """

  model_type = None  # Will be defined in subclasses
  possible_model_types = None  # Will be defined in subclasses
  tuning_criterion = 'rmse'

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

    # Balancing representation training parameters
    self.balancing = args.exp.balancing
    self.alpha = args.exp.alpha  # Used for gradient-reversal
    self.update_alpha = args.exp.update_alpha

    self.alpha_prev_treat = args.exp.alpha_prev_treat
    self.update_alpha_prev_treat = args.exp.update_alpha_prev_treat
    self.alpha_age = args.exp.alpha_age
    self.update_alpha_age = args.exp.update_alpha_age

    self.train_domain_label_adv = args.exp.train_domain_label_adv

  def configure_optimizers(self):
    if (
        self.balancing == 'grad_reverse' and not self.hparams.exp.weights_ema
    ):  # one optimizer
      optimizer = self._get_optimizer(list(self.named_parameters()))

      if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
        return self._get_lr_schedulers(optimizer)

      return optimizer

    else:  # two optimizers - simultaneous gradient descent update
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
          if k not in treatment_head_params
      ]

      assert len(treatment_head_params + non_treatment_head_params) == len(
          list(self.named_parameters())
      )

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

      treatment_head_optimizer = self._get_optimizer(treatment_head_params)
      non_treatment_head_optimizer = self._get_optimizer(
          non_treatment_head_params
      )

      if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
        return self._get_lr_schedulers(
            [non_treatment_head_optimizer, treatment_head_optimizer]
        )

      return [non_treatment_head_optimizer, treatment_head_optimizer]

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
    if hasattr(self, 'prefix_tuning') and self.prefix_tuning:
      return
    if self.hparams.exp.weights_ema and optimizer_idx == 0:
      self.ema_non_treatment.update()
    elif self.hparams.exp.weights_ema and optimizer_idx == 1:
      self.ema_treatment.update()

  def _calculate_bce_weights(self):
    if self.hparams.dataset.treatment_mode == 'multiclass':
      current_treatments = self.dataset_collection.train_f.data[
          'current_treatments'
      ]
      current_treatments = current_treatments.reshape(
          -1, current_treatments.shape[-1]
      )
      current_treatments = current_treatments[
          self.dataset_collection.train_f.data['active_entries']
          .flatten()
          .astype(bool)
      ]
      current_treatments = np.argmax(current_treatments, axis=1)

      self.bce_weights = (
          len(current_treatments)
          / np.bincount(current_treatments)
          / len(np.bincount(current_treatments))
      )
    else:
      raise NotImplementedError()

  def on_fit_start(
      self,
  ):  # Issue with logging not yet existing parameters in MlFlow
    if self.trainer.logger is not None:
      self.trainer.logger.filter_submodels = (
          ['decoder'] if self.model_type == 'encoder' else ['encoder']
      )

  def on_fit_end(
      self,
  ):  # Issue with logging not yet existing parameters in MlFlow
    if self.trainer.logger is not None:
      self.trainer.logger.filter_submodels = ['encoder', 'decoder']

  def training_step(self, batch, batch_ind, optimizer_idx=0):
    _ = batch_ind
    for par in self.parameters():
      par.requires_grad = True

    if (
        optimizer_idx == 0
    ):  # grad reversal or domain confusion representation update
      if self.hparams.exp.weights_ema:
        with self.ema_treatment.average_parameters():
          treatment_pred, outcome_pred, _ = self(batch)
      else:
        treatment_pred, outcome_pred, _ = self(batch)

      if self.alpha_prev_treat > 0 or self.alpha_age > 0:
        treatment_pred, domain_label_pred = treatment_pred
      else:
        domain_label_pred = None

      mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)

      if (
          'treatment_loss_off' in self.hparams.model[self.model_type]
          and self.hparams.model[self.model_type]['treatment_loss_off']
      ):
        bce_loss = 0.0
      elif self.balancing == 'grad_reverse':
        bce_loss = self.bce_loss(
            treatment_pred,
            batch['current_treatments'].to(torch.get_default_dtype()),
            kind='predict',
        )
        if self.train_domain_label_adv:
          if self.alpha_prev_treat > 0:
            domain_label_loss = self.bce_loss(
                domain_label_pred,
                batch['prev_treatments'].to(torch.get_default_dtype()),
                kind='predict',
            )
            bce_loss = bce_loss + domain_label_loss
          if self.alpha_age > 0:
            age = (
                batch['static_features'][Ellipsis, -1:]
                .unsqueeze(1)
                .repeat(1, domain_label_pred.shape[1], 1)
            )
            domain_label_loss = self.mse_loss(
                domain_label_pred,
                age.to(torch.get_default_dtype()),
                kind='predict',
            )
            bce_loss = bce_loss + domain_label_loss
      elif self.balancing == 'domain_confusion':
        bce_loss = self.bce_loss(
            treatment_pred,
            batch['current_treatments'].to(torch.get_default_dtype()),
            kind='confuse',
        )
        bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
        if self.train_domain_label_adv:
          if self.alpha_prev_treat > 0:
            domain_label_loss = self.bce_loss(
                domain_label_pred,
                batch['prev_treatments'].to(torch.get_default_dtype()),
                kind='confuse',
            )
            bce_loss = (
                bce_loss
                + self.br_treatment_outcome_head.alpha_prev_treat
                * domain_label_loss
            )
          if self.alpha_age > 0:
            age = (
                batch['static_features'][Ellipsis, -1:]
                .unsqueeze(1)
                .repeat(1, domain_label_pred.shape[1], 1)
            )
            domain_label_loss = self.mse_loss(
                domain_label_pred,
                age.to(torch.get_default_dtype()),
                kind='confuse',
            )
            bce_loss = (
                bce_loss
                + self.br_treatment_outcome_head.alpha_age * domain_label_loss
            )
      else:
        raise NotImplementedError()

      # Masking for shorter sequences
      # Attention! Averaging across all
      # the active entries (= sequence masks) for full batch
      bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch[
          'active_entries'
      ].sum()
      mse_loss = (batch['active_entries'] * mse_loss).sum() / batch[
          'active_entries'
      ].sum()

      loss = bce_loss + mse_loss

      self.log(
          f'{self.model_type}_train_loss',
          loss,
          on_epoch=True,
          on_step=False,
          sync_dist=True,
      )
      self.log(
          f'{self.model_type}_train_bce_loss',
          bce_loss,
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
      self.log(
          f'{self.model_type}_alpha',
          self.br_treatment_outcome_head.alpha,
          on_epoch=True,
          on_step=False,
          sync_dist=True,
      )

      return loss

    elif optimizer_idx == 1:  # domain classifier update
      if self.hparams.exp.weights_ema:
        with self.ema_non_treatment.average_parameters():
          treatment_pred, _, _ = self(batch, detach_treatment=True)
      else:
        treatment_pred, _, _ = self(batch, detach_treatment=True)

      if self.alpha_prev_treat > 0 or self.alpha_age > 0:
        treatment_pred, domain_label_pred = treatment_pred
      else:
        domain_label_pred = None

      bce_loss = self.bce_loss(
          treatment_pred,
          batch['current_treatments'].to(torch.get_default_dtype()),
          kind='predict',
      )
      if self.balancing == 'domain_confusion':
        bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
      if self.train_domain_label_adv:
        if self.alpha_prev_treat > 0:
          domain_label_loss = self.bce_loss(
              domain_label_pred,
              batch['prev_treatments'].to(torch.get_default_dtype()),
              kind='predict',
          )
          if self.balancing == 'domain_confusion':
            domain_label_loss = (
                self.br_treatment_outcome_head.alpha_prev_treat
                * domain_label_loss
            )
          bce_loss = bce_loss + domain_label_loss
        if self.alpha_age > 0:
          age = (
              batch['static_features'][Ellipsis, -1:]
              .unsqueeze(1)
              .repeat(1, domain_label_pred.shape[1], 1)
          )
          domain_label_loss = self.mse_loss(
              domain_label_pred,
              age.to(torch.get_default_dtype()),
              kind='predict',
          )
          if self.balancing == 'domain_confusion':
            domain_label_loss = (
                self.br_treatment_outcome_head.alpha_prev_treat
                * domain_label_loss
            )
          bce_loss = bce_loss + domain_label_loss

      # Masking for shorter sequences
      # Attention! Averaging across all the
      # active entries (= sequence masks) for full batch
      bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch[
          'active_entries'
      ].sum()
      self.log(
          f'{self.model_type}_train_bce_loss_cl',
          bce_loss,
          on_epoch=True,
          on_step=False,
          sync_dist=True,
      )

      return bce_loss

  def _eval_step(self, batch, batch_ind, subset_name, **kwargs):
    _, _ = batch_ind, kwargs
    if self.hparams.exp.weights_ema:
      with self.ema_non_treatment.average_parameters():
        with self.ema_treatment.average_parameters():
          treatment_pred, outcome_pred, _ = self(batch)
    else:
      treatment_pred, outcome_pred, _ = self(batch)

    if self.alpha_prev_treat > 0 or self.alpha_age > 0:
      treatment_pred, _ = treatment_pred

    if (
        'treatment_loss_off' in self.hparams.model[self.model_type]
        and self.hparams.model[self.model_type]['treatment_loss_off']
    ):
      bce_loss = 0.0
    elif self.balancing == 'grad_reverse':
      bce_loss = self.bce_loss(
          treatment_pred,
          batch['current_treatments'].to(torch.get_default_dtype()),
          kind='predict',
      )
    elif self.balancing == 'domain_confusion':
      bce_loss = self.bce_loss(
          treatment_pred,
          batch['current_treatments'].to(torch.get_default_dtype()),
          kind='confuse',
      )
    else:
      raise NotImplementedError()

    mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)

    # Masking for shorter sequences
    # Attention! Averaging across all
    # the active entries (= sequence masks) for full batch
    bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch[
        'active_entries'
    ].sum()
    mse_loss = (batch['active_entries'] * mse_loss).sum() / batch[
        'active_entries'
    ].sum()
    loss = bce_loss + mse_loss

    self.log(
        f'{self.model_type}_{subset_name}_loss',
        loss,
        on_epoch=True,
        on_step=False,
        sync_dist=True,
    )
    self.log(
        f'{self.model_type}_{subset_name}_bce_loss',
        bce_loss,
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
          '{}-val_metric'.format(self.model_type),
          mse_loss,
          on_epoch=True,
          on_step=False,
          sync_dist=True,
      )

  def validation_step(self, batch, batch_ind, **kwargs):
    subset_name = self.val_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name, **kwargs)

  def test_step(self, batch, batch_ind, **kwargs):
    subset_name = self.test_dataloader().dataset.subset_name
    self._eval_step(batch, batch_ind, subset_name, **kwargs)

  def predict_step(self, batch, batch_idx, dataset_idx=None):
    _, _ = batch_idx, dataset_idx
    if self.hparams.exp.weights_ema:
      with self.ema_non_treatment.average_parameters():
        _, outcome_pred, br = self(batch)
    else:
      _, outcome_pred, br = self(batch)
    return outcome_pred.cpu(), br.cpu()

  def get_representations(self, dataset):
    logger.info(
        '%s', f'Balanced representations inference for {dataset.subset_name}.'
    )
    # Creating Dataloader
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
    )
    _, br = [
        torch.cat(arrs)
        for arrs in zip(*self.trainer.predict(self, data_loader))
    ]
    return br.numpy()

  def get_causal_representations(self, dataset):
    br = self.get_representations(dataset)
    return {'rep_br': br}

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

  def get_predictions_generator(self, dataset):
    logger.info('%s', f'Predictions for {dataset.subset_name}.')
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
        outcome_pred, _ = self.predict_step(batch, batch_idx)
        outputs = batch['outputs']
        active_entries = batch['active_entries']
        yield outcome_pred.cpu().numpy(), outputs.cpu().numpy(), active_entries.cpu().numpy()

  def get_autoregressive_predictions_generator(self, dataset):
    logger.info('%s', f'Autoregressive Prediction for {dataset.subset_name}.')
    if self.model_type == 'decoder':  # CRNDecoder / EDCTDecoder / RMSN Decoder
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
            predicted_outputs[:, t] = outputs_scaled[:, t].cpu().numpy()
            if t < (self.hparams.dataset.projection_horizon - 1):
              batch['prev_outputs'][:, t + 1, :] = outputs_scaled[:, t, :]
          yield predicted_outputs, None, None
    else:
      raise NotImplementedError()

  def get_normalised_masked_rmse(
      self, dataset, one_step_counterfactual=False
  ):
    logger.info('%s', f'RMSE calculation for {dataset.subset_name}.')
    outputs_scaled_gen = self.get_predictions_generator(dataset)
    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse

    mse_orig_list = []
    mse_orig_weights = []
    mse_all_list = []
    mse_all_weights = []
    mse_last_list = []
    mse_last_weights = []

    for outputs_scaled, data_outputs_scaled, active_entries in tqdm(
        outputs_scaled_gen,
        total=int(np.round(len(dataset) / self.hparams.dataset.val_batch_size)),
    ):
      outputs_unscaled, data_outputs_unscaled = None, None
      if unscale:
        output_stds, output_means = (
            dataset.scaling_params['output_stds'],
            dataset.scaling_params['output_means'],
        )
        outputs_unscaled = outputs_scaled * output_stds + output_means
        data_outputs_unscaled = data_outputs_scaled * output_stds + output_means
        # However we must use batch-wise masked-MSE to avoid OOM!
        mse = ((outputs_unscaled - data_outputs_unscaled) ** 2) * active_entries
      else:
        mse = ((outputs_scaled - data_outputs_scaled) ** 2) * active_entries

      # Calculation like in original paper (Masked-Averaging over
      # datapoints (& outputs) and then non-masked time axis)
      mse_orig = mse.sum(0).sum(-1) / active_entries.sum(0).sum(-1)
      mse_orig_list.append(mse_orig)
      mse_orig_weights.append(active_entries.sum(0).sum(-1))

      # Masked averaging over all dimensions at once
      mse_all = mse.sum() / active_entries.sum()
      mse_all_list.append(mse_all)
      mse_all_weights.append(active_entries.sum())

      if one_step_counterfactual:
        # Only considering last active entry with actual counterfactuals
        num_samples, _, output_dim = active_entries.shape
        last_entries = active_entries - np.concatenate(
            [active_entries[:, 1:, :], np.zeros((num_samples, 1, output_dim))],
            axis=1,
        )
        if unscale:
          mse_last = (
              (outputs_unscaled - data_outputs_unscaled) ** 2
          ) * last_entries
        else:
          mse_last = (
              (outputs_scaled - data_outputs_scaled) ** 2
          ) * last_entries

        mse_last = mse_last.sum() / last_entries.sum()
        mse_last_list.append(mse_last)
        mse_last_weights.append(last_entries.sum())

    try:
      mse_orig = np.average(
          np.stack(mse_orig_list, axis=0),
          weights=np.stack(mse_orig_weights, axis=0),
          axis=0,
      ).mean()
    except ZeroDivisionError:
      mse_orig = 0.0
    mse_all = np.average(mse_all_list, weights=mse_all_weights)
    rmse_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const
    rmse_normalised_all = np.sqrt(mse_all) / dataset.norm_const
    if percentage:
      rmse_normalised_orig *= 100.0
      rmse_normalised_all *= 100.0
    if one_step_counterfactual:
      mse_last = np.average(mse_last_list, weights=mse_last_weights)
      rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const
      if percentage:
        rmse_normalised_last *= 100.0
      return rmse_normalised_orig, rmse_normalised_all, rmse_normalised_last
    return rmse_normalised_orig, rmse_normalised_all

  def get_normalised_n_step_rmses(
      self, dataset, datasets_mc = None
  ):
    logger.info('%s', f'RMSE calculation for {dataset.subset_name}.')
    assert (
        self.model_type == 'decoder'
        or self.model_type == 'multi'
        or self.model_type == 'g_net'
    )
    assert hasattr(dataset, 'data_processed_seq')

    unscale = self.hparams.exp.unscale_rmse
    percentage = self.hparams.exp.percentage_rmse
    outputs_scaled_gen = self.get_autoregressive_predictions_generator(
        dataset if datasets_mc is None else datasets_mc
    )

    mse_orig_list = []
    mse_orig_weights = []
    for batch_idx, (outputs_scaled, _, _) in tqdm(
        enumerate(outputs_scaled_gen),
        total=int(np.round(len(dataset) / self.hparams.dataset.val_batch_size)),
    ):
      start_idx = batch_idx * self.hparams.dataset.val_batch_size
      end_idx = start_idx + outputs_scaled.shape[0]
      active_entries = dataset.data_processed_seq['active_entries'][
          start_idx:end_idx
      ]
      if unscale:
        output_stds, output_means = (
            dataset.scaling_params['output_stds'],
            dataset.scaling_params['output_means'],
        )
        outputs_unscaled = outputs_scaled * output_stds + output_means
        data_outputs_unscaled = dataset.data_processed_seq['unscaled_outputs'][
            start_idx:end_idx
        ]
        mse = ((outputs_unscaled - data_outputs_unscaled) ** 2) * active_entries
      else:
        data_outputs_scaled = dataset.data_processed_seq['outputs'][
            start_idx:end_idx
        ]
        mse = ((outputs_scaled - data_outputs_scaled) ** 2) * active_entries

      nan_idx = np.unique(
          np.where(
              np.isnan(dataset.data_processed_seq['outputs'][start_idx:end_idx])
          )[0]
      )
      not_nan = np.array(
          [i for i in range(outputs_scaled.shape[0]) if i not in nan_idx]
      )

      # Calculation like in original paper (Masked-Averaging over
      # datapoints (& outputs) and then non-masked time axis)
      mse_orig = mse[not_nan].sum(0).sum(-1) / active_entries[not_nan].sum(
          0
      ).sum(-1)
      mse_orig_list.append(mse_orig)
      mse_orig_weights.append(active_entries[not_nan].sum(0).sum(-1))

    mse_orig_list = np.stack(mse_orig_list, axis=0)
    mse_orig_weights = np.stack(mse_orig_weights, axis=0)
    mse_orig = np.average(mse_orig_list, axis=0, weights=mse_orig_weights)
    rmses_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const
    if percentage:
      rmses_normalised_orig *= 100.0
    return rmses_normalised_orig

  def attr_forward_func(
      self,
      vitals,
      prev_treatments,
      prev_outputs,
      current_treatments,
      batch,
      step,
  ):
    input_batch = {}
    input_batch['vitals'] = vitals
    input_batch['prev_treatments'] = prev_treatments
    input_batch['prev_outputs'] = prev_outputs
    input_batch['current_treatments'] = current_treatments
    for k in batch:
      if k not in input_batch:
        input_batch[k] = batch[k]
    _, outcome_pred, _ = self(input_batch)

    if step == 1:
      num_samples, _, output_dim = outcome_pred.shape
      last_entries = input_batch['active_entries'] - np.concatenate(
          [
              input_batch['active_entries'][:, 1:, :],
              np.zeros((num_samples, 1, output_dim)),
          ],
          axis=1,
      )
      outcome_pred_last = (outcome_pred * last_entries).sum(dim=1)
      outcome_pred = outcome_pred_last
    else:
      outcome_pred = outcome_pred.mean(dim=1)
    return outcome_pred

  def get_interpret(self, dataset, step):
    logger.info('%s', f'Interpretation for {dataset.subset_name}')
    self.prepare_data()
    data_loader = DataLoader(
        dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=True
    )
    attr = IntegratedGradients(self.attr_forward_func, multiply_by_inputs=True)
    attributions = []
    seq_lengths = []

    batch_num = 30
    count = 0
    for batch in tqdm(data_loader, total=len(data_loader), desc='interpret'):
      vitals = batch['vitals']
      prev_treatments = batch['prev_treatments']
      prev_outputs = batch['prev_outputs']
      current_treatments = batch['current_treatments']
      seq_lengths.append(batch['sequence_lengths'])

      batch_attr = attr.attribute(
          inputs=(vitals, prev_treatments, prev_outputs, current_treatments),
          target=0,
          additional_forward_args=(batch, step),
          n_steps=30,
          internal_batch_size=vitals.shape[0],
      )
      attributions.append(batch_attr)
      count += 1
      if count >= batch_num:
        break
    attributions = [torch.cat(x, dim=0) for x in list(zip(*attributions))]
    attributions = {
        'vitals': attributions[0],
        'prev_treatments': attributions[1],
        'prev_outputs': attributions[2],
        'current_treatments': attributions[3],
        'sequence_lengths': torch.cat(seq_lengths, dim=0),
    }
    return attributions
