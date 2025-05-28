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

"""Script of training representation learning models."""

import ast
import logging
import os
import time

import hydra
from hydra.utils import instantiate
import numpy as np
import omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from src.models.utils import FilteringWandbLogger
from src.run.utils import check_exp_status
from src.run.utils import load_saved_data
import torch
from utils.ckpt import find_ckpt
from utils.ckpt import load_checkpoint
import wandb

DictConfig = omegaconf.DictConfig
OmegaConf = omegaconf.OmegaConf
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# torch.set_default_dtype(torch.double)
torch.autograd.set_detect_anomaly(True)


@hydra.main(config_name='config.yaml', config_path='../config/')
def main(args):
  """Training / evaluation script for models with representation-head.

  Args:
      args: arguments of run as DictConfig

  Returns:
      dict with results (one and nultiple-step-ahead RMSEs)
  """

  results = {}

  # Non-strict access to fields
  OmegaConf.set_struct(args, False)
  OmegaConf.register_new_resolver('sum', lambda x, y: x + y, replace=True)
  logger.info('%s', '\n' + OmegaConf.to_yaml(args, resolve=True))

  # Initialisation of data
  seed_everything(args.exp.seed)
  dataset_collection = load_saved_data(
      args.dataset.name,
      args.exp.seed,
      args.dataset.use_few_shot,
      max_number=args.dataset.max_number,
      few_shot_sample_num=args.dataset.few_shot_sample_num,
  )
  if dataset_collection is None:
    dataset_collection = instantiate(args.dataset, _recursive_=True)
  if args.exp.gen_data_only:
    # dataset_collection.save_data_in_crn_format(args.exp.data_save_path)
    dataset_collection.save_to_pkl(args.exp.data_save_path)
    return

  dataset_collection.process_data_rep_est()
  args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
  args.model.dim_treatments = dataset_collection.train_f.data[
      'current_treatments'
  ].shape[-1]
  args.model.dim_vitals = (
      dataset_collection.train_f.data['vitals'].shape[-1]
      if dataset_collection.has_vitals
      else 0
  )
  args.model.dim_static_features = dataset_collection.train_f.data[
      'static_features'
  ].shape[-1]

  # Train_callbacks
  rep_callbacks = []

  # MlFlow Logger
  def find_finished_wandb_exp(experiment_name, finetune_tag, seed):
    while True:
      pretrain_run, run_state = check_exp_status(
          'causal_over_time',
          tag=finetune_tag,
          expname=experiment_name,
          seed=seed,
      )
      if run_state == 'finished':
        return pretrain_run.id
      elif run_state == 'running':
        logger.info('Pretraining exp is running... check after 1 min')
        time.sleep(60)
      else:
        logger.info('Pretraining exp has not started! Check after 1 min')
        time.sleep(60)

  pretrain_run_id = None
  if args.exp.logging:
    experiment_name = f'{args.model.name}/{args.dataset.name}'
    tags = [s for s in args.exp.tags.split(',') if s]
    mlf_logger = FilteringWandbLogger(
        filter_submodels=['rep_encoder', 'est_head'],
        name=experiment_name,
        project='causal_over_time',
        tags=tags,
    )
    rep_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    if args.exp.finetune_tag:
      pretrain_run_id = find_finished_wandb_exp(
          experiment_name, args.exp.finetune_tag, args.exp.seed
      )
    elif args.exp.pretrain_tag:
      pretrain_run_id = find_finished_wandb_exp(
          experiment_name, args.exp.pretrain_tag, args.exp.seed
      )
    elif args.exp.pretrain_src_tag:
      pretrain_run_id = find_finished_wandb_exp(
          experiment_name, args.exp.pretrain_src_tag, args.exp.seed
      )
  else:
    mlf_logger = None

  if args.exp.early_stopping:
    rep_callbacks += [
        EarlyStopping(
            monitor='rep_encoder-val_metric',
            mode='min',
            patience=args.exp.early_stopping_patience,
        )
    ]

  rep_checkpoint_callback = ModelCheckpoint(
      monitor='rep_encoder-val_metric',
      mode='min',
      save_top_k=1,
      save_last=True,
      filename='rep_encoder-{epoch:d}',
      save_weights_only=True,
  )
  rep_checkpoint_callback.CHECKPOINT_NAME_LAST = 'rep_encoder-last'
  rep_callbacks.append(rep_checkpoint_callback)

  # Initialisation & Training of rep
  rep = instantiate(
      args.model.rep_encoder, args, dataset_collection, _recursive_=False
  )

  if args.model.rep_encoder.tune_hparams:
    rep.finetune(resources_per_trial=args.model.rep_encoder.resources_per_trial)

  if (
      args.exp.logging
      and args.exp.watch_model
      and isinstance(mlf_logger, FilteringWandbLogger)
  ):
    mlf_logger.experiment.watch(rep, log='all', log_freq=5, idx=1)

  rep_trainer = Trainer(
      gpus=ast.literal_eval(str(args.exp.gpus)),
      logger=mlf_logger,
      max_epochs=args.exp.max_epochs,
      callbacks=rep_callbacks,
      terminate_on_nan=True,
      check_val_every_n_epoch=args.exp.check_val_every_n_epoch,
      num_sanity_val_steps=0,
  )

  if args.exp.skip_train_rep:
    pass
  elif args.exp.eval_only:
    eval_ckpt_path = find_ckpt(
        args.exp.eval_ckpt_dir,
        prefix=rep.model_type,
        ckpt_type='last',
        run_id=pretrain_run_id,
    )
    rep = load_checkpoint(
        rep, eval_ckpt_path, load_ema=args.exp.weights_ema, strict=False
    )
  elif args.exp.finetune_ckpt_dir or args.exp.finetune_tag:
    finetune_ckpt_path = find_ckpt(
        args.exp.finetune_ckpt_dir,
        prefix=rep.model_type,
        ckpt_type='last',
        run_id=pretrain_run_id,
    )
    rep = load_checkpoint(
        rep, finetune_ckpt_path, load_ema=args.exp.weights_ema, strict=False
    )
  elif args.exp.pretrain_tag:
    finetune_ckpt_path = find_ckpt(
        args.exp.finetune_ckpt_dir,
        prefix=rep.model_type,
        ckpt_type='last',
        run_id=pretrain_run_id,
    )
    rep = load_checkpoint(
        rep, finetune_ckpt_path, load_ema=args.exp.weights_ema, strict=False
    )
  elif args.exp.pretrain_src_tag:
    finetune_ckpt_path = find_ckpt(
        args.exp.finetune_ckpt_dir,
        prefix=rep.model_type,
        ckpt_type='last',
        run_id=pretrain_run_id,
    )
    rep = load_checkpoint(
        rep, finetune_ckpt_path, load_ema=args.exp.weights_ema, strict=False
    )
  else:
    rep_trainer.fit(rep)

  representation_save_dir = os.getcwd()
  if args.exp.save_representations:
    rep_to_save = rep.get_representations(dataset_collection.train_f)
    np.savez_compressed(
        os.path.join(representation_save_dir, 'pretrained_reps.npz'),
        **rep_to_save,
    )

  # Initialisation & Training of head on original data
  def train_eval_head(
      dataset_collection, init_weight=None, dataset_name='', zero_shot=False
  ):
    if args.exp.logging:
      mlf_logger.filter_submodel = 'rep_encoder'  # Disable Logging to Mlflow

    head_callbacks = []
    if args.exp.logging:
      head_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    if args.exp.early_stopping:
      head_callbacks += [
          EarlyStopping(
              monitor='{}-est_head-val_metric'.format(dataset_name),
              mode='min',
              patience=args.exp.early_stopping_patience,
          )
      ]
    head_checkpoint_callback = ModelCheckpoint(
        monitor='{}-est_head-val_metric'.format(dataset_name),
        mode='min',
        save_top_k=1,
        save_last=True,
        filename=dataset_name + '-est_head-{epoch:d}',
        save_weights_only=True,
    )
    head_checkpoint_callback.CHECKPOINT_NAME_LAST = '{}-est_head-last'.format(
        dataset_name
    )
    head_callbacks.append(head_checkpoint_callback)

    head = instantiate(
        args.model.est_head,
        args,
        rep,
        dataset_collection,
        prefix=dataset_name,
        _recursive_=False,
    )
    if init_weight is not None:
      head.load_state_dict(init_weight)

    if (
        args.exp.logging
        and args.exp.watch_model
        and isinstance(mlf_logger, FilteringWandbLogger)
    ):
      mlf_logger.experiment.watch(head, log='all', log_freq=5, idx=2)

    head_trainer = Trainer(
        gpus=ast.literal_eval(str(args.exp.gpus)),
        logger=mlf_logger,
        max_epochs=(0 if zero_shot else args.exp.max_epochs),
        callbacks=head_callbacks,
        terminate_on_nan=True,
        check_val_every_n_epoch=args.exp.check_val_every_n_epoch,
        num_sanity_val_steps=0,
    )

    if args.exp.eval_only or (
        dataset_name == 'src' and args.exp.pretrain_src_tag
    ):
      eval_ckpt_path = find_ckpt(
          args.exp.eval_ckpt_dir,
          prefix='src-' + head.model_type,
          ckpt_type=args.exp.eval_ckpt_type,
          run_id=pretrain_run_id,
      )
      head = load_checkpoint(
          head, eval_ckpt_path, load_ema=args.exp.weights_ema, strict=False
      )
      head.trainer = head_trainer
    elif getattr(args.exp, 'eval_each_step_only', False):
      if dataset_name in ['src', 'dst-zero-shot']:
        ckpt_prefix = 'src-'
      elif dataset_name in ['dst']:
        ckpt_prefix = 'dst-'
      else:
        raise NotImplementedError()
      logger.info(
          '%s',
          f'eval_each_step_only! load from {ckpt_prefix + head.model_type}',
      )
      eval_ckpt_path = find_ckpt(
          args.exp.eval_ckpt_dir,
          prefix=ckpt_prefix + head.model_type,
          ckpt_type=args.exp.eval_ckpt_type,
          run_id=pretrain_run_id,
      )
      head = load_checkpoint(
          head, eval_ckpt_path, load_ema=args.exp.weights_ema, strict=False
      )
      head.trainer = head_trainer
    else:
      if args.exp.finetune_ckpt_dir or args.exp.finetune_tag:
        finetune_ckpt_path = find_ckpt(
            args.exp.finetune_ckpt_dir,
            prefix='src-' + head.model_type,
            ckpt_type=args.exp.finetune_ckpt_type,
            run_id=pretrain_run_id,
        )
        head = load_checkpoint(
            head,
            finetune_ckpt_path,
            load_ema=args.exp.weights_ema,
            strict=False,
        )
      head_trainer.fit(head)
      if not zero_shot and args.model.use_best_model:
        head = load_checkpoint(
            head,
            head_checkpoint_callback.best_model_path,
            load_ema=args.exp.weights_ema,
            strict=False,
        )

    test_rmses = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):
      test_rmses['{}-encoder_test_rmse_last'.format(dataset_name)] = (
          head.get_normalised_1_step_rmse_syn(
              dataset_collection.test_cf_one_step,
              prefix=f'{dataset_name}-test_cf_one_step',
          )
      )
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):
      rmses = head.get_normalised_n_step_rmses_syn(
          dataset_collection.test_cf_treatment_seq,
          prefix=f'{dataset_name}-test_cf_treatment_seq',
      )
      for k, v in enumerate(rmses):
        test_rmses[
            '{}-decoder_test_rmse_{}-step'.format(dataset_name, k + 2)
        ] = v
    if hasattr(dataset_collection, 'test_f'):
      rmses = head.get_normalised_n_step_rmses_real(
          dataset_collection.test_f, prefix=f'{dataset_name}-test_f'
      )
      for k, v in enumerate(rmses):
        test_rmses[
            '{}-decoder_test_rmse_{}-step'.format(dataset_name, k + 1)
        ] = v

    logger.info('%s', f'Test normalised RMSE (n-step prediction): {test_rmses}')
    if args.exp.loggings:
      mlf_logger.log_metrics(test_rmses)
    results.update(test_rmses)

    if args.exp.save_representations:
      rep_to_save = rep.get_representations(dataset_collection.train_f)
      np.savez_compressed(
          os.path.join(representation_save_dir, f'{dataset_name}_reps.npz'),
          **rep_to_save,
      )

    return head

  if args.model.train_head:
    pretrained_head = train_eval_head(
        dataset_collection, init_weight=None, dataset_name='src'
    )
    # Initialisation of test data if required
    if hasattr(args, 'target_dataset'):
      args.dataset = args.target_dataset
      del dataset_collection
      seed_everything(args.exp.seed)
      dataset_collection = instantiate(args.dataset, _recursive_=True)
    else:
      # replace train_f, val_f with test_train_f, test_val_f
      dataset_collection.train_f = dataset_collection.test_train_f
      dataset_collection.val_f = dataset_collection.test_val_f
    dataset_collection.process_data_rep_est()
    train_eval_head(
        dataset_collection,
        init_weight=pretrained_head.state_dict(),
        dataset_name='dst-zero-shot',
        zero_shot=True,
    )
    if not args.exp.eval_only:
      train_eval_head(
          dataset_collection,
          init_weight=pretrained_head.state_dict(),
          dataset_name='dst',
      )

  wandb.finish()

  return results


if __name__ == '__main__':
  main(DictConfig({}))
