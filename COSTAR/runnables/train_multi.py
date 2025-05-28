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

"""Script training Causal Transformer."""

import ast
import logging
import os
import pathlib
import re
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
from src.models.utils import AlphaRise
from src.models.utils import FilteringWandbLogger
from src.run.utils import check_exp_status
from src.run.utils import load_saved_data
import torch
from utils.ckpt import find_ckpt
from utils.ckpt import load_checkpoint
import wandb

Path = pathlib.Path
DictConfig = omegaconf.DictConfig
OmegaConf = omegaconf.OmegaConf
DataLoader = torch.utils.data.DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name='config.yaml', config_path='../config/')
def main(args):
  """Training / evaluation script for CT (Causal Transformer).

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

  # Train_callbacks
  multimodel_callbacks = [AlphaRise(rate=args.exp.alpha_rate)]

  # MlFlow Logger
  pretrain_run_id = None
  if args.exp.logging:
    experiment_name = f'{args.model.name}/{args.dataset.name}'
    tags = [s for s in args.exp.tags.split(',') if s]
    mlf_logger = FilteringWandbLogger(
        filter_submodels=[],
        name=experiment_name,
        project='causal_over_time',
        tags=tags,
    )
    multimodel_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    if args.exp.finetune_tag:
      while True:
        pretrain_run, run_state = check_exp_status(
            'causal_over_time',
            tag=args.exp.finetune_tag,
            expname=experiment_name,
            seed=args.exp.seed,
            args=args,
        )
        if run_state == 'finished':
          pretrain_run_id = pretrain_run.id
          break
        elif run_state == 'running':
          logger.info('Pretraining exp is running... check after 1 min')
          time.sleep(60)
        else:
          logger.info('Pretraining exp has not started! Check after 1 min')
          time.sleep(60)
  else:
    mlf_logger = None

  # Initialisation of data
  seed_everything(args.exp.seed)
  try:
    gt_causal_prediction_for = args.dataset.gt_causal_prediction_for
  except AttributeError:
    gt_causal_prediction_for = None
  dataset_collection = load_saved_data(
      args.dataset.name,
      args.exp.seed,
      args.dataset.use_few_shot,
      max_number=args.dataset.max_number,
      few_shot_sample_num=args.dataset.few_shot_sample_num,
      gt_causal_prediction_for=gt_causal_prediction_for,
  )
  if dataset_collection is None:
    dataset_collection = instantiate(args.dataset, _recursive_=True)
  dataset_collection.process_data_multi(
      generative_style_predict=args.model.generative_style_predict
  )
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

  if args.exp.early_stopping:
    multimodel_callbacks += [
        EarlyStopping(
            monitor='multi-val_metric',
            mode='min',
            patience=args.exp.early_stopping_patience,
        )
    ]

  multimodel_checkpoint_callback = ModelCheckpoint(
      monitor='multi-val_metric',
      mode='min',
      save_top_k=1,
      save_last=True,
      save_weights_only=True,
  )
  multimodel_callbacks.append(multimodel_checkpoint_callback)
  interval_checkpoint_callback = ModelCheckpoint(
      filename='intckpt-{epoch:04d}',
      save_top_k=-1,
      every_n_epochs=args.exp.save_ckpt_int,
      save_on_train_epoch_end=True,
      save_weights_only=True,
  )
  multimodel_callbacks.append(interval_checkpoint_callback)

  interval_checkpoint_callback = ModelCheckpoint(
      filename='intckpt-{epoch:04d}',
      save_top_k=-1,
      every_n_epochs=10,
      save_on_train_epoch_end=True,
      save_weights_only=True,
  )
  multimodel_callbacks.append(interval_checkpoint_callback)

  # Initialisation & Training of multimodel
  multimodel = instantiate(
      args.model.multi, args, dataset_collection, _recursive_=False
  )

  if args.model.multi.tune_hparams:
    multimodel.finetune(
        resources_per_trial=args.model.multi.resources_per_trial
    )

  if (
      args.exp.logging
      and args.exp.watch_model
      and isinstance(mlf_logger, FilteringWandbLogger)
  ):
    mlf_logger.experiment.watch(multimodel, log='all', log_freq=5)

  multimodel_trainer = Trainer(
      gpus=ast.literal_eval(str(args.exp.gpus)),
      logger=mlf_logger,
      max_epochs=args.exp.max_epochs,
      callbacks=multimodel_callbacks,
      terminate_on_nan=True,
      check_val_every_n_epoch=args.exp.check_val_every_n_epoch,
      num_sanity_val_steps=0,
  )

  eval_ckpt_path = None
  if args.exp.eval_only:
    eval_ckpt_path = find_ckpt(
        args.exp.eval_ckpt_dir,
        prefix='',
        ckpt_type=args.exp.eval_ckpt_type,
        run_id=pretrain_run_id,
    )
    multimodel = load_checkpoint(
        multimodel, eval_ckpt_path, load_ema=args.exp.weights_ema, strict=False
    )
  else:
    if args.exp.finetune_ckpt_dir or args.exp.finetune_tag:
      finetune_ckpt_path = find_ckpt(
          args.exp.finetune_ckpt_dir,
          prefix='',
          ckpt_type=args.exp.finetune_ckpt_type,
          run_id=pretrain_run_id,
      )
      multimodel = load_checkpoint(
          multimodel,
          finetune_ckpt_path,
          load_ema=args.exp.weights_ema,
          strict=False,
      )
    multimodel_trainer.fit(multimodel)
    if args.model.use_best_model:
      multimodel = load_checkpoint(
          multimodel,
          multimodel_checkpoint_callback.best_model_path,
          load_ema=args.exp.weights_ema,
          strict=False,
      )

  # Validation factual rmse
  val_dataloader = DataLoader(
      dataset_collection.val_f,
      batch_size=args.dataset.val_batch_size,
      shuffle=False,
  )
  multimodel_trainer.test(multimodel, test_dataloaders=val_dataloader)
  val_rmse_orig, val_rmse_all = multimodel.get_normalised_masked_rmse(
      dataset_collection.val_f
  )
  logger.info(
      '%s',
      f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig):'
      f' {val_rmse_orig}',
  )

  if args.exp.eval_only:
    interval_ckptpaths = [Path(eval_ckpt_path)]
    eval_epoch_nums = [0]
  else:
    if args.exp.save_ckpt_int == 0:
      interval_ckptpaths = [None]
      eval_epoch_nums = [0]
    else:
      interval_ckptpaths = sorted(
          list(
              Path(interval_checkpoint_callback.dirpath).glob('intckpt-*.ckpt')
          )
      )
      eval_epoch_nums = [
          int(re.match(r'intckpt-epoch=(\d+)\.ckpt', interval_ckptpath.name)[1])
          for interval_ckptpath in interval_ckptpaths
      ]

  for interval_ckptpath, eval_epoch in zip(interval_ckptpaths, eval_epoch_nums):
    if interval_ckptpath is not None:
      logger.info('%s', 'Evaluating ckpt at {}'.format(interval_ckptpath.name))
      multimodel = load_checkpoint(
          multimodel,
          str(interval_ckptpath),
          load_ema=args.exp.weights_ema,
          strict=False,
      )

    encoder_results = {}
    if hasattr(
        dataset_collection, 'test_cf_one_step'
    ):  # Test one_step_counterfactual rmse
      test_rmse_orig, test_rmse_all, test_rmse_last = (
          multimodel.get_normalised_masked_rmse(
              dataset_collection.test_cf_one_step, one_step_counterfactual=True
          )
      )
      logger.info(
          '%s',
          f'Test normalised RMSE (all): {test_rmse_all}; '
          f'Test normalised RMSE (orig): {test_rmse_orig}; '
          f'Test normalised RMSE (only counterfactual): {test_rmse_last}',
      )
      encoder_results = {
          'encoder_val_rmse_all': val_rmse_all,
          'encoder_val_rmse_orig': val_rmse_orig,
          'encoder_test_rmse_all': test_rmse_all,
          'encoder_test_rmse_orig': test_rmse_orig,
          'encoder_test_rmse_last': test_rmse_last,
          'epoch': eval_epoch,
      }
      if args.exp.interpret_only:
        encoder_attrs = multimodel.get_interpret(
            dataset_collection.test_cf_one_step, step=1
        )
        np.savez_compressed(
            os.path.join(os.getcwd(), 'attrs_encoder.npz'), **encoder_attrs
        )
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
      test_rmse_orig, test_rmse_all = multimodel.get_normalised_masked_rmse(
          dataset_collection.test_f
      )
      logger.info(
          '%s',
          f'Test normalised RMSE (all): {test_rmse_all}; '
          f'Test normalised RMSE (orig): {test_rmse_orig}.',
      )
      encoder_results = {
          'encoder_val_rmse_all': val_rmse_all,
          'encoder_val_rmse_orig': val_rmse_orig,
          'encoder_test_rmse_all': test_rmse_all,
          'encoder_test_rmse_orig': test_rmse_orig,
          'epoch': eval_epoch,
      }

    if args.exp.logging:
      mlf_logger.log_metrics(encoder_results)
    results.update(encoder_results)

    test_rmses = {}
    if hasattr(
        dataset_collection, 'test_cf_treatment_seq'
    ):  # Test n_step_counterfactual rmse
      test_rmses = multimodel.get_normalised_n_step_rmses(
          dataset_collection.test_cf_treatment_seq
      )
    elif hasattr(
        dataset_collection, 'test_f_multi'
    ):  # Test n_step_factual rmse
      test_rmses = multimodel.get_normalised_n_step_rmses(
          dataset_collection.test_f_multi
      )
    test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

    logger.info('%s', f'Test normalised RMSE (n-step prediction): {test_rmses}')
    decoder_results = {
        'decoder_val_rmse_all': val_rmse_all,
        'decoder_val_rmse_orig': val_rmse_orig,
        'epoch': eval_epoch,
    }
    decoder_results.update(
        {('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()}
    )

    if args.exp.logging:
      mlf_logger.log_metrics(decoder_results)
    results.update(decoder_results)

  wandb.finish()

  return results


if __name__ == '__main__':
  main(DictConfig({}))
