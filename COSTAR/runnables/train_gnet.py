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

"""Script training G-Net."""

import ast
import logging
import time
import hydra
from hydra.utils import instantiate
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
torch.set_default_dtype(torch.double)


@hydra.main(config_name='config.yaml', config_path='../config/')
def main(args):
  """Training / evaluation script for G-Net.

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

  model_callbacks = []

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
    model_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    if args.exp.finetune_tag:
      while True:
        pretrain_run, run_state = check_exp_status(
            'causal_over_time',
            tag=args.exp.finetune_tag,
            expname=experiment_name,
            seed=args.exp.seed,
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
  # dataset_collection = instantiate(args.dataset, _recursive_=True)
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
  dataset_collection.process_data_multi()
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
    model_callbacks += [
        EarlyStopping(
            monitor='g_net-val_metric',
            mode='min',
            patience=args.exp.early_stopping_patience,
        )
    ]

  model_checkpoint_callback = ModelCheckpoint(
      monitor='g_net-val_metric',
      mode='min',
      save_top_k=1,
      save_last=True,
      save_weights_only=True,
  )
  model_callbacks.append(model_checkpoint_callback)
  interval_checkpoint_callback = ModelCheckpoint(
      filename='intckpt-{epoch:04d}',
      save_top_k=-1,
      every_n_epochs=args.exp.save_ckpt_int,
      save_on_train_epoch_end=True,
      save_weights_only=True,
  )
  model_callbacks.append(interval_checkpoint_callback)

  interval_checkpoint_callback = ModelCheckpoint(
      filename='intckpt-{epoch:04d}',
      save_top_k=-1,
      every_n_epochs=10,
      save_on_train_epoch_end=True,
      save_weights_only=True,
  )
  model_callbacks.append(interval_checkpoint_callback)

  # Conditional networks outputs are of uniform dim
  # between (dim_outcomes + dim_vitals)
  args.model.g_net.comp_sizes = [
      (args.model.dim_outcomes + args.model.dim_vitals)
      // args.model.g_net.num_comp
  ] * args.model.g_net.num_comp

  # Initialisation & Training of G-Net
  model = instantiate(
      args.model.g_net, args, dataset_collection, _recursive_=False
  )
  if args.model.g_net.tune_hparams:
    model.finetune(resources_per_trial=args.model.g_net.resources_per_trial)

  trainer = Trainer(
      gpus=ast.literal_eval(str(args.exp.gpus)),
      logger=mlf_logger,
      max_epochs=args.exp.max_epochs,
      callbacks=model_callbacks,
      terminate_on_nan=True,
      check_val_every_n_epoch=args.exp.check_val_every_n_epoch,
      num_sanity_val_steps=0,
  )
  # trainer.fit(model)
  if args.exp.eval_only:
    eval_ckpt_path = find_ckpt(
        args.exp.eval_ckpt_dir,
        prefix='',
        ckpt_type=args.exp.eval_ckpt_type,
        run_id=pretrain_run_id,
    )
    model = load_checkpoint(
        model, eval_ckpt_path, load_ema=args.exp.weights_ema, strict=False
    )
  else:
    if args.exp.finetune_ckpt_dir or args.exp.finetune_tag:
      finetune_ckpt_path = find_ckpt(
          args.exp.finetune_ckpt_dir,
          prefix='',
          ckpt_type=args.exp.finetune_ckpt_type,
          run_id=pretrain_run_id,
      )
      model = load_checkpoint(
          model, finetune_ckpt_path, load_ema=args.exp.weights_ema, strict=False
      )
    trainer.fit(model)
    if args.model.use_best_model:
      model = load_checkpoint(
          model,
          model_checkpoint_callback.best_model_path,
          load_ema=args.exp.weights_ema,
          strict=False,
      )

  # Validation factual rmse
  val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(
      dataset_collection.val_f
  )
  logger.info(
      '%s',
      f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig):'
      f' {val_rmse_orig}',
  )

  encoder_results = {}
  if hasattr(
      dataset_collection, 'test_cf_one_step'
  ):  # Test one_step_counterfactual rmse
    test_rmse_orig, test_rmse_all, test_rmse_last = (
        model.get_normalised_masked_rmse(
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
    }
  elif hasattr(dataset_collection, 'test_f_multi'):  # Test factual rmse
    test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(
        dataset_collection.test_f_multi
    )
    logger.info(
        '%s',
        f'Test normalised RMSE (all): {test_rmse_all}; Test normalised RMSE'
        f' (orig): {test_rmse_orig}.',
    )
    encoder_results = {
        'encoder_val_rmse_all': val_rmse_all,
        'encoder_val_rmse_orig': val_rmse_orig,
        'encoder_test_rmse_all': test_rmse_all,
        'encoder_test_rmse_orig': test_rmse_orig,
    }

  if args.exp.logging:
    mlf_logger.log_metrics(encoder_results)
  results.update(encoder_results)

  test_rmses = {}
  if hasattr(
      dataset_collection, 'test_cf_treatment_seq_mc'
  ):  # Test n_step_counterfactual rmse
    test_rmses = model.get_normalised_n_step_rmses(
        dataset_collection.test_cf_treatment_seq,
        dataset_collection.test_cf_treatment_seq_mc,
    )
  elif hasattr(dataset_collection, 'test_f_mc'):  # Test n_step_factual rmse
    test_rmses = model.get_normalised_n_step_rmses(
        dataset_collection.test_f_multi, dataset_collection.test_f_mc
    )

  test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

  logger.info('%s', f'Test normalised RMSE (n-step prediction): {test_rmses}')
  decoder_results = {
      'decoder_val_rmse_all': val_rmse_all,
      'decoder_val_rmse_orig': val_rmse_orig,
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
