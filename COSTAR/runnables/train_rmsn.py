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

"""Script training RMSN."""

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
from src.models.rmsn import RMSN
from src.models.utils import FilteringWandbLogger
from src.run.utils import check_exp_status
from src.run.utils import load_saved_data
import torch
from utils.ckpt import find_ckpt
from utils.ckpt import load_checkpoint
import wandb

DictConfig = omegaconf.DictConfig
OmegaConf = omegaconf.OmegaConf
DataLoader = torch.utils.data.DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name='config.yaml', config_path='../config/')
def main(args):
  """Training / evaluation script for RMSNs.

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
  (
      prop_treatment_callbacks,
      propensity_history_callbacks,
      encoder_callbacks,
      decoder_callbacks,
  ) = ([], [], [], [])

  pretrain_run_id = None
  if args.exp.logging:
    experiment_name = f'{args.model.name}/{args.dataset.name}'
    tags = [s for s in args.exp.tags.split(',') if s]
    mlf_logger = FilteringWandbLogger(
        filter_submodels=RMSN.possible_model_types,
        name=experiment_name,
        project='causal_over_time',
        tags=tags,
    )
    encoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    decoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    prop_treatment_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    propensity_history_callbacks += [
        LearningRateMonitor(logging_interval='epoch')
    ]
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

  # Initialisation of data to calculate dim_outcomes, dim_treatments,
  # dim_vitals and dim_static_features
  seed_everything(args.exp.seed)
  if args.dataset.treatment_mode == 'multiclass':
    args.dataset.treatment_mode = 'multilabel'
  # Only binary multilabel regime possible
  assert (
      args.dataset.treatment_mode == 'multilabel'
  )  # Only binary multilabel regime possible

  dataset_collection = load_saved_data(
      args.dataset.name,
      args.exp.seed,
      args.dataset.use_few_shot,
      max_number=args.dataset.max_number,
      few_shot_sample_num=args.dataset.few_shot_sample_num,
  )
  if dataset_collection is None:
    dataset_collection = instantiate(args.dataset, _recursive_=True)
  dataset_collection.process_data_encoder()
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
    encoder_callbacks += [
        EarlyStopping(
            monitor='encoder-val_metric',
            mode='min',
            patience=args.exp.early_stopping_patience,
        )
    ]
    decoder_callbacks += [
        EarlyStopping(
            monitor='decoder-val_metric',
            mode='min',
            patience=args.exp.early_stopping_patience,
        )
    ]
    prop_treatment_callbacks += [
        EarlyStopping(
            monitor='propensity_treatment-val_metric',
            mode='min',
            patience=args.exp.early_stopping_patience,
        )
    ]
    propensity_history_callbacks += [
        EarlyStopping(
            monitor='propensity_history-val_metric',
            mode='min',
            patience=args.exp.early_stopping_patience,
        )
    ]

  encoder_callbacks += [
      ModelCheckpoint(
          monitor='encoder-val_metric',
          mode='min',
          save_top_k=1,
          save_last=True,
          filename='encoder-{epoch:d}',
          save_weights_only=True,
      )
  ]
  encoder_checkpoint_callback = encoder_callbacks[-1]
  encoder_checkpoint_callback.CHECKPOINT_NAME_LAST = 'encoder-last'
  decoder_callbacks += [
      ModelCheckpoint(
          monitor='decoder-val_metric',
          mode='min',
          save_top_k=1,
          save_last=True,
          filename='decoder-{epoch:d}',
          save_weights_only=True,
      )
  ]
  decoder_checkpoint_callback = decoder_callbacks[-1]
  decoder_checkpoint_callback.CHECKPOINT_NAME_LAST = 'decoder-last'
  prop_treatment_callbacks += [
      ModelCheckpoint(
          monitor='propensity_treatment-val_metric',
          mode='min',
          save_top_k=1,
          save_last=True,
          filename='prop_treatment-{epoch:d}',
          save_weights_only=True,
      )
  ]
  prop_treatment_checkpoint_callback = prop_treatment_callbacks[-1]
  prop_treatment_checkpoint_callback.CHECKPOINT_NAME_LAST = (
      'prop_treatment-last'
  )
  propensity_history_callbacks += [
      ModelCheckpoint(
          monitor='propensity_history-val_metric',
          mode='min',
          save_top_k=1,
          save_last=True,
          filename='propensity_history-{epoch:d}',
          save_weights_only=True,
      )
  ]
  propensity_history_checkpoint_callback = propensity_history_callbacks[-1]
  propensity_history_checkpoint_callback.CHECKPOINT_NAME_LAST = (
      'propensity_history-last'
  )

  # Nominator (treatment propensity network)
  propensity_treatment = instantiate(
      args.model.propensity_treatment,
      args,
      dataset_collection,
      _recursive_=False,
  )
  if args.model.propensity_treatment.tune_hparams:
    propensity_treatment.finetune(
        resources_per_trial=args.model.propensity_treatment.resources_per_trial
    )

  propensity_treatment_trainer = Trainer(
      gpus=ast.literal_eval(str(args.exp.gpus)),
      logger=mlf_logger,
      max_epochs=args.exp.max_epochs,
      callbacks=prop_treatment_callbacks,
      gradient_clip_val=args.model.propensity_treatment.max_grad_norm,
      terminate_on_nan=True,
      check_val_every_n_epoch=args.exp.check_val_every_n_epoch,
      num_sanity_val_steps=0,
  )

  if args.exp.eval_only:
    eval_ckpt_path = find_ckpt(
        args.exp.eval_ckpt_dir,
        prefix='prop_treatment',
        ckpt_type=args.exp.eval_ckpt_type,
        run_id=pretrain_run_id,
    )
    propensity_treatment = load_checkpoint(
        propensity_treatment, eval_ckpt_path, load_ema=args.exp.weights_ema
    )
    propensity_treatment.trainer = propensity_treatment_trainer
  else:
    if args.exp.finetune_ckpt_dir or args.exp.finetune_tag:
      finetune_ckpt_path = find_ckpt(
          args.exp.finetune_ckpt_dir,
          prefix='prop_treatment',
          ckpt_type='last',
          run_id=pretrain_run_id,
      )
      propensity_treatment = load_checkpoint(
          propensity_treatment,
          finetune_ckpt_path,
          load_ema=args.exp.weights_ema,
      )
    propensity_treatment_trainer.fit(propensity_treatment)
    if args.model.use_best_model:
      propensity_treatment = load_checkpoint(
          propensity_treatment,
          prop_treatment_checkpoint_callback.best_model_path,
          load_ema=args.exp.weights_ema,
      )

  # Validation BCE
  val_bce_orig, val_bce_all = propensity_treatment.get_masked_bce(
      dataset_collection.val_f
  )
  logger.info(
      '%s',
      f'Val normalised BCE (all): {val_bce_all}; Val normalised RMSE (orig):'
      f' {val_bce_orig}',
  )

  # Test BCE
  if hasattr(
      dataset_collection, 'test_cf_one_step'
  ):  # Test one_step_counterfactual
    test_bce_orig, test_bce_all = propensity_treatment.get_masked_bce(
        dataset_collection.test_cf_one_step
    )
  elif hasattr(dataset_collection, 'test_f'):  # Test factual
    test_bce_orig, test_bce_all = propensity_treatment.get_masked_bce(
        dataset_collection.test_f
    )
  else:
    raise NotImplementedError()

  logger.info(
      '%s',
      f'Test normalised RMSE (all): {test_bce_orig}; Test normalised RMSE'
      f' (orig): {test_bce_all}.',
  )
  prop_treatment_results = {
      'propensity_treatment_val_bce_all': val_bce_all,
      'propensity_treatment_val_bce_orig': val_bce_orig,
      'propensity_treatment_test_bce_all': test_bce_all,
      'propensity_treatment_test_bce_orig': test_bce_orig,
  }

  if args.exp.logging:
    mlf_logger.log_metrics(prop_treatment_results)
  results.update(prop_treatment_results)

  # Denominator (history propensity network)
  propensity_history = instantiate(
      args.model.propensity_history, args, dataset_collection, _recursive_=False
  )
  if args.model.propensity_history.tune_hparams:
    propensity_history.finetune(
        resources_per_trial=args.model.propensity_history.resources_per_trial
    )

  propensity_history_trainer = Trainer(
      gpus=ast.literal_eval(str(args.exp.gpus)),
      logger=mlf_logger,
      max_epochs=args.exp.max_epochs,
      callbacks=propensity_history_callbacks,
      gradient_clip_val=args.model.propensity_history.max_grad_norm,
      terminate_on_nan=True,
      check_val_every_n_epoch=args.exp.check_val_every_n_epoch,
      num_sanity_val_steps=0,
  )

  if args.exp.eval_only:
    eval_ckpt_path = find_ckpt(
        args.exp.eval_ckpt_dir,
        prefix=propensity_history.model_type,
        ckpt_type=args.exp.eval_ckpt_type,
        run_id=pretrain_run_id,
    )
    propensity_history = load_checkpoint(
        propensity_history, eval_ckpt_path, load_ema=args.exp.weights_ema
    )
    propensity_history.trainer = propensity_history_trainer
  else:
    if args.exp.finetune_ckpt_dir or args.exp.finetune_tag:
      finetune_ckpt_path = find_ckpt(
          args.exp.finetune_ckpt_dir,
          prefix=propensity_history.model_type,
          ckpt_type='last',
          run_id=pretrain_run_id,
      )
      propensity_history = load_checkpoint(
          propensity_history, finetune_ckpt_path, load_ema=args.exp.weights_ema
      )
    propensity_history_trainer.fit(propensity_history)
    if args.model.use_best_model:
      propensity_history = load_checkpoint(
          propensity_history,
          propensity_history_checkpoint_callback.best_model_path,
          load_ema=args.exp.weights_ema,
      )

  # Validation BCE
  val_bce_orig, val_bce_all = propensity_history.get_masked_bce(
      dataset_collection.val_f
  )
  logger.info(
      '%s',
      f'Val normalised BCE (all): {val_bce_all}; Val normalised RMSE (orig):'
      f' {val_bce_orig}',
  )

  # Test BCE
  if hasattr(
      dataset_collection, 'test_cf_one_step'
  ):  # Test one_step_counterfactual
    test_bce_orig, test_bce_all = propensity_history.get_masked_bce(
        dataset_collection.test_cf_one_step
    )
  elif hasattr(dataset_collection, 'test_f'):  # Test factual
    test_bce_orig, test_bce_all = propensity_history.get_masked_bce(
        dataset_collection.test_f
    )

  logger.info(
      '%s',
      f'Test normalised BCE (all): {test_bce_orig}; Test normalised BCE (orig):'
      f' {test_bce_all}.',
  )
  propensity_history_results = {
      'propensity_history_val_bce_all': val_bce_all,
      'propensity_history_val_bce_orig': val_bce_orig,
      'propensity_history_test_bce_all': test_bce_all,
      'propensity_history_test_bce_orig': test_bce_orig,
  }

  if args.exp.logging:
    mlf_logger.log_metrics(propensity_history_results)
  results.update(propensity_history_results)

  # Initialisation & Training of Encoder
  encoder = instantiate(
      args.model.encoder,
      args,
      propensity_treatment,
      propensity_history,
      dataset_collection,
      _recursive_=False,
  )
  if args.model.encoder.tune_hparams:
    encoder.finetune(resources_per_trial=args.model.encoder.resources_per_trial)

  encoder_trainer = Trainer(
      gpus=ast.literal_eval(str(args.exp.gpus)),
      logger=mlf_logger,
      max_epochs=args.exp.max_epochs,
      callbacks=encoder_callbacks,
      gradient_clip_val=args.model.encoder.max_grad_norm,
      terminate_on_nan=True,
      check_val_every_n_epoch=args.exp.check_val_every_n_epoch,
      num_sanity_val_steps=0,
  )

  if args.exp.eval_only:
    eval_ckpt_path = find_ckpt(
        args.exp.eval_ckpt_dir,
        prefix=encoder.model_type,
        ckpt_type=args.exp.eval_ckpt_type,
        run_id=pretrain_run_id,
    )
    encoder = load_checkpoint(
        encoder, eval_ckpt_path, load_ema=args.exp.weights_ema
    )
  else:
    if args.exp.finetune_ckpt_dir or args.exp.finetune_tag:
      finetune_ckpt_path = find_ckpt(
          args.exp.finetune_ckpt_dir,
          prefix=encoder.model_type,
          ckpt_type='last',
          run_id=pretrain_run_id,
      )
      encoder = load_checkpoint(
          encoder, finetune_ckpt_path, load_ema=args.exp.weights_ema
      )
    encoder_trainer.fit(encoder)
    if args.model.use_best_model:
      encoder = load_checkpoint(
          encoder,
          encoder_checkpoint_callback.best_model_path,
          load_ema=args.exp.weights_ema,
      )

  encoder_results = {}
  # Validation factual rmse
  val_dataloader = DataLoader(
      dataset_collection.val_f,
      batch_size=args.dataset.val_batch_size,
      shuffle=False,
  )
  encoder_trainer.test(encoder, test_dataloaders=val_dataloader)
  val_rmse_orig, val_rmse_all = encoder.get_normalised_masked_rmse(
      dataset_collection.val_f
  )
  logger.info(
      '%s',
      f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig):'
      f' {val_rmse_orig}'
  )

  if hasattr(
      dataset_collection, 'test_cf_one_step'
  ):  # Test one_step_counterfactual rmse
    test_rmse_orig, test_rmse_all, test_rmse_last = (
        encoder.get_normalised_masked_rmse(
            dataset_collection.test_cf_one_step, one_step_counterfactual=True
        )
    )
    logger.info(
        '%s',
        f'Test normalised RMSE (all): {test_rmse_all}; '
        f'Test normalised RMSE (orig): {test_rmse_orig}; '
        f'Test normalised RMSE (only counterfactual): {test_rmse_last}'
    )
    encoder_results = {
        'encoder_val_rmse_all': val_rmse_all,
        'encoder_val_rmse_orig': val_rmse_orig,
        'encoder_test_rmse_all': test_rmse_all,
        'encoder_test_rmse_orig': test_rmse_orig,
        'encoder_test_rmse_last': test_rmse_last,
    }
  elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
    test_rmse_orig, test_rmse_all = encoder.get_normalised_masked_rmse(
        dataset_collection.test_f
    )
    logger.info(
        '%s',
        f'Test normalised RMSE (all): {test_rmse_all}; '
        f'Test normalised RMSE (orig): {test_rmse_orig}.'
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

  #  Initialisation & Training of Decoder
  if args.model.train_decoder:
    decoder = instantiate(
        args.model.decoder, args, encoder, dataset_collection, _recursive_=False
    )

    if args.model.decoder.tune_hparams:
      decoder.finetune(
          resources_per_trial=args.model.decoder.resources_per_trial
      )

    decoder_trainer = Trainer(
        gpus=ast.literal_eval(str(args.exp.gpus)),
        logger=mlf_logger,
        max_epochs=args.exp.max_epochs,
        gradient_clip_val=args.model.decoder.max_grad_norm,
        callbacks=decoder_callbacks,
        terminate_on_nan=True,
        check_val_every_n_epoch=args.exp.check_val_every_n_epoch,
        num_sanity_val_steps=0,
    )

    if args.exp.eval_only:
      eval_ckpt_path = find_ckpt(
          args.exp.eval_ckpt_dir,
          prefix=decoder.model_type,
          ckpt_type=args.exp.eval_ckpt_type,
          run_id=pretrain_run_id,
      )
      decoder = load_checkpoint(
          decoder, eval_ckpt_path, load_ema=args.exp.weights_ema
      )
    else:
      if args.exp.finetune_ckpt_dir or args.exp.finetune_tag:
        finetune_ckpt_path = find_ckpt(
            args.exp.finetune_ckpt_dir,
            prefix=decoder.model_type,
            ckpt_type=args.exp.finetune_ckpt_type,
            run_id=pretrain_run_id,
        )
        decoder = load_checkpoint(
            decoder, finetune_ckpt_path, load_ema=args.exp.weights_ema
        )
      decoder_trainer.fit(decoder)
      if args.model.use_best_model:
        decoder = load_checkpoint(
            decoder,
            decoder_checkpoint_callback.best_model_path,
            load_ema=args.exp.weights_ema,
        )

    # Validation factual rmse
    val_dataloader = DataLoader(
        dataset_collection.val_f,
        batch_size=10 * args.dataset.val_batch_size,
        shuffle=False,
    )
    decoder_trainer.test(decoder, test_dataloaders=val_dataloader)
    val_rmse_orig, val_rmse_all = decoder.get_normalised_masked_rmse(
        dataset_collection.val_f
    )
    logger.info(
        '%s',
        f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE'
        f' (orig): {val_rmse_orig}'
    )

    test_rmses = {}
    if hasattr(
        dataset_collection, 'test_cf_treatment_seq'
    ):  # Test n_step_counterfactual rmse
      test_rmses = decoder.get_normalised_n_step_rmses(
          dataset_collection.test_cf_treatment_seq
      )
    elif hasattr(dataset_collection, 'test_f'):  # Test n_step_factual rmse
      test_rmses = decoder.get_normalised_n_step_rmses(
          dataset_collection.test_f
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
