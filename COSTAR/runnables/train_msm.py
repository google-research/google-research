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

"""Script training MSM model."""

import logging
import hydra
from hydra.utils import instantiate
import omegaconf
from pytorch_lightning.utilities.seed import seed_everything
from src.models.utils import FilteringWandbLogger
from src.run.utils import load_saved_data
import torch
import wandb

DictConfig = omegaconf.DictConfig
OmegaConf = omegaconf.OmegaConf
DataLoader = torch.utils.data.DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name='config.yaml', config_path='../config/')
def main(args):
  """Training / evaluation script for MSMs.

  Args:
      args: arguments of run as DictConfig

  Returns:
      dict with results (one and multiple-step-ahead RMSEs)
  """

  results = {}

  # Non-strict access to fields
  OmegaConf.set_struct(args, False)
  OmegaConf.register_new_resolver('sum', lambda x, y: x + y, replace=True)
  logger.info('%s', '\n' + OmegaConf.to_yaml(args, resolve=True))

  # Initialisation of data to calculate dim_outcomes, dim_treatments,
  # dim_vitals and dim_static_features
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

  # assert args.dataset.treatment_mode == 'multilabel'
  # Only binary multilabel regime possible
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

  # processing to fit MSM into our previous ways of constructing dataset
  # very ad-hoc...
  def merge_dataset(dataset_list):
    merged_batches = []
    for dataset in dataset_list:
      dataloader = DataLoader(
          dataset, shuffle=False, batch_size=args.dataset.val_batch_size
      )
      for batch in dataloader:
        merged_batches.append(batch)

    merged_data = {}
    for k in merged_batches[0].keys():
      merged_data[k] = torch.cat([x[k] for x in merged_batches], dim=0).numpy()
    return merged_data

  tr_te_type = getattr(args.dataset, 'tr_te_type', None)
  if tr_te_type is not None:
    if args.dataset.name.startswith('tumor_generator'):
      if tr_te_type == 'src_src':
        pass
      else:
        seed_everything(args.exp.seed)
        target_dataset_collection = instantiate(
            args.target_dataset, _recursive_=True
        )
        target_dataset_collection.process_data_multi()
        if tr_te_type == 'src_tgt':
          dataset_collection.test_cf_one_step = (
              target_dataset_collection.test_cf_one_step
          )
          dataset_collection.test_cf_treatment_seq = (
              target_dataset_collection.test_cf_treatment_seq
          )
        elif tr_te_type == 'srctgt_tgt':
          dataset_collection.train_f.data = merge_dataset(
              [dataset_collection.train_f, target_dataset_collection.train_f]
          )
          dataset_collection.test_cf_one_step = (
              target_dataset_collection.test_cf_one_step
          )
          dataset_collection.test_cf_treatment_seq = (
              target_dataset_collection.test_cf_treatment_seq
          )
    elif args.dataset.name.startswith('mimic3_synthetic_age_domain'):
      if tr_te_type == 'src_src':
        dataset_collection.train_f.data = merge_dataset(
            [dataset_collection.src_train_f]
        )
      elif tr_te_type == 'src_tgt':
        dataset_collection.train_f.data = merge_dataset(
            [dataset_collection.src_train_f]
        )
      elif tr_te_type == 'srctgt_tgt':
        dataset_collection.train_f.data = merge_dataset(
            [dataset_collection.src_train_f, dataset_collection.test_train_f]
        )
      dataset_collection.test_cf_one_step.data = merge_dataset(
          [dataset_collection.test_cf_one_step]
      )
      dataset_collection.test_cf_treatment_seq.data = merge_dataset(
          [dataset_collection.test_cf_treatment_seq]
      )
    elif args.dataset.name.startswith('m5_real_foods_household'):
      if tr_te_type == 'src_src':
        dataset_collection.train_f.data = merge_dataset(
            [dataset_collection.src_train_f]
        )
      elif tr_te_type == 'src_tgt':
        dataset_collection.train_f.data = merge_dataset(
            [dataset_collection.src_train_f]
        )
      elif tr_te_type == 'srctgt_tgt':
        dataset_collection.train_f.data = merge_dataset(
            [dataset_collection.src_train_f, dataset_collection.test_train_f]
        )
      dataset_collection.test_f.data = merge_dataset(
          [dataset_collection.test_f]
      )
      dataset_collection.test_f_multi.data = merge_dataset(
          [dataset_collection.test_f_multi]
      )

  # MlFlow Logger
  if args.exp.logging:
    experiment_name = f'{args.model.name}/{args.dataset.name}'
    tags = [s for s in args.exp.tags.split(',') if s]
    mlf_logger = FilteringWandbLogger(
        filter_submodels=[],
        name=experiment_name,
        project='causal_over_time',
        tags=tags,
    )
  else:
    mlf_logger = None

  # Nominator (treatment propensity network)
  propensity_treatment = instantiate(
      args.model.propensity_treatment,
      args,
      dataset_collection,
      _recursive_=False,
  )
  mlf_logger.log_hyperparams(propensity_treatment.hparams)
  propensity_treatment.fit()

  # Denominator (history propensity network)
  propensity_history = instantiate(
      args.model.propensity_history, args, dataset_collection, _recursive_=False
  )
  mlf_logger.log_hyperparams(propensity_history.hparams)
  propensity_history.fit()

  # Initialisation & Training of Encoder
  msm_regressor = instantiate(
      args.model.msm_regressor,
      args,
      propensity_treatment,
      propensity_history,
      dataset_collection,
      _recursive_=False,
  )
  mlf_logger.log_hyperparams(msm_regressor.hparams)
  msm_regressor.fit()
  encoder_results = {}

  if hasattr(
      dataset_collection, 'test_cf_one_step'
  ):  # Test one_step_counterfactual rmse
    test_rmse_orig, test_rmse_all, test_rmse_last = (
        msm_regressor.get_normalised_masked_rmse(
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
        'encoder_test_rmse_all': test_rmse_all,
        'encoder_test_rmse_orig': test_rmse_orig,
        'encoder_test_rmse_last': test_rmse_last,
    }
  elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
    test_rmse_orig, test_rmse_all = msm_regressor.get_normalised_masked_rmse(
        dataset_collection.test_f
    )
    logger.info(
        '%s',
        f'Test normalised RMSE (all): {test_rmse_all}; '
        f'Test normalised RMSE (orig): {test_rmse_orig}.'
    )
    encoder_results = {
        # 'encoder_val_rmse_all': val_rmse_all,
        # 'encoder_val_rmse_orig': val_rmse_orig,
        'encoder_test_rmse_all': test_rmse_all,
        'encoder_test_rmse_orig': test_rmse_orig,
    }

  if args.exp.logging:
    mlf_logger.log_metrics(encoder_results)
  results.update(encoder_results)

  test_rmses = {}
  if hasattr(
      dataset_collection, 'test_cf_treatment_seq'
  ):  # Test n_step_counterfactual rmse
    test_rmses = msm_regressor.get_normalised_n_step_rmses(
        dataset_collection.test_cf_treatment_seq
    )
  elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
    test_rmses = msm_regressor.get_normalised_n_step_rmses(
        dataset_collection.test_f_multi
    )
  test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

  logger.info('%s', f'Test normalised RMSE (n-step prediction): {test_rmses}')
  decoder_results = {
      ('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()
  }

  if args.exp.logging:
    mlf_logger.log_metrics(decoder_results)
  results.update(decoder_results)

  wandb.finish()

  return results


if __name__ == '__main__':
  main(DictConfig({}))
