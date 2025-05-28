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

"""Utilities."""

import gzip
import logging
import os
import pickle as pkl
import time

import numpy as np
from src import ROOT_PATH
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_exp_status(project_name, tag, expname, seed, args=None):
  """Check experiment status."""
  api = wandb.Api()
  runs = api.runs(
      project_name,
      filters={
          '$and': [
              {'tags': {'$eq': tag}},
              {'displayName': {'$eq': expname}},
              {'config.exp/seed': {'$eq': seed}},
          ]
      },
  )

  if tag.startswith('conclude_tcs_cancer_sim'):
    runs = api.runs(
        project_name,
        filters={
            '$and': [
                {'tags': {'$eq': tag}},
                {'displayName': {'$eq': expname}},
                {'config.exp/seed': {'$eq': seed}},
                {'config.dataset/coeff': {'$eq': args.dataset.src_coeff}},
            ]
        },
    )

  if runs:
    if 'm5_real_foods_household' in expname:
      expname = expname.replace('5k', '10k')
      runs = api.runs(
          project_name,
          filters={
              '$and': [
                  {'tags': {'$eq': tag}},
                  {'displayName': {'$eq': expname}},
                  {'config.exp/seed': {'$eq': seed}},
              ]
          },
      )
    elif '_srctest' in expname:
      expname = expname.replace('_srctest', '')
      runs = api.runs(
          project_name,
          filters={
              '$and': [
                  {'tags': {'$eq': tag}},
                  {'displayName': {'$eq': expname}},
                  {'config.exp/seed': {'$eq': seed}},
              ]
          },
      )
  for run in runs:
    if run.state in ['running', 'finished']:
      return run, run.state
  return None, 'notstarted'


def load_saved_data(
    dataset_name,
    seed,
    use_few_shot,
    max_number,
    few_shot_sample_num,
    gt_causal_prediction_for=None,
):
  """Load saved data."""
  _ = max_number
  if dataset_name.startswith('mimic3_synthetic'):
    if dataset_name.endswith('_srctest'):
      data_dir_path = 'multirun/230910_mimicsyn_all_data/mimic3_synthetic_age_domain_0-3_all_srctest/CRN_NONCAUSAL_TROFF/{}/data'.format(
          seed
      )
    else:
      data_dir_path = 'multirun/230814_mimicsyn_all_data/mimic3_synthetic_age_domain_0-3_all/CRN_NONCAUSAL_TROFF/{}/data'.format(
          seed
      )
    data_path = os.path.join(ROOT_PATH, data_dir_path, 'dataset_collection.pt')
    finish_flag_path = os.path.join(ROOT_PATH, data_dir_path, 'finished.txt')
    if os.path.exists(data_dir_path):
      while not os.path.exists(finish_flag_path):
        logger.info(
            '%s', 'Waiting for data at {} to finish saving...'.format(data_path)
        )
        time.sleep(60)
    try:
      f = open(data_path, 'rb')
      dataset_collection = pkl.load(f)
    except FileNotFoundError:
      try:
        f = gzip.open(data_path, 'rb')
        dataset_collection = pkl.load(f)
      except FileNotFoundError:
        return None
    # with gzip.open(data_path, 'rb') as f:
    #     dataset_collection = pkl.load(f)
    if few_shot_sample_num > 0:
      if isinstance(few_shot_sample_num, float):
        test_tr_num = int(
            len(dataset_collection.test_train_f) * few_shot_sample_num
        )
      elif isinstance(few_shot_sample_num, int):
        test_tr_num = few_shot_sample_num
      else:
        raise NotImplementedError()
      total = len(dataset_collection.test_train_f)
      if test_tr_num < total:
        sampled_idx = np.random.choice(total, size=test_tr_num, replace=False)
        for k in dataset_collection.test_train_f.data:
          if len(dataset_collection.test_train_f.data[k]) == total:
            dataset_collection.test_train_f.data[k] = (
                dataset_collection.test_train_f.data[k][sampled_idx]
            )

    if use_few_shot:
      dataset_collection.train_f = dataset_collection.test_train_f
      dataset_collection.val_f = dataset_collection.test_val_f

    if gt_causal_prediction_for is not None:
      if dataset_collection.train_f.gt_causal_prediction_for is None:
        dataset_collection.train_f.mask_features(gt_causal_prediction_for)
        dataset_collection.val_f.mask_features(gt_causal_prediction_for)
        dataset_collection.test_cf_one_step.mask_features(
            gt_causal_prediction_for
        )
        dataset_collection.test_cf_treatment_seq.mask_features(
            gt_causal_prediction_for
        )
      else:
        raise AssertionError('loaded data has been masked!')
    f.close()
    return dataset_collection
  else:
    return None
