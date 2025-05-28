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

"""Given a set of predictions, computes evaluation metrics.

After model has been trained using main.py and predictions have been saved,
this script can be used to further evaluate the predictions under various
configurations (e.g. cost tradeoffs).
"""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import argparse
import itertools
import multiprocessing
import os
import pickle
import pprint
import time
from data_formatting.datasets import get_favorita_data
from data_formatting.datasets import get_m3_data
from lib.evaluator import Evaluator
from lib.naive_scaling_baseline import NaiveScalingBaseline
from main import get_full_test_metrics
from main import get_learned_alpha
import numpy as np
import pandas as pd
import torch
from utils.log_utils import get_summary
from utils.log_utils import tprint
import wandb


def eval_preds(
    wandb_log,
    parallel,
    num_workers,
    dataset_name,
    model_name,
    optimization_obj,
    single_rollout,
    no_safety_stock,
    max_steps,
    N,
    preds_path,
    model_path,
    project_name,
    run_name,
    tags,
    just_convert_to_cpu,
    device_name,
    cpu_checkpt_folder,
    unit_holding_costs,
    unit_stockout_costs,
    unit_var_o_costs,
    do_scaling,
):
  """Evaluate model predictions.

  Args:
    wandb_log: whether to log the metrics to wandb
    parallel: whether to evaluate in parallel
    num_workers: number of workers for parallel dataloading
    dataset_name: name of dataset
    model_name: name of model
    optimization_obj: name of optimization obj
    single_rollout: whether single rollout (vs. double rollout)
    no_safety_stock: whether to include safety stock in order-up-to policy
    max_steps: num steps per timepoint per batch
    N: number of series
    preds_path: path to tensor of predictions
    model_path: path to model checkpoint
    project_name: name of project (for wandb and logging)
    run_name: name of run (for wandb and logging)
    tags: list of tags describing experiment
    just_convert_to_cpu: whether to move predictions from gpu to cpu
    device_name: device to perform computations on
    cpu_checkpt_folder: folder to put cpu checkpoints
    unit_holding_costs: list of costs per unit held
    unit_stockout_costs: list of costs per unit stockout
    unit_var_o_costs: list of costs per unit order variance
    do_scaling: whether to additionally scale predictions (for sktime only)

  Raises:
    NotImplementedError: if a preds_path with unsupported filetype is provided
  """
  print('preds path: ', preds_path)
  print('model path: ', model_path)
  print('tags: ', tags)
  if 'sktime' not in tags:
    assert f'{dataset_name}_{model_name}_{optimization_obj}' in preds_path

  target_dims = [0]
  if dataset_name == 'm3':
    test_t_min = 36
    test_t_max = 144
    valid_t_start = 72
    test_t_start = 108

    forecasting_horizon = 12
    input_window_size = 24

    lead_time = 6
    scale01 = True
    target_service_level = 0.95
    periodicity = 12
    data_fpath = '../data/m3/m3_industry_monthly_shuffled.csv'
    idx_range = (20, 334)

    if not just_convert_to_cpu:
      tprint('getting dataset factory...')
      dataset_factory = get_m3_data(
          forecasting_horizon=forecasting_horizon,
          minmax_scaling=scale01,
          input_window_size=input_window_size,
          csv_fpath=data_fpath,
          default_nan_value=1e15,
          rolling_evaluation=True,
          idx_range=idx_range,
          N=N,
      )
  else:
    assert dataset_name == 'favorita'
    test_t_max = 396
    valid_t_start = 334
    test_t_start = 364
    test_t_min = 180
    forecasting_horizon = 30
    if single_rollout:
      forecasting_horizon = 7
    input_window_size = 90

    lead_time = 7
    scale01 = True
    target_service_level = 0.95
    idx_range = None
    periodicity = 7
    data_fpath = '../data/favorita/favorita_tensor_full.npy'

    if not just_convert_to_cpu:
      tprint('getting dataset factory...')
      dataset_factory = get_favorita_data(
          forecasting_horizon=forecasting_horizon,
          minmax_scaling=scale01,
          input_window_size=input_window_size,
          data_fpath=data_fpath,
          default_nan_value=1e15,
          rolling_evaluation=True,
          N=N,
          test_t_max=test_t_max,
      )

  device = torch.device(device_name)
  naive_model = NaiveScalingBaseline(
      forecasting_horizon=lead_time,
      init_alpha=1.0,
      periodicity=periodicity,
      frozen=True,
  ).to(device)
  evaluator = Evaluator(
      0, scale01, device, target_dims, no_safety_stock=no_safety_stock
  )

  scale_by_naive_model = False
  quantile_loss = None
  use_wandb = True

  # Load predictions
  if preds_path.endswith('.pkl'):
    with open(preds_path, 'rb') as fin:
      test_preds = pickle.load(fin)
    if isinstance(test_preds, list):
      test_preds = torch.cat(test_preds, dim=1)
  elif preds_path.endswith('.npy'):
    test_preds = torch.from_numpy(np.load(preds_path))
    if len(test_preds.shape) == 3:
      test_preds = test_preds.unsqueeze(-1)
  elif preds_path.endswith('test_preds.pt'):
    test_preds = torch.load(preds_path)
  else:
    raise NotImplementedError('Unrecognized file type: ' + preds_path)
  test_preds = test_preds.to(device)
  print('shape of orig test_preds: ', test_preds.shape)
  test_preds = test_preds[:, :, :lead_time, :]
  print('shape of truncated test_preds: ', test_preds.shape)

  # Load alpha (if exists)
  learned_alpha = None
  if 'naive' in model_name and model_path is not None:
    checkpoint = torch.load(model_path)
    model_class = NaiveScalingBaseline
    model_args = {
        'forecasting_horizon': forecasting_horizon,
        'periodicity': periodicity,
        'device': device,
        'target_dims': target_dims,
    }
    model = model_class(**model_args)
    if 'cuda' in model_path:
      model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    learned_alpha = get_learned_alpha(
        model_name=model_name, per_series_models=None, model=model
    )

  if just_convert_to_cpu:
    cpu_checkpoint = {
        'test_preds': test_preds.cpu(),
        'learned_alpha': learned_alpha,
    }
    torch.save(
        cpu_checkpoint, os.path.join(cpu_checkpt_folder, 'cpu_checkpoint.pt')
    )
    tprint(
        'Saved CPU checkpoint: '
        + os.path.join(cpu_checkpt_folder, 'cpu_checkpoint.pt')
    )
    return

  unit_costs = list(
      itertools.product(
          unit_holding_costs,
          unit_stockout_costs,
          unit_var_o_costs,
      )
  )
  for unit_cost in unit_costs:
    unit_holding_cost, unit_stockout_cost, unit_var_o_cost = unit_cost
    print(f'============== {unit_cost} ==============')
    config = {
        'model_name': model_name,
        'unit_holding_cost': unit_holding_cost,
        'unit_stockout_cost': unit_stockout_cost,
        'unit_var_o_cost': unit_var_o_cost,
        'dataset_name': dataset_name,
        'optimization_obj': optimization_obj,
        'later_eval': True,
        'N': len(dataset_factory),
    }
    pprint.pprint(config)
    if wandb_log:
      wandb.init(
          name=run_name,
          project=project_name,
          reinit=True,
          tags=tags,
          config=config,
      )
    start = time.time()
    test_metrics, expanded_test_metrics = get_full_test_metrics(
        dataset_factory,
        test_preds,
        num_workers,
        parallel,
        device,
        test_t_min,
        valid_t_start,
        test_t_start,
        evaluator,
        target_service_level,
        lead_time,
        unit_holding_cost,
        unit_stockout_cost,
        unit_var_o_cost,
        naive_model,
        scale_by_naive_model,
        quantile_loss,
        use_wandb=False,
        sum_ct_metrics=True,
        do_scaling=do_scaling,
    )

    test_results = get_summary(
        test_metrics,
        model_name,
        optimization_obj,
        max_steps,
        start,
        unit_holding_cost,
        unit_stockout_cost,
        unit_var_o_cost,
        valid_t_start,
        learned_alpha,
        quantile_loss,
        naive_model,
        use_wandb,
        expanded_test_metrics,
        idx_range,
    )
    summary = test_results['summary']

    print('runtime: ', time.time() - start)

    if wandb_log:
      wandb.log(
          {
              'combined_test_perfs': wandb.Table(
                  dataframe=pd.DataFrame([summary])
              )
          }
      )
      wandb.finish()


def main():
  multiprocessing.set_start_method('spawn')
  parser = argparse.ArgumentParser()
  parser.add_argument('--parallel', action='store_true')
  parser.add_argument('--dataset_name', choices=['m3', 'favorita'])
  parser.add_argument('--model_name', type=str)
  parser.add_argument('--optimization_obj', type=str)
  parser.add_argument('--max_steps', type=int)
  parser.add_argument('--N', type=int, default=None)
  parser.add_argument('--num_workers', type=int, default=0)
  parser.add_argument('--preds_path', type=str)
  parser.add_argument('--model_path', type=str, default=None)
  parser.add_argument('--project_name', type=str)
  parser.add_argument('--run_name', type=str)
  parser.add_argument('--tags', type=str, action='append')
  parser.add_argument('--unit_holding_costs', type=int, action='append')
  parser.add_argument('--unit_stockout_costs', type=int, action='append')
  parser.add_argument('--unit_var_o_costs', type=float, action='append')
  parser.add_argument('--single_rollout', action='store_true')
  parser.add_argument('--no_safety_stock', action='store_true')
  parser.add_argument('--device', type=str, default='cpu')
  parser.add_argument('--do_scaling', action='store_true')
  parser.add_argument('--just_convert_to_cpu', action='store_true')
  parser.add_argument('--cpu_checkpt_folder', type=str, default='./')

  args = parser.parse_args()
  eval_preds(
      wandb_log=True,
      parallel=args.parallel,
      num_workers=args.num_workers,
      dataset_name=args.dataset_name,
      model_name=args.model_name,
      optimization_obj=args.optimization_obj,
      single_rollout=args.single_rollout,
      no_safety_stock=args.no_safety_stock,
      max_steps=args.max_steps,
      N=args.N,
      preds_path=args.preds_path,
      model_path=args.model_path,
      project_name=args.project_name,
      run_name=args.run_name,
      tags=args.tags,
      just_convert_to_cpu=args.just_convert_to_cpu,
      device_name=args.device,
      cpu_checkpt_folder=args.cpu_checkpt_folder,
      unit_holding_costs=args.unit_holding_costs,
      unit_stockout_costs=args.unit_stockout_costs,
      unit_var_o_costs=args.unit_var_o_costs,
      do_scaling=args.do_scaling,
  )


if __name__ == '__main__':
  main()
