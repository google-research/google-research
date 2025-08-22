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

"""Compares the metrics from simulation vs. differentiable computation.

After model has been trained using main.py and predictions have been saved,
this script can be used to further evaluate the predictions under various
configurations (e.g. cost tradeoffs) and simulations.
"""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import argparse
import collections
import itertools
import multiprocessing as mp
import pickle
import pprint
import time
from typing import Dict

from data_formatting.datasets import get_favorita_data
from data_formatting.datasets import get_m3_data
from lib.evaluator import Evaluator
from lib.naive_scaling_baseline import NaiveScalingBaseline
from main import get_full_test_metrics
from main import get_learned_alpha
import numpy as np
import pandas as pd
import scipy
import torch
from utils.log_utils import get_summary
from utils.log_utils import tprint
import wandb


def compute_safety_stock(
    service_level,
    prior_preds,
    t,
    y,
):
  """Computes the amount of safety stock assuming normally distributed errors.

  Args:
    service_level: float target service level
    prior_preds: dictionary mapping each timepoint to predictions for its demand
    t: current timepoint
    y: entire pandas series corresponding to demand

  Returns:
    float value for quantity of safety stock
  """
  squared_errs = []
  for prev_t in range(t):
    if prev_t in prior_preds:
      errs = list([(y[prev_t] - prd) ** 2 for prd in prior_preds[prev_t]])
      squared_errs += errs
  if not squared_errs:
    return 0
  std_e = np.sqrt(np.mean(squared_errs))
  ss_t = scipy.stats.norm.ppf(service_level) * std_e
  return ss_t


def get_lagged_val(lag, prev_vals, default_val=0):
  if len(prev_vals) >= lag:
    return prev_vals[-lag]
  else:
    return default_val


def eval_simulation(
    wandb_log,
    parallel,
    num_workers,
    dataset_name,
    model_name,
    optimization_obj,
    single_rollout,
    max_steps,
    N,
    preds_path,
    model_path,
    project_name,
    run_name,
    tags,
    batch_size,
    just_convert_to_cpu,
    device_name,
    unit_holding_costs,
    unit_stockout_costs,
    unit_var_o_costs,
    do_scaling,
):
  """Given predictions, simulate inventory system and evaluate performance.

  Also evaluate performance computed by differentiable computation.

  Args:
    wandb_log: whether to log to wandb
    parallel: whether to perform evaluation in parallel
   num_workers: number of workers for parallel dataloading
   dataset_name: name of dataset
   model_name: name of model
   optimization_obj: name of optimization objective
   single_rollout: whether to perform a single rollout (vs. double rollout)
   max_steps: num gradient steps per timepoint per batch
   N: number of series to include
    preds_path: path to saved predictions
    model_path: path to saved model checkpoint
    project_name: name of project (for logging on wandb)
    run_name: name of run (for logging on wandb)
    tags: tags (for logging on wandb)
    batch_size: number of series per batch
    just_convert_to_cpu: whether to move tensors from GPU to CPU
    device_name: device to perform computation on
    unit_holding_costs: list of costs per unit held
    unit_stockout_costs: list of costs per unit stockout
    unit_var_o_costs: list of costs per unit order variance
    do_scaling: whether to perform additional scaling (relevant for sktime)

  Returns:
    simulated metrics and metrics from differentiable computation
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
          train_prop=None,
          val_prop=None,
          batch_size=None,
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
    valid_t_start = None
    test_t_start = None
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
  evaluator = Evaluator(0, scale01, device, target_dims)

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

    complete_dataset = dataset_factory.complete_dataset
    metrics = {}
    for i in range(len(complete_dataset)):
      print('series: ', i)
      series = complete_dataset[i]
      x = series['x']
      x_offset = series['x_offset']
      x_scale = series['x_scale']
      preds = test_preds[i, :, :, :]
      x_scaled = (x * x_scale + x_offset)[test_t_min:, 0]
      preds_scaled = preds * x_scale + x_offset
      preds_scaled = preds_scaled[:, :, 0]
      preds_scaled = torch.maximum(preds_scaled, torch.FloatTensor([0.0]))

      # mse
      mse = ((x_scaled - preds_scaled[:, 0]) ** 2).mean()

      # compute inventory performance
      net_inventory_levels = []
      orders = []
      safety_stocks = []
      lead_forecast_errors = []
      lead_forecasts = []
      prior_preds = collections.defaultdict(list)
      ip_t = 0
      i_t = 0
      w_t = 0
      o_t = 0
      for t in range(x_scaled.shape[0]):
        true_lead_demand = x_scaled[t : t + lead_time].sum()
        pred_lead_demand = preds_scaled[t].sum()

        lead_forecast_errors.append(pred_lead_demand - true_lead_demand)
        lead_forecasts.append(pred_lead_demand)

        # safety stock, updated in Newsvendor fashion
        ss_t = compute_safety_stock(
            target_service_level, prior_preds, t, x_scaled
        )
        safety_stocks.append(ss_t)
        for ti, val in zip(range(t, t + lead_time), preds_scaled[t]):
          prior_preds[ti].append(val)

        # inventory position
        d_t = x_scaled[t]
        ip_t = ip_t + o_t - d_t
        o_t_minus_lead = get_lagged_val(lead_time, orders, default_val=0)
        i_t = i_t + o_t_minus_lead - d_t
        w_t = w_t + o_t - o_t_minus_lead

        # orders
        o_t = pred_lead_demand + ss_t - ip_t

        orders.append(o_t)
        net_inventory_levels.append(i_t)
      var_o = np.var(orders)
      net_inventory_levels = np.array(net_inventory_levels)
      achieved_service_level = (np.array(net_inventory_levels) >= 0).mean()
      holding_cost = (
          unit_holding_cost * np.maximum(net_inventory_levels, 0).sum()
      )
      stockout_cost = (
          unit_stockout_cost * np.maximum(-net_inventory_levels, 0).sum()
      )

      metrics['var_o'] = metrics.get('var_o', []) + [var_o]
      metrics['achieved_service_level'] = metrics.get(
          'achieved_service_level', []
      ) + [achieved_service_level]
      metrics['holding_cost'] = metrics.get('holding_cost', []) + [holding_cost]
      metrics['stockout_cost'] = metrics.get('stockout_cost', []) + [
          stockout_cost
      ]
      metrics['mse'] = metrics.get('mse', []) + [mse]
    simulated_metrics = metrics

    # now, see what neural version gets
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
        batch_size=batch_size,
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
    differentiable_metrics = test_metrics

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
    return simulated_metrics, differentiable_metrics


def main():
  mp.set_start_method('spawn')
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
  parser.add_argument('--device', type=str, default='cpu')
  parser.add_argument('--batch_size', type=int, default=100)
  parser.add_argument('--do_scaling', action='store_true')
  parser.add_argument('--just_convert_to_cpu', action='store_true')

  args = parser.parse_args()
  eval_simulation(
      wandb_log=True,
      parallel=args.parallel,
      num_workers=args.num_workers,
      dataset_name=args.dataset_name,
      model_name=args.model_name,
      optimization_obj=args.optimization_obj,
      single_rollout=args.single_rollout,
      max_steps=args.max_steps,
      N=args.N,
      preds_path=args.preds_path,
      model_path=args.model_path,
      project_name=args.project_name,
      run_name=args.run_name,
      tags=args.tags,
      batch_size=args.batch_size,
      just_convert_to_cpu=args.just_convert_to_cpu,
      device_name=args.device,
      unit_holding_costs=args.unit_holding_costs,
      unit_stockout_costs=args.unit_stockout_costs,
      unit_var_o_costs=args.unit_var_o_costs,
      do_scaling=args.do_scaling,
  )


if __name__ == '__main__':
  main()
