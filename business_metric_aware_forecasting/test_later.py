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

"""Test a collection of model checkpoints.

For each timestep, loads the corresponding model checkpoint and makes
predictions using the checkpoint.
After predictions are made for all timesteps,
"""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import argparse
import copy
import datetime
import multiprocessing as mp
import os
import pickle
import time

from data_formatting.datasets import get_favorita_data
from lib.encoder_decoder import LstmEncoderLstmDecoder
from lib.evaluator import Evaluator
from lib.naive_scaling_baseline import NaiveScalingBaseline
from lib.rolling_evaluation_dataset import TimeIndexedDataset
from main import get_batch_test_metrics
import pandas as pd
import torch
from utils.viz_utils import plot_series_preds
import wandb


def tprint(msg):
  now = datetime.datetime.now()
  print(f'[{now}]\t{msg}')


def get_blank_model(
    model_name,
    forecasting_horizon,
    periodicity,
    device,
    target_dims,
    input_size,
    hidden_size,
    num_layers,
    cuda=True,
):
  """Create a blank model (to later be loaded with checkpointed weights).

  Args:
    model_name: name of model
    forecasting_horizon: number of timepoints to forecast
    periodicity: number of timepoints in a season
    device: device to perform computations on
    target_dims: dimensions corresponding to target
    input_size: size of input
    hidden_size: number of hidden units
    num_layers: number of layers
    cuda: whether to use GPU

  Returns:
    Blank model with desired structure
  """
  if model_name == 'naive_seasonal':
    model_class = NaiveScalingBaseline
    model_args = {
        'forecasting_horizon': forecasting_horizon,
        'periodicity': periodicity,
        'device': device,
        'target_dims': target_dims,
    }
  elif model_name == 'lstm_windowed':
    model_class = LstmEncoderLstmDecoder
    model_args = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'forecasting_horizon': forecasting_horizon,
        'training_prediction': 'teacher_forcing',
        'teacher_forcing_ratio': 0.5,
        'device': device,
        'target_dims': target_dims,
    }
  elif model_name == 'naive_alpha1_seasonal_baseline':
    model_class = NaiveScalingBaseline
    model_args = {
        'forecasting_horizon': forecasting_horizon,
        'periodicity': periodicity,
        'device': device,
        'frozen': True,
        'init_alpha': 1,
        'target_dims': target_dims,
    }
  elif model_name == 'naive_alpha1_baseline':
    model_class = NaiveScalingBaseline
    model_args = {
        'forecasting_horizon': forecasting_horizon,
        'periodicity': 1,
        'device': device,
        'frozen': True,
        'init_alpha': 1,
        'target_dims': target_dims,
    }
  else:
    raise NotImplementedError('invalid model name: ', model_name)
  model = model_class(**model_args).to(device)
  if cuda:
    model = torch.nn.DataParallel(model)
  return model


def do_prediction(model, cur_batch, device, start):
  """Make predictions on a batch.

  Args:
    model: trained model
    cur_batch: batch to make predictions on
    device: device to perform computations on
    start: starting unix time (to keep track of runtime)

  Returns:
    predictions
  """
  tprint(f'testing batch... ({round(time.time() - start, 2)} sec)')
  test_N, test_T_input, test_window, test_D_input = cur_batch[
      'test_inputs'
  ].shape
  test_N, test_T_output, test_fh, test_D_target = cur_batch[
      'test_targets'
  ].shape
  test_batch = {
      'model_inputs': (
          cur_batch['test_inputs']
          .view(test_N * test_T_input, test_window, test_D_input)
          .to(device)
      ),  # model inputs does batching over all encoding points
      'model_targets': (
          cur_batch['test_targets']
          .view(test_N * test_T_output, test_fh, test_D_target)
          .to(device)
      ),
  }
  with torch.no_grad():
    test_preds = model(test_batch)
    if len(test_preds.shape) == 4:
      print('test_preds dim 4 now... in favorita naive scaling it was 3...')
    else:
      assert len(test_preds.shape) == 3
      test_preds = test_preds.unsqueeze(1)
  return test_preds


def test_checkpts(
    dataset_factory,
    input_window_size,
    optimization_obj='mse',
    model_name='naive_seasonal',
    unit_holding_cost=1,
    unit_stockout_cost=1,
    unit_var_o_cost=1.0 / 100000,
    naive_model=None,
    scale_by_naive_model=False,
    quantile_loss=None,
    forecasting_horizon=12,
    hidden_size=50,
    num_layers=2,
    lead_time=6,
    scale01=True,
    target_service_level=0.95,
    max_steps=1000,
    test_t_min=36,
    test_t_max=144,
    idx_range=None,
    target_dims=(0,),
    input_size=1,
    use_wandb=True,
    device=torch.device('cpu'),
    periodicity=12,
    exp_folder=None,
    batch_size=500,
    parallel=False,
    valid_t_start=None,
    test_t_start=None,
    no_safety_stock=False,
):
  """Loads model checkpoints, makes predictions, and evaluates performance.

  Args:
    dataset_factory: factory for datasets
    input_window_size: encoding input window size
    optimization_obj: name of optimization objective
    model_name: name of model
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    naive_model: baseline model
    scale_by_naive_model: whether to scale performance by baseline
    quantile_loss: quantile loss object, if relevant
    forecasting_horizon: number of timepoints to forecast
    hidden_size: number of hidden units
    num_layers: number of layers
    lead_time: number of timepoints it takes for inventory to come in
    scale01: whether model predictions are scaled between 0 and 1
    target_service_level: service level for safety stock computation
    max_steps: num steps per batch per timepoint
    test_t_min: minimum timepoint for evaluation
    test_t_max: maximum timepoint for evaluation
    idx_range: range of indices of series of interest
    target_dims: dimensions corresponding to target
    input_size: size of input
    use_wandb: whether to use wandb for logging
    device: device for computation
    periodicity: number of timepoints per period
    exp_folder: folder for model checkpoints and predictions
    batch_size: number of series per batch
    parallel: whether to do parallel computation
    valid_t_start: start timepoint for validation time period
    test_t_start: start timepoint for test time period
    no_safety_stock: whether to include safety stock in order up to policy

  Returns:
    performance metrics dictionary
  """
  pool = None
  num_proc = int(mp.cpu_count())
  print('Number of processors: ', num_proc)
  num_workers = int(num_proc / 2)
  if parallel:
    pool = mp.Pool(num_proc)

  start = time.time()
  evaluator = Evaluator(0, scale01, device, list(target_dims),
                        no_safety_stock=no_safety_stock)
  complete_dataset = dataset_factory.complete_dataset
  all_test_preds = []
  for t in range(test_t_min, test_t_max):
    tprint(f'---- t: {t} ---- ({round(time.time() - start, 2)} sec)')
    # load model checkpoint
    model = get_blank_model(
        model_name,
        forecasting_horizon,
        periodicity,
        device,
        list(target_dims),
        input_size,
        hidden_size,
        num_layers,
    )
    model_path = os.path.join(exp_folder, f'model_{t}.pt')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # for each timepoint, make prediction
    time_idx = t - input_window_size
    test_dataset = TimeIndexedDataset(
        complete_dataset, time_idx, prefix='test_'
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_datagen = iter(test_dataloader)
    cur_batch = next(test_datagen, None)
    test_batch_preds = []
    batch_idx = 0
    pause_freq = 10
    while cur_batch is not None:
      tprint(f'kicked off batch: {batch_idx}')
      if pool is not None:
        test_batch_preds.append((
            batch_idx,
            pool.apply_async(do_prediction, [model, cur_batch, device, start]),
        ))
        if batch_idx % pause_freq == 0:
          for bi, preds in test_batch_preds:
            while not preds.ready():
              tprint(f'waiting for batch: {bi}')
              time.sleep(30)

      else:
        test_batch_preds.append(
            (batch_idx, do_prediction(model, cur_batch, device, start))
        )
      cur_batch = next(test_datagen, None)
      batch_idx += 1

    if pool is not None:
      for bi, preds in test_batch_preds:
        while not preds.ready():
          tprint(f'waiting for batch: {bi}')
          time.sleep(30)

    batch_preds = []
    for bi, preds in test_batch_preds:
      print(bi)
      if pool is not None:
        batch_preds.append(preds.get())
      else:
        batch_preds.append(preds)
    test_preds_t = torch.concat(batch_preds, dim=0)
    tensor_t_path = os.path.join(exp_folder, f'test_preds_t{t}.pt')
    torch.save(test_preds_t, tensor_t_path)
    all_test_preds.append(test_preds_t)
  test_preds = torch.cat(all_test_preds, dim=1)
  tensor_path = os.path.join(exp_folder, 'test_preds.pt')
  torch.save(test_preds, tensor_path)

  tprint('Given predictions, evaluate the performance...')
  test_dataloader = torch.utils.data.DataLoader(
      dataset=test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
  )
  test_datagen = iter(test_dataloader)
  cur_batch = next(test_datagen, None)
  batches_test_metrics = []
  start_idx = 0
  pool = None
  if parallel:
    num_proc = int(mp.cpu_count())
    pool = mp.Pool(num_proc)
    print('Number of processors: ', num_proc)
  test_batch = None
  batch_test_preds = None
  while cur_batch is not None:
    tprint(f'kicking off batch with start_idx: {start_idx}')
    test_batch = cur_batch
    batch_test_preds = test_preds[
        start_idx : start_idx + cur_batch['inputs'].shape[0]
    ]
    if pool is None:
      batches_test_metrics.append((
          start_idx,
          get_batch_test_metrics(
              cur_batch,
              batch_test_preds,
              test_t_min,
              valid_t_start,
              test_t_start,
              device,
              evaluator,
              target_service_level,
              lead_time,
              unit_holding_cost,
              unit_stockout_cost,
              unit_var_o_cost,
              naive_model,
              scale_by_naive_model,
              quantile_loss,
              start_idx,
          ),
      ))
    else:
      batches_test_metrics.append((
          start_idx,
          pool.apply_async(
              get_batch_test_metrics,
              [
                  cur_batch,
                  batch_test_preds,
                  test_t_min,
                  valid_t_start,
                  test_t_start,
                  device,
                  evaluator,
                  target_service_level,
                  lead_time,
                  unit_holding_cost,
                  unit_stockout_cost,
                  unit_var_o_cost,
                  copy.deepcopy(naive_model),
                  scale_by_naive_model,
                  quantile_loss,
                  start_idx,
              ],
          ),
      ))
    start_idx += cur_batch['inputs'].shape[0]
    cur_batch = next(test_datagen, None)
    print(type(cur_batch))

  if pool is not None:
    for si, test_metrics in batches_test_metrics:
      while not test_metrics.ready():
        print('waiting for batch with start_idx: ', si)
        time.sleep(100)

  all_test_metrics = {}
  for si, test_metrics in batches_test_metrics:
    if pool is not None:
      test_metrics = test_metrics.get()
    print(si)
    for k, v in test_metrics.items():
      all_test_metrics[k] = all_test_metrics.get(k, []) + [v]
  test_metrics = {
      k: torch.concat(v, dim=0).mean()
      for (k, v) in all_test_metrics.items()
      if 'inventory_values' not in k
  }
  log_dict = {
      'Test ' + k: v.detach().item()
      for (k, v) in test_metrics.items()
      if 'inventory_values' not in k
  }

  if use_wandb:
    wandb.log(log_dict)
    if test_batch is not None and batch_test_preds is not None:
      fig = plot_series_preds(test_batch, batch_test_preds)
    wandb.log({'series 0 predictions': fig})

  if parallel:
    pool.close()
    pool.join()

  tprint('getting summary')
  summary = {
      'model_name': model_name,
      'optimization objective': optimization_obj,
      'max_steps': max_steps,
      'runtime': time.time() - start,
      'unit_holding_cost': unit_holding_cost,
      'unit_stockout_cost': unit_stockout_cost,
      'unit_var_o_cost': unit_var_o_cost,
  }
  test_obj = test_metrics[optimization_obj].detach().item()
  final_metrics = test_metrics
  final_metric_names = [
      'mse',
      'mpe',
      'smape',
      'mase',
      'holding_cost',
      'var_o',
      'var_o_cost',
      'prop_neg_orders',
      'achieved_service_level',
      'soft_achieved_service_level',
      'stockout_cost',
      'rms',
      'scaled_rms',
      'rel_rms_avg',
      'rel_rms_2',
      'rel_rms_3',
      'rel_rms_5',
      'rel_rms_logsumexp',
      'total_cost',
      'rel_rms_stockout_2',
      'rel_rms_stockout_3',
      'rel_rms_stockout_5',
      'quantile_loss',
      'scale_by_naive_model',
  ]
  for name in final_metric_names:
    if name in final_metrics:
      val = final_metrics[name]
      summary[name] = val.detach().item()

  if 'naive' in model_name:
    try:
      summary['learned alpha'] = model.alpha.detach().item()
    except AttributeError as e:
      print(e)
      summary['learned alpha'] = model.module.alpha.detach().item()
  if quantile_loss:
    summary['quantile'] = quantile_loss.get_quantile().detach().item()

  summary['naive_model'] = str(naive_model)

  if use_wandb:
    wandb.log({
        'idx_range_test_perfs': wandb.Table(dataframe=pd.DataFrame([summary])),
        'idx_range_test_obj': test_obj,
    })

  test_results = {
      'test_metrics': test_metrics,
      'test_preds': test_preds,
      'test_batch': cur_batch,
      'idx_range': idx_range,
      'summary': summary,
      'test_obj': test_obj,
  }
  pickle.dump(test_results, open(f'{exp_folder}/test_results.pkl', 'wb'))
  return test_results


def main():
  mp.set_start_method('spawn')

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', choices=['m3', 'favorita'])
  parser.add_argument(
      '--model_name',
      choices=['naive_seasonal', 'lstm_windowed', 'naive_alpha1_baseline'],
  )
  parser.add_argument(
      '--optimization_obj',
      choices=[
          'mse',
          'total_cost',
          'rel_rms_stockout_2',
          'mpe',
          'smape',
          'mase',
          'soft_holding_cost',
          'holding_cost',
          'var_o',
          'soft_achieved_service_level',
          'rms',
          'quantile_loss',
          'scaled_rms',
          'rel_rms_avg',
          'rel_rms_2',
          'quantile_loss_and_rel_rms_stockout_2',
          'quantile_loss_and_total_cost',
          # 'no_learning'
      ],
  )
  parser.add_argument('--unit_holding_cost', type=int)
  parser.add_argument('--unit_stockout_cost', type=int)
  parser.add_argument('--unit_var_o_cost', type=float)
  parser.add_argument('--hidden_size', type=int, default=None)
  parser.add_argument('--max_steps', type=int, default=None)
  parser.add_argument('--batch_size', type=int, default=None)
  parser.add_argument('--exp_folder', type=str)
  parser.add_argument('--single_rollout', action='store_true')
  parser.add_argument('--parallel', action='store_true')
  parser.add_argument('--N', type=int, default=None)
  parser.add_argument('--device', type=str, default='cpu')
  parser.add_argument('--no_safety_stock', action='store_true')
  args = parser.parse_args()

  use_wandb = False

  dataset_name = args.dataset_name
  if dataset_name == 'favorita':
    test_t_max = 396
    valid_t_start = 334
    test_t_start = 364

    test_t_min = 180
    forecasting_horizon = 30
    if args.single_rollout:
      forecasting_horizon = 7
    input_window_size = 90

    embedding_dim = 10
    input_size = 20 * embedding_dim
    num_layers = 2
    lead_time = 7
    scale01 = True
    target_service_level = 0.95
    N = args.N
    periodicity = 7
    data_fpath = '../data/favorita/favorita_tensor_full.npy'
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
  else:
    raise NotImplementedError('Dataset not implemented: ', dataset_name)

  device = torch.device(args.device)
  naive_model = NaiveScalingBaseline(
      forecasting_horizon=lead_time,
      init_alpha=1.0,
      periodicity=periodicity,
      frozen=True,
  ).to(device)

  test_checkpts(
      dataset_factory,
      input_window_size,
      optimization_obj=args.optimization_obj,
      model_name=args.model_name,
      unit_holding_cost=args.unit_holding_cost,
      unit_stockout_cost=args.unit_stockout_cost,
      unit_var_o_cost=args.unit_var_o_cost,
      naive_model=naive_model,
      scale_by_naive_model=False,
      quantile_loss=None,
      forecasting_horizon=forecasting_horizon,
      hidden_size=args.hidden_size,
      num_layers=num_layers,
      lead_time=lead_time,
      scale01=scale01,
      target_service_level=target_service_level,
      max_steps=args.max_steps,
      test_t_min=test_t_min,
      test_t_max=test_t_max,
      idx_range=None,
      target_dims=[0],
      input_size=input_size,
      use_wandb=use_wandb,
      device=device,
      periodicity=periodicity,
      exp_folder=args.exp_folder,
      batch_size=args.batch_size,
      parallel=args.parallel,
      valid_t_start=valid_t_start,
      test_t_start=test_t_start,
      no_safety_stock=args.no_safety_stock,
  )


if __name__ == '__main__':
  main()
