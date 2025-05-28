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

"""Main script for training lib."""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import argparse
import datetime
import itertools
import json
import multiprocessing as mp
import os
import pickle
import pprint
import time

from data_formatting.datasets import get_favorita_data
from data_formatting.datasets import get_m3_data
from lib.encoder_decoder import LstmEncoderLstmDecoder
from lib.evaluator import Evaluator
from lib.naive_scaling_baseline import NaiveScalingBaseline
import numpy as np
import pandas as pd
import torch
from utils.log_utils import get_summary
from utils.log_utils import proc_print
from utils.log_utils import tprint
from utils.preprocessing_utils import collapse_first2dims
from utils.preprocessing_utils import get_mask_from_target_times
from utils.preprocessing_utils import get_rollout
from utils.quantile_loss import QuantileLoss
from utils.viz_utils import plot_series_preds
import wandb


def get_optimization_obj(metrics, optimization_obj, prefix=None):
  """Retrieves the optimization objective of interest.

  Args:
    metrics: dictionary containing all metrics.
    optimization_obj: name of the optimization objective of interest
    prefix: any existing prefix on all metrics (e.g. 'test_' in 'test_mse')

  Returns:
    the desired optimization objective

  Raises:
    KeyError: if given optimization objective name is not in the metrics dict
  """
  if prefix:
    metrics = {
        k.replace(prefix, ''): v
        for (k, v) in metrics.items()
        if k.startswith(prefix)
    }
  if 'quantile_loss_and_' in optimization_obj:
    other_obj = optimization_obj.replace('quantile_loss_and_', '')
    obj = metrics['quantile_loss'] + metrics[other_obj]
  elif (optimization_obj == 'quantile_loss') or (
      optimization_obj.startswith('fixed_quantile_loss_')
  ):
    obj = metrics['quantile_loss']
  elif optimization_obj in metrics:
    obj = metrics[optimization_obj]
  else:
    raise KeyError('unknown optimization objective: ', optimization_obj)
  return obj


def get_batch_test_metrics(
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
    sum_ct_metrics=False,
):
  """Retrieves the metrics for the batch.

  If there are train, validation, and test cutoff timepoints, then
  data are split and evaluation metrics are calculated separately.

  Args:
    cur_batch: batch data dict
    batch_test_preds: tensor of predictions for the batch
    test_t_min: minimum t used for general evaluation (start t for train split)
    valid_t_start: start t used for validation split
    test_t_start: start t used for test split
    device: device on which computation is performed
    evaluator: Evaluator object to be used for computing metrics
    target_service_level: target service level used in safety stock calculation
      if safety stock is used. Otherwise, has no effect.
    lead_time: number of timepoints for inventory to come in (lead-time)
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stock-out
    unit_var_o_cost: cost per unit order variance
    naive_model: model object to be used as naive comparison
    scale_by_naive_model: whether to scale metrics by naive model performance
    quantile_loss: quantile loss object, if used
    start_idx: index that the batch starts at
    sum_ct_metrics: whether to compute sums and counts, to later aggregate with
      other batches

  Returns:
    dictionary of metrics
  """
  tprint(f'evaluating batch with start_idx: {start_idx}')
  offset = int((test_t_min + 1) - (cur_batch['target_times'].min() - 1)) - 1
  if (
      valid_t_start is not None and test_t_start is not None
  ):  # split into tr, vl, te time period batches
    tprint(f'evaluating on tr, vl, and te: {start_idx}')
    offset_vl_t_start = int(valid_t_start - cur_batch['target_times'].min())
    offset_te_t_start = int(test_t_start - cur_batch['target_times'].min())
    tr_t_idxs = (offset, offset_vl_t_start)
    vl_t_idxs = (offset_vl_t_start, offset_te_t_start)
    te_t_idxs = (offset_te_t_start, -1)

    tr_batch = {
        k: (
            v[:, tr_t_idxs[0] : tr_t_idxs[1], :lead_time]
            if (len(v.shape) > 2) and (v.shape[2] >= lead_time) and k != 'x'
            else v
        )  # added equal to here
        for (k, v) in cur_batch.items()
    }
    tr_batch = {k: v.to(device) for (k, v) in tr_batch.items()}
    tr_batch['target_times_mask'] = tr_batch['target_times_mask'][
        :, :, :, :, tr_t_idxs[0] : tr_t_idxs[1]
    ].to(device)
    tr_batch['eval_inputs'] = tr_batch['inputs'].to(
        device
    )  # eval inputs does batching as normal
    tr_batch['eval_targets'] = tr_batch['targets'].to(device)
    tr_batch['eval_target_times_mask'] = tr_batch['target_times_mask'].to(
        device
    )
    tr_batch['max_t_cutoff'] = valid_t_start - 1

    vl_batch = {
        k: (
            v[:, vl_t_idxs[0] : vl_t_idxs[1], :lead_time]
            if (len(v.shape) > 2) and (v.shape[2] >= lead_time) and k != 'x'
            else v
        )
        for (k, v) in cur_batch.items()
    }
    vl_batch = {k: v.to(device) for (k, v) in vl_batch.items()}
    vl_batch['target_times_mask'] = vl_batch['target_times_mask'][
        :, :, :, :, vl_t_idxs[0] : vl_t_idxs[1]
    ].to(device)
    vl_batch['eval_inputs'] = vl_batch['inputs'].to(
        device
    )  # eval inputs does batching as normal
    vl_batch['eval_targets'] = vl_batch['targets'].to(device)
    vl_batch['eval_target_times_mask'] = vl_batch['target_times_mask'].to(
        device
    )
    vl_batch['max_t_cutoff'] = test_t_start - 1

    te_batch = {
        k: (
            v[:, te_t_idxs[0] : te_t_idxs[1], :lead_time]
            if (len(v.shape) > 2) and (v.shape[2] >= lead_time) and k != 'x'
            else v
        )
        for (k, v) in cur_batch.items()
    }
    te_batch = {k: v.to(device) for (k, v) in te_batch.items()}
    te_batch['target_times_mask'] = te_batch['target_times_mask'][
        :, :, :, :, te_t_idxs[0] : te_t_idxs[1]
    ].to(device)
    te_batch['eval_inputs'] = te_batch['inputs'].to(
        device
    )  # eval inputs does batching as normal
    te_batch['eval_targets'] = te_batch['targets'].to(device)
    te_batch['eval_target_times_mask'] = te_batch['target_times_mask'].to(
        device
    )

    # Compute metrics
    idx_base = tr_t_idxs[0]
    tr_idxs = (tr_t_idxs[0] - idx_base, tr_t_idxs[1] - idx_base)
    vl_idxs = (vl_t_idxs[0] - idx_base, vl_t_idxs[1] - idx_base)
    te_idxs = (te_t_idxs[0] - idx_base, batch_test_preds.shape[1])
    batch_test_preds_tr = batch_test_preds[:, tr_idxs[0] : tr_idxs[1]]
    batch_test_preds_vl = batch_test_preds[:, vl_idxs[0] : vl_idxs[1]]
    batch_test_preds_te = batch_test_preds[:, te_idxs[0] : te_idxs[1]]

    tprint(f'Computing test_tr_metrics for start_idx {start_idx}')
    test_tr_metrics = evaluator.compute_all_metrics(
        preds=batch_test_preds_tr,
        actual_batch=tr_batch,
        target_service_level=target_service_level,
        unit_holding_cost=unit_holding_cost,
        unit_stockout_cost=unit_stockout_cost,
        unit_var_o_cost=unit_var_o_cost,
        series_mean=False,
        naive_model=naive_model,
        scale_by_naive_model=scale_by_naive_model,
        quantile_loss=quantile_loss,
        rolling_eval=False,
    )
    test_tr_metrics = {'test_tr_' + k: v for (k, v) in test_tr_metrics.items()}

    tprint(f'Computing test_vl_metrics for start_idx {start_idx}')
    test_vl_metrics = evaluator.compute_all_metrics(
        preds=batch_test_preds_vl,
        actual_batch=vl_batch,
        target_service_level=target_service_level,
        unit_holding_cost=unit_holding_cost,
        unit_stockout_cost=unit_stockout_cost,
        unit_var_o_cost=unit_var_o_cost,
        series_mean=False,
        naive_model=naive_model,
        scale_by_naive_model=scale_by_naive_model,
        quantile_loss=quantile_loss,
        rolling_eval=False,
    )
    test_vl_metrics = {'test_vl_' + k: v for (k, v) in test_vl_metrics.items()}

    tprint(f'Computing test_te_metrics for start_idx {start_idx}')
    test_te_metrics = evaluator.compute_all_metrics(
        preds=batch_test_preds_te,
        actual_batch=te_batch,
        target_service_level=target_service_level,
        unit_holding_cost=unit_holding_cost,
        unit_stockout_cost=unit_stockout_cost,
        unit_var_o_cost=unit_var_o_cost,
        series_mean=False,
        naive_model=naive_model,
        scale_by_naive_model=scale_by_naive_model,
        quantile_loss=quantile_loss,
        rolling_eval=False,
    )
    test_te_metrics = {'test_te_' + k: v for (k, v) in test_te_metrics.items()}

    test_metrics = {}
    test_metrics.update(test_tr_metrics)
    test_metrics.update(test_vl_metrics)
    test_metrics.update(test_te_metrics)
  else:
    test_batch = cur_batch
    test_batch = {
        k: (
            v[:, offset:-1, :lead_time]
            if (len(v.shape) > 2) and (v.shape[2] > lead_time) and k != 'x'
            else v
        )
        for (k, v) in test_batch.items()
    }
    test_batch = {k: v.to(device) for (k, v) in test_batch.items()}
    test_batch['target_times_mask'] = test_batch['target_times_mask'][
        :, :, :, :, offset:-1
    ].to(device)
    test_batch['eval_inputs'] = test_batch['inputs'].to(
        device
    )  # eval inputs does batching as normal
    test_batch['eval_targets'] = test_batch['targets'].to(device)
    test_batch['eval_target_times_mask'] = test_batch['target_times_mask'].to(
        device
    )

    # Compute test metrics
    test_metrics = evaluator.compute_all_metrics(
        preds=batch_test_preds,
        actual_batch=test_batch,
        target_service_level=target_service_level,
        unit_holding_cost=unit_holding_cost,
        unit_stockout_cost=unit_stockout_cost,
        unit_var_o_cost=unit_var_o_cost,
        series_mean=False,
        naive_model=naive_model,
        scale_by_naive_model=scale_by_naive_model,
        quantile_loss=quantile_loss,
        rolling_eval=False,
    )

  if sum_ct_metrics:
    sum_metrics = {}
    ct_metrics = {}
    for k, v in test_metrics.items():
      if 'inventory_values' in k:
        continue
      sum_metrics[k] = sum_metrics.get(k, 0) + v.sum()
      ct_metrics[k] = ct_metrics.get(k, 0) + v.numel()
    return {'sums': sum_metrics, 'cts': ct_metrics}
  else:
    return test_metrics


def train_per_series(
    model_class,
    model_args,
    dataset_factory,
    evaluator,
    max_steps,
    optimization_obj,
    learning_rate,
    device,
    use_wandb,
    N,
    lead_time,
    forecasting_horizon,
    test_t_min,
    test_t_max,
    target_service_level,
    unit_holding_cost,
    unit_stockout_cost,
    unit_var_o_cost,
    naive_model,
    scale_by_naive_model,
    quantile_loss,
):
  """Trains a model for each series (relevant for univariate data).

  Models are trained in a roll-forward fashion, getting test predictions
  along the way.

  Args:
    model_class: model class to train
    model_args: arguments used to instantiate model object from model class
    dataset_factory: dataset factory containing data to train/evaluate on
    evaluator: evaluator object used to compute metrics
    max_steps: num gradient steps per timepoint per batch
    optimization_obj: optimization objective name
    learning_rate: learning rate for optimizer
    device: device to perform computations on
    use_wandb: whether to log to wandb
    N: number of series
    lead_time: number of timepoints for inventory to come in
    forecasting_horizon: number of timepoints to forecast
    test_t_min: minimum time point for prediction
    test_t_max: maximum time point for prediction
    target_service_level: target service level for safety stock calculation
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    naive_model: baseline model
    scale_by_naive_model: whether to scale performance by naive model
    quantile_loss: quantile loss object

  Returns:
    per_series_models: dictionary of lib for each series
    test_preds: tensor of predictions
  """

  start = time.time()

  per_series_models = {
      i: model_class(**model_args).to(device) for i in range(N)
  }
  if quantile_loss:
    per_series_optims = {
        i: torch.optim.Adam(
            list(model.parameters()) + list(quantile_loss.parameters()),
            lr=learning_rate,
        )
        for (i, model) in per_series_models.items()
    }
  else:
    per_series_optims = {
        i: torch.optim.Adam(model.parameters(), lr=learning_rate)
        for (i, model) in per_series_models.items()
    }
  per_series_test_preds = {i: [] for i in range(N)}
  proc_print('initialized lib')
  for t in range(test_t_min, test_t_max):
    dataset = dataset_factory.get_data_for_timepoint(t, test_per_series=True)
    for i in range(len(dataset)):
      proc_print(f't: {t}, i: {i}, {(time.time() - start) / 60.:.2f} min.')
      series = dataset[i]
      series = {
          k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
          for (k, v) in series.items()
      }
      model = per_series_models[i]
      optimizer = per_series_optims[i]

      inputs_rollout = get_rollout(
          series['inputs'],
          dimension=2,
          rollout_size=lead_time,
          keep_dim_size=True,
          device=device,
      )
      targets_rollout = get_rollout(
          series['targets'],
          dimension=2,
          rollout_size=lead_time,
          keep_dim_size=True,
          device=device,
      )
      target_times_rollout = get_rollout(
          series['target_times'],
          dimension=2,
          rollout_size=lead_time,
          keep_dim_size=True,
          device=device,
      )

      inputs = collapse_first2dims(inputs_rollout)
      targets = collapse_first2dims(targets_rollout)
      target_times = collapse_first2dims(target_times_rollout)
      target_times_mask = get_mask_from_target_times(target_times, device)

      _, T_x, D_x = series['x'].shape
      N, T_input, window, D = series['inputs'].shape
      N, T_output, fh, target_D = series['targets'].shape
      assert T_input == T_output
      assert target_D == 1
      # model inputs treat samples and start time points as iid samples
      train_batch = {
          'x': (
              series['x']
              .unsqueeze(1)
              .repeat(1, T_output, 1, 1)
              .view(N * T_output, T_x, D_x)
              .to(device)
          ),
          'x_scale': (
              series['x_scale']
              .unsqueeze(1)
              .repeat(1, T_output, 1)
              .view(N * T_output, D_x)
              .to(device)
          ),
          'x_offset': (
              series['x_offset']
              .unsqueeze(1)
              .repeat(1, T_output, 1)
              .view(N * T_output, D_x)
              .to(device)
          ),
          'model_inputs': (
              series['inputs'].view(N * T_input, window, D).to(device)
          ),
          'model_targets': (
              series['targets'].view(N * T_output, fh, target_D).to(device)
          ),
          'target_times': target_times.to(
              device
          ),  # NOTE: THIS DOESN'T ACTUALLY CORRESPOND TO TARGET TIMES MASK
          'eval_inputs': inputs.to(device),
          'eval_targets': targets.to(device),
          'eval_target_times_mask': target_times_mask.to(device),
      }

      for itr in range(max_steps):
        if optimization_obj == 'no_learning':
          break
        optimizer.zero_grad()
        train_preds = model(train_batch)

        train_preds = get_rollout(
            train_preds,
            dimension=1,
            rollout_size=lead_time,
            keep_dim_size=True,
            device=device,
        )
        _, _, num_l, _ = train_preds.shape  # n, t, l, d
        assert num_l == lead_time
        metrics = evaluator.compute_all_metrics(
            preds=train_preds,
            actual_batch=train_batch,
            target_service_level=target_service_level,
            unit_holding_cost=unit_holding_cost,
            unit_stockout_cost=unit_stockout_cost,
            unit_var_o_cost=unit_var_o_cost,
            series_mean=True,
            naive_model=naive_model,
            scale_by_naive_model=scale_by_naive_model,
            quantile_loss=quantile_loss,
            rolling_eval=True,
        )

        obj = get_optimization_obj(metrics, optimization_obj)
        obj.backward()
        optimizer.step()

        log_dict = {
            'Train ' + k: v.detach().item()
            for (k, v) in metrics.items()
            if k != 'inventory_values'
        }
        if (itr % 100 == 0) and use_wandb:
          wandb.log(log_dict)
      with torch.no_grad():
        test_N, test_T_input, test_window, test_D_input = series[
            'test_inputs'
        ].shape
        test_N, test_T_output, test_fh, test_D_target = series[
            'test_targets'
        ].shape
        test_batch = {
            'model_inputs': series['test_inputs'].view(
                test_N * test_T_input, test_window, test_D_input
            ),  # model inputs does batching over all encoding points
            'model_targets': series['test_targets'].view(
                test_N * test_T_output, test_fh, test_D_target
            ),
        }
        test_preds = model(test_batch)
        per_series_test_preds[i].append(test_preds)

  proc_print('for loop done')
  test_preds = torch.zeros(
      (len(dataset), test_t_max - test_t_min, forecasting_horizon, D)
  ).to(device)
  for i, test_preds_list in per_series_test_preds.items():
    test_preds[i, :, :] = torch.cat(test_preds_list, axis=0)
  test_preds = test_preds[:, :, :lead_time]

  return per_series_models, test_preds


def train_multivariate(
    model_class,
    model_args,
    dataset_factory,
    evaluator,
    max_steps,
    num_batches,
    batch_size,
    optimization_obj,
    learning_rate,
    device,
    use_wandb,
    N,
    lead_time,
    test_t_min,
    test_t_max,
    target_service_level,
    unit_holding_cost,
    unit_stockout_cost,
    unit_var_o_cost,
    naive_model,
    scale_by_naive_model,
    quantile_loss,
    test_later,
    exp_folder,
    target_dims,
    num_workers=0,
    save='none',
):
  """Roll-forward training of lib.

  Gets test predictions along the way.

  Performs double-rollout of multivariate series.

  Args:
    model_class: model class to train
    model_args: arguments used to instantiate model object from model class
    dataset_factory: dataset factory containing data to train/evaluate on
    evaluator: evaluator object used to compute metrics
    max_steps: num gradient steps per timepoint per batch
    num_batches: number of batches to train on each epoch
    batch_size: size of each batch
    optimization_obj: objective to optimize
    learning_rate: learning rate for optimizer
    device: device to perform computations on
    use_wandb: whether to log to wandb
    N: number of series
    lead_time: number of timepoints for inventory to come in
    test_t_min: minimum time point for prediction
    test_t_max: maximum time point for prediction
    target_service_level: target service level for safety stock calculation
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    naive_model: baseline model
    scale_by_naive_model: whether to scale performance by naive model
    quantile_loss: quantile loss object
    test_later: whether to skip prediction and simply save the model instead
    exp_folder: folder that houses predictions, lib, etc. for the experiment
    target_dims: dimensions corresponding to target
    num_workers: number of workers to use in dataloader
    save: whether to save the 'latest' (most recent model and predictions),
      'all' (lib from all timepoints and latest predictions), or 'none'

  Returns:
    trained model and predictions
  """

  start = time.time()
  model = model_class(**model_args).to(device)
  if str(device) == 'cuda':
    model = torch.nn.DataParallel(model)
  if quantile_loss:
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(quantile_loss.parameters()),
        lr=learning_rate,
    )
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  start_t = test_t_min
  all_test_preds = []

  # pre-load if available
  model_path = os.path.join(exp_folder, 'model.pt')
  all_test_preds_path = os.path.join(exp_folder, 'all_test_preds.pkl')
  checkpoint_t = None
  needs_eval_first = False
  if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    naive_model.load_state_dict(checkpoint['naive_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if quantile_loss:
      quantile_loss.load_state_dict(checkpoint['quantile_loss_state_dict'])
    checkpoint_t = checkpoint['t']
    start_t = checkpoint_t

    all_test_preds = pickle.load(open(all_test_preds_path, 'rb'))
    num_model_saves = checkpoint_t - test_t_min + 1
    print('test preds: ', len(all_test_preds), 'start_t: ', start_t)
    print(
        f'previously saved after {checkpoint_t} - {test_t_min} + 1 ='
        f' {num_model_saves} timepoints'
    )
    if num_model_saves != len(all_test_preds):
      assert num_model_saves == len(all_test_preds) + 1
      needs_eval_first = True
    tprint(f'loaded model: {model_path}.\nresuming at t: {start_t}')

  proc_print('initialized lib')
  obj = torch.Tensor([-1])
  for t in range(start_t, test_t_max):
    dataset = dataset_factory.get_data_for_timepoint(t, test_per_series=False)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    train_gen = iter(train_loader)
    if t != checkpoint_t:
      for b in range(num_batches):
        tprint(
            f't: {t}, b: {b}, obj:'
            f' {round(obj.cpu().detach().item(), 2)} '
            f'({round(time.time() - start, 2)} sec)'
        )
        # get next batch
        try:
          train_batch = next(train_gen)
        except StopIteration:
          train_gen = iter(train_loader)
          train_batch = next(train_gen)

        series = train_batch
        series = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for (k, v) in series.items()
        }

        # double roll-out
        inputs_rollout = get_rollout(
            series['inputs'],
            dimension=2,
            rollout_size=lead_time,
            keep_dim_size=True,
            device=device,
        )
        targets_rollout = get_rollout(
            series['targets'],
            dimension=2,
            rollout_size=lead_time,
            keep_dim_size=True,
            device=device,
        )
        target_times_rollout = get_rollout(
            series['target_times'],
            dimension=2,
            rollout_size=lead_time,
            keep_dim_size=True,
            device=device,
        )

        # batch together different decoding points, and re-compute time mask
        inputs = collapse_first2dims(inputs_rollout)
        targets = collapse_first2dims(targets_rollout)
        target_times = collapse_first2dims(target_times_rollout)
        target_times_mask = get_mask_from_target_times(target_times, device)

        _, T_x, D_x = series['x'].shape  # N_x, T_x, D_x
        N, T_input, window, D = series['inputs'].shape
        N, T_output, fh, target_D = series['targets'].shape
        assert T_input == T_output

        # model inputs treat samples and start time points as iid samples
        train_batch = {
            'x': (
                series['x']
                .unsqueeze(1)
                .repeat(1, T_output, 1, 1)
                .view(N * T_output, T_x, D_x)
                .to(device)
            ),
            'x_scale': (
                series['x_scale']
                .unsqueeze(1)
                .repeat(1, T_output, 1)
                .view(N * T_output, len(target_dims))
                .to(device)
            ),
            'x_offset': (
                series['x_offset']
                .unsqueeze(1)
                .repeat(1, T_output, 1)
                .view(N * T_output, len(target_dims))
                .to(device)
            ),
            'model_inputs': (
                series['inputs'].view(N * T_input, window, D).to(device)
            ),
            'model_targets': (
                series['targets'].view(N * T_output, fh, target_D).to(device)
            ),
            'target_times': target_times.to(
                device
            ),  # note: doesn't correspond to target times mask
            'eval_inputs': inputs.to(device),
            'eval_targets': targets.to(device),
            'eval_target_times_mask': target_times_mask.to(device),
        }
        for itr in range(max_steps):
          if optimization_obj == 'no_learning':
            break
          optimizer.zero_grad()
          train_preds = model(train_batch)

          train_preds = get_rollout(
              train_preds,
              dimension=1,
              rollout_size=lead_time,
              keep_dim_size=True,
              device=device,
          )
          _, _, num_l, _ = train_preds.shape  # n, t, l, d
          assert num_l == lead_time
          metrics = evaluator.compute_all_metrics(
              preds=train_preds,
              actual_batch=train_batch,
              target_service_level=target_service_level,
              unit_holding_cost=unit_holding_cost,
              unit_stockout_cost=unit_stockout_cost,
              unit_var_o_cost=unit_var_o_cost,
              series_mean=True,
              naive_model=naive_model,
              scale_by_naive_model=scale_by_naive_model,
              quantile_loss=quantile_loss,
              rolling_eval=True,
          )

          obj = get_optimization_obj(metrics, optimization_obj)

          obj.backward()
          optimizer.step()

          log_dict = {
              'Train ' + k: v.detach().item()
              for (k, v) in metrics.items()
              if k != 'inventory_values'
          }
          if (itr % 100 == 0) and use_wandb:
            wandb.log(log_dict)

      state_dict = {
          'model_state_dict': model.state_dict(),
          'naive_model_state_dict': naive_model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          't': t,
      }
      if quantile_loss:
        state_dict['quantile_loss_state_dict'] = quantile_loss.state_dict()

      # save checkpoints for future
      if save != 'none':
        model_path = os.path.join(exp_folder, 'model.pt')
        torch.save(state_dict, model_path)
        tprint(f'saved model checkpoint for time {t}: {model_path}')
      if test_later or save == 'all':
        model_path = os.path.join(exp_folder, f'model_{t}.pt')
        torch.save(state_dict, model_path)
        tprint(f'saved model checkpoint: {model_path}')
      if test_later:
        continue
    if (t != checkpoint_t) or needs_eval_first:
      with torch.no_grad():
        test_dataset = dataset.get_test_data()
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=200,
            shuffle=False,
            num_workers=num_workers,
        )
        test_datagen = iter(test_dataloader)
        series = next(test_datagen, None)
        test_pred_batches = []
        batch_idx = 0
        while series is not None:
          tprint(
              f'testing batch {batch_idx}...'
              f' ({round(time.time() - start, 2)} sec)'
          )
          test_N, test_T_input, test_window, test_D_input = series[
              'test_inputs'
          ].shape
          test_N, test_T_output, test_fh, test_D_target = series[
              'test_targets'
          ].shape
          test_batch = {  # model inputs batches over all encoding points
              'model_inputs': (
                  series['test_inputs']
                  .view(test_N * test_T_input, test_window, test_D_input)
                  .to(device)
              ),
              'model_targets': (
                  series['test_targets']
                  .view(test_N * test_T_output, test_fh, test_D_target)
                  .to(device)
              ),
          }
          test_preds = model(test_batch)
          if len(test_preds.shape) == 4:
            print(
                'test_preds dim 4 now... in favorita naive scaling it was 3...'
            )
          else:
            assert len(test_preds.shape) == 3
            test_preds = test_preds.unsqueeze(1)
          test_pred_batches.append(test_preds)
          series = next(test_datagen, None)
          batch_idx += 1
        test_preds = torch.concat(test_pred_batches, dim=0)
        all_test_preds.append(test_preds)
      pickle.dump(all_test_preds, open(all_test_preds_path, 'wb'))
      print(
          f'saved all {len(all_test_preds)} test preds thus far: ',
          all_test_preds_path,
      )

  if test_later:
    return model, None

  proc_print('for loop done')
  test_preds = torch.cat(all_test_preds, dim=1)
  test_preds = test_preds[:, :, :lead_time]

  return model, test_preds


def train_multivariate_single_rollout(
    model_class,
    model_args,
    dataset_factory,
    evaluator,
    max_steps,
    num_batches,
    batch_size,
    optimization_obj,
    learning_rate,
    device,
    use_wandb,
    N,
    lead_time,
    test_t_min,
    test_t_max,
    target_service_level,
    unit_holding_cost,
    unit_stockout_cost,
    unit_var_o_cost,
    naive_model,
    scale_by_naive_model,
    quantile_loss,
    test_later,
    exp_folder,
    num_workers=0,
    save='none',
):
  """Roll-forward training of lib.

  Gets test predictions along the way.

  Performs single-rollout of multivariate series.

  Args:
    model_class: model class to train
    model_args: arguments used to instantiate model object from model class
    dataset_factory: dataset factory containing data to train/evaluate on
    evaluator: evaluator object used to compute metrics
    max_steps: num gradient steps per timepoint per batch
    num_batches: number of batches to train on each epoch
    batch_size: size of each batch
    optimization_obj: objective to optimize
    learning_rate: learning rate for optimizer
    device: device to perform computations on
    use_wandb: whether to log to wandb
    N: number of series
    lead_time: number of timepoints for inventory to come in
    test_t_min: minimum time point for prediction
    test_t_max: maximum time point for prediction
    target_service_level: target service level for safety stock calculation
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    naive_model: baseline model
    scale_by_naive_model: whether to scale performance by naive model
    quantile_loss: quantile loss object
    test_later: whether to skip prediction and simply save the model instead
    exp_folder: folder that houses predictions, lib, etc. for the experiment
    num_workers: number of workers to use in dataloader
    save: whether to save the 'latest' (most recent model and predictions),
      'all' (lib from all timepoints and latest predictions), or 'none'

  Returns:
    trained model and predictions
  """

  print('doing single rollout...')

  start = time.time()
  model = model_class(**model_args).to(device)
  if str(device) == 'cuda':
    model = torch.nn.DataParallel(model)
  if quantile_loss:
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(quantile_loss.parameters()),
        lr=learning_rate,
    )
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  start_t = test_t_min
  all_test_preds = []

  # pre-load if available
  model_path = os.path.join(exp_folder, 'model.pt')
  all_test_preds_path = os.path.join(exp_folder, 'all_test_preds.pkl')
  checkpoint_t = None
  needs_eval_first = False
  if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    naive_model.load_state_dict(checkpoint['naive_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if quantile_loss:
      quantile_loss.load_state_dict(checkpoint['quantile_loss_state_dict'])
    checkpoint_t = checkpoint['t']
    start_t = checkpoint_t

    all_test_preds = pickle.load(open(all_test_preds_path, 'rb'))
    num_model_saves = checkpoint_t - test_t_min + 1
    print('test preds: ', len(all_test_preds), 'start_t: ', start_t)
    print(
        f'previously saved after {checkpoint_t} - {test_t_min} + 1 ='
        f' {num_model_saves} timepoints'
    )
    if num_model_saves != len(all_test_preds):
      assert num_model_saves == len(all_test_preds) + 1
      needs_eval_first = True
    tprint(f'loaded model: {model_path}.\nresuming at t: {start_t}')

  proc_print('initialized lib')
  obj = torch.Tensor([-1])
  for t in range(start_t, test_t_max):
    dataset = dataset_factory.get_data_for_timepoint(t, test_per_series=False)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    train_gen = iter(train_loader)
    if t != checkpoint_t:
      for b in range(num_batches):
        print(
            f't: {t}, b: {b}, obj:'
            f' {round(obj.cpu().detach().item(), 2)} '
            f'({round(time.time() - start, 2)} sec)'
        )
        # get next batch
        try:
          train_batch = next(train_gen)
        except StopIteration:
          train_gen = iter(train_loader)
          train_batch = next(train_gen)
        series = train_batch
        series = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for (k, v) in series.items()
        }
        N, T_input, window, D = series['inputs'].shape
        N, T_output, fh, target_D = series['targets'].shape
        train_batch = {
            'x': series['x'],
            'x_scale': series['x_scale'],
            'x_offset': series['x_offset'],
            'target_times': series['target_times'][:, :, :, :],
            'model_inputs': series['inputs'].view(N * T_input, window, D),
            'model_targets': series['targets'].view(N * T_output, fh, target_D),
            'eval_inputs': series['inputs'][:, :, :, :],
            'eval_targets': series['targets'][:, :, :, :],
            'eval_target_times_mask': series['target_times_mask'][
                :, :, :, :, :
            ],
        }
        for itr in range(max_steps):
          if optimization_obj == 'no_learning':
            break
          optimizer.zero_grad()
          train_preds = model(train_batch)
          train_preds = train_preds.view(N, T_output, fh, -1)
          metrics = evaluator.compute_all_metrics(
              preds=train_preds,
              actual_batch=train_batch,
              target_service_level=target_service_level,
              unit_holding_cost=unit_holding_cost,
              unit_stockout_cost=unit_stockout_cost,
              unit_var_o_cost=unit_var_o_cost,
              series_mean=True,
              naive_model=naive_model,
              scale_by_naive_model=scale_by_naive_model,
              quantile_loss=quantile_loss,
              rolling_eval=True,
          )
          obj = get_optimization_obj(metrics, optimization_obj)
          obj.backward()
          optimizer.step()

          log_dict = {
              'Train ' + k: v.detach().item()
              for (k, v) in metrics.items()
              if k != 'inventory_values'
          }
          if (itr % 100 == 0) and use_wandb:
            wandb.log(log_dict)

      state_dict = {
          'model_state_dict': model.state_dict(),
          'naive_model_state_dict': naive_model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          't': t,
      }
      if quantile_loss:
        state_dict['quantile_loss_state_dict'] = quantile_loss.state_dict()

      # save checkpoints for future
      if save != 'none':
        model_path = os.path.join(exp_folder, 'model.pt')
        torch.save(state_dict, model_path)
        tprint(f'saved model checkpoint: {model_path}')
      if test_later or save == 'all':
        model_path = os.path.join(exp_folder, f'model_{t}.pt')
        torch.save(state_dict, model_path)
        tprint(f'saved model checkpoint: {model_path}')
      if test_later:
        continue

    if (t != checkpoint_t) or needs_eval_first:
      with torch.no_grad():
        test_dataset = dataset.get_test_data()
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=num_workers,
        )
        test_datagen = iter(test_dataloader)
        series = next(test_datagen, None)
        test_pred_batches = []
        batch_idx = 0
        while series is not None:
          print(
              f'testing batch {batch_idx}...'
              f' ({round(time.time() - start, 2)} sec)'
          )
          test_N, test_T_input, test_window, test_D_input = series[
              'test_inputs'
          ].shape
          test_N, test_T_output, test_fh, test_D_target = series[
              'test_targets'
          ].shape
          test_batch = {  # model inputs batches over all encoding points
              'model_inputs': (
                  series['test_inputs']
                  .view(test_N * test_T_input, test_window, test_D_input)
                  .to(device)
              ),
              'model_targets': (
                  series['test_targets']
                  .view(test_N * test_T_output, test_fh, test_D_target)
                  .to(device)
              ),
          }
          test_preds = model(test_batch)
          if len(test_preds.shape) == 4:
            pass
          else:
            assert len(test_preds.shape) == 3
            test_preds = test_preds.unsqueeze(1)
          test_pred_batches.append(test_preds)
          series = next(test_datagen, None)
          batch_idx += 1
        test_preds = torch.concat(test_pred_batches, dim=0)
        all_test_preds.append(test_preds)
      pickle.dump(all_test_preds, open(all_test_preds_path, 'wb'))
      print('saved all test preds thus far: ', all_test_preds_path)

  if test_later:
    return model, None

  proc_print('for loop done')
  test_preds = torch.cat(all_test_preds, dim=1)
  test_preds = test_preds[:, :, :lead_time]
  return model, test_preds


def get_full_test_metrics(
    dataset_factory,
    test_preds,
    num_workers,
    parallel_eval,
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
    use_wandb,
    sum_ct_metrics=False,
    do_scaling=False,
):
  """Given full set of test predictions, computes test metrics.

  Allows for parallel computation on different batches.
  Can conserve memory by tracking sums and counts rather than the entire list
  of metrics for each series.

  Args:
    dataset_factory: dataset
    test_preds: predictions on entire dataset (produced from rolling forward)
    num_workers: number of workers for parallel dataloading
    parallel_eval: whether to perform parallel eval for different batches
    device: device to perform computations on
    test_t_min: minimum time point for evaluation (start of train time period)
    valid_t_start: start of validation time period
    test_t_start: start of test time period
    evaluator: evaluator object
    target_service_level: service level for safety stock calculation
    lead_time: number of time points for inventory to come in
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    naive_model: baseline model
    scale_by_naive_model: whether to scale performance by baseline model
    quantile_loss: quantile loss object
    use_wandb: whether to log to wandb
    sum_ct_metrics: whether to keep track of sums and counts (to conserve
      memory) for mean computation
    do_scaling: whether to apply scaling (usually False, only used on sktime
      baselines)

  Returns:
    aggregate metrics and metrics per series
  """

  # Create ground truth test batch
  proc_print('testing')
  test_dataset = dataset_factory.complete_dataset
  # test_batch = dataset_factory.test_data
  test_dataloader = torch.utils.data.DataLoader(
      dataset=test_dataset,
      batch_size=100,
      shuffle=False,
      num_workers=num_workers,
  )
  test_datagen = iter(test_dataloader)
  cur_batch = next(test_datagen, None)
  batches_test_metrics = []
  start_idx = 0
  pool = None
  if parallel_eval:
    num_proc = int(mp.cpu_count())
    pool = mp.Pool(num_proc)
    print('Number of processors: ', num_proc)
  test_batch = None
  batch_test_preds = None
  while cur_batch is not None:
    tprint(f'kicking off batch with start_idx: {start_idx}')
    if use_wandb:
      test_batch = cur_batch

    batch_test_preds = test_preds[
        start_idx : start_idx + cur_batch['inputs'].shape[0]
    ]
    if do_scaling:
      _, b_t, b_l, _ = batch_test_preds.shape  # b_n, b_t, b_l, b_d
      b_offset = (
          cur_batch['x_offset'].unsqueeze(1).unsqueeze(1).repeat(1, b_t, b_l, 1)
      )
      b_scale = (
          cur_batch['x_scale'].unsqueeze(1).unsqueeze(1).repeat(1, b_t, b_l, 1)
      )
      batch_test_preds = (batch_test_preds - b_offset) / b_scale
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
              sum_ct_metrics=sum_ct_metrics,
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
                  naive_model,
                  scale_by_naive_model,
                  quantile_loss,
                  start_idx,
                  sum_ct_metrics,
              ],
          ),
      ))
    start_idx += cur_batch['inputs'].shape[0]
    cur_batch = next(test_datagen, None)

  if pool is not None:
    for si, test_metrics in batches_test_metrics:
      while not test_metrics.ready():
        print('waiting for batch with start_idx: ', si)
        time.sleep(100)

  all_test_metrics = {}
  if not sum_ct_metrics:
    for si, test_metrics in batches_test_metrics:
      if pool is not None:
        test_metrics = test_metrics.get()
      print(si)
      for k, v in test_metrics.items():
        all_test_metrics[k] = all_test_metrics.get(k, []) + [v]
    expanded_test_metrics = {
        k: torch.concat(v, dim=0)
        for (k, v) in all_test_metrics.items()
        if 'inventory_values' not in k
    }
    test_metrics = {k: v.mean() for (k, v) in expanded_test_metrics.items()}
  else:
    all_test_metric_sums = {}
    all_test_metric_cts = {}
    for si, test_metrics in batches_test_metrics:
      print(si)
      sums = test_metrics['sums']
      cts = test_metrics['cts']
      for k, v in sums.items():
        all_test_metric_sums[k] = all_test_metric_sums.get(k, 0) + v
        all_test_metric_cts[k] = all_test_metric_cts.get(k, 0) + cts[k]
    expanded_test_metrics = None
    test_metrics = {
        k: v / all_test_metric_cts[k] for (k, v) in all_test_metric_sums.items()
    }

  log_dict = {
      'Test ' + k: v.detach().item()
      for (k, v) in test_metrics.items()
      if 'inventory_values' not in k
  }

  if use_wandb:
    offset = int((test_t_min + 1) - (test_batch['target_times'].min() - 1)) - 1
    test_batch = {
        k: (
            v[:, offset:-1, :lead_time]
            if (len(v.shape) > 2) and (v.shape[2] >= lead_time) and k != 'x'
            else v
        )
        for (k, v) in test_batch.items()
    }
    test_batch['target_times_mask'] = test_batch['target_times_mask'][
        :, :, :, :, offset:-1
    ].to(device)
    test_batch['eval_inputs'] = test_batch['inputs'].to(
        device
    )  # eval inputs does batching as normal
    test_batch['eval_targets'] = test_batch['targets'].to(device)
    test_batch['eval_target_times_mask'] = test_batch['target_times_mask'].to(
        device
    )

    wandb.log(log_dict)
    if test_batch is not None and batch_test_preds is not None:
      fig = plot_series_preds(test_batch, batch_test_preds)
      wandb.log({'series 0 predictions': fig})

  if parallel_eval:
    pool.close()
    pool.join()
  return test_metrics, expanded_test_metrics


def get_learned_alpha(model_name, per_series_models, model):
  """Retrieves learned alpha parameter(s) from model(s).

  Args:
    model_name: name of model (e.g. 'naive_seasonal')
    per_series_models: dictionary of series indices to lib, if univariate
    model: if multivariate, the singular model

  Returns:
    either the list of learned alphas (if univariate) or singular learned alpha
    (if multivariate)
  """
  learned_alpha = None
  if 'naive' in model_name:
    try:
      if per_series_models is not None:
        learned_alpha = [
            model.alpha.detach().item()
            for (i, model) in per_series_models.items()
        ]
      else:
        learned_alpha = model.alpha.detach().item()
    except AttributeError as e:
      print(e)
      if per_series_models is not None:
        learned_alpha = [
            model.module.alpha.detach().item()
            for (i, model) in per_series_models.items()
        ]
      else:
        learned_alpha = model.module.alpha.detach().item()
  return learned_alpha


def run_experiment(
    dataset_name,
    dataset_factory,
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
    num_batches=1000,
    batch_size=50,
    test_t_min=36,
    test_t_max=144,
    valid_t_start=None,
    test_t_start=None,
    learning_rate=1e-3,
    idx_range=None,
    target_dims=(0,),
    input_size=1,
    per_series_training=False,
    single_rollout=False,
    no_safety_stock=False,
    use_wandb=True,
    device=torch.device('cpu'),
    periodicity=12,
    test_later=False,
    exp_folder=None,
    embedding_dim=None,
    parallel_eval=False,
    num_workers=0,
    save='none',
    return_test_preds=False,
):
  """Runs the experiment.

  Args:
    dataset_name: name of dataset
    dataset_factory: data object
    optimization_obj: name of optimization objective
    model_name: name of model
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    naive_model: baseline model
    scale_by_naive_model: whether to scale performance by baseline model
    quantile_loss: quantile loss objective, if relevant
    forecasting_horizon: number of time points to forecast
    hidden_size: number of hidden units in encoder and decoder
    num_layers: number of layers in encoder and decoder
    lead_time: number of timepoints it takes for inventory to come in
    scale01: whether to scale predictions between 0 and 1
    target_service_level: service level to use for safety stock calculation
    max_steps: num gradient steps per timepoint per batch
    num_batches: number of batches to train per timepoint
    batch_size: size of batches
    test_t_min: first timepoint for evaluation
    test_t_max: last timepoint for evaluation
    valid_t_start: start of validation time period
    test_t_start: start of test time period
    learning_rate: learning rate
    idx_range: range of indices of series to use
    target_dims: dimensions corresponding to target
    input_size: size of input
    per_series_training: whether to train one model for each series (univariate)
    single_rollout: whether to perform a single rollout (vs. double rollout)
    no_safety_stock: whether to include safety stock in the order-up-to policy
    use_wandb: whether to log to wandb
    device: device to perform computations on
    periodicity: seasonality of the data
    test_later: whether to skip prediction and simply train the model
    exp_folder: folder to save lib and predictions for this experiment
    embedding_dim: number of dimension for each embedding
    parallel_eval: whether to evaluate in parallel
    num_workers: number of workers for parallel dataloading
    save: whether to save the 'latest', 'all', or 'none' lib and predictions
      across timepoints
    return_test_preds: whether to return the roll-forward predictions across all
      time

  Returns:
    dict of metrics and (if return_test_preds) all roll-forward predictions
  """
  evaluator = Evaluator(0, scale01, device, target_dims, no_safety_stock)

  N = len(dataset_factory)
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
    if dataset_name == 'favorita':
      model_args['embedding_dim'] = embedding_dim
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
    raise NotImplementedError('Unrecognized model name: ', model_name)
  naive_model = naive_model.to(device)
  start = time.time()

  # Train lib and make test predictions
  per_series_models = None
  model = None
  if per_series_training:
    assert not test_later
    per_series_models, test_preds = train_per_series(
        model_class,
        model_args,
        dataset_factory,
        evaluator,
        max_steps,
        optimization_obj,
        learning_rate,
        device,
        use_wandb,
        N,
        lead_time,
        forecasting_horizon,
        test_t_min,
        test_t_max,
        target_service_level,
        unit_holding_cost,
        unit_stockout_cost,
        unit_var_o_cost,
        naive_model,
        scale_by_naive_model,
        quantile_loss,
    )
  elif single_rollout:
    model, test_preds = train_multivariate_single_rollout(
        model_class,
        model_args,
        dataset_factory,
        evaluator,
        max_steps,
        num_batches,
        batch_size,
        optimization_obj,
        learning_rate,
        device,
        use_wandb,
        N,
        lead_time,
        test_t_min,
        test_t_max,
        target_service_level,
        unit_holding_cost,
        unit_stockout_cost,
        unit_var_o_cost,
        naive_model,
        scale_by_naive_model,
        quantile_loss,
        test_later,
        exp_folder,
        num_workers=num_workers,
        save=save,
    )
  else:
    model, test_preds = train_multivariate(
        model_class,
        model_args,
        dataset_factory,
        evaluator,
        max_steps,
        num_batches,
        batch_size,
        optimization_obj,
        learning_rate,
        device,
        use_wandb,
        N,
        lead_time,
        test_t_min,
        test_t_max,
        target_service_level,
        unit_holding_cost,
        unit_stockout_cost,
        unit_var_o_cost,
        naive_model,
        scale_by_naive_model,
        quantile_loss,
        test_later,
        exp_folder,
        target_dims,
        num_workers=num_workers,
        save=save,
    )

  if test_later:
    return None

  learned_alpha = get_learned_alpha(model_name, per_series_models, model)

  test_metrics, expanded_test_metrics = get_full_test_metrics(
      dataset_factory,
      test_preds,
      num_workers,
      parallel_eval,
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
      use_wandb,
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

  if return_test_preds:
    return test_results, test_preds
  else:
    return test_results


def get_configs():
  """Register command-line arguments.

  Returns:
    command-line arguments
  """
  parser = argparse.ArgumentParser(
      prog='Roll-forward Training',
      description=(
          'Evaluation and E2E optimization of inventory and forecasting'
          ' performance metrics.'
      ),
  )

  parser.add_argument('--dataset_name', choices=['m3', 'favorita'])
  parser.add_argument(
      '--model_name',
      choices=['naive_seasonal', 'lstm_windowed', 'naive_alpha1_baseline'],
  )
  parser.add_argument(
      '--optimization_objs',
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
      action='append',
  )
  parser.add_argument('--unit_holding_costs', type=int, action='append')
  parser.add_argument('--unit_stockout_costs', type=int, action='append')
  parser.add_argument('--unit_var_o_costs', type=float, action='append')
  parser.add_argument('--parallel', action='store_true')
  parser.add_argument('--parallel_eval', action='store_true')
  parser.add_argument('--sweep_ct', type=int)

  parser.add_argument('--nosweep', action='store_true')
  parser.add_argument('--hidden_size', type=int, default=None)
  parser.add_argument('--max_steps', type=int, default=None)
  parser.add_argument('--num_batches', type=int, default=None)
  parser.add_argument('--batch_size', type=int, default=None)
  parser.add_argument('--learning_rate', type=float, default=None)
  parser.add_argument('--test_later', action='store_true')
  parser.add_argument('--N', type=int, default=None)

  parser.add_argument('--project_name', type=str, default=None)
  parser.add_argument('--num_workers', type=int, default=0)
  parser.add_argument(
      '--save', choices=['all', 'latest', 'none'], default='none'
  )
  parser.add_argument('--exp_folder', type=str, default=None)
  parser.add_argument('--return_test_preds', action='store_true')

  parser.add_argument('--single_rollout', action='store_true')
  parser.add_argument('--no_safety_stock', action='store_true')

  args = parser.parse_args()

  if args.model_name == 'naive_alpha1_baseline':
    assert args.optimization_objs == ['no_learning']

  return args


def get_sweep_id(
    dataset_name,
    model_name,
    optimization_obj,
    unit_holding_cost,
    unit_stockout_cost,
    unit_var_o_cost,
    sweep_lookup_fpath='project_name_to_sweep_id.json',
):
  """Look up the experiment configuration to retrieve the wandb sweep id.

  Args:
    dataset_name: name of dataset
    model_name: name of model
    optimization_obj: name of optimization obj
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    sweep_lookup_fpath: path to lookup from project name to sweep id

  Returns:
  """
  project_name = (
      f'{dataset_name}_{model_name}_{optimization_obj}_'
      + f'{unit_holding_cost}Ch_{unit_stockout_cost}Cs_{unit_var_o_cost}Co'
  )
  with open(sweep_lookup_fpath, 'r') as fin:
    project_name_to_sweep_id = json.load(fin)
  sweep_id = project_name_to_sweep_id[project_name]
  return sweep_id, project_name


def run_sweep(
    project_name,
    tags,
    dataset_name,
    model_name,
    optimization_obj,
    device,
    unit_holding_cost,
    unit_stockout_cost,
    unit_var_o_cost,
    scale_by_naive_model,
    forecasting_horizon,
    num_layers,
    lead_time,
    scale01,
    target_service_level,
    test_t_min,
    test_t_max,
    valid_t_start,
    test_t_start,
    idx_ranges,
    input_window_size,
    target_dims,
    input_size,
    naive_model,
    N,
    per_series_training,
    parallel,
    hidden_size=None,  # leave as none if want wandb to choose
    max_steps=None,
    num_batches=None,
    batch_size=None,
    learning_rate=None,
    periodicity=12,
    single_rollout=False,
    no_safety_stock=False,
    test_later=False,
    data_fpath='../data/favorita/favorita_tensor_full.npy',
    embedding_dim=None,
    parallel_eval=False,
    num_workers=0,
    save='none',
    exp_folder=None,
    return_test_preds=False,
):
  """Run the wandb sweep.

  If wandb will populate hidden_size, max_steps, num_batches, batch_size, and
  learning_rate if left as None and a sweep_lookup_fpath file mapping
  experiment configuration to sweep id has been created (see get_sweep_id
  function).

  Args:
    project_name:
    tags:
    dataset_name: name of dataset
    model_name: name of model
    optimization_obj: name of optimization objective
    device: device to perform computations on
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    scale_by_naive_model: whether to scale performance by baseline model
    forecasting_horizon: number of time points to forecast
    num_layers: number of layers in encoder and decoder
    lead_time: number of timepoints it takes for inventory to come in
    scale01: whether to scale predictions between 0 and 1
    target_service_level: service level to use for safety stock calculation
    test_t_min: first timepoint for evaluation
    test_t_max: last timepoint for evaluation
    valid_t_start: start of validation time period
    test_t_start: start of test time period
    idx_ranges: range of indices of series to use
    input_window_size: encoding window size
    target_dims: list of dimensions corresponding to target dimension
    input_size: size of input
    naive_model: baseline model
    N: number of series
    per_series_training: whether to train one model for each series (univariate)
    parallel: whether to enable parallel computation
    hidden_size: number of hidden units
    max_steps: num gradient steps per timepoint per batch
    num_batches: number of batches per timepoint
    batch_size: size of batches
    learning_rate: optimizer learning rate
    periodicity: number of timepoints in one season
    single_rollout: whether to do single rollout (vs. double rollout)
    no_safety_stock: whether to include safety stock in the order-up-to policy
    test_later: whether to skip prediction and simply train the model
    data_fpath: path to dataset
    embedding_dim: number of dimension for each embedding
    parallel_eval: whether to evaluate in parallel
    num_workers: number of workers for parallel dataloading
    save: whether to save the 'latest', 'all', or 'none' lib and predictions
      across timepoints
    exp_folder: folder to save lib and predictions for this experiment
    return_test_preds: whether to return the roll-forward predictions across all
      time

  Raises:
    KeyError: If unrecognized dataset name
  """
  if parallel:
    num_proc = int(mp.cpu_count())
    pool = mp.Pool(num_proc)
    print('Number of processors: ', num_proc)
  else:
    pool = None

  now = datetime.datetime.now()
  now = now.strftime('%m-%d-%Y-%H:%M:%S')
  tag_str = ''.join(tags)

  if exp_folder is None:
    exp_name = f'{tag_str}_{now}_{dataset_name}_{model_name}_{optimization_obj}_{str(device)}'
    proc_print(exp_name)
    exp_folder = f'experiments/{exp_name}'
  else:
    exp_name = exp_folder.replace('experiments/', '')
    assert f'{dataset_name}_{model_name}_{optimization_obj}' in exp_name

  if optimization_obj.startswith('fixed_quantile_loss_'):
    q = float(optimization_obj.replace('fixed_quantile_loss_', ''))
    quantile_loss = QuantileLoss(init_quantile=q, frozen=True)
  elif 'quantile_loss' in optimization_obj:
    quantile_loss = QuantileLoss(init_quantile=0.5, frozen=False)
  else:
    quantile_loss = None

  wandb_config = {  # for logging purposes
      'tag_str': tag_str,
      'dataset_name': dataset_name,
      'optimization_obj': optimization_obj,
      'model_name': model_name,
      'learning_rate': learning_rate,
      'unit_holding_cost': unit_holding_cost,
      'unit_stockout_cost': unit_stockout_cost,
      'unit_var_o_cost': unit_var_o_cost,
      'scale_by_naive_model': scale_by_naive_model,
      'quantile_loss': str(quantile_loss),
      'forecasting_horizon': forecasting_horizon,
      'hidden_size': hidden_size,
      'num_layers': num_layers,
      'lead_time': lead_time,
      'scale01': scale01,
      'target_service_level': target_service_level,
      'max_steps': max_steps,
      'num_batches': num_batches,
      'batch_size': batch_size,
      'test_t_min': test_t_min,
      'test_t_max': test_t_max,
      'valid_t_start': valid_t_start,
      'test_t_start': test_t_start,
      'single_rollout': single_rollout,
      'no_safety_stock': no_safety_stock,
      'N': N,
      'test_later': test_later,
      'embedding_dim': embedding_dim,
      'save': save,
  }
  proc_print(wandb_config)
  if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)
    with open('wandb_config.json', 'w') as fout:
      json.dump(wandb_config, fout)

  wandb.init(
      name=exp_name,
      project=project_name,
      reinit=True,
      config=wandb_config,
      tags=tags,
      settings=wandb.Settings(start_method='fork'),
  )
  proc_print('wandb started')

  if num_batches is None:
    num_batches = wandb.config.num_batches
  if batch_size is None:
    batch_size = wandb.config.batch_size
  if max_steps is None:
    max_steps = wandb.config.max_steps
  if learning_rate is None:
    learning_rate = wandb.config.learning_rate
  if hidden_size is None:
    hidden_size = wandb.config.hidden_size

  start = time.time()
  results = []
  for ir, idx_range in enumerate(idx_ranges):
    use_wandb = bool(ir == 0)

    if dataset_name == 'favorita':
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
    elif dataset_name == 'm3':
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
      raise KeyError('Unrecognized dataset_name: ', dataset_name)

    print('loaded dataset factory: ', len(dataset_factory))
    config = {
        'optimization_obj': optimization_obj,
        'model_name': model_name,
        'unit_holding_cost': unit_holding_cost,
        'unit_stockout_cost': unit_stockout_cost,
        'unit_var_o_cost': unit_var_o_cost,
        'naive_model': naive_model,
        'scale_by_naive_model': scale_by_naive_model,
        'quantile_loss': quantile_loss,
        'forecasting_horizon': forecasting_horizon,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'lead_time': lead_time,
        'scale01': scale01,
        'target_service_level': target_service_level,
        'max_steps': max_steps,
        'num_batches': num_batches,
        'batch_size': batch_size,
        'test_t_min': test_t_min,
        'test_t_max': test_t_max,
        'valid_t_start': valid_t_start,
        'test_t_start': test_t_start,
        'learning_rate': learning_rate,
        'idx_range': idx_range,
        'target_dims': target_dims,
        'input_size': input_size,
        'per_series_training': per_series_training,
        'use_wandb': use_wandb,
        'device': device,
        'periodicity': periodicity,
        'single_rollout': single_rollout,
        'no_safety_stock': no_safety_stock,
        'test_later': test_later,
        'exp_folder': exp_folder,
        'embedding_dim': embedding_dim,
        'parallel_eval': parallel_eval,
        'num_workers': num_workers,
        'save': save,
        'return_test_preds': return_test_preds,
    }
    pprint.pprint(config)
    print(optimization_obj, model_name)

    if parallel:
      results.append(
          pool.apply_async(
              run_experiment, [dataset_name, dataset_factory], config
          )
      )
    else:
      results.append(run_experiment(dataset_name, dataset_factory, **config))

  if test_later:
    wandb.finish()
    tprint(f'Test later. Model checkpoints are in: {exp_folder}')
    return

  if parallel:
    for i, r in enumerate(results):
      while not r.ready():
        print(f'waiting on chunk {i}...')
        time.sleep(10)
    results = [r.get() for r in results]
    print('finished waiting')

  if len(idx_ranges) == 1:  # mainly for favorita, since single rollout
    assert not return_test_preds
    test_results = results[0]
    summary = test_results['summary']
    summary['dataset_name'] = dataset_name
    summary['batch_size'] = batch_size
    summary['num_batches'] = num_batches
    summary['learning_rate'] = learning_rate
    summary['hidden_size'] = hidden_size
    summary['unit_holding_cost'] = unit_holding_cost
    summary['unit_stockout_cost'] = unit_stockout_cost
    summary['unit_var_o_cost'] = unit_var_o_cost
    summary['N'] = len(dataset_factory)
    summary['forecasting_horizon'] = forecasting_horizon

    test_obj = test_results['test_obj']
  else:
    combined_test_results = {}
    combined_test_preds = []
    for result in results:
      if return_test_preds:
        d, test_preds = result
        combined_test_preds.append(test_preds)
      else:
        d = result
      for k, v in d.items():
        combined_test_results[k] = combined_test_results.get(k, []) + [v]

    if return_test_preds:
      combined_test_preds = torch.cat(combined_test_preds, dim=0)
      preds_path = os.path.join(exp_folder, 'test_preds.pt')
      torch.save(combined_test_preds, preds_path)
      print('saved predictions: ', preds_path)

    combined_test_metrics = {}
    for d in combined_test_results['expanded_test_metrics']:
      for mname, mval in d.items():
        combined_test_metrics[mname] = combined_test_metrics.get(mname, []) + [
            mval
        ]
    expanded_combined_test_metrics = {
        k: torch.cat(v, dim=0) for (k, v) in combined_test_metrics.items()
    }
    combined_test_metrics = {
        k: v.mean() for (k, v) in expanded_combined_test_metrics.items()
    }

    if valid_t_start is None:
      test_obj = (
          get_optimization_obj(combined_test_metrics, optimization_obj)
          .detach()
          .item()
      )
      test_N = get_optimization_obj(
          expanded_combined_test_metrics, optimization_obj
      ).shape[0]
    else:
      test_obj = (
          get_optimization_obj(
              combined_test_metrics, optimization_obj, prefix='test_vl_'
          )
          .detach()
          .item()
      )
      test_N = get_optimization_obj(
          expanded_combined_test_metrics, optimization_obj, prefix='test_vl_'
      ).shape[0]

    summary = {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'optimization objective': optimization_obj,
        'max_steps': max_steps,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'learning_rate': learning_rate,
        'hidden_size': hidden_size,
        'unit_holding_cost': unit_holding_cost,
        'unit_stockout_cost': unit_stockout_cost,
        'unit_var_o_cost': unit_var_o_cost,
        'runtime': time.time() - start,
        'N': test_N,
        'forecasting_horizon': forecasting_horizon,
    }

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
        # 'quantile_loss', 'scale_by_naive_model',
    ]

    if valid_t_start is not None:
      metric_names1 = ['test_tr_' + n for n in final_metric_names]
      metric_names2 = ['test_vl_' + n for n in final_metric_names]
      metric_names3 = ['test_te_' + n for n in final_metric_names]
      final_metric_names = (
          list(metric_names1) + list(metric_names2) + list(metric_names3)
      )

    for name in final_metric_names:
      if name in combined_test_metrics:
        val = combined_test_metrics[name]
        summary[name] = val.detach().item()

    summary['naive_model'] = str(naive_model)

  summary = pd.DataFrame([summary])
  wandb.log({
      'combined_test_perfs': wandb.Table(dataframe=summary),
      'test_obj': test_obj,
  })
  wandb.finish()

  summary.to_csv(
      'all_experiment_results.csv', mode='a', index=False, header=False
  )

  if parallel:
    pool.close()
    pool.join()


def main():
  args = get_configs()

  mp.set_start_method('spawn')
  parallel = args.parallel

  dataset_name = args.dataset_name
  model_name = args.model_name
  optimization_objs = args.optimization_objs
  target_dims = [0]

  unit_holding_costs = args.unit_holding_costs
  unit_stockout_costs = args.unit_stockout_costs
  unit_var_o_costs = args.unit_var_o_costs

  sweep_ct = args.sweep_ct

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
    per_series_training = False
    num_layers = 2
    lead_time = 7
    scale01 = True
    target_service_level = 0.95
    N = args.N
    idx_ranges = [None]
    periodicity = 7
    data_fpath = '../data/favorita/favorita_tensor_full.npy'
  elif dataset_name == 'm3':
    test_t_min = 36
    test_t_max = 144
    valid_t_start = 72
    test_t_start = 108

    input_size = 1
    per_series_training = True

    forecasting_horizon = 12
    input_window_size = 24

    embedding_dim = None
    num_layers = 2
    lead_time = 6
    scale01 = True
    target_service_level = 0.95

    N = args.N
    chunksize = 20
    dataset_size = 334 if N is None else N
    idx_ranges = [
        (int(i), int(i + chunksize))
        for i in np.arange(20, dataset_size, chunksize)
    ]
    if N == 20:
      idx_ranges = [(0, 10), (10, 20)]  # first chunk used for validation

    periodicity = 12
    data_fpath = '../data/m3/m3_industry_monthly_shuffled.csv'
    print('idx_ranges: ', idx_ranges)
  else:
    raise Exception('Unrecognized dataset name: ', dataset_name)

  print('return test preds: ', args.return_test_preds)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  naive_model = NaiveScalingBaseline(
      forecasting_horizon=lead_time,
      init_alpha=1.0,
      periodicity=periodicity,
      frozen=True,
  ).to(device)
  scale_by_naive_model = False

   # filepath for json file mapping experiment configuration to wandb sweep id
  sweep_lookup_fpath = 'project_name_to_sweep_id.json'

  tags = [dataset_name]

  exp_settings = [
      optimization_objs,
      unit_holding_costs,
      unit_stockout_costs,
      unit_var_o_costs,
  ]

  for params in itertools.product(*exp_settings):
    optimization_obj, unit_holding_cost, unit_stockout_cost, unit_var_o_cost = (
        params
    )

    if (
        (len(unit_holding_costs) > 1)
        or (len(unit_stockout_costs) > 1)
        or (len(unit_var_o_costs) > 1)
        or args.project_name is not None
    ):
      assert args.nosweep
      project_name = args.project_name
    else:
      sweep_id, project_name = get_sweep_id(
          dataset_name,
          model_name,
          optimization_obj,
          unit_holding_cost,
          unit_stockout_cost,
          unit_var_o_cost,
          sweep_lookup_fpath=sweep_lookup_fpath,
      )

    def do_sweep(
        proj_name=project_name,
        opt_obj=optimization_obj,
        unit_c_h=unit_holding_cost,
        unit_c_s=unit_stockout_cost,
        unit_c_v=unit_var_o_cost,
    ):
      run_sweep(
          proj_name,
          tags,
          dataset_name,
          model_name,
          opt_obj,
          device,
          unit_c_h,
          unit_c_s,
          unit_c_v,
          scale_by_naive_model,
          forecasting_horizon,
          num_layers,
          lead_time,
          scale01,
          target_service_level,
          test_t_min,
          test_t_max,
          valid_t_start,
          test_t_start,
          idx_ranges,
          input_window_size,
          target_dims,
          input_size,
          naive_model,
          N,
          per_series_training,
          parallel,
          hidden_size=args.hidden_size,
          max_steps=args.max_steps,
          num_batches=args.num_batches,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate,
          periodicity=periodicity,
          single_rollout=args.single_rollout,
          no_safety_stock=args.no_safety_stock,
          test_later=args.test_later,
          data_fpath=data_fpath,
          embedding_dim=embedding_dim,
          parallel_eval=args.parallel_eval,
          num_workers=args.num_workers,
          save=args.save,
          exp_folder=args.exp_folder,
          return_test_preds=args.return_test_preds,
      )

    if args.nosweep:
      do_sweep()
    else:
      wandb.agent(
          sweep_id, function=do_sweep, project=project_name, count=sweep_ct
      )
  wandb.alert(
      title='favorita experiment finished.',
      text='exp_settings: ' + str(exp_settings),
  )
  wandb.finish()
  print('finished.')


if __name__ == '__main__':
  main()
