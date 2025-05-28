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

"""Utility functions for logging purposes."""

import datetime
import os
import time

import pandas as pd
import wandb


def proc_info(title):
  print(title)
  print('module name: ', __name__)
  print('parent_process: ', os.getppid())
  print('process id: ', os.getpid())


def tprint(msg):
  now = datetime.datetime.now()
  print(f'[{now}]\t{msg}')


def proc_print(msg):
  tprint(f'pid: {os.getpid()},\t{msg}')


def get_summary(
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
):
  """Creates summary dictionary of all metrics and basic experiment parameters.

  For logging purposes.

  Args:
    test_metrics: dictionary of metrics
    model_name: name of model
    optimization_obj: name of optimization obj
    max_steps: num gradient steps per timepoint per batch
    start: start time (unix time)
    unit_holding_cost: cost per unit held
    unit_stockout_cost: cost per unit stockout
    unit_var_o_cost: cost per unit order variance
    valid_t_start: start of validation time period
    learned_alpha: learned alpha (if naive scaling)
    quantile_loss: quantile loss object (if used)
    naive_model: baseline model object
    use_wandb: whether to log to wandb
    expanded_test_metrics: per-series test metrics
    idx_range: range of series being summarized

  Returns:
    dictionary summarizing results
  """
  proc_print('getting summary')
  summary = {
      'model_name': model_name,
      'optimization objective': optimization_obj,
      'max_steps': max_steps,
      'runtime': time.time() - start,
      'unit_holding_cost': unit_holding_cost,
      'unit_stockout_cost': unit_stockout_cost,
      'unit_var_o_cost': unit_var_o_cost,
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
      'quantile_loss',
      'scale_by_naive_model',
  ]

  if valid_t_start is None:
    test_obj = test_metrics.get(optimization_obj, None)
  else:
    test_obj = test_metrics.get('test_vl_' + optimization_obj, None)
    metric_names1 = ['test_tr_' + n for n in final_metric_names]
    metric_names2 = ['test_vl_' + n for n in final_metric_names]
    metric_names3 = ['test_te_' + n for n in final_metric_names]
    final_metric_names = (
        list(metric_names1) + list(metric_names2) + list(metric_names3)
    )

  if test_obj is not None:
    test_obj = test_obj.detach().item()

  for name in final_metric_names:
    if name in test_metrics:
      val = test_metrics[name]
      summary[name] = val.detach().item()

  if learned_alpha is not None:
    summary['learned alpha'] = learned_alpha

  if quantile_loss:
    summary['quantile'] = quantile_loss.get_quantile().detach().item()

  summary['naive_model'] = str(naive_model)

  test_results = {
      'test_metrics': test_metrics,
      'expanded_test_metrics': expanded_test_metrics,
      'idx_range': idx_range,
      'summary': summary,
      'test_obj': test_obj,
  }

  if use_wandb:
    wandb.log({
        'idx_range_test_perfs': wandb.Table(dataframe=pd.DataFrame([summary])),
        'idx_range_test_obj': test_obj,
    })

  return test_results
