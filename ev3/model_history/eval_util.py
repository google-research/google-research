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

"""Evaluation methods used by optimize and decide."""

from typing import Any, Dict, Union

import jax.numpy as jnp
import pandas as pd

from ev3 import base
from ev3.model_history import struct


PARAMS = 'params'
STABLE_PARAMS = 'stable_params'


def get_batch(
    state
):
  return next(state.data_iter)


def evaluate_updates(
    model_updates,
    state,
    model,
    add_model_params,
    get_batch_fn = get_batch,
):
  """Evaluates proposed updates and returns the results in a dictionary.

  Args:
    model_updates: The updates to be evaluated.
    state: The EV3 state containing the metrics to be evaluated and a data
      iterator.
    model: An object containing both the model graph and the model parameters.
    add_model_params: If True, the parameters of model are also evaluated.
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.

  Returns:
    A dictionary containing the evaluation results, including the upper and
    lower confidence bounds of the metric evaluation for each update and the
    model parameters that were evaluated.

  Raises:
    ValueError: The variable 'model' does not have its stable parameters
    initialized.
  """
  # Evaluate old and new parameters.
  all_params = {}
  for ind, update in enumerate(model_updates):
    all_params[ind] = (model + update).params
  if model.stable_params is None:
    raise ValueError("The model's stable parameters have not been initialized.")
  if add_model_params:
    all_params.update({
        PARAMS: model.params,
        STABLE_PARAMS: model.stable_params,
    })
  batch = get_batch_fn(state)
  all_eval_results = {}
  for m_ind, metric_fn in enumerate(state.metric_fn_list):
    for k, params in all_params.items():
      all_eval_results[m_ind, k] = metric_fn(params, model.graph, batch)

  eval_results_df = pd.DataFrame(all_eval_results)
  # The rows of eval_results_df correspond to the samples from the metric.
  eval_results_df.rename_axis('samples', axis='index', inplace=True)
  # The columns enumerate the metrics and the evaluated model params.
  eval_results_df.columns.set_names(['metric index', 'param key'], inplace=True)

  # Compute confidence intervals.
  eval_results_mean = eval_results_df.mean(axis='rows')
  eval_results_std = eval_results_df.std(axis='rows')
  eval_results_stderr = eval_results_std / jnp.sqrt(eval_results_df.shape[0])
  eval_results_ucb = eval_results_mean + state.ucb_alpha * eval_results_stderr
  eval_results_lcb = eval_results_mean - state.ucb_alpha * eval_results_stderr
  return {
      'eval_results_ucb': eval_results_ucb,
      'eval_results_lcb': eval_results_lcb,
      'all_params': all_params,
  }


def evaluate_updates_batches(
    model_updates,
    state,
    model,
    add_model_params,
    get_batch_fn = get_batch,
    n_batches=1,
):
  """Evaluates the proposed updates with multiple batches of data.

  This is useful when we need more than one batch of data for optimize or
  decide.

  Args:
    model_updates: The updates to be evaluated.
    state: The EV3 state containing the metrics to be evaluated and a data
      iterator.
    model: An object containing both the model graph and the model parameters.
    add_model_params: If True, the parameters of model are also evaluated.
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.
    n_batches: The number of batches.

  Returns:
    A dictionary containing the evaluation results, including the upper and
    lower confidence bounds of the metric evaluation for each update and the
    model parameters that were evaluated.

  Raises:
    ValueError: The variable 'model' does not have its stable parameters
    initialized.
  """
  # Evaluate old and new parameters.
  all_params = {}
  for ind, model_update in enumerate(model_updates):
    all_params[ind] = (model + model_update).params
  if model.stable_params is None:
    raise ValueError("The model's stable parameters have not been initialized.")
  if add_model_params:
    all_params.update({
        PARAMS: model.params,
        STABLE_PARAMS: model.stable_params,
    })

  all_eval_results = {}
  for _ in range(n_batches):
    batch = get_batch_fn(state)
    for m_ind, metric_fn in enumerate(state.metric_fn_list):
      for k, params in all_params.items():
        if (m_ind, k) in all_eval_results:
          all_eval_results[m_ind, k] = jnp.concatenate([
              all_eval_results[m_ind, k],
              metric_fn(params, model.graph, batch),
          ])
        else:
          all_eval_results[m_ind, k] = metric_fn(params, model.graph, batch)

  eval_results_df = pd.DataFrame(all_eval_results)
  # The rows of eval_results_df correspond to the samples from the metric.
  eval_results_df.rename_axis('samples', axis='index', inplace=True)
  # The columns enumerate the metrics and the evaluated model params.
  eval_results_df.columns.set_names(['metric index', 'param key'], inplace=True)

  # Compute confidence intervals.
  eval_results_mean = eval_results_df.mean(axis='rows')
  eval_results_std = eval_results_df.std(axis='rows')
  eval_results_stderr = eval_results_std / jnp.sqrt(eval_results_df.shape[0])
  eval_results_ucb = eval_results_mean + state.ucb_alpha * eval_results_stderr
  eval_results_lcb = eval_results_mean - state.ucb_alpha * eval_results_stderr
  return {
      'eval_results_ucb': eval_results_ucb,
      'eval_results_lcb': eval_results_lcb,
      'all_params': all_params,
  }
