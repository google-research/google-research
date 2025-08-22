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

"""Functionality to plot predictions and inventory quantitties over time."""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_numpy_arr(tensor, nan_cutoff=1e14):
  vals = tensor.cpu().detach().numpy()
  vals[vals >= nan_cutoff] = np.nan
  return vals


def plot_series_preds(batch, preds, i=0, first_cutoff=0):
  """Plot predictions and true values of time series.

  Args:
    batch: batch data dictionary
    preds: predicted values
    i: index of time series
    first_cutoff: first time point at which predictions are made

  Returns:
    plot figure
  """
  if 'inputs' in batch:
    assert first_cutoff == 0
    inputs = batch['inputs'].detach().clone().cpu()
    input_times = batch['input_times'].detach().clone().cpu()
    targets = batch['targets'].detach().clone().cpu()
    target_times = batch['target_times'].detach().clone().cpu()
    preds = preds.detach().clone()
    preds = get_numpy_arr(
        preds[i, :, :, 0]
    )  # just plots the first one, if there are multiple prediction dims)
    pred_times = get_numpy_arr(target_times[i, :, :, 0])

    inputs = get_numpy_arr(inputs[i, :, 0, 0])
    input_times = get_numpy_arr(input_times[i, :, 0, 0])

    targets = get_numpy_arr(targets[i, :, 0, 0])
    target_times = get_numpy_arr(target_times[i, :, 0, 0])
  else:
    inputs = get_numpy_arr(batch['x_imputed'][i, :first_cutoff])
    input_times = np.array(range(1, first_cutoff + 1))

    targets = get_numpy_arr(batch['x_imputed'][i, first_cutoff:])
    target_times = np.array(
        range(first_cutoff + 1, batch['x_imputed'].shape[1] + 1)
    )

    preds = get_numpy_arr(preds[i, :, :])
    pred_times = get_numpy_arr(batch['unfolded_times'][i, :, :])

  L = preds.shape[-1]

  inputs = pd.Series(inputs, index=input_times, name='inputs')
  targets = pd.Series(targets, index=target_times, name='targets')
  fig, ax = plt.subplots(max(int(np.ceil(L / 2.0)), 2), 2)
  for l in range(L):
    a = int(l / 2.0)
    b = l % 2
    pred_l = pd.Series(preds[:, l], index=pred_times[:, l], name='preds')
    pred_l.plot(ax=ax[a, b])
    inputs.plot(ax=ax[a, b])
    targets.plot(ax=ax[a, b])
    ax[a, b].set_title(f'lead time {l + 1}')
  plt.tight_layout()
  return fig


def _plot_nonnans(series, i, threshold=1e10, label=None):
  plt.plot(series[i][abs(series[i]) < threshold], label=label)


def plot_series_inventory_perf(perf, i, title):
  """Plot inventory quantities over time.

  Args:
    perf: inventory performance dictionary
    i: index of time series
    title: title of plot

  Returns:

  """
  fig = plt.figure()
  _plot_nonnans(perf['inventory_positions'], i, label='ip_t')
  _plot_nonnans(perf['net_inventory_levels'], i, label='i_t')
  _plot_nonnans(perf['work_in_progress'], i, label='w_t')
  _plot_nonnans(perf['demand'], i, label='d_t')
  _plot_nonnans(perf['orders'], i, label='o_t')
  _plot_nonnans(perf['safety_stocks'], i, label='ss_t')
  _plot_nonnans(perf['lead_forecasts'], i, label='pred_DL')
  _plot_nonnans(perf['unfolded_actual_imputed'].sum(dim=-1), i, label='true_DL')

  plt.title(title)
  plt.axhline(0, linestyle='--')
  plt.legend(bbox_to_anchor=(1.05, 1.0))
  return fig
