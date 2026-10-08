# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Plotting utilities for residual coherence and error visualization."""

import itertools
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import scipy.sparse as sp

from dp_clip import dp_clip_lib


def compute_rc_curve(
    data,
    *,
    beta,
    rho_sum,
):
  """Computes candidate bounds, residual coherence values, and threshold.

  Args:
    data: A scipy.sparse matrix of data vectors.
    beta: The geometric base for the sequence of candidate bounds.
    rho_sum: The privacy budget parameter for the final vector sum.

  Returns:
    A tuple (candidate_bounds, rc_vals, threshold) truncated shortly after the
    residual coherence curve first drops below threshold.
  """
  _, d = data.shape
  threshold = math.sqrt(2.0 * d / rho_sum)
  rc = dp_clip_lib.precompute_rc(data, beta)
  if rc.size == 0:
    return np.array([]), np.array([]), threshold

  rc_vals = np.append(rc, 0.0)
  num_bins = len(rc_vals)
  candidate_bounds = beta ** np.arange(num_bins)

  below_thresh_indices = np.where(rc_vals < threshold)[0]
  if below_thresh_indices.size > 0:
    stop_idx = min(below_thresh_indices[0] + 11, num_bins)
  else:
    stop_idx = num_bins

  return candidate_bounds[:stop_idx], rc_vals[:stop_idx], threshold


def plot_rc(
    data,
    *,
    beta,
    rho_sum,
    dataset_name,
    optimal_clip,
    output_dir = '.',
    log_y = False,
):
  """Plots residual coherence against candidate clipping bounds and saves it.

  Args:
    data: A scipy.sparse matrix of data vectors.
    beta: The geometric base for the sequence of candidate bounds.
    rho_sum: The privacy budget parameter for the final vector sum.
    dataset_name: String name of the dataset for the plot title and filename.
    optimal_clip: The MSE-optimal clipping bound to mark on the plot.
    output_dir: String path to directory to save the plot.
    log_y: Boolean whether to use a logarithmic scale on the y-axis.
  """
  x_vals, y_vals, threshold = compute_rc_curve(
      data, beta=beta, rho_sum=rho_sum
  )
  if x_vals.size == 0:
    print('No valid vectors found above the lower bound.')
    return

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(
      x_vals,
      y_vals,
      marker='o',
      markersize=6,
      linewidth=2.5,
      label='Residual Coherence',
  )
  ax.axhline(
      y=threshold,
      color='r',
      linestyle='--',
      linewidth=3,
      label=f'Threshold ({threshold:.2f})',
  )
  ax.axvline(
      x=optimal_clip,
      color='g',
      linestyle=':',
      linewidth=3,
      label=f'Optimal Bound ({optimal_clip:.2f})',
  )
  ax.set_xlabel('Candidate Clipping Bound', fontsize=16)
  ax.set_ylabel('Residual Coherence', fontsize=16)
  if log_y:
    ax.set_yscale('log')
  ax.set_title(f'Dataset: {dataset_name}', fontsize=18)
  ax.tick_params(axis='both', labelsize=14)
  ax.xaxis.set_major_locator(plt.MaxNLocator(4))
  ax.legend(fontsize=14)
  ax.grid(True, linestyle='--', alpha=0.6)

  out_path = os.path.join(output_dir, f'rc_{dataset_name}.png')
  fig.savefig(out_path, dpi=300, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved plot to {out_path}')


def plot_all_rc(
    datasets,
    *,
    rc_curves,
    output_dir = '.',
    filename = 'rc_combined_300dpi.png',
    log_y = True,
):
  """Plots a multi-subplot grid of residual coherence curves across datasets.

  Args:
    datasets: List of dataset names to plot.
    rc_curves: Dict mapping each dataset name to a tuple
      (candidate_bounds, rc_vals, threshold, optimal_clip).
    output_dir: String path to directory to save the plot.
    filename: Output filename for the saved figure.
    log_y: Boolean whether to use a logarithmic scale on the y-axis.
  """
  num_datasets = len(datasets)
  if num_datasets == 0:
    return

  cols = 2
  rows = math.ceil(num_datasets / cols)
  fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
  axes = axes.flatten()

  for i, dataset in enumerate(datasets):
    ax = axes[i]
    x_vals, y_vals, threshold, optimal_clip = rc_curves[dataset]
    ax.plot(
        x_vals,
        y_vals,
        marker='o',
        markersize=5,
        linewidth=2.5,
        label='Residual Coherence',
    )
    ax.axhline(
        y=threshold,
        color='r',
        linestyle='--',
        linewidth=2.5,
        label=f'Threshold ({threshold:.2f})',
    )
    ax.axvline(
        x=optimal_clip,
        color='g',
        linestyle=':',
        linewidth=2.5,
        label=f'Optimal Bound ({optimal_clip:.2f})',
    )
    ax.set_title(f'Dataset: {dataset}', fontsize=16)
    if i >= num_datasets - cols:
      ax.set_xlabel('Candidate Clipping Bound', fontsize=14)
    if i % cols == 0:
      ax.set_ylabel('Residual Coherence', fontsize=14)
    if log_y:
      ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

  for i in range(num_datasets, len(axes)):
    fig.delaxes(axes[i])

  fig.tight_layout()
  out_path = os.path.join(output_dir, filename)
  fig.savefig(out_path, dpi=300, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved plot to {out_path}')


def plot_histogram_errors(
    datasets,
    *,
    clip_rho_range,
    all_errors,
    output_dir = '.',
    filename = 'histogram_errors.png',
    opt_name = 'Optimal',
    show_error_bars = False,
):
  """Plots histogram MSE errors for various clipping budgets and saves to disk.

  Args:
    datasets: List of dataset names.
    clip_rho_range: Numpy array of privacy budgets.
    all_errors: Dict keyed by dataset (and method), where each value is a numpy
      array of histogram errors for each clip rho.
    output_dir: String path to directory to save the plot.
    filename: Output filename for the saved figure.
    opt_name: String name of the optimal estimator.
    show_error_bars: Boolean whether to display error bars.
  """
  num_datasets = len(datasets)
  if num_datasets == 0:
    print('No datasets provided.')
    return

  cols = 2
  rows = math.ceil(num_datasets / cols)

  fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
  axes = axes.flatten()

  line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

  for i, dataset in enumerate(datasets):
    ax = axes[i]
    errors = all_errors[dataset]

    opt_mean_errors = np.mean(errors[opt_name], axis=1)
    opt_mean_errors = np.maximum(opt_mean_errors, 1e-9)

    style_cycler = itertools.cycle(line_styles)

    for name, err_array in errors.items():
      if name == opt_name:
        continue

      mean_errors = np.mean(err_array, axis=1)
      ratio_means = mean_errors / opt_mean_errors

      y_err_vals = None
      if show_error_bars:
        std_errors = np.std(err_array, axis=1)
        y_err_vals = std_errors / opt_mean_errors

      ax.errorbar(
          clip_rho_range,
          ratio_means,
          yerr=y_err_vals,
          label=name,
          marker='o',
          markersize=7,
          capsize=4 if show_error_bars else 0,
          linestyle=next(style_cycler),
          linewidth=2.5,
      )

    ax.set_title(f'Dataset: {dataset}', fontsize=16)
    if i >= num_datasets - cols:
      ax.set_xlabel(
          r'Clipping Privacy Budget ($\rho_{\text{clip}}$)', fontsize=14
      )
    if i % cols == 0:
      ax.set_ylabel('Error Ratio (vs Optimal)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(True, linestyle='--', alpha=0.6)

  for i in range(num_datasets, len(axes)):
    fig.delaxes(axes[i])

  handles, labels = axes[0].get_legend_handles_labels()
  fig.tight_layout()

  fig.legend(
      handles,
      labels,
      loc='upper center',
      bbox_to_anchor=(0.5, -0.02),
      ncol=min(len(labels), 5),
      alignment='center',
      fontsize=13,
  )

  out_path = os.path.join(output_dir, filename)
  fig.savefig(out_path, dpi=300, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved plot to {out_path}')


def plot_clip_estimates(
    datasets,
    *,
    clip_rho_range,
    clip_estimates,
    output_dir = '.',
    filename = 'clip_estimates.png',
    opt_name = 'Optimal',
    show_error_bars = False,
):
  """Plots estimated clipping bounds for various estimators and saves to disk.

  Args:
    datasets: List of dataset names.
    clip_rho_range: Numpy array of privacy budgets.
    clip_estimates: Dict keyed by dataset (and method), where each value is a
      numpy array of clip estimates for each clip rho.
    output_dir: String path to directory to save the plot.
    filename: Output filename for the saved figure.
    opt_name: String name of the optimal estimator.
    show_error_bars: Boolean whether to display error bars.
  """
  num_datasets = len(datasets)
  if num_datasets == 0:
    print('No datasets provided.')
    return

  cols = 2
  rows = math.ceil(num_datasets / cols)

  fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
  axes = axes.flatten()

  line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

  for i, dataset in enumerate(datasets):
    ax = axes[i]
    estimates = clip_estimates[dataset]

    opt_values = np.array(estimates[opt_name])
    opt_means = np.mean(opt_values, axis=1)
    opt_means = np.maximum(opt_means, 1e-9)

    style_cycler = itertools.cycle(line_styles)

    ax.axhline(
        y=1.0,
        color='k',
        linestyle='-',
        linewidth=2.5,
        label=opt_name,
    )

    for name, est_array in estimates.items():
      if name == opt_name:
        continue

      est_array = np.array(est_array)
      mean_vals = np.mean(est_array, axis=1)
      ratio_means = mean_vals / opt_means

      y_err_vals = None
      if show_error_bars:
        std_vals = np.std(est_array, axis=1)
        y_err_vals = std_vals / opt_means

      ax.errorbar(
          clip_rho_range,
          ratio_means,
          yerr=y_err_vals,
          label=name,
          marker='o',
          markersize=7,
          capsize=4 if show_error_bars else 0,
          linestyle=next(style_cycler),
          linewidth=2.5,
      )

    ax.set_title(f'Bound Estimation: {dataset}', fontsize=16)
    if i >= num_datasets - cols:
      ax.set_xlabel(
          r'Clipping Privacy Budget ($\rho_{\text{clip}}$)', fontsize=14
      )
    if i % cols == 0:
      ax.set_ylabel('Estimate Ratio (vs Optimal)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(True, linestyle='--', alpha=0.6)

  for i in range(num_datasets, len(axes)):
    fig.delaxes(axes[i])

  handles, labels = axes[0].get_legend_handles_labels()
  fig.tight_layout()

  fig.legend(
      handles,
      labels,
      loc='upper center',
      bbox_to_anchor=(0.5, -0.02),
      ncol=min(len(labels), 6),
      alignment='center',
      fontsize=13,
  )

  out_path = os.path.join(output_dir, filename)
  fig.savefig(out_path, dpi=300, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved plot to {out_path}')
