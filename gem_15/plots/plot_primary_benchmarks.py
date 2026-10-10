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

"""Generates publication-ready benchmark charts for all primary LiDAR metrics.

Reads metrics directly from compiled CSVs rather than hardcoded values:
1. Tile-Level Debiased RMSE (All 12 Datasets)
2. Overall Direct RMSE (Spaceborne & Continuous National Aerial LiDAR)
3. Mean Vertical Bias (Spaceborne & Continuous National Aerial LiDAR)
"""

import os
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gem_15.plots.benchmark_data import add_category_dividers_and_banners
from gem_15.plots.benchmark_data import filter_dtm_benchmarks
from gem_15.plots.benchmark_data import load_or_compile_benchmark_summary
from gem_15.plots.benchmark_data import save_plot

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_dir',
    None,
    'Directory containing validation dataset results (e.g.'
    ' validation_data_gem15).',
)
flags.DEFINE_string(
    'summary_csv',
    None,
    'Path to pre-compiled benchmark summary CSV.',
)
flags.DEFINE_string(
    'output_dir',
    None,
    'Output directory to save generated plots.',
    required=True,
)
flags.DEFINE_string(
    'title_suffix',
    '',
    'Optional suffix to append to DTM/DSM plot titles (e.g. " (Slope > 5°)").',
)
flags.DEFINE_string(
    'output_suffix',
    '',
    'Optional suffix to append before .png in output filenames (e.g.'
    ' "_slope_gt_5").',
)
flags.DEFINE_float(
    'min_slope_deg',
    0.0,
    'Minimum terrain slope in degrees for filtering tiles (e.g. 5.0).',
)
flags.DEFINE_bool(
    'dtm_only',
    False,
    'If true, only generate DTM benchmark plots.',
)


def plot_tile_debiased_rmse(df, output_dir):
  """Plot Tile-Level Debiased RMSE across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_debiased_rmse'].tolist()
  cop_vals = df['cop_debiased_rmse'].tolist()
  alos_vals = df['alos_debiased_rmse'].tolist()
  nasa_vals = df['nasa_debiased_rmse'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.18

  c_ge = '#ff7f0e'
  c_cop = '#2ca02c'
  c_alos = '#1f77b4'
  c_nasa = '#d62728'

  fig, ax = plt.subplots(figsize=(32, 11), dpi=300)

  r1 = ax.bar(
      x - 1.5 * bar_width,
      ge_vals,
      bar_width,
      label='GEM-15 DSM v1 (15m)',
      color=c_ge,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r2 = ax.bar(
      x - 0.5 * bar_width,
      cop_vals,
      bar_width,
      label='Copernicus DEM (30m)',
      color=c_cop,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r3 = ax.bar(
      x + 0.5 * bar_width,
      alos_vals,
      bar_width,
      label='ALOS AW3D30 (30m)',
      color=c_alos,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r4 = ax.bar(
      x + 1.5 * bar_width,
      nasa_vals,
      bar_width,
      label='NASADEM (30m)',
      color=c_nasa,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )

  all_vals = [
      h for h in ge_vals + cop_vals + alos_vals + nasa_vals if not np.isnan(h)
  ]
  max_val = max(all_vals) if all_vals else 10.0
  y_top = max_val * 1.22
  ax.set_ylim(0, y_top)
  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i], r3[i], r4[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    best_val = min(valid_vals) if valid_vals else None

    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      fw = (
          'bold'
          if best_val is not None and np.isclose(h, best_val, atol=1e-4)
          else 'normal'
      )
      ax.annotate(
          f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, 4),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=12.5,
          fontweight=fw,
      )

  y_banner = y_top * 0.91
  add_category_dividers_and_banners(
      ax, df, y_banner, aerial_label='AERIAL LiDAR'
  )

  ax.set_ylabel(
      'Debiased RMSE (m)',
      fontsize=17,
      fontweight='bold',
      labelpad=12,
  )
  ax.set_title(
      'Overall DSM: Debiased RMSE',
      fontsize=19.5,
      fontweight='bold',
      pad=25,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=13.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(output_dir, 'tile_level_debiased_rmse_benchmark.png')
  save_plot(fig, out_path, dpi=300)


def plot_overall_rmse(df, output_dir):
  """Plot Overall Direct Vertical RMSE across Spaceborne & Continuous Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_direct_rmse'].tolist()
  cop_vals = df_sub['cop_direct_rmse'].tolist()
  alos_vals = df_sub['alos_direct_rmse'].tolist()
  nasa_vals = df_sub['nasa_direct_rmse'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.18

  c_ge = '#ff7f0e'
  c_cop = '#2ca02c'
  c_alos = '#1f77b4'
  c_nasa = '#d62728'

  fig, ax = plt.subplots(figsize=(26, 11), dpi=300)

  r1 = ax.bar(
      x - 1.5 * bar_width,
      ge_vals,
      bar_width,
      label='GEM-15 DSM v1 (15m)',
      color=c_ge,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r2 = ax.bar(
      x - 0.5 * bar_width,
      cop_vals,
      bar_width,
      label='Copernicus DEM (30m)',
      color=c_cop,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r3 = ax.bar(
      x + 0.5 * bar_width,
      alos_vals,
      bar_width,
      label='ALOS AW3D30 (30m)',
      color=c_alos,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r4 = ax.bar(
      x + 1.5 * bar_width,
      nasa_vals,
      bar_width,
      label='NASADEM (30m)',
      color=c_nasa,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )

  all_vals = [
      h for h in ge_vals + cop_vals + alos_vals + nasa_vals if not np.isnan(h)
  ]
  max_val = max(all_vals) if all_vals else 12.0
  y_top = max_val * 1.22
  ax.set_ylim(0, y_top)
  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i], r3[i], r4[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    best_val = min(valid_vals) if valid_vals else None

    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      fw = (
          'bold'
          if best_val is not None and np.isclose(h, best_val, atol=1e-4)
          else 'normal'
      )
      ax.annotate(
          f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, 4),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=12.5,
          fontweight=fw,
      )

  y_banner = y_top * 0.91
  add_category_dividers_and_banners(ax, df_sub, y_banner)

  ax.set_ylabel(
      'RMSE (m)',
      fontsize=17,
      fontweight='bold',
      labelpad=12,
  )
  ax.set_title(
      'Overall DSM: RMSE',
      fontsize=19.5,
      fontweight='bold',
      pad=25,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=14.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(output_dir, 'overall_rmse_benchmark.png')
  save_plot(fig, out_path, dpi=300)


def plot_bias(df, output_dir):
  """Plot Mean Vertical Bias across Spaceborne & Continuous Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_bias'].tolist()
  cop_vals = df_sub['cop_bias'].tolist()
  alos_vals = df_sub['alos_bias'].tolist()
  nasa_vals = df_sub['nasa_bias'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.18

  c_ge = '#ff7f0e'
  c_cop = '#2ca02c'
  c_alos = '#1f77b4'
  c_nasa = '#d62728'

  fig, ax = plt.subplots(figsize=(26, 11), dpi=300)

  r1 = ax.bar(
      x - 1.5 * bar_width,
      ge_vals,
      bar_width,
      label='GEM-15 DSM v1 (15m)',
      color=c_ge,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r2 = ax.bar(
      x - 0.5 * bar_width,
      cop_vals,
      bar_width,
      label='Copernicus DEM (30m)',
      color=c_cop,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r3 = ax.bar(
      x + 0.5 * bar_width,
      alos_vals,
      bar_width,
      label='ALOS AW3D30 (30m)',
      color=c_alos,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r4 = ax.bar(
      x + 1.5 * bar_width,
      nasa_vals,
      bar_width,
      label='NASADEM (30m)',
      color=c_nasa,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )

  all_vals = [
      h for h in ge_vals + cop_vals + alos_vals + nasa_vals if not np.isnan(h)
  ]
  min_val = min(all_vals) if all_vals else -6.5
  max_val = max(all_vals) if all_vals else 2.0
  y_min = np.floor(min_val - 1.5)
  y_max = np.ceil(max_val + 2.5)
  ax.set_ylim(y_min, y_max)
  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i], r3[i], r4[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    best_abs_val = min([abs(h) for h in valid_vals]) if valid_vals else None

    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      if best_abs_val is not None and np.isclose(
          abs(h), best_abs_val, atol=1e-4
      ):
        fw = 'bold'
      else:
        fw = 'normal'
      offset = 4 if h >= 0 else -4
      va = 'bottom' if h >= 0 else 'top'
      sign = '+' if h > 0 else ''
      ax.annotate(
          f'{sign}{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, offset),
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=10.5,
          fontweight=fw,
      )

  ax.axhline(0, color='black', linewidth=1.2, linestyle='-', zorder=2)
  y_banner = y_max - (y_max - y_min) * 0.08
  add_category_dividers_and_banners(ax, df_sub, y_banner)

  ax.set_ylabel(
      'Mean Vertical Bias (m)',
      fontsize=17,
      fontweight='bold',
      labelpad=12,
  )
  ax.set_title(
      'Overall DSM: Bias',
      fontsize=19.5,
      fontweight='bold',
      pad=25,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=14.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='lower right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(output_dir, 'mean_bias_benchmark.png')
  save_plot(fig, out_path, dpi=300)


def plot_tile_debiased_rmse_dtm(df, output_dir):
  """Plot Tile-Level Debiased RMSE for DTM across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_dtm_debiased_rmse'].tolist()
  fab_vals = df['fab_debiased_rmse'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.28

  c_ge = '#ff7f0e'
  c_fab = '#2ca02c'

  fig, ax = plt.subplots(figsize=(32, 11), dpi=300)

  r1 = ax.bar(
      x - 0.5 * bar_width,
      ge_vals,
      bar_width,
      label='GEM-15 DTM v1 (15m)',
      color=c_ge,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r2 = ax.bar(
      x + 0.5 * bar_width,
      fab_vals,
      bar_width,
      label='FABDEM (30m)',
      color=c_fab,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )

  all_vals = [h for h in ge_vals + fab_vals if not np.isnan(h)]
  max_val = max(all_vals) if all_vals else 10.0
  y_top = max_val * 1.22
  ax.set_ylim(0, y_top)
  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    best_val = min(valid_vals) if valid_vals else None

    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      fw = (
          'bold'
          if best_val is not None and np.isclose(h, best_val, atol=1e-4)
          else 'normal'
      )
      ax.annotate(
          f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, 4),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=12.5,
          fontweight=fw,
      )

  y_banner = y_top * 0.91
  add_category_dividers_and_banners(
      ax, df, y_banner, aerial_label='AERIAL LiDAR'
  )

  ax.set_ylabel(
      'Debiased RMSE (m)',
      fontsize=17,
      fontweight='bold',
      labelpad=12,
  )
  ax.set_title(
      f'Overall DTM{FLAGS.title_suffix}: Debiased RMSE',
      fontsize=22,
      fontweight='bold',
      pad=20,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=13.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=2,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(
      output_dir,
      f'tile_level_debiased_rmse_dtm_benchmark{FLAGS.output_suffix}.png',
  )
  save_plot(fig, out_path, dpi=300)


def plot_overall_rmse_dtm(df, output_dir):
  """Plot Overall Direct Vertical RMSE for DTM across Spaceborne & Continuous Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_dtm_direct_rmse'].tolist()
  fab_vals = df_sub['fab_direct_rmse'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.28

  c_ge = '#ff7f0e'
  c_fab = '#2ca02c'

  fig, ax = plt.subplots(figsize=(26, 11), dpi=300)

  r1 = ax.bar(
      x - 0.5 * bar_width,
      ge_vals,
      bar_width,
      label='GEM-15 DTM v1 (15m)',
      color=c_ge,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r2 = ax.bar(
      x + 0.5 * bar_width,
      fab_vals,
      bar_width,
      label='FABDEM (30m)',
      color=c_fab,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )

  all_vals = [h for h in ge_vals + fab_vals if not np.isnan(h)]
  max_val = max(all_vals) if all_vals else 12.0
  y_top = max_val * 1.22
  ax.set_ylim(0, y_top)
  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    best_val = min(valid_vals) if valid_vals else None

    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      fw = (
          'bold'
          if best_val is not None and np.isclose(h, best_val, atol=1e-4)
          else 'normal'
      )
      ax.annotate(
          f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, 4),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=12.5,
          fontweight=fw,
      )

  y_banner = y_top * 0.91
  add_category_dividers_and_banners(ax, df_sub, y_banner)

  ax.set_ylabel(
      'RMSE (m)',
      fontsize=17,
      fontweight='bold',
      labelpad=12,
  )
  ax.set_title(
      f'Overall DTM{FLAGS.title_suffix}: RMSE',
      fontsize=22,
      fontweight='bold',
      pad=20,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=14.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=2,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(
      output_dir, f'overall_rmse_dtm_benchmark{FLAGS.output_suffix}.png'
  )
  save_plot(fig, out_path, dpi=300)


def plot_bias_dtm(df, output_dir):
  """Plot Mean Vertical Bias for DTM across Spaceborne & Continuous Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_dtm_bias'].tolist()
  fab_vals = df_sub['fab_bias'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.28

  c_ge = '#ff7f0e'
  c_fab = '#2ca02c'

  fig, ax = plt.subplots(figsize=(26, 11), dpi=300)

  r1 = ax.bar(
      x - 0.5 * bar_width,
      ge_vals,
      bar_width,
      label='GEM-15 DTM v1 (15m)',
      color=c_ge,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r2 = ax.bar(
      x + 0.5 * bar_width,
      fab_vals,
      bar_width,
      label='FABDEM (30m)',
      color=c_fab,
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )

  ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=2)

  all_vals = [h for h in ge_vals + fab_vals if not np.isnan(h)]
  min_val = min(all_vals) if all_vals else -2.0
  max_val = max(all_vals) if all_vals else 2.0
  abs_max = max(abs(min_val), abs(max_val)) * 1.35
  ax.set_ylim(-abs_max, abs_max)
  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_abs = [abs(h) for h in h_vals if not np.isnan(h)]
    best_abs = min(valid_abs) if valid_abs else None

    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      fw = (
          'bold'
          if best_abs is not None and np.isclose(abs(h), best_abs, atol=1e-4)
          else 'normal'
      )
      offset_y = 5 if h >= 0 else -16
      ax.annotate(
          f'{h:+.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, offset_y),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=12.5,
          fontweight=fw,
      )

  y_banner = abs_max * 0.82
  add_category_dividers_and_banners(ax, df_sub, y_banner)

  ax.set_ylabel(
      'Mean Vertical Bias (m)',
      fontsize=17,
      fontweight='bold',
      labelpad=12,
  )
  ax.set_title(
      f'Overall DTM{FLAGS.title_suffix}: Bias',
      fontsize=22,
      fontweight='bold',
      pad=20,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=14.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='lower right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=2,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(
      output_dir, f'mean_bias_dtm_benchmark{FLAGS.output_suffix}.png'
  )
  save_plot(fig, out_path, dpi=300)


def plot_tile_nmad(df, output_dir):
  """Plot Tile-Level Exact NMAD across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_nmad'].tolist()
  cop_vals = df['cop_nmad'].tolist()
  alos_vals = df['alos_nmad'].tolist()
  nasa_vals = df['nasa_nmad'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.18

  fig, ax = plt.subplots(figsize=(32, 11), dpi=300)
  r1 = ax.bar(
      x - 1.5 * bar_width,
      ge_vals,
      bar_width,
      label='GEM-15 DSM v1 (15m)',
      color='#ff7f0e',
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r2 = ax.bar(
      x - 0.5 * bar_width,
      cop_vals,
      bar_width,
      label='Copernicus DEM (30m)',
      color='#2ca02c',
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r3 = ax.bar(
      x + 0.5 * bar_width,
      alos_vals,
      bar_width,
      label='ALOS AW3D30 (30m)',
      color='#1f77b4',
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r4 = ax.bar(
      x + 1.5 * bar_width,
      nasa_vals,
      bar_width,
      label='NASADEM (30m)',
      color='#d62728',
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )

  all_vals = [
      h for h in ge_vals + cop_vals + alos_vals + nasa_vals if not np.isnan(h)
  ]
  max_val = max(all_vals) if all_vals else 10.0
  y_top = max_val * 1.22
  ax.set_ylim(0, y_top)
  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i], r3[i], r4[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    best_val = min(valid_vals) if valid_vals else None
    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      fw = (
          'bold'
          if best_val is not None and np.isclose(h, best_val, atol=1e-4)
          else 'normal'
      )
      ax.annotate(
          f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, 4),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=12.5,
          fontweight=fw,
      )

  y_banner = y_top * 0.91
  add_category_dividers_and_banners(
      ax, df, y_banner, aerial_label='AERIAL LiDAR'
  )
  ax.set_ylabel('NMAD (m)', fontsize=17, fontweight='bold', labelpad=12)
  ax.set_title(
      'Overall DSM: NMAD (Normalized Median Absolute Deviation)',
      fontsize=19.5,
      fontweight='bold',
      pad=25,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=13.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)
  plt.tight_layout()
  save_plot(
      fig, os.path.join(output_dir, 'tile_level_nmad_benchmark.png'), dpi=300
  )


def plot_tile_nmad_dtm(df, output_dir):
  """Plot Tile-Level Exact NMAD for DTM across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_dtm_nmad'].tolist()
  fab_vals = df['fab_nmad'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.28

  fig, ax = plt.subplots(figsize=(30, 11), dpi=300)
  r1 = ax.bar(
      x - 0.5 * bar_width,
      ge_vals,
      bar_width,
      label='GEM-15 DTM v1 (15m)',
      color='#ff7f0e',
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )
  r2 = ax.bar(
      x + 0.5 * bar_width,
      fab_vals,
      bar_width,
      label='FABDEM (30m)',
      color='#2ca02c',
      edgecolor='black',
      linewidth=0.8,
      zorder=3,
  )

  all_vals = [h for h in ge_vals + fab_vals if not np.isnan(h)]
  max_val = max(all_vals) if all_vals else 10.0
  y_top = max_val * 1.22
  ax.set_ylim(0, y_top)
  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    best_val = min(valid_vals) if valid_vals else None
    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      fw = (
          'bold'
          if best_val is not None and np.isclose(h, best_val, atol=1e-4)
          else 'normal'
      )
      ax.annotate(
          f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, 4),
          textcoords='offset points',
          ha='center',
          va='bottom',
          fontsize=12.5,
          fontweight=fw,
      )

  y_banner = y_top * 0.91
  add_category_dividers_and_banners(
      ax, df, y_banner, aerial_label='AERIAL LiDAR'
  )
  ax.set_ylabel('NMAD (m)', fontsize=17, fontweight='bold', labelpad=12)
  ax.set_title(
      f'Overall DTM{FLAGS.title_suffix}: NMAD',
      fontsize=22,
      fontweight='bold',
      pad=20,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=13.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=2,
  )
  legend.get_frame().set_linewidth(1.3)
  plt.tight_layout()
  save_plot(
      fig,
      os.path.join(
          output_dir, f'tile_level_nmad_dtm_benchmark{FLAGS.output_suffix}.png'
      ),
      dpi=300,
  )


def _draw_grouped_bars_on_ax(
    ax,
    df_sub,
    series_specs,
    ylabel,
    panel_title,
    is_bias = False,
    show_banners = True,
    show_xticks = True,
    show_legend = False,
    bar_width = 0.19,
):
  """Helper to draw a single grouped bar panel inside a stacked benchmark figure."""
  labels = df_sub['label'].tolist()
  n_groups = len(labels)
  x = np.arange(n_groups)
  n_series = len(series_specs)
  offsets = np.linspace(
      -bar_width * (n_series - 1) / 2.0,
      bar_width * (n_series - 1) / 2.0,
      n_series,
  )

  rects = []
  all_vals = []
  for (col, leg_label, color), offset in zip(series_specs, offsets):
    vals = df_sub[col].tolist()
    all_vals.extend([v for v in vals if not np.isnan(v)])
    r = ax.bar(
        x + offset,
        vals,
        bar_width,
        label=leg_label,
        color=color,
        edgecolor='black',
        linewidth=0.85,
        zorder=3,
    )
    rects.append(r)

  if is_bias:
    ax.axhline(0, color='black', linewidth=1.4, linestyle='-', zorder=2)
    min_val = min(all_vals) if all_vals else -3.0
    max_val = max(all_vals) if all_vals else 3.0
    abs_max = max(abs(min_val), abs(max_val)) * 1.56
    y_min, y_max = -abs_max, abs_max
    ax.set_ylim(y_min, y_max)
    y_banner = y_max - (y_max - y_min) * 0.10
  else:
    max_val = max(all_vals) if all_vals else 10.0
    y_top = max_val * 1.38
    ax.set_ylim(0, y_top)
    y_banner = y_top * 0.90

  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r[i] for r in rects]
    h_vals = [rect.get_height() for rect in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    if is_bias:
      best_target = min([abs(h) for h in valid_vals]) if valid_vals else None
    else:
      best_target = min(valid_vals) if valid_vals else None

    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      cmp_val = abs(h) if is_bias else h
      is_best = best_target is not None and np.isclose(
          cmp_val, best_target, atol=1e-4
      )
      fw = 'bold' if is_best else 'normal'
      if is_bias:
        offset_y = 6 if h >= 0 else -6
        va = 'bottom' if h >= 0 else 'top'
        txt = f'{h:+.2f}'
      else:
        offset_y = 6
        va = 'bottom'
        txt = f'{h:.2f}'
      ax.annotate(
          txt,
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, offset_y),
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=16.0 if n_series > 2 else 17.5,
          fontweight=fw,
          rotation=90,
      )

  # If this is a bias plot with Drone rows left empty, add a note in Drone span
  if is_bias and 'category' in df_sub.columns:
    d_idx = [
        idx
        for idx, cat in enumerate(df_sub['category'].tolist())
        if cat == 'Drone'
    ]
    if d_idx:
      ax.text(
          (d_idx[0] + d_idx[-1]) / 2.0,
          0.0,
          'Local Ellipsoidal Datum\n(Mean Bias Excluded)',
          ha='center',
          va='center',
          fontsize=16.5,
          fontstyle='italic',
          fontweight='bold',
          color='#666666',
          bbox=dict(
              boxstyle='round,pad=0.5',
              facecolor='#fafafa',
              edgecolor='#cccccc',
              alpha=0.92,
              lw=1.2,
          ),
          zorder=4,
      )

  if show_banners:
    add_category_dividers_and_banners(
        ax, df_sub, y_banner, aerial_label='AERIAL LiDAR'
    )
  else:
    cats = df_sub['category'].tolist()
    n = len(cats)
    for cat_name in ['Spaceborne', 'Drone']:
      idx_list = [i for i, c in enumerate(cats) if c == cat_name]
      if idx_list and idx_list[-1] < n - 1:
        ax.axvline(
            x=idx_list[-1] + 0.5,
            color='#444444',
            linestyle=':',
            linewidth=2.0,
            zorder=2,
        )

  ax.set_ylabel(ylabel, fontsize=20.5, fontweight='bold', labelpad=12)
  ax.set_title(
      panel_title, fontsize=23.5, fontweight='bold', pad=14, loc='left'
  )
  ax.set_xticks(x)
  if show_xticks:
    ax.set_xticklabels(labels, fontsize=16.0, fontweight='bold')
  else:
    ax.set_xticklabels([])
  ax.tick_params(axis='y', labelsize=17.5)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  if show_legend:
    legend = ax.legend(
        loc='upper right',
        frameon=True,
        facecolor='white',
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=19.0,
        ncol=2 if len(series_specs) >= 4 else len(series_specs),
    )
    legend.get_frame().set_linewidth(1.4)


def plot_stacked_dsm_benchmark(
    df, output_dir, include_direct_rmse = False
):
  """Plot combined vertically stacked DSM benchmark figure (3-panel by default)."""
  # Keep full df layout for bias/direct_rmse and leave Drone region empty (NaN)
  # so that every dataset column aligns 100% vertically across all panels.
  df_aligned_sub = df.copy()
  drone_mask = df_aligned_sub['category'] == 'Drone'
  for col in [
      'ge_direct_rmse',
      'cop_direct_rmse',
      'alos_direct_rmse',
      'nasa_direct_rmse',
      'ge_bias',
      'cop_bias',
      'alos_bias',
      'nasa_bias',
  ]:
    if col in df_aligned_sub.columns:
      df_aligned_sub.loc[drone_mask, col] = np.nan

  dsm_specs_drmse = [
      ('ge_debiased_rmse', 'GEM-15 DSM v1 (15m)', '#ff7f0e'),
      ('cop_debiased_rmse', 'Copernicus DEM (30m)', '#2ca02c'),
      ('alos_debiased_rmse', 'ALOS AW3D30 (30m)', '#1f77b4'),
      ('nasa_debiased_rmse', 'NASADEM (30m)', '#d62728'),
  ]
  dsm_specs_nmad = [
      ('ge_nmad', 'GEM-15 DSM v1 (15m)', '#ff7f0e'),
      ('cop_nmad', 'Copernicus DEM (30m)', '#2ca02c'),
      ('alos_nmad', 'ALOS AW3D30 (30m)', '#1f77b4'),
      ('nasa_nmad', 'NASADEM (30m)', '#d62728'),
  ]
  dsm_specs_rmse = [
      ('ge_direct_rmse', 'GEM-15 DSM v1 (15m)', '#ff7f0e'),
      ('cop_direct_rmse', 'Copernicus DEM (30m)', '#2ca02c'),
      ('alos_direct_rmse', 'ALOS AW3D30 (30m)', '#1f77b4'),
      ('nasa_direct_rmse', 'NASADEM (30m)', '#d62728'),
  ]
  dsm_specs_bias = [
      ('ge_bias', 'GEM-15 DSM v1 (15m)', '#ff7f0e'),
      ('cop_bias', 'Copernicus DEM (30m)', '#2ca02c'),
      ('alos_bias', 'ALOS AW3D30 (30m)', '#1f77b4'),
      ('nasa_bias', 'NASADEM (30m)', '#d62728'),
  ]

  nrows = 4 if include_direct_rmse else 3
  fig, axes = plt.subplots(nrows, 1, figsize=(34, 8.6 * nrows), dpi=300)

  _draw_grouped_bars_on_ax(
      axes[0],
      df,
      dsm_specs_drmse,
      ylabel='Debiased RMSE (m)',
      panel_title='(a) Overall DSM: Tile-Level Debiased RMSE',
      is_bias=False,
      show_banners=True,
      show_xticks=True,
      show_legend=True,
      bar_width=0.19,
  )
  _draw_grouped_bars_on_ax(
      axes[1],
      df,
      dsm_specs_nmad,
      ylabel='NMAD (m)',
      panel_title=(
          '(b) Overall DSM: Median NMAD (Normalized Median Absolute Deviation)'
      ),
      is_bias=False,
      show_banners=True,
      show_xticks=True,
      show_legend=False,
      bar_width=0.19,
  )
  if include_direct_rmse:
    _draw_grouped_bars_on_ax(
        axes[2],
        df_aligned_sub,
        dsm_specs_rmse,
        ylabel='Direct RMSE (m)',
        panel_title=(
            '(c) Overall DSM: Direct RMSE (Spaceborne & Continuous Aerial'
            ' LiDAR)'
        ),
        is_bias=False,
        show_banners=True,
        show_xticks=True,
        show_legend=False,
        bar_width=0.19,
    )
    _draw_grouped_bars_on_ax(
        axes[3],
        df_aligned_sub,
        dsm_specs_bias,
        ylabel='Mean Bias (m)',
        panel_title=(
            '(d) Overall DSM: Mean Vertical Bias (Spaceborne & Continuous'
            ' Aerial LiDAR)'
        ),
        is_bias=True,
        show_banners=True,
        show_xticks=True,
        show_legend=False,
        bar_width=0.19,
    )
    suffix = '4panel'
  else:
    _draw_grouped_bars_on_ax(
        axes[2],
        df_aligned_sub,
        dsm_specs_bias,
        ylabel='Mean Bias (m)',
        panel_title=(
            '(c) Overall DSM: Mean Vertical Bias (Spaceborne & Continuous'
            ' Aerial LiDAR)'
        ),
        is_bias=True,
        show_banners=True,
        show_xticks=True,
        show_legend=False,
        bar_width=0.19,
    )
    suffix = '3panel'

  plt.tight_layout(h_pad=2.2)
  out_path = os.path.join(
      output_dir, f'dsm_benchmark_stacked_{suffix}{FLAGS.output_suffix}.png'
  )
  save_plot(fig, out_path, dpi=300)


def plot_stacked_dtm_benchmark(
    df, output_dir, include_direct_rmse = False
):
  """Plot combined vertically stacked DTM benchmark figure (3-panel by default)."""
  # Keep full df layout for bias/direct_rmse and leave Drone region empty (NaN)
  # so that every dataset column aligns 100% vertically across all panels.
  df_aligned_sub = df.copy()
  drone_mask = df_aligned_sub['category'] == 'Drone'
  for col in [
      'ge_dtm_direct_rmse',
      'fab_direct_rmse',
      'ge_dtm_bias',
      'fab_bias',
  ]:
    if col in df_aligned_sub.columns:
      df_aligned_sub.loc[drone_mask, col] = np.nan

  dtm_specs_drmse = [
      ('ge_dtm_debiased_rmse', 'GEM-15 DTM v1 (15m)', '#ff7f0e'),
      ('fab_debiased_rmse', 'FABDEM (30m)', '#2ca02c'),
  ]
  dtm_specs_nmad = [
      ('ge_dtm_nmad', 'GEM-15 DTM v1 (15m)', '#ff7f0e'),
      ('fab_nmad', 'FABDEM (30m)', '#2ca02c'),
  ]
  dtm_specs_rmse = [
      ('ge_dtm_direct_rmse', 'GEM-15 DTM v1 (15m)', '#ff7f0e'),
      ('fab_direct_rmse', 'FABDEM (30m)', '#2ca02c'),
  ]
  dtm_specs_bias = [
      ('ge_dtm_bias', 'GEM-15 DTM v1 (15m)', '#ff7f0e'),
      ('fab_bias', 'FABDEM (30m)', '#2ca02c'),
  ]

  nrows = 4 if include_direct_rmse else 3
  fig, axes = plt.subplots(nrows, 1, figsize=(34, 8.6 * nrows), dpi=300)

  _draw_grouped_bars_on_ax(
      axes[0],
      df,
      dtm_specs_drmse,
      ylabel='Debiased RMSE (m)',
      panel_title=(
          f'(a) Overall DTM{FLAGS.title_suffix}: Tile-Level Debiased RMSE'
      ),
      is_bias=False,
      show_banners=True,
      show_xticks=True,
      show_legend=True,
      bar_width=0.28,
  )
  _draw_grouped_bars_on_ax(
      axes[1],
      df,
      dtm_specs_nmad,
      ylabel='NMAD (m)',
      panel_title=f'(b) Overall DTM{FLAGS.title_suffix}: Median NMAD',
      is_bias=False,
      show_banners=True,
      show_xticks=True,
      show_legend=False,
      bar_width=0.28,
  )
  if include_direct_rmse:
    _draw_grouped_bars_on_ax(
        axes[2],
        df_aligned_sub,
        dtm_specs_rmse,
        ylabel='Direct RMSE (m)',
        panel_title=(
            f'(c) Overall DTM{FLAGS.title_suffix}: Direct RMSE (Spaceborne &'
            ' Continuous Aerial LiDAR)'
        ),
        is_bias=False,
        show_banners=True,
        show_xticks=True,
        show_legend=False,
        bar_width=0.28,
    )
    _draw_grouped_bars_on_ax(
        axes[3],
        df_aligned_sub,
        dtm_specs_bias,
        ylabel='Mean Bias (m)',
        panel_title=(
            f'(d) Overall DTM{FLAGS.title_suffix}: Mean Vertical Bias'
            ' (Spaceborne & Continuous Aerial LiDAR)'
        ),
        is_bias=True,
        show_banners=True,
        show_xticks=True,
        show_legend=False,
        bar_width=0.28,
    )
    suffix = '4panel'
  else:
    _draw_grouped_bars_on_ax(
        axes[2],
        df_aligned_sub,
        dtm_specs_bias,
        ylabel='Mean Bias (m)',
        panel_title=(
            f'(c) Overall DTM{FLAGS.title_suffix}: Mean Vertical Bias'
            ' (Spaceborne & Continuous Aerial LiDAR)'
        ),
        is_bias=True,
        show_banners=True,
        show_xticks=True,
        show_legend=False,
        bar_width=0.28,
    )
    suffix = '3panel'

  plt.tight_layout(h_pad=2.2)
  out_path = os.path.join(
      output_dir, f'dtm_benchmark_stacked_{suffix}{FLAGS.output_suffix}.png'
  )
  save_plot(fig, out_path, dpi=300)


def main(argv):
  del argv  # Unused.
  plt.style.use(
      'seaborn-v0_8-whitegrid'
      if 'seaborn-v0_8-whitegrid' in plt.style.available
      else 'default'
  )
  df = load_or_compile_benchmark_summary(
      data_dir=FLAGS.data_dir,
      summary_csv_path=FLAGS.summary_csv,
      min_slope_deg=FLAGS.min_slope_deg,
  )
  if not FLAGS.dtm_only:
    # DSM Benchmarks (individual + 3-panel stacked)
    plot_tile_debiased_rmse(df, FLAGS.output_dir)
    plot_overall_rmse(df, FLAGS.output_dir)
    plot_bias(df, FLAGS.output_dir)
    plot_tile_nmad(df, FLAGS.output_dir)
    plot_stacked_dsm_benchmark(df, FLAGS.output_dir, include_direct_rmse=False)

  # DTM Benchmarks (individual + 3-panel stacked)
  dtm_df = filter_dtm_benchmarks(df)
  plot_tile_debiased_rmse_dtm(dtm_df, FLAGS.output_dir)
  plot_overall_rmse_dtm(dtm_df, FLAGS.output_dir)
  plot_bias_dtm(dtm_df, FLAGS.output_dir)
  plot_tile_nmad_dtm(dtm_df, FLAGS.output_dir)
  plot_stacked_dtm_benchmark(
      dtm_df, FLAGS.output_dir, include_direct_rmse=False
  )


if __name__ == '__main__':
  app.run(main)
