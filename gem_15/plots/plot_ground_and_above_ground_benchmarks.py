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

"""Generates publication-ready benchmark charts for Above Ground and Ground subsets.

Reads metrics directly from compiled CSVs rather than hardcoded values:
1. Tile-Level Debiased RMSE - Above Ground (All 12 Datasets)
2. Tile-Level Debiased RMSE - Ground Level (All 12 Datasets)
3. Overall Direct RMSE - Above Ground (Spaceborne & Continuous Aerial LiDAR)
4. Overall Direct RMSE - Ground Level (Spaceborne & Continuous Aerial LiDAR)
5. Mean Vertical Bias - Above Ground (Spaceborne & Continuous Aerial LiDAR)
6. Mean Vertical Bias - Ground Level (Spaceborne & Continuous Aerial LiDAR)
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


def plot_tile_debiased_rmse_above_ground(df, output_dir):
  """Plot Tile-Level Debiased RMSE for Above Ground across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_debiased_rmse_ag'].tolist()
  cop_vals = df['cop_debiased_rmse_ag'].tolist()
  alos_vals = df['alos_debiased_rmse_ag'].tolist()
  nasa_vals = df['nasa_debiased_rmse_ag'].tolist()

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
      'Above Ground DSM: Debiased RMSE',
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
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(
      output_dir, 'tile_level_debiased_rmse_above_ground_benchmark.png'
  )
  save_plot(fig, out_path, dpi=300)


def plot_tile_debiased_rmse_ground(df, output_dir):
  """Plot Tile-Level Debiased RMSE for Ground across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_debiased_rmse_g'].tolist()
  cop_vals = df['cop_debiased_rmse_g'].tolist()
  alos_vals = df['alos_debiased_rmse_g'].tolist()
  nasa_vals = df['nasa_debiased_rmse_g'].tolist()

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
      'Ground Level DSM: Debiased RMSE',
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
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(
      output_dir, 'tile_level_debiased_rmse_ground_benchmark.png'
  )
  save_plot(fig, out_path, dpi=300)


def plot_overall_rmse_above_ground(df, output_dir):
  """Plot Overall Direct RMSE for Above Ground across Spaceborne & Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_direct_rmse_ag'].tolist()
  cop_vals = df_sub['cop_direct_rmse_ag'].tolist()
  alos_vals = df_sub['alos_direct_rmse_ag'].tolist()
  nasa_vals = df_sub['nasa_direct_rmse_ag'].tolist()

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
      'Above Ground DSM: RMSE',
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
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(output_dir, 'overall_rmse_above_ground_benchmark.png')
  save_plot(fig, out_path, dpi=300)


def plot_overall_rmse_ground(df, output_dir):
  """Plot Overall Direct RMSE for Ground across Spaceborne & Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_direct_rmse_g'].tolist()
  cop_vals = df_sub['cop_direct_rmse_g'].tolist()
  alos_vals = df_sub['alos_direct_rmse_g'].tolist()
  nasa_vals = df_sub['nasa_direct_rmse_g'].tolist()

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
      'Ground Level DSM: RMSE',
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
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(output_dir, 'overall_rmse_ground_benchmark.png')
  save_plot(fig, out_path, dpi=300)


def plot_bias_above_ground(df, output_dir):
  """Plot Mean Vertical Bias for Above Ground across Spaceborne & Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_bias_ag'].tolist()
  cop_vals = df_sub['cop_bias_ag'].tolist()
  alos_vals = df_sub['alos_bias_ag'].tolist()
  nasa_vals = df_sub['nasa_bias_ag'].tolist()

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
      'Above Ground DSM: Bias',
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
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(output_dir, 'mean_bias_above_ground_benchmark.png')
  save_plot(fig, out_path, dpi=300)


def plot_bias_ground(df, output_dir):
  """Plot Mean Vertical Bias for Ground across Spaceborne & Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_bias_g'].tolist()
  cop_vals = df_sub['cop_bias_g'].tolist()
  alos_vals = df_sub['alos_bias_g'].tolist()
  nasa_vals = df_sub['nasa_bias_g'].tolist()

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
  y_max = max(20.0, np.ceil(max_val + 2.5))
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
      'Ground Level DSM: Bias',
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
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(output_dir, 'mean_bias_ground_benchmark.png')
  save_plot(fig, out_path, dpi=300)


def plot_tile_debiased_rmse_above_ground_dtm(df, output_dir):
  """Plot Tile-Level Debiased RMSE for Above Ground DTM across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_dtm_debiased_rmse_ag'].tolist()
  fab_vals = df['fab_debiased_rmse_ag'].tolist()

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
      f'Above Ground DTM{FLAGS.title_suffix}: Debiased RMSE',
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
      f'tile_level_debiased_rmse_above_ground_dtm_benchmark{FLAGS.output_suffix}.png',
  )
  save_plot(fig, out_path, dpi=300)


def plot_tile_debiased_rmse_ground_dtm(df, output_dir):
  """Plot Tile-Level Debiased RMSE for Ground DTM across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_dtm_debiased_rmse_g'].tolist()
  fab_vals = df['fab_debiased_rmse_g'].tolist()

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
      f'Ground Level DTM{FLAGS.title_suffix}: Debiased RMSE',
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
      f'tile_level_debiased_rmse_ground_dtm_benchmark{FLAGS.output_suffix}.png',
  )
  save_plot(fig, out_path, dpi=300)


def plot_overall_rmse_above_ground_dtm(df, output_dir):
  """Plot Overall Direct RMSE for Above Ground DTM across Spaceborne & Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_dtm_direct_rmse_ag'].tolist()
  fab_vals = df_sub['fab_direct_rmse_ag'].tolist()

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
      f'Above Ground DTM{FLAGS.title_suffix}: RMSE',
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
      output_dir,
      f'overall_rmse_above_ground_dtm_benchmark{FLAGS.output_suffix}.png',
  )
  save_plot(fig, out_path, dpi=300)


def plot_overall_rmse_ground_dtm(df, output_dir):
  """Plot Overall Direct RMSE for Ground DTM across Spaceborne & Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_dtm_direct_rmse_g'].tolist()
  fab_vals = df_sub['fab_direct_rmse_g'].tolist()

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
      f'Ground Level DTM{FLAGS.title_suffix}: RMSE',
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
      output_dir,
      f'overall_rmse_ground_dtm_benchmark{FLAGS.output_suffix}.png',
  )
  save_plot(fig, out_path, dpi=300)


def plot_bias_above_ground_dtm(df, output_dir):
  """Plot Mean Vertical Bias for Above Ground DTM across Spaceborne & Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_dtm_bias_ag'].tolist()
  fab_vals = df_sub['fab_bias_ag'].tolist()

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
      f'Above Ground DTM{FLAGS.title_suffix}: Bias',
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
      output_dir,
      f'mean_bias_above_ground_dtm_benchmark{FLAGS.output_suffix}.png',
  )
  save_plot(fig, out_path, dpi=300)


def plot_bias_ground_dtm(df, output_dir):
  """Plot Mean Vertical Bias for Ground DTM across Spaceborne & Aerial LiDAR."""
  df_sub = df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
  labels = df_sub['label'].tolist()
  ge_vals = df_sub['ge_dtm_bias_g'].tolist()
  fab_vals = df_sub['fab_bias_g'].tolist()

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
      f'Ground Level DTM{FLAGS.title_suffix}: Bias',
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
      output_dir, f'mean_bias_ground_dtm_benchmark{FLAGS.output_suffix}.png'
  )
  save_plot(fig, out_path, dpi=300)


def plot_tile_nmad_above_ground(df, output_dir):
  """Plot Tile-Level Exact NMAD for Above Ground across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_nmad_ag'].tolist()
  cop_vals = df['cop_nmad_ag'].tolist()
  alos_vals = df['alos_nmad_ag'].tolist()
  nasa_vals = df['nasa_nmad_ag'].tolist()

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
      'Above Ground DSM: NMAD', fontsize=19.5, fontweight='bold', pad=25
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
      fig,
      os.path.join(output_dir, 'tile_level_nmad_above_ground_benchmark.png'),
      dpi=300,
  )


def plot_tile_nmad_ground(df, output_dir):
  """Plot Tile-Level Exact NMAD for Ground Level across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_nmad_g'].tolist()
  cop_vals = df['cop_nmad_g'].tolist()
  alos_vals = df['alos_nmad_g'].tolist()
  nasa_vals = df['nasa_nmad_g'].tolist()

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
      'Bare Earth Ground DSM: NMAD', fontsize=19.5, fontweight='bold', pad=25
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
      fig,
      os.path.join(output_dir, 'tile_level_nmad_ground_benchmark.png'),
      dpi=300,
  )


def plot_tile_nmad_above_ground_dtm(df, output_dir):
  """Plot Tile-Level Exact NMAD for Above Ground DTM across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_dtm_nmad_ag'].tolist()
  fab_vals = df['fab_nmad_ag'].tolist()

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
      f'Above Ground DTM{FLAGS.title_suffix}: NMAD',
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
          output_dir,
          f'tile_level_nmad_above_ground_dtm_benchmark{FLAGS.output_suffix}.png',
      ),
      dpi=300,
  )


def plot_tile_nmad_ground_dtm(df, output_dir):
  """Plot Tile-Level Exact NMAD for Ground Level DTM across all 12 LiDAR datasets."""
  labels = df['label'].tolist()
  ge_vals = df['ge_dtm_nmad_g'].tolist()
  fab_vals = df['fab_nmad_g'].tolist()

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
      f'Bare Earth Ground DTM{FLAGS.title_suffix}: NMAD',
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
          output_dir,
          f'tile_level_nmad_ground_dtm_benchmark{FLAGS.output_suffix}.png',
      ),
      dpi=300,
  )


def _draw_dsm_ag_g_panel(
    ax,
    df_sub,
    metric_col_prefix,
    sfx,
    metric_title,
    y_label,
    is_bias = False,
):
  """Draws a single DSM Ground or Above-Ground panel on the provided Axes."""
  labels = df_sub['label'].tolist()
  ge_vals = df_sub[f'ge_{metric_col_prefix}{sfx}'].tolist()
  cop_vals = df_sub[f'cop_{metric_col_prefix}{sfx}'].tolist()
  alos_vals = df_sub[f'alos_{metric_col_prefix}{sfx}'].tolist()
  nasa_vals = df_sub[f'nasa_{metric_col_prefix}{sfx}'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.18

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
  min_val = min(all_vals) if all_vals else -5.0
  val_range = max(max_val - min_val, 1.0)
  y_max = max(max_val + val_range * 0.38, 1.0)
  y_min = min(min_val - val_range * 0.25, -1.0)
  y_top = max_val * 1.38
  if is_bias:
    ax.set_ylim(y_min, y_max)
  else:
    ax.set_ylim(0, y_top)

  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i], r3[i], r4[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    if is_bias:
      best_val = min([abs(h) for h in valid_vals]) if valid_vals else None
    else:
      best_val = min(valid_vals) if valid_vals else None

    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      test_val = abs(h) if is_bias else h
      fw = (
          'bold'
          if best_val is not None and np.isclose(test_val, best_val, atol=1e-4)
          else 'normal'
      )
      va = 'bottom' if h >= 0 else 'top'
      offset = 6 if h >= 0 else -6
      ax.annotate(
          f'{h:+.2f}' if is_bias else f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, offset),
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=15.0,
          fontweight=fw,
          rotation=90,
      )

  if is_bias:
    ax.axhline(0, color='black', linewidth=1.3, linestyle='-', zorder=2)
    y_banner = y_max - (y_max - y_min) * 0.09
  else:
    y_banner = y_top * 0.88

  add_category_dividers_and_banners(
      ax, df_sub, y_banner, aerial_label='AERIAL LiDAR'
  )

  ax.set_ylabel(y_label, fontsize=20.0, fontweight='bold', labelpad=12)
  ax.set_title(metric_title, fontsize=23.5, fontweight='bold', pad=18)
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=16.0, fontweight='bold')
  ax.tick_params(axis='y', labelsize=17.5)
  ax.grid(True, linestyle='--', alpha=0.4, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right' if not is_bias else 'lower right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=18.0,
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.4)


def _draw_dtm_ag_g_panel(
    ax,
    df_sub,
    metric_col_prefix,
    sfx,
    metric_title,
    y_label,
    is_bias = False,
):
  """Draws a single DTM Ground or Above-Ground panel on the provided Axes."""
  labels = df_sub['label'].tolist()
  ge_vals = df_sub[f'ge_dtm_{metric_col_prefix}{sfx}'].tolist()
  fab_vals = df_sub[f'fab_{metric_col_prefix}{sfx}'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.28

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
  min_val = min(all_vals) if all_vals else -5.0
  val_range = max(max_val - min_val, 1.0)
  y_max = max(max_val + val_range * 0.38, 1.0)
  y_min = min(min_val - val_range * 0.25, -1.0)
  y_top = max_val * 1.38
  if is_bias:
    ax.set_ylim(y_min, y_max)
  else:
    ax.set_ylim(0, y_top)

  ax.set_xlim(-0.55, n_groups - 0.45)

  for i in range(n_groups):
    group_rects = [r1[i], r2[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    if is_bias:
      best_val = min([abs(h) for h in valid_vals]) if valid_vals else None
    else:
      best_val = min(valid_vals) if valid_vals else None

    for rect, h in zip(group_rects, h_vals):
      if np.isnan(h):
        continue
      test_val = abs(h) if is_bias else h
      fw = (
          'bold'
          if best_val is not None and np.isclose(test_val, best_val, atol=1e-4)
          else 'normal'
      )
      va = 'bottom' if h >= 0 else 'top'
      offset = 6 if h >= 0 else -6
      ax.annotate(
          f'{h:+.2f}' if is_bias else f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, offset),
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=16.0,
          fontweight=fw,
          rotation=90,
      )

  if is_bias:
    ax.axhline(0, color='black', linewidth=1.3, linestyle='-', zorder=2)
    y_banner = y_max - (y_max - y_min) * 0.09
  else:
    y_banner = y_top * 0.88

  add_category_dividers_and_banners(
      ax, df_sub, y_banner, aerial_label='AERIAL LiDAR'
  )

  ax.set_ylabel(y_label, fontsize=20.0, fontweight='bold', labelpad=12)
  ax.set_title(metric_title, fontsize=23.5, fontweight='bold', pad=18)
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=16.0, fontweight='bold')
  ax.tick_params(axis='y', labelsize=17.5)
  ax.grid(True, linestyle='--', alpha=0.4, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right' if not is_bias else 'lower right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=18.0,
      ncol=2,
  )
  legend.get_frame().set_linewidth(1.4)


def _plot_combined_dsm_ground_and_above_ground(
    df,
    metric_col_prefix,
    metric_label,
    y_label,
    out_filename,
    output_dir,
    spaceborne_and_aerial_only = False,
    is_bias = False,
):
  """Plots a 2-row figure with Ground Level DSM (Row 1) and Above Ground DSM (Row 2)."""
  df_sub = (
      df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
      if spaceborne_and_aerial_only
      else df.copy()
  )
  fig_w = 26 if spaceborne_and_aerial_only else 32
  fig, axes = plt.subplots(2, 1, figsize=(fig_w, 20), dpi=300)
  _draw_dsm_ag_g_panel(
      axes[0],
      df_sub,
      metric_col_prefix,
      '_ag',
      f'Above Ground DSM (Canopy > 5m){FLAGS.title_suffix}: {metric_label}',
      y_label,
      is_bias=is_bias,
  )
  _draw_dsm_ag_g_panel(
      axes[1],
      df_sub,
      metric_col_prefix,
      '_g',
      f'Ground Level DSM (Canopy <= 5m){FLAGS.title_suffix}: {metric_label}',
      y_label,
      is_bias=is_bias,
  )
  plt.tight_layout(h_pad=3.0)
  save_plot(fig, os.path.join(output_dir, out_filename), dpi=300)


def _plot_combined_dtm_ground_and_above_ground(
    df,
    metric_col_prefix,
    metric_label,
    y_label,
    out_filename,
    output_dir,
    spaceborne_and_aerial_only = False,
    is_bias = False,
):
  """Plots a 2-row figure with Above Ground DTM (Row 1) and Ground DTM (Row 2)."""
  df_sub = (
      df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
      if spaceborne_and_aerial_only
      else df.copy()
  )
  fig_w = 26 if spaceborne_and_aerial_only else 30
  fig, axes = plt.subplots(2, 1, figsize=(fig_w, 20), dpi=300)
  _draw_dtm_ag_g_panel(
      axes[0],
      df_sub,
      metric_col_prefix,
      '_ag',
      f'Above Ground Under-Canopy DTM (Canopy > 5m){FLAGS.title_suffix}:'
      f' {metric_label}',
      y_label,
      is_bias=is_bias,
  )
  _draw_dtm_ag_g_panel(
      axes[1],
      df_sub,
      metric_col_prefix,
      '_g',
      f'Bare Earth Ground DTM (Canopy <= 5m){FLAGS.title_suffix}:'
      f' {metric_label}',
      y_label,
      is_bias=is_bias,
  )
  plt.tight_layout(h_pad=3.0)
  save_plot(fig, os.path.join(output_dir, out_filename), dpi=300)


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
    # 2-Row Combined Ground + Above-Ground DSM Benchmarks
    _plot_combined_dsm_ground_and_above_ground(
        df,
        'debiased_rmse',
        'Debiased RMSE',
        'Debiased RMSE (m)',
        f'tile_level_debiased_rmse_ground_and_above_ground_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
    )
    _plot_combined_dsm_ground_and_above_ground(
        df,
        'nmad',
        'NMAD',
        'NMAD (m)',
        f'tile_level_nmad_ground_and_above_ground_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
    )
    _plot_combined_dsm_ground_and_above_ground(
        df,
        'direct_rmse',
        'RMSE',
        'RMSE (m)',
        f'overall_rmse_ground_and_above_ground_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
        spaceborne_and_aerial_only=True,
    )
    _plot_combined_dsm_ground_and_above_ground(
        df,
        'bias',
        'Bias',
        'Mean Vertical Bias (m)',
        f'mean_bias_ground_and_above_ground_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
        spaceborne_and_aerial_only=True,
        is_bias=True,
    )
    plot_tile_debiased_rmse_above_ground(df, FLAGS.output_dir)
    plot_tile_debiased_rmse_ground(df, FLAGS.output_dir)
    plot_overall_rmse_above_ground(df, FLAGS.output_dir)
    plot_overall_rmse_ground(df, FLAGS.output_dir)
    plot_bias_above_ground(df, FLAGS.output_dir)
    plot_bias_ground(df, FLAGS.output_dir)
    plot_tile_nmad_above_ground(df, FLAGS.output_dir)
    plot_tile_nmad_ground(df, FLAGS.output_dir)

  dtm_df = filter_dtm_benchmarks(df)
  # 2-Row Combined Ground + Above-Ground DTM Benchmarks
  _plot_combined_dtm_ground_and_above_ground(
      dtm_df,
      'debiased_rmse',
      'Debiased RMSE',
      'Debiased RMSE (m)',
      f'tile_level_debiased_rmse_ground_and_above_ground_dtm_benchmark{FLAGS.output_suffix}.png',
      FLAGS.output_dir,
  )
  _plot_combined_dtm_ground_and_above_ground(
      dtm_df,
      'nmad',
      'NMAD',
      'NMAD (m)',
      f'tile_level_nmad_ground_and_above_ground_dtm_benchmark{FLAGS.output_suffix}.png',
      FLAGS.output_dir,
  )
  _plot_combined_dtm_ground_and_above_ground(
      dtm_df,
      'direct_rmse',
      'RMSE',
      'RMSE (m)',
      f'overall_rmse_ground_and_above_ground_dtm_benchmark{FLAGS.output_suffix}.png',
      FLAGS.output_dir,
      spaceborne_and_aerial_only=True,
  )
  _plot_combined_dtm_ground_and_above_ground(
      dtm_df,
      'bias',
      'Bias',
      'Mean Vertical Bias (m)',
      f'mean_bias_ground_and_above_ground_dtm_benchmark{FLAGS.output_suffix}.png',
      FLAGS.output_dir,
      spaceborne_and_aerial_only=True,
      is_bias=True,
  )
  plot_tile_debiased_rmse_above_ground_dtm(dtm_df, FLAGS.output_dir)
  plot_tile_debiased_rmse_ground_dtm(dtm_df, FLAGS.output_dir)
  plot_overall_rmse_above_ground_dtm(dtm_df, FLAGS.output_dir)
  plot_overall_rmse_ground_dtm(dtm_df, FLAGS.output_dir)
  plot_bias_above_ground_dtm(dtm_df, FLAGS.output_dir)
  plot_bias_ground_dtm(dtm_df, FLAGS.output_dir)
  plot_tile_nmad_above_ground_dtm(dtm_df, FLAGS.output_dir)
  plot_tile_nmad_ground_dtm(dtm_df, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
