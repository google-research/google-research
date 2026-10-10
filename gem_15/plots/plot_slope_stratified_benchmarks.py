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

"""Generates publication-ready benchmark charts for Topographic Slope Stratification.

Plots performance across all 12 global LiDAR datasets stratified into:
1. Flat Terrain (Slope < 10°)
2. Steep Terrain (Slope >= 10°)

For each stratum, generates:
- Tile-Level Debiased RMSE (DSM & DTM)
- Overall Direct RMSE (DSM & DTM)
- Mean Vertical Bias (DSM & DTM)
- Exact NMAD (DSM & DTM)
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
    'Optional suffix to append to DTM/DSM plot titles.',
)
flags.DEFINE_string(
    'output_suffix',
    '',
    'Optional suffix to append before .png in output filenames.',
)
flags.DEFINE_bool(
    'dtm_only',
    False,
    'If true, only generate DTM benchmark plots.',
)


def _plot_dsm_benchmark(
    df,
    metric_col_prefix,
    sfx,
    metric_title,
    y_label,
    out_filename,
    output_dir,
    spaceborne_and_aerial_only = False,
    is_bias = False,
):
  """Generic generator for DSM grouped bar chart across all models."""
  df_sub = (
      df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
      if spaceborne_and_aerial_only
      else df.copy()
  )
  if sfx == '_steep':
    df_sub = df_sub.copy()
    trop_mask = (
        (df_sub['dataset_key'] == 'us_tropical')
        if 'dataset_key' in df_sub.columns
        else df_sub['label'].str.contains('Tropical', case=False, na=False)
    )
    for m_col in [
        f'ge_{metric_col_prefix}{sfx}',
        f'cop_{metric_col_prefix}{sfx}',
        f'alos_{metric_col_prefix}{sfx}',
        f'nasa_{metric_col_prefix}{sfx}',
    ]:
      if m_col in df_sub.columns:
        df_sub.loc[trop_mask, m_col] = np.nan

  labels = df_sub['label'].tolist()
  ge_vals = df_sub[f'ge_{metric_col_prefix}{sfx}'].tolist()
  cop_vals = df_sub[f'cop_{metric_col_prefix}{sfx}'].tolist()
  alos_vals = df_sub[f'alos_{metric_col_prefix}{sfx}'].tolist()
  nasa_vals = df_sub[f'nasa_{metric_col_prefix}{sfx}'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.18

  c_ge = '#ff7f0e'
  c_cop = '#2ca02c'
  c_alos = '#1f77b4'
  c_nasa = '#d62728'

  fig_w = 26 if spaceborne_and_aerial_only else 32
  fig, ax = plt.subplots(figsize=(fig_w, 11), dpi=300)

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
  min_val = min(all_vals) if all_vals else -5.0
  val_range = max(max_val - min_val, 1.0)
  y_max = max(max_val + val_range * 0.28, 1.0)
  y_min = min(min_val - val_range * 0.15, -1.0)
  y_top = max_val * 1.22
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
      offset = 4 if h >= 0 else -14
      ax.annotate(
          f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, offset),
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=10.5,
          fontweight=fw,
      )

  if is_bias:
    ax.axhline(0, color='black', linewidth=1.2, linestyle='-', zorder=2)
    y_banner = y_max - (y_max - y_min) * 0.08
  else:
    y_banner = y_top * 0.91

  add_category_dividers_and_banners(
      ax, df_sub, y_banner, aerial_label='AERIAL LiDAR'
  )

  ax.set_ylabel(y_label, fontsize=17, fontweight='bold', labelpad=12)
  ax.set_title(metric_title, fontsize=19.5, fontweight='bold', pad=25)
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=13.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right' if not is_bias else 'lower right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=4,
  )
  legend.get_frame().set_linewidth(1.3)
  plt.tight_layout()

  out_path = os.path.join(output_dir, out_filename)
  save_plot(fig, out_path, dpi=300)


def _plot_dtm_benchmark(
    df,
    metric_col_prefix,
    sfx,
    metric_title,
    y_label,
    out_filename,
    output_dir,
    spaceborne_and_aerial_only = False,
    is_bias = False,
):
  """Generic generator for DTM grouped bar chart (GEM-15 DTM vs FABDEM)."""
  df_sub = (
      df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
      if spaceborne_and_aerial_only
      else df.copy()
  )
  if sfx == '_steep':
    df_sub = df_sub.copy()
    trop_mask = (
        (df_sub['dataset_key'] == 'us_tropical')
        if 'dataset_key' in df_sub.columns
        else df_sub['label'].str.contains('Tropical', case=False, na=False)
    )
    for m_col in [
        f'ge_dtm_{metric_col_prefix}{sfx}',
        f'fab_{metric_col_prefix}{sfx}',
    ]:
      if m_col in df_sub.columns:
        df_sub.loc[trop_mask, m_col] = np.nan

  labels = df_sub['label'].tolist()
  ge_vals = df_sub[f'ge_dtm_{metric_col_prefix}{sfx}'].tolist()
  fab_vals = df_sub[f'fab_{metric_col_prefix}{sfx}'].tolist()

  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.28

  c_ge = '#ff7f0e'
  c_fab = '#2ca02c'

  fig_w = 26 if spaceborne_and_aerial_only else 30
  fig, ax = plt.subplots(figsize=(fig_w, 11), dpi=300)

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
  min_val = min(all_vals) if all_vals else -5.0
  val_range = max(max_val - min_val, 1.0)
  y_max = max(max_val + val_range * 0.28, 1.0)
  y_min = min(min_val - val_range * 0.15, -1.0)
  y_top = max_val * 1.22
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
      offset = 4 if h >= 0 else -14
      ax.annotate(
          f'{h:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, h),
          xytext=(0, offset),
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=12.0,
          fontweight=fw,
      )

  if is_bias:
    ax.axhline(0, color='black', linewidth=1.2, linestyle='-', zorder=2)
    y_banner = y_max - (y_max - y_min) * 0.08
  else:
    y_banner = y_top * 0.91

  add_category_dividers_and_banners(
      ax, df_sub, y_banner, aerial_label='AERIAL LiDAR'
  )

  ax.set_ylabel(y_label, fontsize=17, fontweight='bold', labelpad=12)
  ax.set_title(metric_title, fontsize=21.0, fontweight='bold', pad=20)
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=13.5, fontweight='bold')
  ax.tick_params(axis='y', labelsize=15)
  ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

  legend = ax.legend(
      loc='upper right' if not is_bias else 'lower right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=15.5,
      ncol=2,
  )
  legend.get_frame().set_linewidth(1.3)
  plt.tight_layout()

  out_path = os.path.join(output_dir, out_filename)
  save_plot(fig, out_path, dpi=300)


def _draw_dsm_panel(
    ax,
    df_sub,
    metric_col_prefix,
    sfx,
    metric_title,
    y_label,
    is_bias = False,
):
  """Draws a grouped bar panel comparing DSM products for one metric.

  Args:
    ax: Matplotlib axes to draw on.
    df_sub: Per-dataset metrics for the datasets shown in this panel.
    metric_col_prefix: Metric column prefix (e.g. 'rmse').
    sfx: Slope-bin column suffix (e.g. '_steep').
    metric_title: Panel title.
    y_label: Y-axis label.
    is_bias: Whether the metric is a signed bias rather than a magnitude.
  """
  if sfx == '_steep':
    df_sub = df_sub.copy()
    trop_mask = (
        (df_sub['dataset_key'] == 'us_tropical')
        if 'dataset_key' in df_sub.columns
        else df_sub['label'].str.contains('Tropical', case=False, na=False)
    )
    for m_col in [
        f'ge_{metric_col_prefix}{sfx}',
        f'cop_{metric_col_prefix}{sfx}',
        f'alos_{metric_col_prefix}{sfx}',
        f'nasa_{metric_col_prefix}{sfx}',
    ]:
      if m_col in df_sub.columns:
        df_sub.loc[trop_mask, m_col] = np.nan

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


def _draw_dtm_panel(
    ax,
    df_sub,
    metric_col_prefix,
    sfx,
    metric_title,
    y_label,
    is_bias = False,
):
  """Draws a grouped bar panel comparing DTM products for one metric.

  Args:
    ax: Matplotlib axes to draw on.
    df_sub: Per-dataset metrics for the datasets shown in this panel.
    metric_col_prefix: Metric column prefix (e.g. 'rmse').
    sfx: Slope-bin column suffix (e.g. '_steep').
    metric_title: Panel title.
    y_label: Y-axis label.
    is_bias: Whether the metric is a signed bias rather than a magnitude.
  """
  if sfx == '_steep':
    df_sub = df_sub.copy()
    trop_mask = (
        (df_sub['dataset_key'] == 'us_tropical')
        if 'dataset_key' in df_sub.columns
        else df_sub['label'].str.contains('Tropical', case=False, na=False)
    )
    for m_col in [
        f'ge_dtm_{metric_col_prefix}{sfx}',
        f'fab_{metric_col_prefix}{sfx}',
    ]:
      if m_col in df_sub.columns:
        df_sub.loc[trop_mask, m_col] = np.nan

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


def _plot_combined_dsm_slope_benchmark(
    df,
    metric_col_prefix,
    metric_label,
    y_label,
    out_filename,
    output_dir,
    spaceborne_and_aerial_only = False,
    is_bias = False,
):
  """Plots a 2-row figure with Flat Terrain DSM (Row 1) and Steep Terrain DSM (Row 2)."""
  df_sub = (
      df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
      if spaceborne_and_aerial_only
      else df.copy()
  )
  fig_w = 26 if spaceborne_and_aerial_only else 32
  fig, axes = plt.subplots(2, 1, figsize=(fig_w, 20), dpi=300)
  _draw_dsm_panel(
      axes[0],
      df_sub,
      metric_col_prefix,
      '_steep',
      f'Steep Terrain (Slope >= 10°) DSM: {metric_label}',
      y_label,
      is_bias=is_bias,
  )
  _draw_dsm_panel(
      axes[1],
      df_sub,
      metric_col_prefix,
      '_flat',
      f'Flat Terrain (Slope < 10°) DSM: {metric_label}',
      y_label,
      is_bias=is_bias,
  )
  plt.tight_layout(h_pad=3.0)
  save_plot(fig, os.path.join(output_dir, out_filename), dpi=300)


def _plot_combined_dtm_slope_benchmark(
    df,
    metric_col_prefix,
    metric_label,
    y_label,
    out_filename,
    output_dir,
    spaceborne_and_aerial_only = False,
    is_bias = False,
):
  """Plots a 2-row figure with Steep Terrain DTM (Row 1) and Flat Terrain DTM (Row 2)."""
  df_sub = (
      df[df['category'].isin(['Spaceborne', 'Aerial'])].copy()
      if spaceborne_and_aerial_only
      else df.copy()
  )
  fig_w = 26 if spaceborne_and_aerial_only else 30
  fig, axes = plt.subplots(2, 1, figsize=(fig_w, 20), dpi=300)
  _draw_dtm_panel(
      axes[0],
      df_sub,
      metric_col_prefix,
      '_steep',
      f'Steep Terrain (Slope >= 10°) DTM: {metric_label}',
      y_label,
      is_bias=is_bias,
  )
  _draw_dtm_panel(
      axes[1],
      df_sub,
      metric_col_prefix,
      '_flat',
      f'Flat Terrain (Slope < 10°) DTM: {metric_label}',
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
  )
  dtm_df = filter_dtm_benchmarks(df)

  if not FLAGS.dtm_only:
    # 2-Row Combined Flat + Steep DSM Benchmarks
    _plot_combined_dsm_slope_benchmark(
        df,
        'debiased_rmse',
        'Debiased RMSE',
        'Debiased RMSE (m)',
        f'tile_level_debiased_rmse_flat_and_steep_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
    )
    _plot_combined_dsm_slope_benchmark(
        df,
        'nmad',
        'NMAD (Normalized Median Absolute Deviation)',
        'NMAD (m)',
        f'tile_level_nmad_flat_and_steep_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
    )
    _plot_combined_dsm_slope_benchmark(
        df,
        'direct_rmse',
        'Overall RMSE',
        'Overall RMSE (m)',
        f'overall_rmse_flat_and_steep_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
        spaceborne_and_aerial_only=True,
    )
    _plot_combined_dsm_slope_benchmark(
        df,
        'bias',
        'Mean Vertical Bias',
        'Mean Vertical Bias (m)',
        f'mean_bias_flat_and_steep_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
        spaceborne_and_aerial_only=True,
        is_bias=True,
    )

  # 2-Row Combined Flat + Steep DTM Benchmarks
  _plot_combined_dtm_slope_benchmark(
      dtm_df,
      'debiased_rmse',
      'Debiased RMSE',
      'Debiased RMSE (m)',
      f'tile_level_debiased_rmse_flat_and_steep_dtm_benchmark{FLAGS.output_suffix}.png',
      FLAGS.output_dir,
  )
  _plot_combined_dtm_slope_benchmark(
      dtm_df,
      'nmad',
      'NMAD (Normalized Median Absolute Deviation)',
      'NMAD (m)',
      f'tile_level_nmad_flat_and_steep_dtm_benchmark{FLAGS.output_suffix}.png',
      FLAGS.output_dir,
  )
  _plot_combined_dtm_slope_benchmark(
      dtm_df,
      'direct_rmse',
      'Overall RMSE',
      'Overall RMSE (m)',
      f'overall_rmse_flat_and_steep_dtm_benchmark{FLAGS.output_suffix}.png',
      FLAGS.output_dir,
      spaceborne_and_aerial_only=True,
  )
  _plot_combined_dtm_slope_benchmark(
      dtm_df,
      'bias',
      'Mean Vertical Bias',
      'Mean Vertical Bias (m)',
      f'mean_bias_flat_and_steep_dtm_benchmark{FLAGS.output_suffix}.png',
      FLAGS.output_dir,
      spaceborne_and_aerial_only=True,
      is_bias=True,
  )

  strata = [
      (
          '_flat',
          'Flat Terrain (Slope < 10°)',
          'flat',
      ),
      (
          '_steep',
          'Steep Terrain (Slope >= 10°)',
          'steep',
      ),
  ]

  for sfx, desc, name in strata:
    if not FLAGS.dtm_only:
      # 1. DSM Tile Debiased RMSE
      _plot_dsm_benchmark(
          df,
          'debiased_rmse',
          sfx,
          f'{desc} DSM: Debiased RMSE',
          'Debiased RMSE (m)',
          f'tile_level_debiased_rmse_{name}_benchmark{FLAGS.output_suffix}.png',
          FLAGS.output_dir,
      )
      # 2. DSM Overall Direct RMSE
      _plot_dsm_benchmark(
          df,
          'direct_rmse',
          sfx,
          f'{desc} DSM: Overall RMSE',
          'Overall RMSE (m)',
          f'overall_rmse_{name}_benchmark{FLAGS.output_suffix}.png',
          FLAGS.output_dir,
          spaceborne_and_aerial_only=True,
      )
      # 3. DSM Mean Vertical Bias
      _plot_dsm_benchmark(
          df,
          'bias',
          sfx,
          f'{desc} DSM: Mean Vertical Bias',
          'Mean Vertical Bias (m)',
          f'mean_bias_{name}_benchmark{FLAGS.output_suffix}.png',
          FLAGS.output_dir,
          spaceborne_and_aerial_only=True,
          is_bias=True,
      )
      # 4. DSM NMAD
      _plot_dsm_benchmark(
          df,
          'nmad',
          sfx,
          f'{desc} DSM: NMAD (Normalized Median Absolute Deviation)',
          'NMAD (m)',
          f'tile_level_nmad_{name}_benchmark{FLAGS.output_suffix}.png',
          FLAGS.output_dir,
      )

    # DTM Benchmarks (GEM-15 DTM vs FABDEM)
    # 1. DTM Tile Debiased RMSE
    _plot_dtm_benchmark(
        dtm_df,
        'debiased_rmse',
        sfx,
        f'{desc} DTM: Debiased RMSE',
        'Debiased RMSE (m)',
        f'tile_level_debiased_rmse_{name}_dtm_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
    )
    # 2. DTM Overall Direct RMSE
    _plot_dtm_benchmark(
        dtm_df,
        'direct_rmse',
        sfx,
        f'{desc} DTM: Overall RMSE',
        'Overall RMSE (m)',
        f'overall_rmse_{name}_dtm_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
        spaceborne_and_aerial_only=True,
    )
    # 3. DTM Mean Vertical Bias
    _plot_dtm_benchmark(
        dtm_df,
        'bias',
        sfx,
        f'{desc} DTM: Mean Vertical Bias',
        'Mean Vertical Bias (m)',
        f'mean_bias_{name}_dtm_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
        spaceborne_and_aerial_only=True,
        is_bias=True,
    )
    # 4. DTM NMAD
    _plot_dtm_benchmark(
        dtm_df,
        'nmad',
        sfx,
        f'{desc} DTM: NMAD (Normalized Median Absolute Deviation)',
        'NMAD (m)',
        f'tile_level_nmad_{name}_dtm_benchmark{FLAGS.output_suffix}.png',
        FLAGS.output_dir,
    )


if __name__ == '__main__':
  app.run(main)
