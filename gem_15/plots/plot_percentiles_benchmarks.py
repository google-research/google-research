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

"""Compiles and plots GEM-15 percentiles validation across all LiDAR benchmarks."""

import os
from typing import Any, Dict, Sequence, cast
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gem_15 import metrics
from gem_15.plots.benchmark_data import add_category_dividers_and_banners
from gem_15.plots.benchmark_data import BENCHMARK_DATASETS
from gem_15.plots.benchmark_data import format_pixel_count
from gem_15.plots.benchmark_data import save_plot
from gem_15.plots.benchmark_data import sort_benchmark_df

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_dir',
    None,
    'Directory containing validation dataset results (e.g.'
    ' validation_outputs).',
    required=True,
)
flags.DEFINE_string(
    'summary_csv',
    None,
    'Path to save compiled percentiles benchmark summary CSV.',
)
flags.DEFINE_string(
    'output_dir',
    None,
    'Output directory to save generated plots.',
    required=True,
)

PERCENTILE_BANDS = [
    ('dtm', 'DTM (Ground)', '#2ca02c'),
    ('p5', 'P5 (Base)', '#ff7f0e'),
    ('p50', 'P50 (Median)', '#17becf'),
    ('p95', 'P95 (Crown)', '#9467bd'),
    ('dsm', 'DSM (Top)', '#1f77b4'),
    ('std', 'STD (Dev)', '#8c564b'),
]


def compile_percentiles_summary(data_dir):
  """Compiles percentiles metrics across all 11 LiDAR benchmarks."""
  records = []

  for ds in BENCHMARK_DATASETS:
    key = ds['key']
    if ds['category'] == 'Spaceborne':
      continue

    # US NEON is pooled from four annual samples, like the main benchmark.
    if key == 'neon':
      subdir_groups = [
          [f'lidar_percentiles_neon_{y}' for y in range(2022, 2026)],
          ['lidar_percentiles_neon'],
      ]
    else:
      subdir_groups = [
          [f'lidar_percentiles_{key}'],
          [f'lidar_percentiles/{key}'],
      ]
    csv_files = []
    for group in subdir_groups:
      csv_files = [
          p
          for p in (
              os.path.join(data_dir, s, 'lidar_percentiles_results.csv')
              for s in group
          )
          if os.path.exists(p)
      ]
      if csv_files:
        break

    if not csv_files:
      print(f'Warning: No percentiles results found for {key} in {data_dir}')
      continue

    dfs = []
    for csv_file in csv_files:
      with open(csv_file, 'r') as f:
        dfs.append(pd.read_csv(f))
    df_raw = cast(pd.DataFrame, pd.concat(dfs, ignore_index=True))

    # Total evaluated pixels for the DSM metric.
    total_pixels = 0.0
    for c in [
        'dsm_diff_count',
        'p95_diff_count',
        'dtm_diff_count',
        'valid_pixel_count',
    ]:
      if c in df_raw.columns:
        valid_c = df_raw[c].dropna()
        if (valid_c > 0).any():
          total_pixels = float(valid_c[valid_c > 0].sum())
          break
    pixels_str = format_pixel_count(total_pixels)

    rec: Dict[str, Any] = {
        'dataset_key': key,
        'label': ds['display_name'],
        'category': ds['category'],
        'year': ds['year'],
        'month': ds['month'],
        'month_str': ds['month_str'],
        'num_tiles': len(df_raw),
        'total_pixels': total_pixels,
        'pixels_str': pixels_str,
    }

    band_metrics = metrics.compute_percentiles_metrics_from_df(
        df_raw, bands=[b[0] for b in PERCENTILE_BANDS]
    )
    for b_code, _, _ in PERCENTILE_BANDS:
      bm = band_metrics.get(b_code, {})
      rec[f'{b_code}_debiased_rmse'] = bm.get('debiased_rmse', np.nan)
      rec[f'{b_code}_direct_rmse'] = bm.get('direct_rmse', np.nan)
      rec[f'{b_code}_bias'] = bm.get('bias', np.nan)
      rec[f'{b_code}_nmad'] = bm.get('nmad', np.nan)

    records.append(rec)

  df = pd.DataFrame(records)
  if not df.empty:
    df = sort_benchmark_df(df)
  return df


def plot_percentiles_metric(
    df,
    metric_key,
    metric_label,
    output_png,
    is_bias = False,
):
  """Renders grouped bar chart across all 11 LiDAR datasets for all 6 percentiles bands."""
  if df.empty:
    print(f'Skipping plot for {metric_key}: DataFrame is empty.')
    return

  plt.style.use(
      'seaborn-v0_8-whitegrid'
      if 'seaborn-v0_8-whitegrid' in plt.style.available
      else 'default'
  )
  n_groups = len(df)
  x = np.arange(n_groups)
  n_bands = len(PERCENTILE_BANDS)
  bar_width = 0.135

  fig, ax = plt.subplots(figsize=(32, 11), dpi=300)

  offsets = np.linspace(
      -bar_width * (n_bands - 1) / 2.0, bar_width * (n_bands - 1) / 2.0, n_bands
  )

  all_vals = []
  rects = []
  for (b_code, b_name, b_color), offset in zip(PERCENTILE_BANDS, offsets):
    col = f'{b_code}_{metric_key}'
    vals = df[col].tolist()
    all_vals.extend([v for v in vals if np.isfinite(v)])
    r = ax.bar(
        x + offset,
        vals,
        bar_width,
        label=f'GEM-15 {b_name}',
        color=b_color,
        edgecolor='black',
        linewidth=0.8,
        zorder=3,
    )
    rects.append(r)

  # Category dividers and banners
  y_max = max(all_vals) if all_vals else 10.0
  y_min = min(all_vals) if (all_vals and is_bias) else 0.0

  if is_bias:
    y_top = max(abs(y_max), abs(y_min)) * 1.54
    ax.set_ylim(-y_top, y_top)
    y_banner = y_top * 0.78
    ax.axhline(0, color='black', linewidth=1.2, linestyle='-', zorder=2)
  else:
    y_top = y_max * 1.42
    ax.set_ylim(0, y_top)
    y_banner = y_top * 0.80

  ax.set_xlim(-0.55, n_groups - 0.45)

  add_category_dividers_and_banners(
      ax,
      df,
      y_banner=y_banner,
      aerial_label='AERIAL LiDAR',
  )

  # X-axis labels
  labels = []
  for _, row in df.iterrows():
    labels.append(f"{row['label']}\n[{row['month_str']}, {row['pixels_str']}]")
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=15.0, fontweight='bold')
  ax.set_ylabel(
      f'{metric_label} (meters)', fontsize=19.0, fontweight='bold', labelpad=12
  )
  ax.set_title(
      f'LiDAR Sub-Pixel Vertical Envelope & Surfaces: {metric_label}',
      fontsize=23.0,
      fontweight='bold',
      pad=18,
  )
  ax.tick_params(axis='y', labelsize=16.5)
  legend = ax.legend(
      loc='upper right',
      fontsize=16.5,
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      ncol=3,
  )
  legend.get_frame().set_linewidth(1.3)
  ax.grid(True, linestyle='--', alpha=0.4, zorder=0, axis='y')

  # Bar value annotations
  for group in rects:
    for bar in group:
      h = bar.get_height()
      if np.isnan(h):
        continue
      va = 'bottom' if h >= 0 else 'top'
      offset_y = 5 if h >= 0 else -5
      ax.annotate(
          f'{h:+.2f}' if is_bias else f'{h:.2f}',
          xy=(bar.get_x() + bar.get_width() / 2, h),
          xytext=(0, offset_y),
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=12.5,
          fontweight='bold',
          rotation=90,
      )

  plt.tight_layout()
  save_plot(fig, output_png)


def _draw_percentiles_panel_on_ax(
    ax,
    df_sub,
    metric_key,
    ylabel,
    panel_title,
    is_bias = False,
    show_banners = True,
    show_xticks = True,
    show_legend = False,
):
  """Draws a single percentiles panel on ax inside a stacked figure."""
  n_groups = len(df_sub)
  x = np.arange(n_groups)
  n_bands = len(PERCENTILE_BANDS)
  bar_width = 0.138
  offsets = np.linspace(
      -bar_width * (n_bands - 1) / 2.0, bar_width * (n_bands - 1) / 2.0, n_bands
  )

  all_vals = []
  rects = []
  for (b_code, b_name, b_color), offset in zip(PERCENTILE_BANDS, offsets):
    col = f'{b_code}_{metric_key}'
    vals = df_sub[col].tolist()
    all_vals.extend([v for v in vals if np.isfinite(v)])
    r = ax.bar(
        x + offset,
        vals,
        bar_width,
        label=f'GEM-15 {b_name}',
        color=b_color,
        edgecolor='black',
        linewidth=0.85,
        zorder=3,
    )
    rects.append(r)

  y_max = max(all_vals) if all_vals else 10.0
  y_min = min(all_vals) if (all_vals and is_bias) else 0.0

  if is_bias:
    y_top = max(abs(y_max), abs(y_min)) * 1.65
    ax.set_ylim(-y_top, y_top)
    y_banner = y_top * 0.86
    ax.axhline(0, color='black', linewidth=1.4, linestyle='-', zorder=2)
  elif show_legend:
    y_top = y_max * 1.54
    ax.set_ylim(0, y_top)
    y_banner = y_top * 0.80
  else:
    y_top = y_max * 1.44
    ax.set_ylim(0, y_top)
    y_banner = y_top * 0.88

  ax.set_xlim(-0.55, n_groups - 0.45)

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
          fontsize=20.0,
          fontstyle='italic',
          fontweight='bold',
          color='#666666',
          bbox=dict(
              boxstyle='round,pad=0.55',
              facecolor='#fafafa',
              edgecolor='#cccccc',
              alpha=0.94,
              lw=1.4,
          ),
          zorder=4,
      )

  if show_banners:
    add_category_dividers_and_banners(
        ax,
        df_sub,
        y_banner=y_banner,
        aerial_label='AERIAL LiDAR',
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

  labels = []
  for _, row in df_sub.iterrows():
    labels.append(f"{row['label']}\n[{row['month_str']}, {row['pixels_str']}]")
  ax.set_xticks(x)
  if show_xticks:
    ax.set_xticklabels(labels, fontsize=16.5, fontweight='bold')
  else:
    ax.set_xticklabels([])
  ax.set_ylabel(ylabel, fontsize=23.5, fontweight='bold', labelpad=14)
  ax.set_title(
      panel_title, fontsize=27.0, fontweight='bold', pad=16, loc='left'
  )
  ax.tick_params(axis='y', labelsize=20.0)
  ax.grid(True, linestyle='--', alpha=0.4, zorder=0, axis='y')

  if show_legend:
    legend = ax.legend(
        loc='upper right',
        fontsize=20.5,
        frameon=True,
        facecolor='white',
        framealpha=0.95,
        edgecolor='#cccccc',
        ncol=3,
    )
    legend.get_frame().set_linewidth(1.5)

  for group in rects:
    for bar in group:
      h = bar.get_height()
      if np.isnan(h):
        continue
      va = 'bottom' if h >= 0 else 'top'
      offset_y = 6 if h >= 0 else -6
      ax.annotate(
          f'{h:+.2f}' if is_bias else f'{h:.2f}',
          xy=(bar.get_x() + bar.get_width() / 2, h),
          xytext=(0, offset_y),
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=15.0,
          fontweight='bold',
          rotation=90,
      )


def plot_stacked_percentiles_benchmark(
    df, output_dir, include_direct_rmse = False
):
  """Renders combined vertically stacked percentiles benchmark figure (3-panel by default)."""
  if df.empty:
    return
  # Keep full df layout for bias/direct_rmse and leave Drone region empty (NaN)
  # so that every dataset column aligns 100% vertically across all 3 panels.
  df_aligned_aerial = df.copy()
  drone_mask = df_aligned_aerial['category'] == 'Drone'
  for b_code, _, _ in PERCENTILE_BANDS:
    for m_sfx in ['direct_rmse', 'bias']:
      col = f'{b_code}_{m_sfx}'
      if col in df_aligned_aerial.columns:
        df_aligned_aerial.loc[drone_mask, col] = np.nan

  nrows = 4 if include_direct_rmse else 3
  fig, axes = plt.subplots(nrows, 1, figsize=(40, 9.8 * nrows), dpi=300)

  _draw_percentiles_panel_on_ax(
      axes[0],
      df,
      metric_key='debiased_rmse',
      ylabel='Debiased RMSE (m)',
      panel_title=(
          '(a) Sub-Pixel Vertical Envelope & Surfaces: Tile-Level Debiased RMSE'
      ),
      is_bias=False,
      show_banners=True,
      show_xticks=True,
      show_legend=True,
  )
  _draw_percentiles_panel_on_ax(
      axes[1],
      df,
      metric_key='nmad',
      ylabel='Median NMAD (m)',
      panel_title='(b) Sub-Pixel Vertical Envelope & Surfaces: Median NMAD',
      is_bias=False,
      show_banners=True,
      show_xticks=True,
      show_legend=False,
  )
  if include_direct_rmse:
    _draw_percentiles_panel_on_ax(
        axes[2],
        df_aligned_aerial,
        metric_key='direct_rmse',
        ylabel='Direct RMSE (m)',
        panel_title=(
            '(c) Sub-Pixel Vertical Envelope & Surfaces: Direct RMSE'
            ' (Continuous Aerial LiDAR)'
        ),
        is_bias=False,
        show_banners=True,
        show_xticks=True,
        show_legend=False,
    )
    _draw_percentiles_panel_on_ax(
        axes[3],
        df_aligned_aerial,
        metric_key='bias',
        ylabel='Mean Bias (m)',
        panel_title=(
            '(d) Sub-Pixel Vertical Envelope & Surfaces: Mean Vertical Bias'
            ' (Continuous Aerial LiDAR)'
        ),
        is_bias=True,
        show_banners=True,
        show_xticks=True,
        show_legend=False,
    )
    suffix = '4panel'
  else:
    _draw_percentiles_panel_on_ax(
        axes[2],
        df_aligned_aerial,
        metric_key='bias',
        ylabel='Mean Bias (m)',
        panel_title=(
            '(c) Sub-Pixel Vertical Envelope & Surfaces: Mean Vertical Bias'
            ' (Continuous Aerial LiDAR)'
        ),
        is_bias=True,
        show_banners=True,
        show_xticks=True,
        show_legend=False,
    )
    suffix = '3panel'

  plt.tight_layout(h_pad=2.5)
  out_png = os.path.join(
      output_dir, f'percentiles_benchmark_stacked_{suffix}.png'
  )
  save_plot(fig, out_png, dpi=300)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  os.makedirs(FLAGS.output_dir, exist_ok=True)
  summary_csv = FLAGS.summary_csv or os.path.join(
      FLAGS.output_dir, 'percentiles_benchmark_summary.csv'
  )

  # Always recompile from the per-tile results. --summary_csv is an OUTPUT
  # path only: reading it back would silently plot the previous run's numbers.
  print(f'Compiling percentiles metrics from {FLAGS.data_dir}...')
  df = compile_percentiles_summary(FLAGS.data_dir)
  if not df.empty:
    print(f'Saving summary CSV to: {summary_csv}')
    with open(summary_csv, 'w') as f:
      df.to_csv(f, index=False)

  if df.empty:
    print('No percentiles validation datasets found to plot.')
    return

  print('Generating percentiles publication benchmark plots...')
  plot_percentiles_metric(
      df,
      'debiased_rmse',
      'Tile-Level Debiased RMSE',
      os.path.join(FLAGS.output_dir, 'percentiles_debiased_rmse_benchmark.png'),
      is_bias=False,
  )
  plot_percentiles_metric(
      df,
      'nmad',
      'Normalized Median Absolute Deviation (NMAD)',
      os.path.join(FLAGS.output_dir, 'percentiles_nmad_benchmark.png'),
      is_bias=False,
  )

  # Direct RMSE and Mean Bias are restricted to Aerial LiDAR (continuous
  # national surveys) to avoid uncalibrated drone geoid / local GNSS shifts.
  df_aerial = df[df['category'] == 'Aerial'].copy()
  plot_percentiles_metric(
      df_aerial,
      'direct_rmse',
      'Overall Direct RMSE',
      os.path.join(FLAGS.output_dir, 'percentiles_direct_rmse_benchmark.png'),
      is_bias=False,
  )
  plot_percentiles_metric(
      df_aerial,
      'bias',
      'Mean Vertical Bias',
      os.path.join(FLAGS.output_dir, 'percentiles_mean_bias_benchmark.png'),
      is_bias=True,
  )
  plot_stacked_percentiles_benchmark(
      df, FLAGS.output_dir, include_direct_rmse=False
  )
  print(f'All percentiles plots successfully saved to: {FLAGS.output_dir}')


if __name__ == '__main__':
  app.run(main)
