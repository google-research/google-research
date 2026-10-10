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

"""Generate Continent-Stratified International GNSS CORS Benchmark Plots.

Produces publication-styled continent-stratified grouped bar charts
(Debiased RMSE, Median NMAD, Direct RMSE, Mean Bias, and 4-panel summary),
global spatial error maps, and the markdown summary report from
`cors_checkpoints/results.csv`.
"""

from collections.abc import Sequence
import io
import os
import shutil
from typing import Any
from absl import app
from absl import flags
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gem_15.plots import benchmark_data

matplotlib.use('Agg')

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'data_dir',
    benchmark_data.DEFAULT_VALIDATION_DIR,
    'local root directory containing validation result folders.',
)
flags.DEFINE_string(
    'output_dir',
    f'{benchmark_data.DEFAULT_VALIDATION_DIR}/plots',
    'Output directory for generated plots.',
)
flags.DEFINE_integer(
    'year',
    2025,
    'Observation year for plot titles and summary reports (default: 2025).',
)


def save_plot(fig, out_path, dpi = 300):
  """Saves matplotlib figure to local path to disk."""
  parent_dir = os.path.dirname(out_path)
  if parent_dir and not os.path.exists(parent_dir):
    os.makedirs(parent_dir, exist_ok=True)
  buf = io.BytesIO()
  fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
  plt.close(fig)
  with open(out_path, 'wb') as f:
    f.write(buf.getvalue())
  print(f'Saved: {out_path}')


def plot_cors_continent_benchmarks(
    cors_df,
    output_dir,
    cors_dir = None,
    year = 2025,
):
  """Plot Continent-Stratified International GNSS CORS Benchmark Charts."""
  if cors_df.empty:
    return

  labels = cors_df['label'].tolist()
  n_groups = len(labels)
  x = np.arange(n_groups)
  bar_width = 0.18

  c_ge = '#ff7f0e'
  c_cop = '#2ca02c'
  c_alos = '#1f77b4'
  c_nasa = '#d62728'

  specs = [
      (
          'debiased_rmse',
          'Debiased Vertical RMSE (meters)',
          (
              f'International GNSS CORS ({year}): Debiased RMSE Stratified by'
              ' Continent'
          ),
          'cors_continent_drmse_benchmark.png',
          'timeseries_drmse_plot.png',
          False,
      ),
      (
          'nmad',
          'Normalized Median Absolute Deviation — NMAD (meters)',
          (
              f'International GNSS CORS ({year}): Median NMAD Stratified by'
              ' Continent'
          ),
          'cors_continent_nmad_benchmark.png',
          'timeseries_nmad_plot.png',
          False,
      ),
      (
          'direct_rmse',
          'Direct Vertical RMSE (meters)',
          (
              f'International GNSS CORS ({year}): Direct RMSE Stratified by'
              ' Continent'
          ),
          'cors_continent_rmse_benchmark.png',
          'timeseries_rmse_plot.png',
          False,
      ),
      (
          'bias',
          'Mean Vertical Bias (meters)',
          (
              f'International GNSS CORS ({year}): Mean Bias Stratified by'
              ' Continent'
          ),
          'cors_continent_bias_benchmark.png',
          'timeseries_bias_plot.png',
          True,
      ),
  ]

  for metric, ylabel, title, fname, legacy_fname, is_bias in specs:
    ge_vals = cors_df[f'ge_{metric}'].tolist()
    cop_vals = cors_df[f'cop_{metric}'].tolist()
    alos_vals = cors_df[f'alos_{metric}'].tolist()
    nasa_vals = cors_df[f'nasa_{metric}'].tolist()

    fig, ax = plt.subplots(figsize=(24, 11), dpi=300)
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
    min_val = min(all_vals) if all_vals else 0.0
    max_val = max(all_vals) if all_vals else 10.0
    if is_bias:
      ax.set_ylim(
          min_val * 1.28 if min_val < 0 else -1.0, max(1.0, max_val * 1.25)
      )
      ax.axhline(0.0, color='black', linewidth=1.0, zorder=2)
    else:
      ax.set_ylim(0, max_val * 1.25)
    ax.set_xlim(-0.55, n_groups - 0.45)

    for i in range(n_groups):
      group_rects = [r1[i], r2[i], r3[i], r4[i]]
      h_vals = [r.get_height() for r in group_rects]
      valid_vals = [abs(h) if is_bias else h for h in h_vals if not np.isnan(h)]
      best_val = min(valid_vals) if valid_vals else None
      for rect, height in zip(group_rects, h_vals):
        if np.isnan(height):
          continue
        test_val = abs(height) if is_bias else height
        is_best = best_val is not None and np.isclose(
            test_val, best_val, atol=1e-4
        )
        fw = 'bold' if is_best else 'normal'
        va = 'bottom' if height >= 0 else 'top'
        xytext = (0, 6) if height >= 0 else (0, -18)
        ax.annotate(
            f'{height:.2f}',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=xytext,
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=14.5,
            fontweight=fw,
            color='#111111' if is_best else '#333333',
        )

    ax.axvline(
        x=0.5,
        color='#555555',
        linestyle='--',
        linewidth=1.8,
        alpha=0.65,
        zorder=2,
    )
    ax.set_ylabel(ylabel, fontsize=18.5, fontweight='bold', labelpad=12)
    ax.set_title(title, fontsize=21.5, fontweight='bold', pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16.5, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16.5)
    ax.grid(True, linestyle='--', alpha=0.35, zorder=0, axis='y')

    legend = ax.legend(
        loc='upper right',
        frameon=True,
        facecolor='white',
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=16.5,
        ncol=4,
    )
    legend.get_frame().set_linewidth(1.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, fname)
    save_plot(fig, out_path, dpi=300)
    if cors_dir:
      shutil.copyfile(out_path, os.path.join(cors_dir, fname))
      shutil.copyfile(out_path, os.path.join(cors_dir, legacy_fname))
    if metric == 'debiased_rmse':
      shutil.copyfile(
          out_path,
          os.path.join(output_dir, 'cors_metrics_comparison_barchart.png'),
      )
      if cors_dir:
        shutil.copyfile(
            out_path,
            os.path.join(cors_dir, 'cors_metrics_comparison_barchart.png'),
        )


def plot_cors_4panel_summary(
    cors_df,
    output_dir,
    cors_dir = None,
    year = 2025,
):
  """Generates the 2x2 4-panel continent-stratified CORS summary figure."""
  if cors_df.empty:
    return

  labels = cors_df['label'].tolist()
  n_groups = len(labels)
  x = np.arange(n_groups, dtype=float)
  bar_w = 0.20

  models = [
      ('ge', 'GEM-15 DSM v1 (15m)', '#ff7f0e', -1.5 * bar_w),
      ('cop', 'Copernicus GLO-30 (30m)', '#2ca02c', -0.5 * bar_w),
      ('alos', 'ALOS AW3D30 (30m)', '#1f77b4', 0.5 * bar_w),
      ('nasa', 'NASADEM (30m)', '#d62728', 1.5 * bar_w),
  ]

  metric_specs = [
      (
          'debiased_rmse',
          'Debiased RMSE (meters)',
          (
              f'International GNSS CORS ({year}): Debiased RMSE (DRMSE)'
              ' Stratified by Continent'
          ),
          False,
      ),
      (
          'nmad',
          'Median NMAD (meters)',
          (
              f'International GNSS CORS ({year}): Normalized Median Absolute'
              ' Deviation (NMAD) by Continent'
          ),
          False,
      ),
      (
          'direct_rmse',
          'Direct RMSE (meters)',
          (
              f'International GNSS CORS ({year}): Direct RMSE Stratified by'
              ' Continent'
          ),
          False,
      ),
      (
          'bias',
          'Mean Bias (meters)',
          (
              f'International GNSS CORS ({year}): Mean Bias (DEM - GNSS)'
              ' Stratified by Continent'
          ),
          True,
      ),
  ]

  fig, axes = plt.subplots(2, 2, figsize=(24, 15), dpi=200)
  panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
  for ax, (metric, ylabel, title, is_bias) in zip(panel_axes, metric_specs):
    rects_list: list[Any] = []
    all_vals = []
    for src, label, color, offset in models:
      vals = cors_df[f'{src}_{metric}'].to_numpy(dtype=float)
      all_vals.extend([v for v in vals if not np.isnan(v)])
      rects = ax.bar(
          x + offset,
          vals,
          bar_w,
          label=label,
          color=color,
          edgecolor='white',
          linewidth=0.5,
      )
      rects_list.append(rects)

    for g_idx in range(n_groups):
      group_rects = [r[g_idx] for r in rects_list]
      h_vals = [float(r.get_height()) for r in group_rects]
      valid_vals = [abs(h) if is_bias else h for h in h_vals if not np.isnan(h)]
      best_val = min(valid_vals) if valid_vals else None
      for rect, height in zip(group_rects, h_vals):
        if np.isnan(height):
          continue
        va = 'bottom' if height >= 0 else 'top'
        xytext = (0, 4) if height >= 0 else (0, -14)
        test_val = abs(height) if is_bias else height
        is_best = best_val is not None and np.isclose(
            test_val, best_val, atol=1e-3
        )
        fw = 'bold' if is_best else 'normal'
        ax.annotate(
            f'{height:.2f}',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=xytext,
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=10.5,
            fontweight=fw,
        )

    ax.axvline(0.5, color='#555555', linestyle='--', linewidth=1.0, alpha=0.6)
    if is_bias:
      ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12.5, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13.5, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    valid_vals = [v for v in all_vals if not np.isnan(v)]
    ymin = min(valid_vals) if valid_vals else 0.0
    ymax = max(valid_vals) if valid_vals else 1.0
    if is_bias:
      ax.set_ylim([ymin * 1.28 if ymin < 0 else -1.0, max(1.0, ymax * 1.22)])
    else:
      ax.set_ylim([0.0, ymax * 1.22])

  handles, legend_labels = panel_axes[0].get_legend_handles_labels()
  fig.legend(
      handles,
      legend_labels,
      loc='upper center',
      ncol=4,
      fontsize=12,
      bbox_to_anchor=(0.5, 0.99),
  )
  plt.tight_layout(rect=(0, 0, 1, 0.95))
  out_path = os.path.join(output_dir, 'cors_continent_4panel_benchmark.png')
  save_plot(fig, out_path, dpi=200)
  if cors_dir:
    shutil.copyfile(
        out_path,
        os.path.join(cors_dir, 'cors_continent_4panel_benchmark.png'),
    )


def plot_cors_global_maps_and_report(
    cors_df,
    cors_dir,
    year = 2025,
):
  """Generates global station error maps and report.md in cors_checkpoints."""
  csv_candidates = [
      os.path.join(cors_dir, 'results.csv'),
      os.path.join(cors_dir, f'results_{year}.csv'),
  ]
  results_csv = None
  for cand in csv_candidates:
    if os.path.exists(cand):
      results_csv = cand
      break
  if not results_csv:
    return

  with open(results_csv, 'r') as f:
    df = pd.read_csv(f)
  if df.empty:
    return

  ref_name = f'International GNSS CORS ({year})'
  plots_dir = os.path.join(cors_dir, 'plots')
  if not os.path.exists(plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

  try:
    url = 'https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson'
    world = gpd.read_file(url)
  except Exception:  # pylint: disable=broad-except
    world = None

  df['ge_error'] = df['ge'] - df['ref_elevation']
  df['cop_error'] = df['cop'] - df['ref_elevation']
  df['alos_error'] = df['alos'] - df['ref_elevation']
  df['nasa_error'] = df['nasa'] - df['ref_elevation']

  gdf = gpd.GeoDataFrame(
      df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326'
  )
  error_cmap = 'RdBu_r'
  error_norm = plt.Normalize(vmin=-15, vmax=15)

  plots_config = [
      (
          'ge_error',
          error_cmap,
          error_norm,
          f'GEM-15 DSM v1 Error against {ref_name} (meters)',
          'map_gem15_error.png',
          f'Error (DEM - {ref_name}) [meters]',
      ),
      (
          'cop_error',
          error_cmap,
          error_norm,
          f'Copernicus DEM Error against {ref_name} (meters)',
          'map_copernicus_error.png',
          f'Error (DEM - {ref_name}) [meters]',
      ),
      (
          'alos_error',
          error_cmap,
          error_norm,
          f'ALOS DEM Error against {ref_name} (meters)',
          'map_alos_error.png',
          f'Error (DEM - {ref_name}) [meters]',
      ),
      (
          'nasa_error',
          error_cmap,
          error_norm,
          f'NASADEM Error against {ref_name} (meters)',
          'map_nasa_error.png',
          f'Error (DEM - {ref_name}) [meters]',
      ),
  ]

  for col_name, cmap, norm, title, filename, label in plots_config:
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    if world is not None:
      world.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)
    gdf.plot(
        ax=ax,
        column=col_name,
        cmap=cmap,
        norm=norm,
        markersize=8,
        alpha=0.75,
        legend=False,
    )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([-180, 180])
    ax.set_ylim([-60, 65])
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # pylint: disable=protected-access
    cbar = fig.colorbar(
        sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.7
    )
    cbar.set_label(label, fontsize=11, fontweight='bold')
    save_plot(fig, os.path.join(plots_dir, filename), dpi=150)

  if not cors_df.empty:
    global_row = cors_df.iloc[0]
    table_rows = []
    for _, r in cors_df.iterrows():
      c_name = r['continent']
      cnt = int(r['count'])
      table_rows.append(
          f'| **{c_name}** (`N={cnt:,}`) |'
          f" **{r['ge_debiased_rmse']:.2f}m** ({r['ge_nmad']:.2f}m) |"
          f" {r['cop_debiased_rmse']:.2f}m ({r['cop_nmad']:.2f}m) |"
          f" {r['alos_debiased_rmse']:.2f}m ({r['alos_nmad']:.2f}m) |"
          f" {r['nasa_debiased_rmse']:.2f}m ({r['nasa_nmad']:.2f}m) |"
      )
    report_lines = [
        f'# DEM vs {ref_name} Global Continent-Stratified Validation Report',
        '',
        (
            'This report presents the global validation of GEM-15 DSM v1'
            f' (15m) and public global DEMs against `{len(df):,}` active'
            f' {ref_name} geodetic survey stations across all 6 inhabited'
            ' continents.'
        ),
        '',
        '## 1. Continent-Stratified Debiased RMSE & Median NMAD',
        '![Continent DRMSE](cors_continent_drmse_benchmark.png)',
        '![Continent 4-Panel](cors_continent_4panel_benchmark.png)',
        '',
        (
            '| Continent / Region | **GEM-15 DSM (15m)** DRMSE (NMAD) |'
            ' Copernicus GLO-30 | ALOS AW3D30 | NASADEM |'
        ),
        '| :--- | :---: | :---: | :---: | :---: |',
    ]
    report_lines.extend(table_rows)
    report_lines.extend([
        '',
        '## 2. Global Overall Summary Metrics',
        (
            '- **GEM-15 DSM (15m)**: DRMSE ='
            f" **{global_row['ge_debiased_rmse']:.2f}m**, RMSE ="
            f" **{global_row['ge_direct_rmse']:.2f}m**, Median NMAD ="
            f" **{global_row['ge_nmad']:.2f}m**, Bias ="
            f" **{global_row['ge_bias']:.2f}m**"
        ),
        (
            '- **Copernicus GLO-30**: DRMSE ='
            f" {global_row['cop_debiased_rmse']:.2f}m, RMSE ="
            f" {global_row['cop_direct_rmse']:.2f}m, Median NMAD ="
            f" {global_row['cop_nmad']:.2f}m, Bias ="
            f" {global_row['cop_bias']:.2f}m"
        ),
        (
            '- **ALOS AW3D30**: DRMSE ='
            f" {global_row['alos_debiased_rmse']:.2f}m, RMSE ="
            f" {global_row['alos_direct_rmse']:.2f}m, Median NMAD ="
            f" {global_row['alos_nmad']:.2f}m, Bias ="
            f" {global_row['alos_bias']:.2f}m"
        ),
        (
            f"- **NASADEM**: DRMSE = {global_row['nasa_debiased_rmse']:.2f}m,"
            f" RMSE = {global_row['nasa_direct_rmse']:.2f}m,"
            f" Median NMAD = {global_row['nasa_nmad']:.2f}m,"
            f" Bias = {global_row['nasa_bias']:.2f}m"
        ),
    ])
    report_path = os.path.join(cors_dir, 'report.md')
    with open(report_path, 'w') as f:
      f.write('\n'.join(report_lines) + '\n')
    print(f'Saved summary report to: {report_path}')


def main(argv):
  del argv  # Unused.
  plt.style.use(
      'seaborn-v0_8-whitegrid'
      if 'seaborn-v0_8-whitegrid' in plt.style.available
      else 'default'
  )
  cors_df = benchmark_data.compile_cors_continent_summary(
      data_dir=FLAGS.data_dir,
      save_summary_csv=True,
  )
  cors_dir = os.path.join(FLAGS.data_dir, 'cors_checkpoints')
  plot_cors_continent_benchmarks(
      cors_df=cors_df,
      output_dir=FLAGS.output_dir,
      cors_dir=cors_dir,
      year=FLAGS.year,
  )
  plot_cors_4panel_summary(
      cors_df=cors_df,
      output_dir=FLAGS.output_dir,
      cors_dir=cors_dir,
      year=FLAGS.year,
  )
  plot_cors_global_maps_and_report(
      cors_df=cors_df,
      cors_dir=cors_dir,
      year=FLAGS.year,
  )


if __name__ == '__main__':
  app.run(main)
