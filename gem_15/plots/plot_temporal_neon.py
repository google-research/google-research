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

"""Generates timeseries comparison plots for US NEON annual aerial LiDAR surveys (2012-2025).

Evaluates and plots DSM performance metrics (DRMSE, NMAD, Direct RMSE, and Mean
Bias)
year-by-year for GEM-15 DSM v1 (GEM-15), Copernicus DEM GLO-30, ALOS AW3D30,
and NASADEM, mirroring the NOAA/NGL CORS timeseries evaluation structure.
"""

from collections.abc import Sequence
import io
import os
from typing import Any, Dict, Optional, Set, Tuple
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import pandas as pd
from gem_15.plots.benchmark_data import compute_metrics_from_csv
from gem_15.plots.benchmark_data import compute_metrics_from_df
from gem_15.plots.benchmark_data import save_plot

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_dir',
    './validation_outputs',
    'Root directory containing per-year NEON validation subdirectories.',
)
flags.DEFINE_string(
    'output_dir',
    None,
    'Directory where generated plots and compiled CSV will be saved.',
    required=True,
)
flags.DEFINE_integer('min_year', 2013, 'Earliest NEON survey year to evaluate.')
flags.DEFINE_integer('max_year', 2025, 'Latest NEON survey year to evaluate.')

# Bounding boxes (min_lon, min_lat, max_lon, max_lat) for 57 NEON survey sites
NEON_SITE_BOUNDS = {
    'ABBY': (-122.4142, 45.686, -122.2227, 45.823),
    'ARIK': (-102.6153, 39.6535, -102.4249, 39.8203),
    'BARR': (-156.8897, 71.1958, -156.2136, 71.4268),
    'BART': (-71.3661, 43.9541, -71.177, 44.111),
    'BLAN': (-78.1177, 39.0019, -77.928, 39.1693),
    'BLUE': (-96.6977, 34.4301, -96.5991, 34.4958),
    'BONA': (-147.7113, 65.1059, -147.3144, 65.2608),
    'CHEQ': (-90.15, 45.7579, -90.0196, 45.8608),
    'CLBJ': (-97.6839, 33.2905, -97.5318, 33.4554),
    'CPER': (-104.8036, 40.7625, -104.6627, 40.8899),
    'CUPE': (-67.0112, 18.082, -66.9532, 18.1557),
    'DEJU': (-145.8491, 63.7548, -145.5859, 63.9733),
    'DELA': (-87.911, 32.4662, -87.7294, 32.6304),
    'DSNY': (-81.5143, 28.0013, -81.351, 28.1833),
    'GRSM': (-83.6707, 35.5244, -83.2344, 35.7757),
    'GUAN': (-66.928, 17.9265, -66.7845, 18.0464),
    'GUIL': (-66.8217, 18.1249, -66.7543, 18.1988),
    'HARV': (-72.3094, 42.3476, -72.0819, 42.6055),
    'HEAL': (-149.352, 63.8144, -149.0879, 63.9537),
    'HOPB': (-72.3902, 42.4428, -72.3046, 42.5265),
    'JERC': (-84.5923, 31.1246, -84.3354, 31.3646),
    'JORN': (-106.9905, 32.5168, -106.7199, 32.765),
    'KONZ': (-96.6831, 39.0149, -96.4689, 39.2538),
    'LAJA': (-67.1351, 17.9734, -66.9346, 18.1388),
    'LENO': (-88.2739, 31.7593, -88.1143, 31.9059),
    'LIRO': (-89.7315, 45.9711, -89.6672, 46.0363),
    'MCDI': (-96.5365, 38.9146, -96.4202, 38.9989),
    'MCRA': (-122.191, 44.2359, -122.1175, 44.3003),
    'MLBS': (-80.6327, 37.2923, -80.441, 37.4651),
    'MOAB': (-109.4636, 38.1739, -109.3136, 38.321),
    'NIWO': (-105.6737, 39.9412, -105.4749, 40.1053),
    'NOGP': (-100.999, 46.7153, -100.827, 46.8628),
    'OAES': (-99.2033, 35.265, -99.006, 35.4645),
    'ONAQ': (-112.5924, 40.1035, -112.3913, 40.26),
    'ORNL': (-84.4579, 35.8448, -84.1542, 36.0324),
    'OSBS': (-82.112, 29.5772, -81.883, 29.7873),
    'PRIN': (-97.8986, 33.302, -97.7579, 33.4394),
    'PUUM': (-155.4651, 19.4769, -155.2053, 19.6346),
    'REDB': (-111.8234, 40.7511, -111.7178, 40.8427),
    'RMNP': (-105.5935, 40.1128, -105.4292, 40.3217),
    'SCBI': (-78.2292, 38.8075, -78.0764, 38.9378),
    'SERC': (-76.644, 38.8143, -76.481, 38.9705),
    'SJER': (-119.8087, 37.0262, -119.6596, 37.1564),
    'SOAP': (-119.346, 36.965, -119.1634, 37.1215),
    'SRER': (-110.9944, 31.693, -110.7199, 31.9379),
    'STEI': (-89.6328, 45.4155, -89.3602, 45.5927),
    'STER': (-103.1193, 40.4029, -102.9636, 40.5593),
    'SYCA': (-111.5781, 33.7127, -111.362, 33.8768),
    'TALL': (-87.4868, 32.8381, -87.3476, 33.0199),
    'TEAK': (-119.1211, 36.9057, -118.9146, 37.1252),
    'TOOL': (-149.7446, 68.5038, -149.0768, 68.7277),
    'UKFS': (-95.2718, 38.9301, -95.118, 39.1137),
    'UNDE': (-89.6147, 46.109, -89.4023, 46.3031),
    'WLOU': (-105.9646, 39.8491, -105.8601, 39.9319),
    'WOOD': (-99.3355, 47.056, -99.0082, 47.2735),
    'WREF': (-122.1176, 45.754, -121.7574, 45.9111),
    'YELL': (-110.6886, 44.8407, -110.3606, 45.005),
}

# Series 0: 3 California Sierra Sites (2013-2018)
NEON_SERIES_0_SITES = set(['SJER', 'SOAP', 'TEAK'])

# Series 1: 14 Common Sites (2016-2018)
NEON_SERIES_1_SITES = set([
    'BART',
    'CLBJ',
    'DELA',
    'DSNY',
    'GRSM',
    'HARV',
    'JERC',
    'KONZ',
    'LENO',
    'OAES',
    'ORNL',
    'OSBS',
    'TALL',
    'UKFS',
])

# Series 2: 12 Common Sites (2019-2025)
NEON_SERIES_2_SITES = set([
    'ABBY',
    'BONA',
    'CLBJ',
    'DEJU',
    'DSNY',
    'HEAL',
    'NOGP',
    'OAES',
    'OSBS',
    'SCBI',
    'WOOD',
    'WREF',
])


def get_site_for_coords(lon, lat):
  """Maps longitude/latitude to NEON site code using spatial bounding boxes."""
  for site, (min_x, min_y, max_x, max_y) in NEON_SITE_BOUNDS.items():
    if min_x <= lon <= max_x and min_y <= lat <= max_y:
      return site
  return None


def _timeseries_row(m, **extra):
  """Maps canonical metrics.py keys onto this plot's timeseries row schema.

  Uses direct indexing rather than `.get(..., 0)` so a missing or renamed
  metric raises instead of being silently plotted as a real 0.00 m value.

  Args:
    m: Dictionary of pooled metrics from metrics.py.
    **extra: Additional metadata key-values (e.g. year, site).

  Returns:
    Row dictionary matching the timeseries DataFrame schema.
  """
  row: Dict[str, Any] = dict(extra)
  row['num_samples'] = float(m['num_samples'])
  for src in ['ge', 'cop', 'alos', 'nasa']:
    row[f'{src}_drmse'] = float(m[f'{src}_debiased_rmse'])
    row[f'{src}_rmse'] = float(m[f'{src}_direct_rmse'])
    row[f'{src}_bias'] = float(m[f'{src}_bias'])
    row[f'{src}_nmad'] = float(m[f'{src}_nmad'])
  return row


def load_neon_cohort_series(
    data_dir,
):
  """Loads metrics for Series 1 (14 sites: 2016-18) and Series 2 (12 sites: 2019-25)."""
  s1_years = [2016, 2017, 2018]
  s2_years = [2019, 2021, 2023, 2025]

  def _load_single_year(
      year, series_idx, target_sites
  ):
    series_name = f'Series {series_idx} ({len(target_sites)} Sites)'
    # 1. Check dedicated series directory
    dedicated_dir = os.path.join(
        data_dir, f'neon_series_{series_idx}_{year}_lidar_gem15'
    )
    dedicated_csv = os.path.join(dedicated_dir, 'results.csv')
    if os.path.exists(dedicated_csv):
      print(
          f'Loading dedicated {series_name} metrics for {year}: {dedicated_csv}'
      )
      m = compute_metrics_from_csv(dedicated_csv)
      return _timeseries_row(m, year=year, series=series_name)

    # 2. Fallback: Filter from existing neon_{year}_lidar_gem15/results.csv
    base_csv = os.path.join(data_dir, f'neon_{year}_lidar_gem15', 'results.csv')
    if os.path.exists(base_csv):
      print(f'Filtering {series_name} from {base_csv} for {year}...')
      with open(base_csv, 'rb') as f:
        df_raw = pd.read_csv(io.BytesIO(f.read()))

      if 'lon' in df_raw.columns and 'lat' in df_raw.columns:
        sites_list = [
            get_site_for_coords(x, y)
            for x, y in zip(df_raw['lon'], df_raw['lat'])
        ]
        df_raw['site'] = pd.Series(sites_list, index=df_raw.index)
        mask = df_raw['site'].isin(target_sites)
        df_filtered = df_raw[mask].copy()
        if not df_filtered.empty:
          m = compute_metrics_from_df(df_filtered)
          return _timeseries_row(m, year=year, series=series_name)
    return None

  s1_rows = [_load_single_year(y, 1, NEON_SERIES_1_SITES) for y in s1_years]
  s2_rows = [_load_single_year(y, 2, NEON_SERIES_2_SITES) for y in s2_years]

  df_s1 = pd.DataFrame([r for r in s1_rows if r is not None])
  df_s2 = pd.DataFrame([r for r in s2_rows if r is not None])
  return df_s1, df_s2


def load_neon_yearly_metrics(
    data_dir, min_year, max_year
):
  """Scans data_dir for yearly NEON validation results and compiles timeseries metrics."""
  rows = []
  years = list(range(min_year, max_year + 1))

  for year in years:
    csv_path = os.path.join(data_dir, f'neon_{year}_lidar_gem15', 'results.csv')
    if not os.path.exists(csv_path):
      csv_path = None

    if csv_path and csv_path.endswith('results.csv'):
      print(f'Computing metrics for NEON {year} from: {csv_path}...')
      m = compute_metrics_from_csv(csv_path)
      rows.append(_timeseries_row(m, year=year))

  df = pd.DataFrame(rows)
  if not df.empty:
    df = df.sort_values(by='year').reset_index(drop=True)
  return df


def plot_neon_timeseries(df, output_dir):
  """Generates publication-ready timeseries plots for NEON survey years."""
  if df.empty:
    print('Warning: No NEON yearly data available to plot.')
    return

  colors = {
      'ge': '#ff7f0e',  # GEM-15 (orange)
      'cop': '#2ca02c',  # Copernicus (green)
      'alos': '#1f77b4',  # ALOS (blue)
      'nasa': '#d62728',  # NASA (red)
  }
  labels = {
      'ge': 'GEM-15 DSM v1 (Ours)',
      'cop': 'Copernicus DEM GLO-30',
      'alos': 'ALOS World 3D (AW3D30)',
      'nasa': 'NASADEM',
  }
  years = df['year'].tolist()

  plots_def = [
      (
          'drmse',
          'Debiased RMSE (meters)',
          'neon_timeseries_drmse_plot.png',
          'US NEON Aerial LiDAR: Debiased RMSE (DRMSE) Timeseries (2013-2025)',
      ),
      (
          'nmad',
          'NMAD (meters)',
          'neon_timeseries_nmad_plot.png',
          (
              'US NEON Aerial LiDAR: Normalized Median Absolute Deviation'
              ' (NMAD) Timeseries'
          ),
      ),
      (
          'rmse',
          'Direct RMSE (meters)',
          'neon_timeseries_rmse_plot.png',
          'US NEON Aerial LiDAR: Direct RMSE Timeseries (2013-2025)',
      ),
      (
          'bias',
          'Mean Bias (meters)',
          'neon_timeseries_bias_plot.png',
          'US NEON Aerial LiDAR: Mean Vertical Bias Timeseries (2013-2025)',
      ),
  ]

  # 1. Generate Individual Timeseries Plots
  for metric, ylabel, filename, title in plots_def:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)

    for src in ['ge', 'cop', 'alos', 'nasa']:
      col = f'{src}_{metric}'
      if col in df.columns:
        ax.plot(
            df['year'],
            df[col],
            marker='o',
            markersize=7,
            linewidth=2.5,
            color=colors[src],
            label=labels[src],
            zorder=4 if src == 'ge' else 3,
        )

    ax.set_xlabel(
        'NEON Survey Year', fontsize=12, fontweight='bold', labelpad=8
    )
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(years)
    ax.grid(True, linestyle=':', alpha=0.6)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + 0.28 * y_range)
    ax.legend(
        loc='upper left',
        fontsize=10,
        frameon=True,
        facecolor='white',
        framealpha=0.9,
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, filename)
    save_plot(fig, out_path)
    plt.close(fig)
    print(f'Saved NEON plot: {out_path}')

  # 2. Generate 4-Panel Combined Figure
  fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=200)
  panel_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

  for (metric, ylabel, _, title), (r, c) in zip(plots_def, panel_coords):
    ax = axes[r, c]
    for src in ['ge', 'cop', 'alos', 'nasa']:
      col = f'{src}_{metric}'
      if col in df.columns:
        ax.plot(
            df['year'],
            df[col],
            marker='o',
            markersize=6,
            linewidth=2.2,
            color=colors[src],
            label=labels[src],
            zorder=4 if src == 'ge' else 3,
        )

    ax.set_xlabel('NEON Survey Year', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(years)
    ax.grid(True, linestyle=':', alpha=0.6)
    if r == 0 and c == 0:
      ax.legend(
          loc='best',
          fontsize=9,
          frameon=True,
          facecolor='white',
          framealpha=0.9,
      )

  plt.tight_layout()
  combined_path = os.path.join(output_dir, 'neon_timeseries_4panel_plot.png')
  save_plot(fig, combined_path)
  plt.close(fig)
  print(f'Saved combined 4-panel NEON plot: {combined_path}')

  # 3. Save Compiled CSV
  csv_out = os.path.join(output_dir, 'neon_timeseries_results.csv')
  with open(csv_out, 'w') as f:
    df.to_csv(f, index=False)
  print(f'Saved compiled NEON timeseries CSV: {csv_out}')


def plot_neon_series_comparison(
    df_s1,
    df_s2,
    output_dir,
):
  """Plots Series 1 and Series 2 as two distinct lines for all comparisons."""
  if df_s1.empty and df_s2.empty:
    print('Warning: No NEON series data available to plot.')
    return

  colors = {
      'ge': '#ff7f0e',  # GEM-15 (orange)
      'cop': '#2ca02c',  # Copernicus (green)
      'alos': '#1f77b4',  # ALOS (blue)
      'nasa': '#d62728',  # NASA (red)
  }
  labels = {
      'ge': 'GEM-15 DSM v1 (Ours)',
      'cop': 'Copernicus DEM GLO-30',
      'alos': 'ALOS World 3D (AW3D30)',
      'nasa': 'NASADEM',
  }

  plots_def = [
      (
          'drmse',
          'Debiased RMSE (meters)',
          'neon_series_drmse_plot.png',
          (
              'US NEON Aerial LiDAR: Constant Cohort Debiased RMSE (DRMSE)'
              ' Timeseries (2016-2025)'
          ),
      ),
      (
          'nmad',
          'NMAD (meters)',
          'neon_series_nmad_plot.png',
          'US NEON Aerial LiDAR: Constant Cohort NMAD Timeseries (2016-2025)',
      ),
      (
          'rmse',
          'Direct RMSE (meters)',
          'neon_series_rmse_plot.png',
          (
              'US NEON Aerial LiDAR: Constant Cohort Direct RMSE Timeseries'
              ' (2016-2025)'
          ),
      ),
      (
          'bias',
          'Mean Bias (meters)',
          'neon_series_bias_plot.png',
          (
              'US NEON Aerial LiDAR: Constant Cohort Mean Vertical Bias'
              ' Timeseries (2016-2025)'
          ),
      ),
  ]

  all_years = list(range(2016, 2026))

  # 1. Generate Individual Timeseries Plots
  for metric, ylabel, filename, title in plots_def:
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)

    for src in ['ge', 'cop', 'alos', 'nasa']:
      col = f'{src}_{metric}'
      # Plot Series 1 (dashed line, circle marker, 2016-2018)
      if not df_s1.empty and col in df_s1.columns:
        ax.plot(
            df_s1['year'],
            df_s1[col],
            marker='o',
            markersize=7,
            linewidth=2.4,
            linestyle='--',
            color=colors[src],
            label=f'{labels[src]} (Series 1: 14 Sites, 2016–18)',
            zorder=4 if src == 'ge' else 3,
        )
      # Plot Series 2 (solid line, square marker, 2019-2025)
      if not df_s2.empty and col in df_s2.columns:
        ax.plot(
            df_s2['year'],
            df_s2[col],
            marker='s',
            markersize=7.5,
            linewidth=2.8,
            linestyle='-',
            color=colors[src],
            label=f'{labels[src]} (Series 2: 12 Sites, 2019–25)',
            zorder=5 if src == 'ge' else 4,
        )

    ax.set_xlabel(
        'NEON Survey Year', fontsize=12, fontweight='bold', labelpad=8
    )
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(all_years)
    ax.grid(True, linestyle=':', alpha=0.6)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + 0.28 * y_range)
    ax.legend(
        loc='upper left',
        fontsize=8.5,
        frameon=True,
        facecolor='white',
        framealpha=0.9,
        ncol=2,
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, filename)
    save_plot(fig, out_path)
    plt.close(fig)
    print(f'Saved NEON series plot: {out_path}')

  # 2. Generate 4-Panel Combined Figure
  fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=200)
  panel_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

  for (metric, ylabel, _, title), (r, c) in zip(plots_def, panel_coords):
    ax = axes[r, c]
    for src in ['ge', 'cop', 'alos', 'nasa']:
      col = f'{src}_{metric}'
      # Plot Series 1 (dashed line, circle marker)
      if not df_s1.empty and col in df_s1.columns:
        ax.plot(
            df_s1['year'],
            df_s1[col],
            marker='o',
            markersize=6,
            linewidth=2.2,
            linestyle='--',
            color=colors[src],
            label=f'{labels[src]} (Series 1: 14 Sites)',
            zorder=4 if src == 'ge' else 3,
        )
      # Plot Series 2 (solid line, square marker)
      if not df_s2.empty and col in df_s2.columns:
        ax.plot(
            df_s2['year'],
            df_s2[col],
            marker='s',
            markersize=6.5,
            linewidth=2.6,
            linestyle='-',
            color=colors[src],
            label=f'{labels[src]} (Series 2: 12 Sites)',
            zorder=5 if src == 'ge' else 4,
        )

    ax.set_xlabel('NEON Survey Year', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=10.5, fontweight='bold')
    ax.set_xticks(all_years)
    ax.grid(True, linestyle=':', alpha=0.6)
    if r == 0 and c == 0:
      ax.legend(
          loc='upper right',
          fontsize=7.5,
          frameon=True,
          facecolor='white',
          framealpha=0.9,
          ncol=2,
      )

  plt.tight_layout()
  series_4panel_path = os.path.join(output_dir, 'neon_series_4panel_plot.png')
  save_plot(fig, series_4panel_path)
  plt.close(fig)
  print(f'Saved combined 4-panel NEON series plot: {series_4panel_path}')

  # 3. Generate 2-Panel Side-by-Side DRMSE and NMAD Combined Figure
  fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
  for idx, (metric, ylabel, title_prefix) in enumerate([
      (
          'drmse',
          'Debiased RMSE (meters)',
          '(a) Constant Cohort Debiased RMSE (DRMSE)',
      ),
      ('nmad', 'NMAD (meters)', '(b) Constant Cohort NMAD Precision'),
  ]):
    ax = axes[idx]
    for src in ['ge', 'cop', 'alos', 'nasa']:
      col = f'{src}_{metric}'
      if not df_s1.empty and col in df_s1.columns:
        ax.plot(
            df_s1['year'],
            df_s1[col],
            marker='o',
            markersize=7,
            linewidth=2.2,
            linestyle='--',
            color=colors[src],
            label=f'{labels[src]} (Series 1: 14 Sites)',
            zorder=4 if src == 'ge' else 3,
        )
      if not df_s2.empty and col in df_s2.columns:
        ax.plot(
            df_s2['year'],
            df_s2[col],
            marker='s',
            markersize=7.5,
            linewidth=2.6,
            linestyle='-',
            color=colors[src],
            label=f'{labels[src]} (Series 2: 12 Sites)',
            zorder=5 if src == 'ge' else 4,
        )
    ax.set_xlabel(
        'NEON Survey Year', fontsize=12, fontweight='bold', labelpad=8
    )
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(title_prefix, fontsize=12.5, fontweight='bold', pad=12)
    ax.set_xticks(all_years)
    ax.grid(True, linestyle=':', alpha=0.6)
    y_min, y_max = ax.get_ylim()
    headroom = 0.35 if idx == 0 else 0.1
    ax.set_ylim(y_min, y_max + headroom * (y_max - y_min))
    if idx == 0:
      ax.legend(
          loc='upper right',
          fontsize=8.5,
          frameon=True,
          facecolor='white',
          framealpha=0.9,
          ncol=2,
      )
  plt.tight_layout()
  series_2panel_path = os.path.join(
      output_dir, 'neon_series_drmse_nmad_side_by_side.png'
  )
  save_plot(fig, series_2panel_path)
  plt.close(fig)
  print(
      'Saved 2-panel NEON series DRMSE & NMAD side-by-side plot:'
      f' {series_2panel_path}'
  )

  # 3. Save Compiled Series CSV
  df_series_all = pd.concat([df_s1, df_s2], ignore_index=True)
  csv_out = os.path.join(output_dir, 'neon_series_results.csv')
  with open(csv_out, 'w') as f:
    df_series_all.to_csv(f, index=False)
  print(f'Saved compiled NEON series CSV: {csv_out}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  os.makedirs(FLAGS.output_dir, exist_ok=True)

  # 1. Full pooled annual surveys (2013-2025)
  print(
      f'Scanning NEON yearly results in: {FLAGS.data_dir} for years'
      f' {FLAGS.min_year}-{FLAGS.max_year}...'
  )
  df_yearly = load_neon_yearly_metrics(
      FLAGS.data_dir, FLAGS.min_year, FLAGS.max_year
  )
  if not df_yearly.empty:
    print(
        f'Successfully loaded {len(df_yearly)} survey years of pooled NEON'
        ' metrics.'
    )
    plot_neon_timeseries(df_yearly, FLAGS.output_dir)

  # 2. Constant-cohort 2-series evaluation (Series 1 & Series 2: 12 sites)
  print(
      'Loading NEON constant-cohort series (Series 1: 12 sites, Series 2: 12'
      ' sites)...'
  )
  df_s1, df_s2 = load_neon_cohort_series(FLAGS.data_dir)
  if not df_s1.empty or not df_s2.empty:
    print(
        f'Loaded Series 1 ({len(df_s1)} points) and Series 2 ({len(df_s2)}'
        ' points).'
    )
    plot_neon_series_comparison(df_s1, df_s2, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
