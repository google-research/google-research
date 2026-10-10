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

"""Benchmark data module to compile and load validation metrics from CSVs."""

import io
import os
from typing import Any, Dict, List, Optional, cast
import numpy as np
import pandas as pd
import ee
from gem_15 import ee_assets_config
from gem_15 import metrics
from gem_15 import utils

DEFAULT_VALIDATION_DIR = (
    './validation_outputs'
)

DTM_EXCLUDED_DATASETS = frozenset({
    'brazil',
    'canada',
    'mozambique',
    'us_tropical',
})


def _sort_date(key, year, month):
  """Returns the acquisition-span midpoint (decimal year) used for ordering."""
  ds = next((d for d in BENCHMARK_DATASETS if d['key'] == key), {})
  default_x = year + (month - 0.5) / 12.0
  return (ds.get('year_min', default_x) + ds.get('year_max', default_x)) / 2.0


def sort_benchmark_df(df):
  """Sorts benchmark dataframe: Spaceborne first, then Drone by recency, then Aerial by recency."""
  if df.empty or 'category' not in df.columns:
    return df
  cat_order = {'Spaceborne': 0, 'Drone': 1, 'Aerial': 2}
  df = df.copy()
  df['_cat_rank'] = df['category'].map(lambda c: cat_order.get(c, 99))
  df['_neg_date'] = [
      -_sort_date(k, int(y), int(m))
      for k, y, m in zip(df['dataset_key'], df['year'], df['month'])
  ]
  # Preserve stable tie-breaking order from BENCHMARK_DATASETS
  key_order = {ds['key']: i for i, ds in enumerate(BENCHMARK_DATASETS)}
  df['_key_rank'] = df['dataset_key'].map(lambda k: key_order.get(k, 99))
  df = df.sort_values(by=['_cat_rank', '_neg_date', '_key_rank']).drop(
      columns=['_cat_rank', '_neg_date', '_key_rank']
  )
  return df.reset_index(drop=True)


def filter_dtm_benchmarks(df):
  """Filters out datasets with invalid/noisy ground-truth DTM from DTM benchmark plots."""
  df = sort_benchmark_df(df)
  if 'dataset_key' in df.columns:
    df = df[~df['dataset_key'].isin(DTM_EXCLUDED_DATASETS)]
  if 'ge_dtm_debiased_rmse' in df.columns:
    df = df[df['ge_dtm_debiased_rmse'].notna()]
  return df.reset_index(drop=True)


def add_category_dividers_and_banners(
    ax,
    df,
    y_banner,
    aerial_label = 'AERIAL LiDAR',
):
  """Dynamically adds vertical dotted category dividers and category banners based on df."""
  if 'category' not in df.columns or df.empty:
    return
  cats = df['category'].tolist()
  n = len(cats)

  s_idx = [i for i, c in enumerate(cats) if c == 'Spaceborne']
  d_idx = [i for i, c in enumerate(cats) if c == 'Drone']
  a_idx = [i for i, c in enumerate(cats) if c == 'Aerial']

  if s_idx:
    if s_idx[-1] < n - 1:
      ax.axvline(
          x=s_idx[-1] + 0.5,
          color='#444444',
          linestyle=':',
          linewidth=2.0,
          zorder=2,
      )
    ax.text(
        (s_idx[0] + s_idx[-1]) / 2.0,
        y_banner,
        'SPACEBORNE\nLiDAR',
        ha='center',
        va='center',
        fontsize=17.5,
        fontweight='bold',
        color='#1a5276',
        bbox=dict(
            boxstyle='round,pad=0.45',
            facecolor='#ebf5fb',
            edgecolor='#aed6f1',
            lw=1.6,
        ),
    )

  if d_idx:
    if d_idx[-1] < n - 1:
      ax.axvline(
          x=d_idx[-1] + 0.5,
          color='#444444',
          linestyle=':',
          linewidth=2.0,
          zorder=2,
      )
    ax.text(
        (d_idx[0] + d_idx[-1]) / 2.0,
        y_banner,
        'DRONE LiDAR',
        ha='center',
        va='center',
        fontsize=17.5,
        fontweight='bold',
        color='#7d6608',
        bbox=dict(
            boxstyle='round,pad=0.45',
            facecolor='#fef9e7',
            edgecolor='#f9e79f',
            lw=1.6,
        ),
    )

  if a_idx:
    ax.text(
        (a_idx[0] + a_idx[-1]) / 2.0,
        y_banner,
        aerial_label,
        ha='center',
        va='center',
        fontsize=17.5,
        fontweight='bold',
        color='#196f3d',
        bbox=dict(
            boxstyle='round,pad=0.45',
            facecolor='#eafaf1',
            edgecolor='#a9dfbf',
            lw=1.6,
        ),
    )


def format_pixel_count(n_pixels):
  """Formats pixel count dynamically (e.g. 1.1M px, 348k px)."""
  if n_pixels >= 1_000_000:
    return f'{n_pixels / 1_000_000:.1f}M px'
  elif n_pixels >= 1_000:
    return f'{n_pixels / 1_000:.0f}k px'
  else:
    return f'{int(n_pixels)} px'


BENCHMARK_DATASETS: List[Dict[str, Any]] = [
    # 1. Spaceborne LiDAR
    {
        'key': 'gedi',
        'display_name': 'GEDI',
        'category': 'Spaceborne',
        'subdirs': [
            'gedi_global_100k',
        ],
        'year': 2025,
        'month': 6,
        'month_str': 'Jun 2025',
        'plus_year': False,
        'desc_template': '{year_str}, {pixels_str}',
    },
    # 2. Drone LiDAR (Ordered by recency: recent first -> older)
    {
        'key': 'india_tech_mahindra',
        'display_name': 'India Drone',
        'category': 'Drone',
        'subdirs': [
            'india_tech_mahindra_drone_lidar_gem15',
            'india_tech_mahindra_gem15',
        ],
        'year': 2024,
        'month': 9,
        'month_str': 'Aug–Oct 2024',
        'year_min': 2024.62,
        'year_max': 2024.80,
        'plus_year': False,
        'desc_template': '2024, {pixels_str}',
    },
    {
        'key': 'global_tech_mahindra',
        'display_name': 'Global Drone',
        'category': 'Drone',
        'subdirs': [
            'global_tech_mahindra_drone_lidar_gem15',
            'global_tech_mahindra_gem15',
        ],
        'year': 2023,
        'month': 6,
        'month_str': 'Jun 2023',
        'plus_year': False,
        'desc_template': '{year_str}, {pixels_str}',
    },
    {
        'key': '3dtrees',
        'display_name': '3D Trees Drone',
        'category': 'Drone',
        'subdirs': ['3dtrees_drone_lidar_gem15'],
        'year': 2023,
        'month': 6,
        'month_str': '2023–2025',
        'year_min': 2023.45,
        'year_max': 2025.55,
        'plus_year': False,
        'desc_template': '2023–25, {pixels_str}',
    },
    # 3. Aerial LiDAR (Ordered by recency: recent first -> older)
    {
        'key': 'new_zealand',
        'display_name': 'New Zealand',
        'category': 'Aerial',
        'subdirs': ['new_zealand_lidar_gem15'],
        'year': 2024,
        'month': 5,
        'month_str': '2024–2025',
        'year_min': 2024.00,
        'year_max': 2025.08,
        'plus_year': False,
        'desc_template': '2024–25, {pixels_str}',
    },
    {
        'key': 'neon',
        'display_name': 'US NEON',
        'category': 'Aerial',
        'subdirs': [
            'neon_2022_lidar_gem15',
            'neon_2023_lidar_gem15',
            'neon_2024_lidar_gem15',
            'neon_2025_lidar_gem15',
        ],
        'year': 2023,
        'month': 11,
        'month_str': '2022–2025',
        'year_min': 2022.40,
        'year_max': 2025.55,
        'plus_year': False,
        'desc_template': '2022–25, {pixels_str}',
    },
    {
        'key': 'canada',
        'display_name': 'Canada',
        'category': 'Aerial',
        'subdirs': ['canada_lidar_gem15'],
        'year': 2021,
        'month': 5,
        'month_str': '2020–2023',
        'year_min': 2020.00,
        'year_max': 2023.92,
        'plus_year': False,
        'desc_template': '2020–23, {pixels_str}',
    },
    {
        'key': 'uk_ea',
        'display_name': 'UK England',
        'category': 'Aerial',
        'subdirs': ['uk_ea_lidar_gem15', 'england_lidar_gem15'],
        'year': 2021,
        'month': 1,
        'month_str': '2020–2022',
        'year_min': 2020.00,
        'year_max': 2022.50,
        'plus_year': False,
        'desc_template': '2020–22, {pixels_str}',
    },
    {
        'key': 'us_tropical',
        'display_name': 'US Tropical',
        'category': 'Aerial',
        'subdirs': ['us_tropical_lidar_gem15'],
        'year': 2020,
        'month': 3,
        'month_str': 'Mar 2020',
        'plus_year': False,
        'desc_template': '{year_str}, {pixels_str}',
    },
    {
        'key': 'spain',
        'display_name': 'Spain',
        'category': 'Aerial',
        'subdirs': ['spain_lidar_gem15', 'spain_lidar_comparison_15m_gem15'],
        'year': 2018,
        'month': 7,
        'month_str': '2018',
        'plus_year': False,
        'desc_template': '{year_str}, {pixels_str}',
    },
    {
        'key': 'brazil',
        'display_name': 'Brazil',
        'category': 'Aerial',
        'subdirs': [
            'brazil_lidar_gem15',
            'brazil_lidar_comparison_15m_badblocks_filtered',
        ],
        'year': 2016,
        'month': 1,
        'month_str': '2012–2018',
        'year_min': 2012.00,
        'year_max': 2018.50,
        'plus_year': False,
        'desc_template': '2012–18, {pixels_str}',
    },
    {
        'key': 'indonesia',
        'display_name': 'Indonesia',
        'category': 'Aerial',
        'subdirs': ['indonesia_lidar_gem15'],
        'year': 2014,
        'month': 10,
        'month_str': 'Oct 2014',
        'plus_year': False,
        'desc_template': '{year_str}, {pixels_str}',
    },
    {
        'key': 'mozambique',
        'display_name': 'Mozambique',
        'category': 'Aerial',
        'subdirs': ['mozambique_lidar_gem15'],
        'year': 2014,
        'month': 5,
        'month_str': 'May 2014',
        'plus_year': False,
        'desc_template': '{year_str}, {pixels_str}',
    },
]


def _ensure_slope_col(
    df, ee_project = None
):
  """Ensures slope_cop_mean column is present, computing via Earth Engine if missing."""
  if 'slope_cop_mean' in df.columns:
    return df
  if df.empty or 'lon' not in df.columns or 'lat' not in df.columns:
    return df
  print('Computing terrain slope (slope_cop_mean) via Earth Engine...')

  try:
    ee.Number(1).getInfo()
  except Exception:  # pylint: disable=broad-except
    utils.authenticate_earth_engine(project=ee_project)

  cop_dem = (
      ee.ImageCollection(ee_assets_config.COPERNICUS_GLO30_ASSET_ID)
      .mosaic()
      .select('DEM')
      .reproject(crs='EPSG:3857', scale=30)
  )
  slope_cop = ee.Terrain.slope(cop_dem).rename('slope_cop')
  batch_size = 400
  cop_means = []
  for start in range(0, len(df), batch_size):
    sub = df.iloc[start : start + batch_size]
    feats = [
        ee.Feature(
            ee.Geometry.Point([float(r['lon']), float(r['lat'])])
            .buffer(750)
            .bounds(),
            {'idx': int(idx)},
        )
        for idx, r in sub.iterrows()
    ]
    res = slope_cop.reduceRegions(
        collection=ee.FeatureCollection(feats),
        reducer=ee.Reducer.mean(),
        scale=30,
    ).getInfo()
    features = res.get('features', []) if isinstance(res, dict) else []
    res_map = {}
    for f in features:
      if isinstance(f, dict) and 'properties' in f and f['properties']:
        res_map[f['properties']['idx']] = f['properties']
    for idx in sub.index:
      cop_means.append(res_map.get(int(idx), {}).get('mean', np.nan))
  df = df.copy()
  df['slope_cop_mean'] = cop_means
  return df


def compute_metrics_from_df(
    df, min_slope_deg = 0.0
):
  """Computes summary statistics from a results DataFrame."""
  if min_slope_deg > 0.0:
    df = _ensure_slope_col(df)
  return metrics.compute_benchmark_metrics_from_df(
      df, min_slope_deg=min_slope_deg
  )


def compute_metrics_from_csv(
    csv_path, min_slope_deg = 0.0
):
  """Computes summary statistics from a results.csv file using Pooled Pixel-Weighted RMSE."""
  with open(csv_path, 'rb') as f:
    df = pd.read_csv(io.BytesIO(f.read()))
  return compute_metrics_from_df(df, min_slope_deg=min_slope_deg)


def load_or_compile_benchmark_summary(
    data_dir = None,
    summary_csv_path = None,
    save_summary_csv = True,
    min_slope_deg = 0.0,
):
  """Compiles the benchmark summary from the per-dataset results.csv files.

  Args:
    data_dir: Root validation directory containing dataset subdirectories.
    summary_csv_path: Optional explicit output CSV path.
    save_summary_csv: Whether to write the compiled summary CSV to disk.
    min_slope_deg: Minimum terrain slope threshold in degrees.

  Returns:
    Compiled benchmark summary DataFrame.
  """
  search_dir = data_dir or DEFAULT_VALIDATION_DIR
  print(f'Compiling benchmark summary from results in: {search_dir}...')

  rows = []
  for ds in BENCHMARK_DATASETS:
    csv_founds = []
    for subdir in ds['subdirs']:
      candidate = os.path.join(search_dir, subdir, 'results.csv')
      if os.path.exists(candidate):
        csv_founds.append(candidate)
        if ds['key'] != 'neon':
          break

    if not csv_founds:
      print(f"Warning: No results.csv found for dataset '{ds['display_name']}'")
      continue

    if len(csv_founds) > 1:
      print(
          f"Reading {ds['display_name']} from {len(csv_founds)} subdirs:"
          f' {csv_founds}'
      )
      dfs = []
      for p in csv_founds:
        with open(p, 'rb') as f:
          dfs.append(pd.read_csv(io.BytesIO(f.read())))
      combined_df = cast(pd.DataFrame, pd.concat(dfs, ignore_index=True))
      ds_metrics = compute_metrics_from_df(
          combined_df, min_slope_deg=min_slope_deg
      )
    else:
      csv_found = csv_founds[0]
      print(f"Reading {ds['display_name']} from: {csv_found}")
      with open(csv_found, 'rb') as f:
        raw_df = pd.read_csv(io.BytesIO(f.read()))
      ds_metrics = compute_metrics_from_df(raw_df, min_slope_deg=min_slope_deg)
      if ds['key'] == 'us_tropical':
        # US Tropical (Florida Everglades) is a flat sea-level wetland with
        # insufficient steep terrain data (only 200 pixels / 0.016% of dataset).
        # Set steep terrain metrics to NaN so steep plots are empty.
        for m_prefix in ['ge', 'cop', 'alos', 'nasa', 'ge_dtm', 'fab']:
          for stat in ['debiased_rmse', 'direct_rmse', 'bias', 'nmad']:
            ds_metrics[f'{m_prefix}_{stat}_steep'] = np.nan
        ds_metrics['total_pixels_steep'] = 0
    samples = int(ds_metrics.get('num_samples', 0))
    samples_k = samples / 1000.0
    total_pixels = ds_metrics.get('total_pixels', 0.0)
    pixels_str = format_pixel_count(total_pixels)
    pixels_k = total_pixels / 1000.0
    pixels_m = total_pixels / 1000000.0

    samples_str = (
        f'{samples_k:.0f}k tiles'
        if samples_k.is_integer()
        else f'{samples_k:.1f}k tiles'
    )
    year_str = (
        f"{ds['year']}+" if ds.get('plus_year', False) else f"{ds['year']}"
    )

    desc = ds['desc_template'].format(
        year=ds['year'],
        year_str=year_str,
        samples_k=samples_k,
        samples=samples,
        samples_str=samples_str,
        pixels_str=pixels_str,
        pixels_k=pixels_k,
        pixels_m=pixels_m,
        total_pixels=int(total_pixels),
    )

    default_x = ds['year'] + (ds['month'] - 0.5) / 12.0
    row = {
        'dataset_key': ds['key'],
        'dataset_name': ds['display_name'],
        'category': ds['category'],
        'year': ds['year'],
        'month': ds['month'],
        'month_str': ds['month_str'],
        'year_min': ds.get('year_min', default_x),
        'year_max': ds.get('year_max', default_x),
        'desc': desc,
        'label': f"{ds['display_name']}\n[{desc}]",
        'num_samples': samples,
        'total_pixels': int(total_pixels),
        'total_pixels_ag': int(ds_metrics.get('total_pixels_ag', 0.0)),
        'total_pixels_g': int(ds_metrics.get('total_pixels_g', 0.0)),
    }
    row.update(ds_metrics)
    rows.append(row)

  summary_df = sort_benchmark_df(pd.DataFrame(rows))

  if save_summary_csv and search_dir:
    out_csv = (
        summary_csv_path
        if summary_csv_path
        else os.path.join(search_dir, 'benchmark_summary.csv')
    )
    print(f'Saving compiled benchmark summary CSV to: {out_csv}...')
    with open(out_csv, 'w') as f:
      summary_df.to_csv(f, index=False)

  return summary_df


def compile_cors_continent_summary(
    data_dir = None,
    save_summary_csv = True,
):
  """Compiles continent-stratified CORS summary metrics from cors_checkpoints/results.csv."""
  search_dir = data_dir if data_dir else DEFAULT_VALIDATION_DIR
  cors_dir = os.path.join(search_dir, 'cors_checkpoints')
  csv_candidates = [
      os.path.join(cors_dir, 'results.csv'),
      os.path.join(cors_dir, 'results_2025.csv'),
      os.path.join(cors_dir, 'results_2024.csv'),
  ]
  cors_csv = None
  for cand in csv_candidates:
    if os.path.exists(cand):
      cors_csv = cand
      break
  if not cors_csv:
    print(f'Warning: No CORS results CSV found in {cors_dir}.')
    return pd.DataFrame()

  print(f'Reading International GNSS CORS results from: {cors_csv}')
  with open(cors_csv, 'r') as f:
    df = pd.read_csv(f)

  continent_order = [
      'Global (All)',
      'North America',
      'Europe',
      'Asia',
      'Oceania',
      'South America',
      'Africa',
  ]
  rows = []
  for cont in continent_order:
    sub = df if cont == 'Global (All)' else df[df['continent'] == cont]
    if sub.empty:
      continue
    point_m = metrics.compute_point_metrics_from_df(sub)
    station_count = int(point_m.get('count', len(sub)))
    row: dict[str, Any] = {
        'continent': cont,
        'count': station_count,
        'label': f'{cont}\n[N={station_count:,} stations]',
    }
    row.update(point_m)
    rows.append(row)

  cont_df = pd.DataFrame(rows)
  if save_summary_csv and search_dir:
    out_csv = os.path.join(search_dir, 'cors_continent_summary.csv')
    print(f'Saving compiled CORS continent summary CSV to: {out_csv}...')
    with open(out_csv, 'w') as f:
      cont_df.to_csv(f, index=False)
  return cont_df


def save_plot(
    plot_obj,
    out_path,
    dpi = 300,
):
  """Saves a matplotlib Figure or plt module to a local path with auto-created directories."""
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  parent_dir = os.path.dirname(out_path)
  if parent_dir and not os.path.exists(parent_dir):
    os.makedirs(parent_dir, exist_ok=True)
  with open(out_path, 'wb') as f:
    plot_obj.savefig(f, dpi=dpi)
  if plot_obj != plt:
    plt.close(plot_obj)
  else:
    plt.close()
  print('Saved:', out_path)
