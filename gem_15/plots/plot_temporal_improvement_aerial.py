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

"""Plot percentage improvement of GEM-15 over Copernicus DEM vs Survey Date for Aerial LiDAR."""

import os
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def plot_temporal_improvement(df, output_dir):
  """Plots temporal improvement of GEM-15 over Copernicus DEM from aerial & drone benchmark data."""
  aerial_df = df[df['category'].isin(['Aerial', 'Drone'])].copy()
  aerial_df = aerial_df.sort_values(
      by=['year', 'month', 'dataset_key']
  ).reset_index(drop=True)

  years = np.array(
      [r['year'] + (r['month'] - 0.5) / 12.0 for _, r in aerial_df.iterrows()]
  )
  year_mins = np.array([
      float(r.get('year_min', y))
      for (_, r), y in zip(aerial_df.iterrows(), years)
  ])
  year_maxs = np.array([
      float(r.get('year_max', y))
      for (_, r), y in zip(aerial_df.iterrows(), years)
  ])
  xerr_low = np.maximum(0.0, years - year_mins)
  xerr_high = np.maximum(0.0, year_maxs - years)

  ge_rmse = aerial_df['ge_debiased_rmse'].to_numpy()
  cop_rmse = aerial_df['cop_debiased_rmse'].to_numpy()
  pct_improvement = (cop_rmse - ge_rmse) / cop_rmse * 100.0

  fig, ax = plt.subplots(figsize=(18, 10), dpi=300)

  # Fill background regions
  ax.axhspan(
      0,
      38,
      facecolor='#eafaf1',
      alpha=0.5,
      label='GEM-15 Outperforms Copernicus DEM',
  )
  ax.axhspan(
      -18,
      0,
      facecolor='#fdedec',
      alpha=0.5,
      label='Copernicus DEM Edge (Legacy Surveys)',
  )

  # Horizontal zero parity line
  ax.axhline(0, color='#2c3e50', linestyle='--', linewidth=1.5, zorder=2)
  ax.text(
      2013.7,
      0.6,
      'Parity Line (0%)',
      fontsize=14,
      fontweight='bold',
      color='#2c3e50',
  )

  # Fit linear trend line excluding Indonesia & Brazil (closed-canopy
  # tropical rainforests).
  is_forest = aerial_df['dataset_key'].isin(['indonesia', 'brazil']).to_numpy()
  non_forest_mask = ~is_forest
  x_fit = years[non_forest_mask]
  y_fit = pct_improvement[non_forest_mask]
  slope, intercept = np.polyfit(x_fit, y_fit, 1)
  pearson_r = float(np.corrcoef(x_fit, y_fit)[0, 1])
  spearman_rho = float(
      pd.Series(x_fit).corr(pd.Series(y_fit), method='spearman')
  )
  print(
      f'OLS Fit (n={len(x_fit)}): slope={slope:.4f}%/yr,'
      f' intercept={intercept:.4f}, zero_crossing={-intercept/slope:.4f},'
      f' Pearson_r={pearson_r:.4f} (R^2={pearson_r**2:.4f}),'
      f' Spearman_rho={spearman_rho:.4f}'
  )
  linear_x = np.linspace(2014.2, 2024.8, 200)
  linear_y = slope * linear_x + intercept

  # Plot red dotted linear trend line
  ax.plot(
      linear_x,
      linear_y,
      color='#d62728',
      linestyle=':',
      linewidth=3.5,
      label='Linear Trend (excl. Tropical Forests)',
      zorder=3,
  )

  # Plot horizontal acquisition date range bars (xerr) for multi-year mosaics
  has_span = (xerr_low > 0.25) | (xerr_high > 0.25)
  span_non_forest = has_span & non_forest_mask
  span_forest = has_span & is_forest
  if np.any(span_non_forest):
    ax.errorbar(
        years[span_non_forest],
        pct_improvement[span_non_forest],
        xerr=[xerr_low[span_non_forest], xerr_high[span_non_forest]],
        fmt='none',
        ecolor='#5d6d7e',
        elinewidth=2.0,
        capsize=5,
        capthick=1.6,
        alpha=0.7,
        label='Multi-Year Tile Acquisition Span',
        zorder=4,
    )
  if np.any(span_forest):
    ax.errorbar(
        years[span_forest],
        pct_improvement[span_forest],
        xerr=[xerr_low[span_forest], xerr_high[span_forest]],
        fmt='none',
        ecolor='#aeb6bf',
        elinewidth=1.8,
        capsize=4,
        capthick=1.4,
        alpha=0.45,
        zorder=3,
    )

  # Plot Aerial (circles), Drone (diamonds), and Greyed-out Tropical Forest
  # benchmarks.
  is_drone = (aerial_df['category'] == 'Drone').to_numpy() & non_forest_mask
  is_aerial = (~(aerial_df['category'] == 'Drone').to_numpy()) & non_forest_mask
  colors_aerial = [
      '#27ae60' if p >= 0 else '#c0392b' for p in pct_improvement[is_aerial]
  ]
  colors_drone = [
      '#27ae60' if p >= 0 else '#c0392b' for p in pct_improvement[is_drone]
  ]

  ax.scatter(
      years[is_aerial],
      pct_improvement[is_aerial],
      s=170,
      marker='o',
      c=colors_aerial,
      edgecolor='black',
      linewidth=1.5,
      zorder=5,
  )
  # Dummy scatter for clean green legend marker for Aerial LiDAR
  ax.scatter(
      [],
      [],
      s=170,
      marker='o',
      c='#27ae60',
      edgecolor='black',
      linewidth=1.5,
      label='Aerial LiDAR Benchmark',
  )
  ax.scatter(
      years[is_drone],
      pct_improvement[is_drone],
      s=190,
      marker='D',
      c=colors_drone,
      edgecolor='black',
      linewidth=1.5,
      zorder=6,
  )
  # Dummy scatter for clean green rhombus legend marker for Drone LiDAR
  ax.scatter(
      [],
      [],
      s=190,
      marker='D',
      c='#27ae60',
      edgecolor='black',
      linewidth=1.5,
      label='Drone LiDAR Benchmark',
  )
  # Greyed-out Tropical Forest benchmarks (Indonesia, Brazil)
  ax.scatter(
      years[is_forest],
      pct_improvement[is_forest],
      s=165,
      marker='o',
      c='#bdc3c7',
      edgecolor='#7f8c8d',
      linewidth=1.4,
      alpha=0.65,
      label='Tropical Forest LiDAR (Excluded from Trend)',
      zorder=4,
  )

  # Custom callout offsets across all 12 Aerial + Drone datasets placed away
  # from horizontal bars.
  offset_map = {
      'mozambique': ((24, 14), 'left'),
      'indonesia': ((22, 14), 'left'),
      'brazil': ((18, 16), 'left'),
      'spain': ((-22, -12), 'right'),
      'us_tropical': ((-22, -10), 'right'),
      'uk_ea': ((-12, -14), 'right'),
      'canada': ((-18, 16), 'right'),
      '3dtrees': ((0, -14), 'center'),
      'global_tech_mahindra': ((-24, 14), 'right'),
      'neon': ((0, -15), 'center'),
      'new_zealand': ((0, 16), 'center'),
      'india_tech_mahindra': ((0, -15), 'center'),
  }

  for (_, row), x_pos, pct, forest_flag in zip(
      aerial_df.iterrows(), years, pct_improvement, is_forest
  ):
    ds_key = row['dataset_key']
    offset, ha = offset_map.get(ds_key, ((15, 15), 'left'))
    va = 'top' if offset[1] < 0 else 'bottom'
    sign = '+' if pct > 0 else ''
    lbl = f"{row['dataset_name']} ({row['month_str']})\n{sign}{pct:.1f}%"
    text_col = '#7f8c8d' if forest_flag else '#1b2631'
    box_edge = '#d5d8dc' if forest_flag else '#bdc3c7'
    box_face = '#f8f9f9' if forest_flag else 'white'
    box_alpha = 0.80 if forest_flag else 0.95
    arr_col = '#95a5a6' if forest_flag else '#444444'
    ax.annotate(
        lbl,
        xy=(x_pos, pct),
        xytext=offset,
        textcoords='offset points',
        ha=ha,
        va=va,
        fontsize=12.5,
        fontweight='bold',
        color=text_col,
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor=box_face,
            edgecolor=box_edge,
            alpha=box_alpha,
            lw=1.0,
        ),
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            color=arr_col,
            lw=1.3,
        ),
        zorder=7 if not forest_flag else 4,
    )

  # Axis and Title styling
  ax.set_xlabel(
      'Aerial & Drone LiDAR Survey Date',
      fontsize=17,
      fontweight='bold',
      labelpad=12,
  )
  ax.set_ylabel(
      'Improvement over Copernicus DEM (%)',
      fontsize=17,
      fontweight='bold',
      labelpad=12,
  )
  ax.set_title(
      'GEM-15 Accuracy Improvement over Copernicus DEM vs. Aerial & Drone LiDAR'
      ' Survey Date',
      fontsize=19.5,
      fontweight='bold',
      pad=22,
  )

  ax.set_xlim(2013.5, 2025.6)
  xticks = list(range(2014, 2026))
  ax.set_xticks(xticks)
  ax.set_xticklabels(
      [str(y) for y in xticks],
      fontsize=15,
      fontweight='bold',
  )
  ax.tick_params(axis='y', labelsize=15)
  ax.set_ylim(-18, 38)
  ax.grid(True, linestyle='--', alpha=0.45, zorder=0)

  # Legend
  legend = ax.legend(
      loc='lower right',
      frameon=True,
      facecolor='white',
      framealpha=0.95,
      edgecolor='#cccccc',
      fontsize=14.5,
  )
  legend.get_frame().set_linewidth(1.3)

  plt.tight_layout()

  out_path = os.path.join(
      output_dir, 'gem15_vs_copernicus_temporal_improvement.png'
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
      save_summary_csv=False,
  )
  plot_temporal_improvement(df, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
