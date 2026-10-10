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

"""Validate GEM-15 Percentile Elevation & STD Bands against Aerial/Drone LiDAR.

Evaluates 15m GEM-15 downsampled percentile bands against high-resolution
(50cm/1m) reference LiDAR downsampled to 15m with identical percentile
operations:
1. 5th Percentile: GEM-15 P5 vs GT LiDAR DSM P5 (Overall, Ground <= 5m, Above
Ground > 5m)
2. 50th Percentile: GEM-15 P50 vs GT LiDAR DSM P50 (Overall, Ground <= 5m, Above
Ground > 5m)
3. 95th Percentile: GEM-15 P95 vs GT LiDAR DSM P95 (Overall, Ground <= 5m, Above
Ground > 5m)
4. Sub-pixel Standard Deviation: GEM-15 dsm_std vs GT LiDAR DSM stdDev
5. Mean Surface & Bare-Earth: GEM-15 DSM vs GT DSM, GEM-15 DTM vs GT DTM
"""

import os
import time
from typing import Any, Dict, Sequence, cast
import uuid

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ee
from gem_15 import metrics
from gem_15.lidar_configs import get_lidar_config
from gem_15.utils import authenticate_earth_engine
from gem_15.utils import encode_sparse_histograms
from gem_15.utils import export_and_download_feature_collection
from gem_15.utils import get_gem15_layers
from gem_15.utils import get_global_reference_dems

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'ee_project',
    None,
    'Google Cloud project ID for Earth Engine.',
)
flags.DEFINE_string(
    'ee_service_account',
    None,
    'Service account principal to impersonate for Earth Engine authentication.',
)
flags.DEFINE_string(
    'output_dir',
    './validation_outputs/lidar_percentiles',
    'Directory path (local) where all outputs will be saved.',
)
flags.DEFINE_string(
    'load_csv',
    None,
    'Optional path to pre-computed results.csv to regenerate plots and report'
    ' without running Earth Engine export.',
)
flags.DEFINE_string(
    'lidar_dataset',
    'uk_ea',
    'Key of the LiDAR dataset to validate against (e.g. "uk_ea", "canada",'
    ' "spain", "new_zealand", "3dtrees", "global_tech_mahindra",'
    ' "india_tech_mahindra", "indonesia", "brazil", "mozambique",'
    ' "us_tropical").',
)
flags.DEFINE_integer(
    'max_regions_limit',
    1000,
    'Maximum number of spatial evaluation tiles to sample.',
)
flags.DEFINE_integer(
    'random_seed',
    42,
    'Random seed for deterministic tile selection.',
)


def plot_percentiles_comparison(
    df, output_dir, ref_name
):
  """Generates grouped bar chart of GEM-15 bands vs GT LiDAR bands."""
  plt.style.use(
      'seaborn-v0_8-whitegrid'
      if 'seaborn-v0_8-whitegrid' in plt.style.available
      else 'default'
  )
  _, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

  c_deb = '#1f77b4'  # Steel blue
  c_nmad = '#2ca02c'  # Forest green
  c_bias = '#e67e22'  # Amber

  bands_labels = [
      'DSM\n(Top)',
      'DTM\n(Ground)',
      'P95\n(Crown)',
      'P50\n(Median)',
      'P5\n(Base)',
      'STD\n(Dev)',
  ]
  sources = ['dsm', 'dtm', 'p95', 'p50', 'p5', 'std']

  nmads = []
  deb_rmses = []
  biases = []

  band_metrics = metrics.compute_percentiles_metrics_from_df(df, bands=sources)
  for src in sources:
    bm = band_metrics.get(src, {})
    deb_rmses.append(bm.get('debiased_rmse', np.nan))
    nmads.append(bm.get('nmad', np.nan))
    biases.append(bm.get('bias', np.nan))

  x = np.arange(len(bands_labels))
  w = 0.25

  r1 = ax.bar(
      x - w,
      deb_rmses,
      w,
      label='Debiased RMSE',
      color=c_deb,
      edgecolor='black',
      lw=0.6,
  )
  r2 = ax.bar(
      x, nmads, w, label='NMAD', color=c_nmad, edgecolor='black', lw=0.6
  )
  r3 = ax.bar(
      x + w, biases, w, label='Bias', color=c_bias, edgecolor='black', lw=0.6
  )

  ax.set_title(
      f'Overall GEM-15 vs Ground Truth {ref_name} (N = {len(df)} tiles)',
      fontsize=12,
      fontweight='bold',
      pad=12,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(bands_labels, fontsize=10, fontweight='bold')
  ax.set_ylabel('Error (meters)', fontsize=11)
  ax.axhline(0, color='black', lw=0.8, linestyle='--')
  ax.grid(True, linestyle='--', alpha=0.35)
  ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

  for group in [r1, r2, r3]:
    for bar in group:
      h = bar.get_height()
      if np.isnan(h):
        continue
      va = 'bottom' if h >= 0 else 'top'
      ax.annotate(
          f'{h:.2f}m',
          xy=(bar.get_x() + bar.get_width() / 2, h),
          xytext=(0, 3 if h >= 0 else -8),
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=8,
          fontweight='bold',
      )

  out_chart = os.path.join(output_dir, 'percentiles_comparison.png')
  with open(out_chart, 'wb') as f:
    plt.savefig(f, bbox_inches='tight')
  plt.close()
  print(f'Saved comparison plot to: {out_chart}')


def generate_report(df, output_dir, ref_name):
  """Generates markdown validation report."""
  report_path = os.path.join(output_dir, 'report.md')
  lines = [
      (
          f'# GEM-15 Downsampled Bands vs Ground Truth {ref_name} Validation'
          ' Report\n'
      ),
      f'Evaluates GEM-15 15m bands against {ref_name} LiDAR downsampled',
      (
          'from native resolution (50cm/1m) using identical reductions across'
          ' 15m pixels.\n'
      ),
      '## Validation Statistics (Pooled Pixel-Weighted)\n',
      '| Surface Band | Bias (m) | DRMSE (m) | NMAD (m) | Direct RMSE (m) |',
      '| :--- | :---: | :---: | :---: | :---: |',
  ]

  bands = [
      ('dsm', 'dsm', 'DSM (Mean Surface)'),
      ('dtm', 'dtm', 'DTM (Bare Earth)'),
      ('dsm_percentile_95', 'p95', 'P95 (Crown Top)'),
      ('dsm_percentile_50', 'p50', 'P50 (Median Surface)'),
      ('dsm_percentile_5', 'p5', 'P5 (Sub-Canopy / Base)'),
      ('dsm_std', 'std', 'STD (Surface Dispersion)'),
  ]

  band_metrics = metrics.compute_percentiles_metrics_from_df(
      df, bands=[b[1] for b in bands]
  )
  for _, b_code, b_label in bands:
    bm = band_metrics.get(b_code, {})
    bias = bm.get('bias', np.nan)
    deb = bm.get('debiased_rmse', np.nan)
    nmad = bm.get('nmad', np.nan)
    dir_r = bm.get('direct_rmse', np.nan)
    lines.append(
        f'| **{b_label}** | {bias:+.2f} | **{deb:.2f}** | **{nmad:.2f}** |'
        f' {dir_r:.2f} |'
    )

  with open(report_path, 'w') as f:
    f.write('\n'.join(lines))
  print(f'Saved report to: {report_path}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  os.makedirs(FLAGS.output_dir, exist_ok=True)
  lidar_cfg = get_lidar_config(FLAGS.lidar_dataset)
  ref_name = lidar_cfg['display_name']

  if FLAGS.load_csv:
    print(f'Loading pre-computed results from: {FLAGS.load_csv}...')
    with open(FLAGS.load_csv, 'r') as f:
      df = pd.read_csv(f)
    plot_percentiles_comparison(df, FLAGS.output_dir, ref_name)
    generate_report(df, FLAGS.output_dir, ref_name)
    return

  authenticate_earth_engine(
      ee_service_account=FLAGS.ee_service_account,
      project=FLAGS.ee_project,
  )

  # 1. Base Datasets
  # GEM-15 elevation and percentile layers with bilinear resampling across tiles
  gem15_layers = get_gem15_layers(resample_bilinear=True)
  gem15_collection = gem15_layers['collection']
  ge_dsm = gem15_layers['dsm']
  ge_dtm = gem15_layers['dtm']
  ge_p5 = gem15_layers['p5']
  ge_p50 = gem15_layers['p50']
  ge_p95 = gem15_layers['p95']
  ge_std = gem15_layers['std']

  ref_dems = get_global_reference_dems(resample_bilinear=True)
  egm96 = ref_dems['egm96']

  # 2. GT LiDAR Setup
  lidar_asset = lidar_cfg['asset_id']
  print(f'Loading ground truth LiDAR asset: {lidar_asset} ({ref_name})...')
  asset_info = cast(Dict[str, Any], ee.data.getAsset(lidar_asset))
  asset_type = asset_info.get('type')

  lidar_col = None
  sample_geometry = None
  if asset_type == 'IMAGE_COLLECTION':
    lidar_col = ee.ImageCollection(lidar_asset)
    if lidar_cfg['date_range']:
      lidar_col = lidar_col.filterDate(*lidar_cfg['date_range'])
    if lidar_cfg['collection_filter']:
      lidar_col = lidar_cfg['collection_filter'](lidar_col)
    if lidar_cfg['image_preprocess_fn']:
      lidar_col = lidar_col.map(lidar_cfg['image_preprocess_fn'])
    lidar_image = lidar_col.mosaic()
    sample_geometry = lidar_col.geometry().bounds()
    native_proj = ee.Image(lidar_col.first()).select(0).projection()
  else:
    lidar_image = ee.Image(lidar_asset)
    sample_geometry = lidar_image.geometry().bounds()
    native_proj = lidar_image.select(lidar_cfg['dsm_band']).projection()

  if lidar_cfg['geoid_asset'] is None:
    local_geoid = ee.Image.constant(0)
  else:
    local_geoid = ee.Image(lidar_cfg['geoid_asset']).select(
        lidar_cfg['geoid_band']
    )
    if lidar_cfg['geoid_scale'] != 1.0:
      local_geoid = local_geoid.multiply(lidar_cfg['geoid_scale'])

  if asset_type == 'IMAGE_COLLECTION' and lidar_cfg.get(
      'per_image_downsample', False
  ):
    print(
        'Applying per-image reduceResolution downsampling for multi-CRS'
        ' collection...'
    )

    def _prep_and_downsample_tile(img):
      dsm = img.select(lidar_cfg['dsm_band'])
      dtm = img.select(lidar_cfg['dtm_band'])
      vmask = dtm.neq(0)
      dsm = dsm.updateMask(vmask).add(local_geoid).subtract(egm96)
      dtm = dtm.updateMask(vmask).add(local_geoid).subtract(egm96)
      native_p = img.select(0).projection()
      p5 = (
          dsm.setDefaultProjection(native_p)
          .reduceResolution(
              reducer=ee.Reducer.percentile([5]),
              maxPixels=2048,
              bestEffort=True,
          )
          .reproject(crs='EPSG:3857', scale=15)
          .rename('gt_p5')
      )
      p50 = (
          dsm.setDefaultProjection(native_p)
          .reduceResolution(
              reducer=ee.Reducer.percentile([50]),
              maxPixels=2048,
              bestEffort=True,
          )
          .reproject(crs='EPSG:3857', scale=15)
          .rename('gt_p50')
      )
      p95 = (
          dsm.setDefaultProjection(native_p)
          .reduceResolution(
              reducer=ee.Reducer.percentile([95]),
              maxPixels=2048,
              bestEffort=True,
          )
          .reproject(crs='EPSG:3857', scale=15)
          .rename('gt_p95')
      )
      std = (
          dsm.setDefaultProjection(native_p)
          .reduceResolution(
              reducer=ee.Reducer.stdDev(), maxPixels=2048, bestEffort=True
          )
          .reproject(crs='EPSG:3857', scale=15)
          .rename('gt_std')
      )
      dsm_15m = (
          dsm.setDefaultProjection(native_p)
          .reduceResolution(
              reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
          )
          .reproject(crs='EPSG:3857', scale=15)
          .rename('gt_dsm')
      )
      dtm_15m = (
          dtm.setDefaultProjection(native_p)
          .reduceResolution(
              reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
          )
          .reproject(crs='EPSG:3857', scale=15)
          .rename('gt_dtm')
      )
      return ee.Image.cat([dsm_15m, dtm_15m, p95, p50, p5, std])

    assert lidar_col is not None
    downsampled_col = lidar_col.map(_prep_and_downsample_tile)
    downsampled_mosaic = downsampled_col.mosaic()
    gt_dsm = downsampled_mosaic.select('gt_dsm')
    gt_dtm = downsampled_mosaic.select('gt_dtm')
    gt_p95 = downsampled_mosaic.select('gt_p95')
    gt_p50 = downsampled_mosaic.select('gt_p50')
    gt_p5 = downsampled_mosaic.select('gt_p5')
    gt_std = downsampled_mosaic.select('gt_std')
    ref_dsm_geoid = gt_dsm
    ref_dtm_geoid = gt_dtm
  else:
    raw_dsm = lidar_image.select(lidar_cfg['dsm_band'])
    raw_dtm = lidar_image.select(lidar_cfg['dtm_band'])
    valid_mask = raw_dtm.neq(0)
    raw_dsm = raw_dsm.updateMask(valid_mask)
    raw_dtm = raw_dtm.updateMask(valid_mask)

    ref_dsm_geoid = raw_dsm.add(local_geoid).subtract(egm96)
    ref_dtm_geoid = raw_dtm.add(local_geoid).subtract(egm96)

    # 3. Resample Ground Truth to 15m with identical percentile operations
    gt_p5 = (
        ref_dsm_geoid.setDefaultProjection(native_proj)
        .reduceResolution(
            reducer=ee.Reducer.percentile([5]), maxPixels=2048, bestEffort=True
        )
        .reproject(crs='EPSG:3857', scale=15)
        .rename('gt_p5')
    )
    gt_p50 = (
        ref_dsm_geoid.setDefaultProjection(native_proj)
        .reduceResolution(
            reducer=ee.Reducer.percentile([50]), maxPixels=2048, bestEffort=True
        )
        .reproject(crs='EPSG:3857', scale=15)
        .rename('gt_p50')
    )
    gt_p95 = (
        ref_dsm_geoid.setDefaultProjection(native_proj)
        .reduceResolution(
            reducer=ee.Reducer.percentile([95]), maxPixels=2048, bestEffort=True
        )
        .reproject(crs='EPSG:3857', scale=15)
        .rename('gt_p95')
    )
    gt_std = (
        ref_dsm_geoid.setDefaultProjection(native_proj)
        .reduceResolution(
            reducer=ee.Reducer.stdDev(), maxPixels=2048, bestEffort=True
        )
        .reproject(crs='EPSG:3857', scale=15)
        .rename('gt_std')
    )
    gt_dsm = (
        ref_dsm_geoid.setDefaultProjection(native_proj)
        .reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
        )
        .reproject(crs='EPSG:3857', scale=15)
        .rename('gt_dsm')
    )
    gt_dtm = (
        ref_dtm_geoid.setDefaultProjection(native_proj)
        .reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
        )
        .reproject(crs='EPSG:3857', scale=15)
        .rename('gt_dtm')
    )

  # 4. Difference Bands
  common_mask = (
      ge_dsm.mask()
      .And(ge_dtm.mask())
      .And(ge_p5.mask())
      .And(ge_p50.mask())
      .And(ge_p95.mask())
      .And(ge_std.mask())
      .And(gt_dsm.mask())
      .And(gt_dtm.mask())
      .And(gt_p5.mask())
      .And(gt_p50.mask())
      .And(gt_p95.mask())
      .And(gt_std.mask())
  )

  pairs = [
      ('dsm', ge_dsm, gt_dsm),
      ('dtm', ge_dtm, gt_dtm),
      ('p95', ge_p95, gt_p95),
      ('p50', ge_p50, gt_p50),
      ('p5', ge_p5, gt_p5),
      ('std', ge_std, gt_std),
  ]

  eval_mask = common_mask

  diff_bands = []
  for name, ge_img, gt_img in pairs:
    diff = ge_img.subtract(gt_img)
    diff_bands.append(diff.updateMask(eval_mask).rename(f'{name}_diff'))

  all_diffs = ee.Image.cat(diff_bands)

  # 5. Tile Sampling
  print('Sampling spatial evaluation tiles...')
  buffer_radius = lidar_cfg['buffer_radius']

  if asset_type == 'IMAGE_COLLECTION':
    assert lidar_col is not None
    num_tiles = lidar_col.size()
    sample_col = ee.ImageCollection(
        ee.Algorithms.If(
            num_tiles.gt(500),
            lidar_col.randomColumn('rand', FLAGS.random_seed)
            .sort('rand')
            .limit(500),
            lidar_col,
        )
    )
    pts_per_tile = (
        ee.Number(FLAGS.max_regions_limit)
        .divide(sample_col.size())
        .ceil()
        .max(1)
    )
    if lidar_cfg['pts_per_tile_cap']:
      pts_per_tile = pts_per_tile.min(lidar_cfg['pts_per_tile_cap'])

    def sample_tile_img(img):
      return img.select(lidar_cfg['dsm_band']).sample(
          region=img.geometry(),
          scale=15,
          numPixels=pts_per_tile,
          seed=FLAGS.random_seed,
          geometries=True,
      )

    raw_pts = sample_col.map(sample_tile_img).flatten()
  else:
    assert sample_geometry is not None
    tiles_feats = gem15_collection.filterBounds(sample_geometry).map(
        lambda img: ee.Feature(img.geometry())
    )
    pts_per_tile = (
        ee.Number(FLAGS.max_regions_limit)
        .divide(tiles_feats.size())
        .ceil()
        .max(1)
    )

    def sample_tile_feat(f):
      return ref_dsm_geoid.sample(
          region=f.geometry(),
          scale=15,
          numPixels=pts_per_tile,
          seed=FLAGS.random_seed,
          geometries=True,
      )

    raw_pts = tiles_feats.map(sample_tile_feat).flatten()

  def make_eval_region(feat):
    coords = feat.geometry().coordinates()
    square = feat.geometry().buffer(buffer_radius).bounds()
    return (
        feat.setGeometry(square)
        .set('orig_lon', coords.get(0))
        .set('orig_lat', coords.get(1))
    )

  valid_mask = common_mask.selfMask().rename('valid_px')

  def count_valid_px(feat):
    geom = feat.geometry()
    cnt_dict = valid_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geom,
        crs='EPSG:3857',
        scale=15,
        maxPixels=100000000,
    )
    cnt = ee.Number(
        ee.Algorithms.If(
            cnt_dict.contains('valid_px'),
            cnt_dict.get('valid_px'),
            0,
        )
    )
    return feat.set('valid_pixel_count', cnt)

  # Cap candidate tiles, then drop tiles with < 10 valid pixels.
  eval_regions = (
      raw_pts.map(make_eval_region)
      .limit(FLAGS.max_regions_limit)
      .map(count_valid_px)
      .filter(ee.Filter.gte('valid_pixel_count', 10))
  )

  diff_band_names = [f'{name}_diff' for name, _, _ in pairs]

  # Pass 1 reducer: mean (bias), variance, count, median
  p1_reducer = (
      ee.Reducer.mean()
      .combine(reducer2=ee.Reducer.variance(), sharedInputs=True)
      .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
      .combine(reducer2=ee.Reducer.count(), sharedInputs=True)
  )

  def get_tile_metrics(feat):
    geom = feat.geometry()
    p1_stats = all_diffs.reduceRegion(
        reducer=p1_reducer,
        geometry=geom,
        crs='EPSG:3857',
        scale=15,
        maxPixels=100000000,
    )
    # Pass 2: NMAD = 1.4826 * median(|diff - median(diff)|)
    med_vals = [
        ee.Algorithms.If(
            p1_stats.contains(f'{b}_median'),
            ee.Algorithms.If(
                p1_stats.get(f'{b}_median'),
                p1_stats.get(f'{b}_median'),
                0.0,
            ),
            0.0,
        )
        for b in diff_band_names
    ]
    med_img = ee.Image.constant(med_vals).rename(diff_band_names)
    abs_diffs = all_diffs.select(diff_band_names).subtract(med_img).abs()
    mad_stats = abs_diffs.reduceRegion(
        reducer=ee.Reducer.median().combine(
            reducer2=ee.Reducer.fixedHistogram(0.0, 100.0, 10000),
            sharedInputs=True,
        ),
        geometry=geom,
        crs='EPSG:3857',
        scale=15,
        maxPixels=100000000,
    )
    med_keys = [f'{b}_median' for b in diff_band_names]
    renamed_mads = mad_stats.select(med_keys).rename(
        med_keys,
        [f'{b}_mad' for b in diff_band_names],
    )

    hist_dict = encode_sparse_histograms(mad_stats, diff_band_names)
    return feat.set(p1_stats.combine(renamed_mads).combine(hist_dict))

  evaluated = eval_regions.map(get_tile_metrics)

  asset_name = f'lidar_percentiles_{FLAGS.lidar_dataset}_{int(time.time())}_{uuid.uuid4().hex[:8]}'
  print(f'Exporting feature collection to: {asset_name}...')
  features = export_and_download_feature_collection(
      collection=evaluated,
      asset_name=asset_name,
      project=FLAGS.ee_project,
  )

  records = [f['properties'] for f in features]
  df = pd.DataFrame(records)
  out_csv = os.path.join(FLAGS.output_dir, 'lidar_percentiles_results.csv')
  with open(out_csv, 'w') as f:
    df.to_csv(f, index=False)
  print(f'Saved raw tile results to: {out_csv}')

  plot_percentiles_comparison(df, FLAGS.output_dir, ref_name)
  generate_report(df, FLAGS.output_dir, ref_name)
  print(
      '\nLidar percentiles validation completed successfully! Outputs:'
      f' {FLAGS.output_dir}'
  )


if __name__ == '__main__':
  app.run(main)
