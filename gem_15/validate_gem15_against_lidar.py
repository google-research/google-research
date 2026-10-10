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

"""GEM-15 DEM & DTM Validation Engine (GEDI Spaceborne & National Airborne Lidar).

This script performs validation of:
1. DSM: GEM-15 DSM v1 (15m) and public global DEMs (Copernicus GLO-30,
   ALOS AW3D30, and NASADEM) against reference DSM surfaces.
2. DTM: GEM-15 DTM v1 (15m) and FABDEM against reference bare-earth DTM
   surfaces.

Validation sources:
1. GEDI spaceborne lidar (global point footprint observations: RH98 for DSM,
   elev_lowestmode for DTM).
2. High-resolution national airborne Lidar (from UK, Canada, Spain, New Zealand,
   Mozambique, Indonesia, Brazil, US Tropical, Tech Mahindra, 3D Trees).


To resolve vertical datum mismatches, the script performs geoid alignment:
- GEDI elevation values (natively relative to WGS84 ellipsoid) are converted
  to EGM96 geoid heights by subtracting the local EGM96 geoid undulation.
- Airborne Lidar elevations (natively relative to local vertical datums) are
  aligned to EGM96 by applying local geoid offset grids (OSGM15, CGVD2013,
  REDNAP, NZVD2016, or EGM2008).
- GEM-15 DSM/DTM, Copernicus, ALOS, NASADEM, and FABDEM are kept in their
  native geoid frames (aligned to EGM96).

Evaluation metrics (Bias, Direct RMSE, and Debiased RMSE) are computed at a
resampled 15m resolution within Earth Engine, exported in batch mode, and
compiled locally.
"""

import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, Sequence, cast
import uuid
from absl import app
from absl import flags
import numpy as np
import pandas as pd
import ee
from gem_15 import ee_assets_config
from gem_15.lidar_configs import get_lidar_config
from gem_15.utils import apply_common_mask
from gem_15.utils import authenticate_earth_engine
from gem_15.utils import encode_sparse_histograms
from gem_15.utils import export_and_download_chunked_feature_collections
from gem_15.utils import export_and_download_feature_collection
from gem_15.utils import generate_plots_and_report
from gem_15.utils import get_clean_gedi_collection
from gem_15.utils import get_gedi_reference_layers
from gem_15.utils import get_gem15_layers
from gem_15.utils import get_global_reference_dems
from gem_15.utils import sample_gedi_evaluation_tiles

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'mode',
    'gedi',
    ['gedi', 'lidar'],
    'Validation mode: "gedi" for global GEDI point returns, "lidar" for gridded'
    ' national Lidar assets.',
)
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
    './validation_outputs',
    'Directory path (local) where all outputs will be saved.',
)
flags.DEFINE_string(
    'load_csv',
    None,
    'Optional path to pre-computed results.csv to regenerate plots and report'
    ' without running Earth Engine export.',
)
flags.DEFINE_integer(
    'num_points_per_tile',
    None,
    'Optional cap on GEDI reference points to sample per spatial granule tile'
    ' (only applicable in "gedi" mode).',
)
flags.DEFINE_integer(
    'max_regions_limit',
    100000,
    'Hard limit on the number of evaluation regions to process.',
)
flags.DEFINE_string(
    'lidar_dataset',
    'uk_ea',
    'Key or Earth Engine asset ID of the lidar dataset to validate against'
    ' (e.g. "3dtrees", "uk_ea", "canada", "spain", "new_zealand",'
    ' "mozambique", "indonesia", "brazil", "us_tropical",'
    ' "india_tech_mahindra", "global_tech_mahindra"). Only applicable in'
    ' "lidar" mode.',
)
flags.DEFINE_bool(
    'exclude_india',
    True,
    'Whether to exclude India from GEDI tile sampling due to regulatory noise'
    ' compliance.',
)
flags.DEFINE_integer(
    'random_seed',
    42,
    'Random seed for deterministic tile selection and point sampling.',
)
flags.DEFINE_float(
    'gedi_max_srtm_diff_m',
    100.0,
    'Maximum allowed difference in meters between GEDI lowest mode and SRTM'
    ' DEM to filter cloud reflections (defaults to 100.0m). Applicable in'
    ' "gedi" mode.',
)
flags.DEFINE_integer(
    'gedi_year',
    2025,
    'Evaluation year for monthly GEDI collection (defaults to 2025).',
)
flags.DEFINE_bool(
    'gedi_nighttime_only',
    True,
    'Whether to restrict GEDI observations to nighttime only (solar_elevation'
    ' < 0).',
)
flags.DEFINE_bool(
    'gedi_power_beams_only',
    False,
    'Whether to restrict GEDI observations to full-power beams only (beams 5,'
    ' 6, 8, 11).',
)
flags.DEFINE_float(
    'gedi_min_sensitivity',
    0.90,
    'Minimum sensitivity threshold for GEDI shots (defaults to 0.90).',
)
flags.DEFINE_bool(
    'gedi_no_elevation_bias',
    True,
    'Whether to filter out shots with potential 4-bin ranging error'
    ' (elevation_bias_flag == 0).',
)
flags.DEFINE_integer(
    'num_export_chunks',
    32,
    'Number of parallel Earth Engine export chunks for large region sets'
    ' (avoids EE batch worker memory overflow on heavy fixedHistogram'
    ' reductions). If 1, a single export task is used.',
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Ensure output directory exists
  os.makedirs(FLAGS.output_dir, exist_ok=True)

  is_gedi_mode = FLAGS.mode == 'gedi'
  if is_gedi_mode:
    ref_name = 'GEDI'
  else:
    lidar_cfg_init = get_lidar_config(FLAGS.lidar_dataset)
    ref_name = lidar_cfg_init['display_name']

  # If load_csv is specified, skip EE export and directly replot and report
  if FLAGS.load_csv:
    print(f'Loading pre-computed results from: {FLAGS.load_csv}...')
    with open(FLAGS.load_csv, 'r') as f:
      df = pd.read_csv(f)
    print(f'Loaded {len(df)} records from {FLAGS.load_csv}.')
    generate_plots_and_report(
        df, FLAGS.output_dir, ref_name=ref_name, is_lidar=not is_gedi_mode
    )
    print(f'\nAll outputs successfully saved to directory: {FLAGS.output_dir}')
    return

  # ---------------------------------------------------------
  # 1. Earth Engine Authentication and Impersonation
  # ---------------------------------------------------------
  authenticate_earth_engine(
      ee_service_account=FLAGS.ee_service_account,
      project=FLAGS.ee_project,
  )

  # ---------------------------------------------------------
  # 2. Base Dataset Definitions (GEM-15 & Public DEMs)
  # ---------------------------------------------------------
  # Earth Engine Collections (GEM-15 DSM & DTM)
  gem15_layers = get_gem15_layers(resample_bilinear=True)
  gem15_collection = gem15_layers['collection']
  ge_eval_dsm = gem15_layers['dsm']
  ge_eval_dtm = gem15_layers['dtm']

  # Public Reference DEMs & Geoids (aligned to EGM96)
  ref_dems = get_global_reference_dems(resample_bilinear=True)
  copernicus_dem = ref_dems['copernicus']
  alos_dem = ref_dems['alos']
  nasa_dem = ref_dems['nasa']
  fabdem = ref_dems['fabdem']
  egm96 = ref_dems['egm96']

  # ---------------------------------------------------------
  # 3. Setup Reference Elevation Assets (GEDI vs Lidar)
  # ---------------------------------------------------------

  asset_type = None
  lidar_col = None
  clean_gedi: ee.ImageCollection | None = None
  ref_dsm_geoid: ee.Image | None = None
  ref_dtm_geoid: ee.Image | None = None
  lidar_cfg: Dict[str, Any] | None = None
  already_downsampled = False
  if is_gedi_mode:
    print('\nRunning in GEDI validation mode (2025 monthly raster images)...')
    # Static native projection of LARSE/GEDI/GEDI02_A_002_MONTHLY (EPSG:4326,
    # 25m grid spacing = 0.0002245788210298804 deg, origin at [-180, 52]).
    native_proj = ee.Projection(
        'EPSG:4326',
        [0.0002245788210298804, 0, -180, 0, -0.0002245788210298804, 52],
    )
    sample_geometry = None
    asset_name = (
        f'global_gedi_validation_temp_{int(time.time())}_{uuid.uuid4().hex[:8]}'
    )

    clean_gedi, gedi_mosaic = get_clean_gedi_collection(
        year=FLAGS.gedi_year,
        gedi_max_srtm_diff_m=FLAGS.gedi_max_srtm_diff_m,
        nighttime_only=FLAGS.gedi_nighttime_only,
        power_beams_only=FLAGS.gedi_power_beams_only,
        min_sensitivity=FLAGS.gedi_min_sensitivity,
        no_elevation_bias=FLAGS.gedi_no_elevation_bias,
    )
    ref_dsm_geoid, ref_dtm_geoid = get_gedi_reference_layers(gedi_mosaic, egm96)
  else:
    lidar_cfg = get_lidar_config(FLAGS.lidar_dataset)
    lidar_asset = lidar_cfg['asset_id']
    ref_name = lidar_cfg['display_name']
    print(
        f'\nRunning in Lidar validation mode for: {ref_name} (key:'
        f' {FLAGS.lidar_dataset}, asset: {lidar_asset})...'
    )
    try:
      asset_info = cast(Dict[str, Any], ee.data.getAsset(lidar_asset))
      asset_type = asset_info.get('type')
      print(f'Asset type: {asset_type}')
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Failed to get asset info for {lidar_asset}: {e}')
      sys.exit(1)

    # 1. Prepare Collection or Image
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
    elif asset_type == 'IMAGE':
      lidar_image = ee.Image(lidar_asset)
      sample_geometry = lidar_image.geometry().bounds()
      native_proj = lidar_image.select(lidar_cfg['dsm_band']).projection()
    else:
      print(f'Unsupported asset type: {asset_type}')
      sys.exit(1)

    # 2. Local Geoid Alignment
    print(lidar_cfg['description'])
    if lidar_cfg['geoid_asset'] is None:
      local_geoid = ee.Image.constant(0)
    else:
      local_geoid = ee.Image(lidar_cfg['geoid_asset']).select(
          lidar_cfg['geoid_band']
      )
      if lidar_cfg['geoid_scale'] != 1.0:
        local_geoid = local_geoid.multiply(lidar_cfg['geoid_scale'])

    if asset_type == 'IMAGE_COLLECTION' and lidar_cfg['per_image_downsample']:
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
        dsm_15m = dsm.reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
        ).reproject(crs='EPSG:3857', scale=15)
        dtm_15m = dtm.reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
        ).reproject(crs='EPSG:3857', scale=15)
        return dsm_15m.rename('ref_dsm').addBands(dtm_15m.rename('ref_dtm'))

      assert lidar_col is not None
      downsampled_col = lidar_col.map(_prep_and_downsample_tile)
      downsampled_mosaic = downsampled_col.mosaic()
      ref_dsm_geoid = downsampled_mosaic.select('ref_dsm')
      ref_dtm_geoid = downsampled_mosaic.select('ref_dtm')
      already_downsampled = True
    else:
      # 3. Mask out unmasked 0 dropouts in raw Lidar
      raw_dsm = lidar_image.select(lidar_cfg['dsm_band'])
      raw_dtm = lidar_image.select(lidar_cfg['dtm_band'])
      valid_lidar_mask = raw_dtm.neq(0)
      raw_dsm = raw_dsm.updateMask(valid_lidar_mask)
      raw_dtm = raw_dtm.updateMask(valid_lidar_mask)

      # 4. Convert Lidar DSM & DTM to EGM96
      ref_dsm_geoid = raw_dsm.add(local_geoid).subtract(egm96)
      ref_dtm_geoid = raw_dtm.add(local_geoid).subtract(egm96)
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', lidar_asset.split('/')[-1])
    asset_name = (
        f'{clean_name}_validation_temp_{int(time.time())}_{uuid.uuid4().hex[:8]}_{FLAGS.lidar_dataset}'
    )

  # ---------------------------------------------------------
  # 4. Metric Evaluation Logic (Per-Tile/Footprint)
  # ---------------------------------------------------------
  eval_scale = 25 if is_gedi_mode else 15
  eval_crs = 'EPSG:3857'
  reduce_scale = eval_scale

  cop_eval_dem = copernicus_dem
  alos_eval_dem = alos_dem
  nasa_eval_dem = nasa_dem
  fab_eval_dem = fabdem
  slope_cop_img = ee.Terrain.slope(
      cop_eval_dem.reproject(crs='EPSG:3857', scale=30)
  ).rename('slope_cop')
  slope_fab_img = ee.Terrain.slope(
      fab_eval_dem.reproject(crs='EPSG:3857', scale=30)
  ).rename('slope_fab')

  # DTM validation land cover mask for GEDI:
  # Class 50 is built-up (building roofs).
  # Class 10 is tree cover (dense forest canopy where GEDI laser pulses cannot
  # reliably penetrate to the true ground surface).
  # Class 80 is permanent water bodies.
  worldcover_map = (
      ee.ImageCollection(ee_assets_config.ESA_WORLDCOVER_ASSET_ID)
      .mosaic()
      .select('Map')
  )
  worldcover_dtm_mask = (
      worldcover_map.neq(50)
      .And(worldcover_map.neq(10))
      .And(worldcover_map.neq(80))
  )

  def get_metrics_for_feature(feature):
    region = feature.geometry()
    buffered_region = region.buffer(30).bounds()

    # Clip GEM-15 DSM & DTM
    local_ge_dsm = ge_eval_dsm.clip(buffered_region).rename('ge_dsm')
    local_ge_dtm = ge_eval_dtm.clip(buffered_region).rename('ge_dtm')

    # Clip global reference DEMs
    local_copernicus = cop_eval_dem.clip(buffered_region).rename('copernicus')
    local_alos = alos_eval_dem.clip(buffered_region).rename('alos')
    local_nasa = nasa_eval_dem.clip(buffered_region).rename('nasa')
    local_fabdem = fab_eval_dem.clip(buffered_region).rename('fabdem')
    local_slope_cop = slope_cop_img.clip(buffered_region).rename('slope_cop')
    local_slope_fab = slope_fab_img.clip(buffered_region).rename('slope_fab')

    # Clip and process Reference Elevation Layers
    # For dense airborne/drone Lidar grids, downsample using mean reduction.
    assert ref_dsm_geoid is not None
    assert ref_dtm_geoid is not None
    if is_gedi_mode or already_downsampled:
      local_ref_dsm = ref_dsm_geoid.clip(buffered_region).rename('ref_dsm')
      local_ref_dtm = ref_dtm_geoid.clip(buffered_region).rename('ref_dtm')
    else:
      local_ref_dsm = (
          ref_dsm_geoid.setDefaultProjection(native_proj)
          .reduceResolution(
              reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
          )
          .reproject(crs='EPSG:3857', scale=15)
          .clip(buffered_region)
          .rename('ref_dsm')
      )
      local_ref_dtm = (
          ref_dtm_geoid.setDefaultProjection(native_proj)
          .reduceResolution(
              reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
          )
          .reproject(crs='EPSG:3857', scale=15)
          .clip(buffered_region)
          .rename('ref_dtm')
      )

    # 1. Above ground (canopy/structures) vs bare earth, by height above ground.
    height_above_ground = local_ref_dsm.subtract(local_ref_dtm)
    above_ground_mask = height_above_ground.gt(5.0)
    ground_mask = height_above_ground.lte(5.0)

    # 2. Flat (< 10 deg) vs steep (>= 10 deg) terrain.
    if is_gedi_mode:
      ref_slope = local_slope_cop
    else:
      ref_slope = ee.Terrain.slope(
          local_ref_dtm.reproject(crs='EPSG:3857', scale=eval_scale)
      )
    flat_mask = ref_slope.lt(10.0)
    steep_mask = ref_slope.gte(10.0)

    local_ref_dsm_eval = local_ref_dsm

    # 1. DSM Common Mask: evaluate on the intersection of valid, unmasked
    # pixels across all DSM models (GEM-15 DSM, Copernicus, ALOS, NASADEM)
    # and reference ground truth (ref_dsm).
    dsm_layers = apply_common_mask({
        'ref_dsm': local_ref_dsm_eval,
        'ge_dsm': local_ge_dsm,
        'copernicus': local_copernicus,
        'alos': local_alos,
        'nasa': local_nasa,
    })
    local_ref_dsm = dsm_layers['ref_dsm']
    local_ge_dsm = dsm_layers['ge_dsm']
    local_copernicus = dsm_layers['copernicus']
    local_alos = dsm_layers['alos']
    local_nasa = dsm_layers['nasa']

    # 2. DTM Common Mask (shared across ref_dtm, ge_dtm, and fabdem).
    # In GEDI mode the WorldCover land-cover filter (exclude built-up 50, tree
    # cover 10 and water 80) is applied to the GEDI reference DTM where
    # elev_lowestmode is affected by canopy or roof blockage.
    local_ref_dtm_eval = local_ref_dtm
    if is_gedi_mode:
      local_ref_dtm_eval = local_ref_dtm.updateMask(
          worldcover_dtm_mask.clip(buffered_region)
      )
    dtm_layers = apply_common_mask({
        'ref_dtm': local_ref_dtm_eval,
        'ge_dtm': local_ge_dtm,
        'fabdem': local_fabdem,
    })
    local_ref_dtm = dtm_layers['ref_dtm']
    local_ge_dtm = dtm_layers['ge_dtm']
    local_fabdem = dtm_layers['fabdem']

    # Compute elevation difference bands for DSM (Overall)
    ge_diff = local_ge_dsm.subtract(local_ref_dsm).rename('ge_diff')
    cop_diff = local_copernicus.subtract(local_ref_dsm).rename('cop_diff')
    alos_diff = local_alos.subtract(local_ref_dsm).rename('alos_diff')
    nasa_diff = local_nasa.subtract(local_ref_dsm).rename('nasa_diff')

    # Segment DSM differences by land type (Above Ground, HAG > 5m).
    ge_diff_ag = ge_diff.updateMask(above_ground_mask).rename('ge_diff_ag')
    cop_diff_ag = cop_diff.updateMask(above_ground_mask).rename('cop_diff_ag')
    alos_diff_ag = alos_diff.updateMask(above_ground_mask).rename(
        'alos_diff_ag'
    )
    nasa_diff_ag = nasa_diff.updateMask(above_ground_mask).rename(
        'nasa_diff_ag'
    )

    # Segment DSM differences by land type (Bare Earth Ground, HAG <= 5m)
    ge_diff_g = ge_diff.updateMask(ground_mask).rename('ge_diff_g')
    cop_diff_g = cop_diff.updateMask(ground_mask).rename('cop_diff_g')
    alos_diff_g = alos_diff.updateMask(ground_mask).rename('alos_diff_g')
    nasa_diff_g = nasa_diff.updateMask(ground_mask).rename('nasa_diff_g')

    # Compute elevation difference bands for DTM (Overall vs ref_dtm)
    ge_dtm_diff = local_ge_dtm.subtract(local_ref_dtm).rename('ge_dtm_diff')
    fab_diff = local_fabdem.subtract(local_ref_dtm).rename('fab_diff')

    # Segment DTM differences by land type (Above Ground Canopy / Structures)
    ge_dtm_diff_ag = ge_dtm_diff.updateMask(above_ground_mask).rename(
        'ge_dtm_diff_ag'
    )
    fab_diff_ag = fab_diff.updateMask(above_ground_mask).rename('fab_diff_ag')

    # Segment DTM differences by land type (Bare Earth Ground vs ref_dtm)
    ge_dtm_diff_g = ge_dtm_diff.updateMask(ground_mask).rename('ge_dtm_diff_g')
    fab_diff_g = fab_diff.updateMask(ground_mask).rename('fab_diff_g')

    # Segment DSM differences by slope
    ge_diff_flat = ge_diff.updateMask(flat_mask).rename('ge_diff_flat')
    cop_diff_flat = cop_diff.updateMask(flat_mask).rename('cop_diff_flat')
    alos_diff_flat = alos_diff.updateMask(flat_mask).rename('alos_diff_flat')
    nasa_diff_flat = nasa_diff.updateMask(flat_mask).rename('nasa_diff_flat')

    ge_diff_steep = ge_diff.updateMask(steep_mask).rename('ge_diff_steep')
    cop_diff_steep = cop_diff.updateMask(steep_mask).rename('cop_diff_steep')
    alos_diff_steep = alos_diff.updateMask(steep_mask).rename('alos_diff_steep')
    nasa_diff_steep = nasa_diff.updateMask(steep_mask).rename('nasa_diff_steep')

    # Segment DTM differences by slope
    ge_dtm_diff_flat = ge_dtm_diff.updateMask(flat_mask).rename(
        'ge_dtm_diff_flat'
    )
    fab_diff_flat = fab_diff.updateMask(flat_mask).rename('fab_diff_flat')

    ge_dtm_diff_steep = ge_dtm_diff.updateMask(steep_mask).rename(
        'ge_dtm_diff_steep'
    )
    fab_diff_steep = fab_diff.updateMask(steep_mask).rename('fab_diff_steep')

    combined_diffs = ee.Image.cat([
        ge_diff,
        cop_diff,
        alos_diff,
        nasa_diff,
        ge_diff_ag,
        cop_diff_ag,
        alos_diff_ag,
        nasa_diff_ag,
        ge_diff_g,
        cop_diff_g,
        alos_diff_g,
        nasa_diff_g,
        ge_diff_flat,
        cop_diff_flat,
        alos_diff_flat,
        nasa_diff_flat,
        ge_diff_steep,
        cop_diff_steep,
        alos_diff_steep,
        nasa_diff_steep,
        ge_dtm_diff,
        fab_diff,
        ge_dtm_diff_ag,
        fab_diff_ag,
        ge_dtm_diff_g,
        fab_diff_g,
        ge_dtm_diff_flat,
        fab_diff_flat,
        ge_dtm_diff_steep,
        fab_diff_steep,
        local_slope_cop,
        local_slope_fab,
    ])

    diff_bands = [
        'ge_diff',
        'cop_diff',
        'alos_diff',
        'nasa_diff',
        'ge_diff_ag',
        'cop_diff_ag',
        'alos_diff_ag',
        'nasa_diff_ag',
        'ge_diff_g',
        'cop_diff_g',
        'alos_diff_g',
        'nasa_diff_g',
        'ge_diff_flat',
        'cop_diff_flat',
        'alos_diff_flat',
        'nasa_diff_flat',
        'ge_diff_steep',
        'cop_diff_steep',
        'alos_diff_steep',
        'nasa_diff_steep',
        'ge_dtm_diff',
        'fab_diff',
        'ge_dtm_diff_ag',
        'fab_diff_ag',
        'ge_dtm_diff_g',
        'fab_diff_g',
        'ge_dtm_diff_flat',
        'fab_diff_flat',
        'ge_dtm_diff_steep',
        'fab_diff_steep',
    ]

    # Pass 1: Compute mean (bias), variance, median, and valid pixel count
    p1_stats = combined_diffs.reduceRegion(
        reducer=ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.variance(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.count(), sharedInputs=True),
        geometry=region,
        crs=eval_crs,
        scale=reduce_scale,
        maxPixels=100000000,
    )

    # Pass 2: Formal exact NMAD = 1.4826 * median(|diff - median(diff)|)
    # Subtract the per-band median to obtain absolute deviations from median
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
        for b in diff_bands
    ]
    med_img = ee.Image.constant(med_vals).rename(diff_bands)
    abs_diffs = combined_diffs.select(diff_bands).subtract(med_img).abs()
    mad_stats = abs_diffs.reduceRegion(
        reducer=ee.Reducer.median().combine(
            reducer2=ee.Reducer.fixedHistogram(0.0, 100.0, 10000),
            sharedInputs=True,
        ),
        geometry=region,
        crs=eval_crs,
        scale=reduce_scale,
        maxPixels=100000000,
    )
    med_keys = [f'{b}_median' for b in diff_bands]
    renamed_mads = mad_stats.select(med_keys).rename(
        med_keys,
        [f'{b}_mad' for b in diff_bands],
    )
    hist_dict = encode_sparse_histograms(mad_stats, diff_bands)
    stats = p1_stats.combine(renamed_mads).combine(hist_dict)

    return feature.set(stats)

  def set_coords(f):
    coords = f.geometry().coordinates()
    return f.set({'orig_lon': coords.get(0), 'orig_lat': coords.get(1)})

  def make_region(feat):
    buffer_radius = 750 if lidar_cfg is None else lidar_cfg['buffer_radius']
    square = feat.geometry().buffer(buffer_radius).bounds()
    return feat.setGeometry(square)

  if is_gedi_mode:
    cached_sample_path = (
        f'/tmp/gedi_sample_features_y{FLAGS.gedi_year}_n{FLAGS.max_regions_limit}'
        f'_s{FLAGS.random_seed}_ind{int(FLAGS.exclude_india)}'
        f'_ppt{FLAGS.num_points_per_tile}.json'
    )
    if os.path.exists(cached_sample_path):
      print(
          f'Loading pre-sampled evaluation tiles from {cached_sample_path}...'
      )
      with open(cached_sample_path) as f:
        sample_features = cast(list[dict[str, Any]], json.load(f))
      print(
          f'Successfully loaded {len(sample_features)} pre-sampled candidate'
          ' evaluation tiles.'
      )
    else:
      print(
          'Sampling 1.5km x 1.5km non-overlapping evaluation tiles directly'
          ' from 2025 monthly raster images...'
      )
      assert clean_gedi is not None
      sample_regions_raw = sample_gedi_evaluation_tiles(
          clean_gedi=clean_gedi,
          max_samples=FLAGS.max_regions_limit,
          random_seed=FLAGS.random_seed,
          exclude_india=FLAGS.exclude_india,
          num_points_per_tile=FLAGS.num_points_per_tile,
      )
      print('Materializing candidate evaluation tiles to detach query DAG...')
      raw_sample_info = cast(dict[str, Any], sample_regions_raw.getInfo())
      sample_features = cast(
          list[dict[str, Any]], raw_sample_info.get('features', [])
      )
      print(
          f'Successfully materialized {len(sample_features)} candidate'
          ' evaluation tiles.'
      )
      try:
        with open(cached_sample_path, 'w') as f:
          json.dump(sample_features, f)
      except OSError:
        pass
    sample_regions = ee.FeatureCollection(sample_features)
  elif asset_type == 'IMAGE_COLLECTION' and lidar_col is not None:
    sample_features = None
    print('Sampling evaluation grid over airborne Lidar footprint...')
    assert lidar_cfg is not None
    dsm_band = lidar_cfg['dsm_band']
    sample_col = lidar_col
    num_tiles = sample_col.size()
    sample_col = ee.Algorithms.If(
        num_tiles.gt(500),
        sample_col.randomColumn('rand', FLAGS.random_seed)
        .sort('rand')
        .limit(500),
        sample_col,
    )
    sample_col = ee.ImageCollection(sample_col)
    eff_tiles = sample_col.size()
    pts_per_tile = (
        ee.Number(FLAGS.max_regions_limit).divide(eff_tiles).ceil().max(1)
    )
    if lidar_cfg['pts_per_tile_cap']:
      pts_per_tile = pts_per_tile.min(lidar_cfg['pts_per_tile_cap'])

    def sample_lidar_tile(img):
      return img.select(dsm_band).sample(
          region=img.geometry(),
          scale=15,
          numPixels=pts_per_tile,
          seed=FLAGS.random_seed,
          geometries=True,
      )

    sample_points_raw = sample_col.map(sample_lidar_tile).flatten()
    points_with_coords = sample_points_raw.map(set_coords)
    sample_regions = points_with_coords.map(make_region).limit(
        FLAGS.max_regions_limit
    )
  else:
    sample_features = None
    # For Image assets (like England), sample points across intersecting
    # GEM-15 tiles.
    print('Sampling evaluation grid over airborne Lidar footprint...')
    assert sample_geometry is not None
    tiles_features = gem15_collection.filterBounds(sample_geometry).map(
        lambda img: ee.Feature(img.geometry())
    )
    num_tiles = tiles_features.size()
    pts_per_tile = (
        ee.Number(FLAGS.max_regions_limit).divide(num_tiles).ceil().max(1)
    )

    def sample_gem_tile(tile_feat):
      return ref_dsm_geoid.sample(  # pyrefly: ignore[missing-attribute]
          region=tile_feat.geometry(),
          scale=15,
          numPixels=pts_per_tile,
          seed=FLAGS.random_seed,
          geometries=True,
      )

    sample_points_raw = tiles_features.map(sample_gem_tile).flatten()
    points_with_coords = sample_points_raw.map(set_coords)
    sample_regions = points_with_coords.map(make_region).limit(
        FLAGS.max_regions_limit
    )

  # Drop tiles with < 10 valid unmasked pixels initially:
  # Intersect across all DSM sources (GEM-15, Copernicus, ALOS, NASADEM)
  # and ground-truth reference DSM.
  valid_mask = (
      ref_dsm_geoid.mask()  # pyrefly: ignore[missing-attribute]
      .And(ge_eval_dsm.mask())
      .And(cop_eval_dem.mask())
      .And(alos_eval_dem.mask())
      .And(nasa_eval_dem.mask())
      .selfMask()
      .rename('valid_px')
  )

  def count_valid_px(feat):
    geom = feat.geometry()
    cnt_dict = valid_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geom,
        crs=eval_crs,
        scale=reduce_scale,
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

  # ---------------------------------------------------------
  # 5. Apply Metrics Reducer & Export to Earth Engine Table Assets
  # ---------------------------------------------------------
  if sample_features is None and FLAGS.num_export_chunks > 1:
    print(
        'Materializing candidate evaluation tiles to enable'
        f' {FLAGS.num_export_chunks}-chunk parallel export...'
    )
    try:
      raw_sample_info = cast(dict[str, Any], sample_regions.getInfo())
      sample_features = cast(
          list[dict[str, Any]], raw_sample_info.get('features', [])
      )
      print(
          f'Successfully materialized {len(sample_features)} candidate'
          ' evaluation tiles.'
      )
    except ee.EEException as e:
      print(
          f'Warning: Could not materialize sample regions ({e}); proceeding'
          ' with single export.'
      )
      sample_features = None

  num_chunks = (
      FLAGS.num_export_chunks
      if (sample_features is not None and len(sample_features) > 50)
      else 1
  )

  if num_chunks > 1:
    print(
        f'Chunking evaluation into {num_chunks} parallel Earth Engine export'
        ' tasks to prevent worker memory limits...'
    )
    assert sample_features is not None
    chunk_collections = []
    chunk_size = int(math.ceil(len(sample_features) / num_chunks))
    for k in range(num_chunks):
      chunk_feats = sample_features[k * chunk_size : (k + 1) * chunk_size]
      chunk_regions = ee.FeatureCollection(chunk_feats)
      chunk_valid = chunk_regions.map(count_valid_px).filter(
          ee.Filter.gte('valid_pixel_count', 10)
      )
      chunk_results = chunk_valid.map(get_metrics_for_feature).filter(
          ee.Filter.notNull([
              'ge_diff_mean',
              'ge_diff_variance',
              'cop_diff_mean',
              'cop_diff_variance',
              'alos_diff_mean',
              'alos_diff_variance',
              'nasa_diff_mean',
              'nasa_diff_variance',
          ])
      )
      chunk_collections.append(chunk_results)

    features = export_and_download_chunked_feature_collections(
        collections=chunk_collections,
        base_asset_name=asset_name,
        project=FLAGS.ee_project,
        poll_interval_sec=15 if is_gedi_mode else 10,
    )
  else:
    sample_regions = sample_regions.map(count_valid_px).filter(
        ee.Filter.gte('valid_pixel_count', 10)
    )
    results_collection = sample_regions.map(get_metrics_for_feature)
    valid_results = results_collection.filter(
        ee.Filter.notNull([
            'ge_diff_mean',
            'ge_diff_variance',
            'cop_diff_mean',
            'cop_diff_variance',
            'alos_diff_mean',
            'alos_diff_variance',
            'nasa_diff_mean',
            'nasa_diff_variance',
        ])
    )
    features = export_and_download_feature_collection(
        collection=valid_results,
        asset_name=asset_name,
        project=FLAGS.ee_project,
        poll_interval_sec=15 if is_gedi_mode else 10,
    )

  # Parse features into DataFrame records
  records = []
  for feat in features:
    props = feat['properties']
    lon = props['orig_lon']
    lat = props['orig_lat']

    record = {
        'lon': lon,
        'lat': lat,
        'n_px': props.get('ge_diff_count', np.nan),
        'n_px_ag': props.get('ge_diff_ag_count', np.nan),
        'n_px_g': props.get('ge_diff_g_count', np.nan),
        'n_px_flat': props.get('ge_diff_flat_count', np.nan),
        'n_px_steep': props.get('ge_diff_steep_count', np.nan),
        'slope_cop_mean': props.get('slope_cop_mean', np.nan),
        'slope_fab_mean': props.get('slope_fab_mean', np.nan),
    }
    for src in ['ge', 'cop', 'alos', 'nasa', 'ge_dtm', 'fab']:
      for _, suffix in [
          ('overall', ''),
          ('ag', '_ag'),
          ('g', '_g'),
          ('flat', '_flat'),
          ('steep', '_steep'),
      ]:
        mean_val = props.get(f'{src}_diff{suffix}_mean')
        var_val = props.get(f'{src}_diff{suffix}_variance')
        cnt_val = props.get(f'{src}_diff{suffix}_count')
        mad_val = props.get(f'{src}_diff{suffix}_mad')
        median_val = props.get(f'{src}_diff{suffix}_median')
        if mean_val is not None and var_val is not None:
          direct_rmse = math.sqrt(var_val + mean_val**2)
          tile_deb_rmse = math.sqrt(max(0.0, var_val))
        else:
          mean_val, direct_rmse, tile_deb_rmse = (
              np.nan,
              np.nan,
              np.nan,
          )

        if mad_val is not None:
          tile_nmad = 1.4826 * float(mad_val)
        else:
          tile_nmad = np.nan

        record[f'{src}_bias{suffix}'] = mean_val
        record[f'{src}_median{suffix}'] = median_val
        record[f'{src}_direct_rmse{suffix}'] = direct_rmse
        record[f'{src}_debiased_rmse{suffix}'] = tile_deb_rmse
        record[f'{src}_nmad{suffix}'] = tile_nmad
        record[f'{src}_count{suffix}'] = cnt_val
        record[f'{src}_hist{suffix}'] = props.get(
            f'{src}_diff{suffix}_hist', '[]'
        )
    records.append(record)

  df = pd.DataFrame(records)
  print(f'Successfully compiled {len(df)} raw records.')

  # Retain records where at least DSM or DTM has valid count (no blunder filter)
  valid_mask = (df['ge_count'] > 0) | (df.get('ge_dtm_count', 0) > 0)
  df = df[valid_mask].copy()

  print(f'Retained {len(df)} valid records (no blunder filter applied).')

  # Save CSV output to local folder
  csv_path = os.path.join(FLAGS.output_dir, 'results.csv')
  print(f'Saving compiled CSV results to: {csv_path}...')
  with open(csv_path, 'w') as f:
    df.to_csv(f, index=False)

  # Generate spatial visualization plots and the report
  generate_plots_and_report(
      df, FLAGS.output_dir, ref_name=ref_name, is_lidar=not is_gedi_mode
  )
  print(f'\nAll outputs successfully saved to directory: {FLAGS.output_dir}')


if __name__ == '__main__':
  app.run(main)
