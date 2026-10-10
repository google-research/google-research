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

"""Utility functions for DEM vs GEDI Lidar validation."""

import math
import os
import time
from typing import Any, Sequence, cast
import geopandas as gpd
import google.auth
import google.auth.impersonated_credentials
from matplotlib import patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ee
from gem_15 import ee_assets_config
from gem_15 import metrics


def authenticate_earth_engine(
    ee_service_account = None,
    project = None,
    opt_url = None,
):
  """Authenticates and initializes Google Earth Engine."""
  if not project:
    raise ValueError(
        'Earth Engine Cloud project (--ee_project) must be specified.'
    )
  init_kwargs: dict[str, Any] = {'project': project}
  if opt_url is not None:
    init_kwargs['opt_url'] = opt_url

  if ee_service_account:
    print('Obtaining default developer credentials...')
    creds, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    print(f'Impersonating service account: {ee_service_account}...')
    impersonated_creds = google.auth.impersonated_credentials.Credentials(
        source_credentials=creds,
        target_principal=ee_service_account,
        target_scopes=[
            'https://www.googleapis.com/auth/earthengine',
            'https://www.googleapis.com/auth/cloud-platform',
        ],
        lifetime=3600,
    )
    init_kwargs['credentials'] = impersonated_creds
    print('Initializing Earth Engine...')
    ee.Initialize(**init_kwargs)
    print('Successfully authenticated.')
    return

  print('Initializing Earth Engine with default credentials...')
  try:
    ee.Initialize(**init_kwargs)
  except Exception:  # pylint: disable=broad-except
    creds, _ = google.auth.default(
        scopes=[
            'https://www.googleapis.com/auth/earthengine',
            'https://www.googleapis.com/auth/cloud-platform',
        ]
    )
    init_kwargs['credentials'] = creds
    ee.Initialize(**init_kwargs)
  print('Successfully authenticated.')


def call_ee_with_retries(
    ee_object, max_retries = 3, initial_delay = 15.0
):
  """Calls getInfo() on an Earth Engine object with exponential backoff retries."""
  delay = initial_delay
  for attempt in range(max_retries):
    try:
      return ee_object.getInfo()
    except Exception as e:  # pylint: disable=broad-except
      if attempt == max_retries - 1:
        raise e
      print(
          f'Warning: Earth Engine call failed due to: {e}. Retrying in'
          f' {delay}s (attempt {attempt + 1}/{max_retries})...'
      )
      time.sleep(delay)
      delay *= 2


def get_gem15_col():
  """Loads the GEM-15 ImageCollection with standard elevation and percentile bands."""
  gem15_col = ee.ImageCollection(ee_assets_config.GEM15_ASSET_ID)
  return gem15_col.select([
      'dsm',
      'dsm_std',
      'dsm_percentile_5',
      'dsm_percentile_50',
      'dsm_percentile_95',
      'dtm',
  ])


def get_gem15_layers(resample_bilinear = True):
  """Loads GEM-15 elevation & percentile layers as bilinearly resampled mosaics.

  Resamples individual tiles prior to mosaic to ensure continuous spatial
  interpolation at arbitrary evaluation scales (e.g. 15m or 25m).

  Args:
    resample_bilinear: Whether to apply bilinear resampling across tiles before
      mosaic.

  Returns:
    Dict containing:
      - 'dsm': Resampled 15m DSM mosaic.
      - 'dtm': Resampled 15m DTM mosaic.
      - 'p5': Resampled 15m P5 percentile mosaic.
      - 'p50': Resampled 15m P50 percentile mosaic.
      - 'p95': Resampled 15m P95 percentile mosaic.
      - 'std': Resampled 15m standard deviation dispersion mosaic.
      - 'collection': The underlying ImageCollection.
  """
  col = get_gem15_col()
  if resample_bilinear:
    proc_col = col.map(lambda img: img.resample('bilinear'))
  else:
    proc_col = col

  dsm = proc_col.select('dsm').mosaic()
  dtm = proc_col.select('dtm').mosaic().min(dsm)
  p5 = proc_col.select('dsm_percentile_5').mosaic()
  p50 = proc_col.select('dsm_percentile_50').mosaic()
  p95 = proc_col.select('dsm_percentile_95').mosaic()
  std = proc_col.select('dsm_std').mosaic()

  return {
      'dsm': dsm,
      'dtm': dtm,
      'p5': p5,
      'p50': p50,
      'p95': p95,
      'std': std,
      'collection': col,
  }


def get_global_reference_dems(
    resample_bilinear = True,
):
  """Loads and aligns Copernicus GLO-30, ALOS AW3D30, NASADEM, and FABDEM to EGM96.

  Args:
    resample_bilinear: Whether to apply bilinear resampling on tiles before
      mosaic to support continuous interpolation at arbitrary evaluation scales.

  Returns:
    Dict containing:
      - 'copernicus': Copernicus DEM GLO-30 aligned to EGM96 geoid.
      - 'copernicus_raw': Raw Copernicus DEM (WGS84 EGM2008).
      - 'alos': ALOS AW3D30 DSM.
      - 'nasa': NASADEM elevation.
      - 'fabdem': FABDEM (forest/buildings removed) aligned to EGM96 geoid.
      - 'egm96': EGM96 geoid undulation model.
      - 'egm2008': EGM2008 geoid undulation model.
  """
  cop_col = ee.ImageCollection(ee_assets_config.COPERNICUS_GLO30_ASSET_ID)
  alos_col = ee.ImageCollection(ee_assets_config.ALOS_AW3D30_ASSET_ID)
  nasa_raw = ee.Image(ee_assets_config.NASADEM_ASSET_ID).select('elevation')
  fab_col = ee.ImageCollection(ee_assets_config.FABDEM_ASSET_ID)

  if resample_bilinear:
    copernicus_raw = (
        cop_col.map(lambda img: img.resample('bilinear')).mosaic().select('DEM')
    )
    alos_raw = (
        alos_col.map(lambda img: img.resample('bilinear'))
        .select('DSM')
        .mosaic()
    )
    nasa_raw = nasa_raw.resample('bilinear')
    fabdem_raw = (
        fab_col.map(lambda img: img.resample('bilinear')).mosaic().select('b1')
    )
  else:
    copernicus_raw = cop_col.mosaic().select('DEM')
    alos_raw = alos_col.select('DSM').mosaic()
    fabdem_raw = fab_col.mosaic().select('b1')

  egm96 = ee.Image(ee_assets_config.GEOID_EGM96_ASSET_ID).select('b1')
  egm2008 = ee.Image(ee_assets_config.GEOID_EGM2008_ASSET_ID).select('b1')
  if resample_bilinear:
    egm96 = egm96.resample('bilinear')
    egm2008 = egm2008.resample('bilinear')

  copernicus = copernicus_raw.add(egm2008).subtract(egm96).rename('copernicus')
  fabdem = fabdem_raw.add(egm2008).subtract(egm96).rename('fabdem')

  return {
      'copernicus': copernicus,
      'copernicus_raw': copernicus_raw,
      'alos': alos_raw.rename('alos'),
      'nasa': nasa_raw.rename('nasa'),
      'fabdem': fabdem,
      'egm96': egm96,
      'egm2008': egm2008,
  }


def export_and_download_feature_collection(
    collection,
    asset_name,
    project,
    chunk_size = 2500,
    poll_interval_sec = 15,
):
  """Exports an EE FeatureCollection to an asset, polls status, and downloads records.

  Args:
    collection: Earth Engine FeatureCollection to export.
    asset_name: Name of the asset (without project prefix).
    project: Earth Engine cloud project.
    chunk_size: Number of features to download per chunk (default 2,500).
    poll_interval_sec: Seconds to wait between polling task status.

  Returns:
    List of GeoJSON Feature dictionaries.

  Raises:
    ValueError: If `project` is not provided.
    RuntimeError: If the Earth Engine export task or feature download fails.
  """
  if not project:
    raise ValueError(
        'Earth Engine Cloud project (--ee_project) must be specified.'
    )
  folder_id = f'projects/{project}/assets/temp'
  try:
    ee.data.createFolder(folder_id)
  except Exception:  # pylint: disable=broad-except
    pass

  clean_asset_name = asset_name.replace('temp/', '')
  asset_id = f'{folder_id}/{clean_asset_name}'

  print(f'Checking for pre-existing asset: {asset_id}...')
  try:
    ee.data.deleteAsset(asset_id)
    print(f'Deleted pre-existing asset: {asset_id}')
  except Exception:  # pylint: disable=broad-except
    pass

  print(f'Exporting results to Earth Engine Asset: {asset_id}...')
  task = ee.batch.Export.table.toAsset(
      collection=collection,
      description=clean_asset_name.replace('/', '_'),
      assetId=asset_id,
  )
  task.start()
  task_id = task.id
  print(f'Export task started. Task ID: {task_id}.')

  # Poll status until completion
  print('Polling task status...')
  start_time = time.time()
  while True:
    status = ee.data.getTaskStatus(task_id)[0]
    state = status['state']
    elapsed = time.time() - start_time
    print(f'[{elapsed:.0f}s] Task state: {state}')

    if state == 'COMPLETED':
      print('\nExport task completed successfully!')
      break
    elif state in ['FAILED', 'CANCEL_REQUESTED', 'CANCELLED']:
      err_msg = status.get('error_message', 'No error message provided.')
      raise RuntimeError(
          f'Export task failed or was cancelled. Details: {err_msg}'
      )

    time.sleep(poll_interval_sec)

  # Download FeatureCollection in chunks
  print('\nRetrieving results from exported Asset...')
  try:
    exported_fc = ee.FeatureCollection(asset_id)
    raw_size = call_ee_with_retries(exported_fc.size())
    if raw_size is None:
      raise ValueError('Failed to retrieve size of exported asset.')
    total_size = int(cast(Any, raw_size))
    print(f'Total features in asset: {total_size}')

    features = []
    num_chunks = int(math.ceil(total_size / chunk_size))
    for i in range(num_chunks):
      offset = i * chunk_size
      print(
          f'Downloading chunk {i+1}/{num_chunks} (offset {offset}, limit'
          f' {chunk_size})...'
      )
      chunk_list = call_ee_with_retries(exported_fc.toList(chunk_size, offset))
      if chunk_list is None:
        raise ValueError(f'Failed to retrieve chunk list at offset {offset}.')
      features.extend(cast(list[Any], chunk_list))
    print(f'Success! Retrieved {len(features)} valid records.')
  except Exception as e:
    raise RuntimeError(f'Failed to retrieve asset data: {e}') from e
  finally:
    # Cleanup temporary asset
    print('Cleaning up temporary Earth Engine asset...')
    try:
      ee.data.deleteAsset(asset_id)
      print('Temporary asset deleted successfully.')
    except Exception as e:  # pylint: disable=broad-except
      print('Warning: Failed to delete temporary asset:', e)

  return features


def export_and_download_chunked_feature_collections(
    collections,
    base_asset_name,
    project,
    chunk_size = 2500,
    poll_interval_sec = 15,
    max_retries = 3,
):
  """Exports multiple EE FeatureCollections in parallel, polls status, and downloads combined records.

  Avoids worker OOM on batch export for large collections with heavy
  fixedHistogram reductions. A failed chunk is resubmitted up to
  `max_retries` times; if it still fails the run aborts, so results can never
  silently miss a chunk of evaluation tiles.

  Args:
    collections: List of Earth Engine FeatureCollections to export in parallel.
    base_asset_name: Base asset name (without project prefix).
    project: Earth Engine cloud project.
    chunk_size: Number of features to download per chunk (default 2,500).
    poll_interval_sec: Seconds to wait between polling task status.
    max_retries: Number of times to resubmit a failed chunk before aborting.

  Returns:
    List of GeoJSON Feature dictionaries across all chunks.

  Raises:
    ValueError: If `project` is not provided.
    RuntimeError: If a chunk export task or feature download fails.
  """
  if not project:
    raise ValueError(
        'Earth Engine Cloud project (--ee_project) must be specified.'
    )
  folder_id = f'projects/{project}/assets/temp'
  try:
    ee.data.createFolder(folder_id)
  except Exception:  # pylint: disable=broad-except
    pass

  def _start_export(col, asset_id, clean_name):
    try:
      ee.data.deleteAsset(asset_id)
    except Exception:  # pylint: disable=broad-except
      pass
    task = ee.batch.Export.table.toAsset(
        collection=col,
        description=clean_name.replace('/', '_'),
        assetId=asset_id,
    )
    task.start()
    return task.id

  # task_id -> (asset_id, name, collection, attempt)
  active_tasks = {}
  for i, col in enumerate(collections):
    clean_name = f"{base_asset_name.replace('temp/', '')}_chunk{i+1}"
    asset_id = f'{folder_id}/{clean_name}'
    print(
        f'Exporting chunk {i+1}/{len(collections)} to Earth Engine Asset:'
        f' {asset_id}...'
    )
    task_id = _start_export(col, asset_id, clean_name)
    print(f'Chunk {i+1} export task started. Task ID: {task_id}.')
    active_tasks[task_id] = (asset_id, clean_name, col, 0)

  print(f'Polling status for all {len(active_tasks)} parallel export tasks...')
  start_time = time.time()
  all_features = []
  while active_tasks:
    time.sleep(poll_interval_sec)
    elapsed = time.time() - start_time
    for tid in list(active_tasks):
      status = ee.data.getTaskStatus(tid)[0]
      state = status['state']
      aid, name, col, attempt = active_tasks[tid]
      print(f'[{elapsed:.0f}s] Task {name} ({tid[:8]}) state: {state}')
      if state == 'COMPLETED':
        print(f'Task {name} completed successfully! Downloading features...')
        try:
          exported_fc = ee.FeatureCollection(aid)
          raw_size = call_ee_with_retries(exported_fc.size())
          total_size = int(cast(Any, raw_size)) if raw_size is not None else 0
          print(f'{name} ({aid}): {total_size} features')
          if total_size > 0:
            num_chunks = int(math.ceil(total_size / chunk_size))
            for j in range(num_chunks):
              offset = j * chunk_size
              chunk_list = call_ee_with_retries(
                  exported_fc.toList(chunk_size, offset)
              )
              if chunk_list is not None:
                all_features.extend(cast(list[Any], chunk_list))
        except ee.EEException as e:
          raise RuntimeError(f'Failed to retrieve {name} data: {e}') from e
        finally:
          try:
            ee.data.deleteAsset(aid)
          except ee.EEException:
            pass
        del active_tasks[tid]
      elif state in ['FAILED', 'CANCEL_REQUESTED', 'CANCELLED']:
        err_msg = status.get('error_message', 'No error message provided.')
        del active_tasks[tid]
        if 'Table is empty' in str(err_msg):
          # Every tile in this chunk was filtered out; nothing to download.
          print(f'Task {name}: no valid tiles in this chunk (0 features).')
          continue
        if attempt < max_retries:
          print(
              f'\nWarning: Export task {name} failed or was cancelled:'
              f' {err_msg}. Retrying ({attempt + 1}/{max_retries})...'
          )
          new_tid = _start_export(col, aid, name)
          active_tasks[new_tid] = (aid, name, col, attempt + 1)
        else:
          try:
            ee.data.deleteAsset(aid)
          except ee.EEException:
            pass
          raise RuntimeError(
              f'Export task {name} failed after {max_retries} retries:'
              f' {err_msg}. Aborting so results are not missing evaluation'
              ' tiles.'
          )

  print(
      f'Success! Retrieved total {len(all_features)} valid records across all'
      f' {len(collections)} chunks.'
  )
  return all_features


def get_clean_gedi_collection(
    year = 2025,
    gedi_max_srtm_diff_m = 100.0,
    nighttime_only = True,
    power_beams_only = False,
    min_sensitivity = 0.90,
    no_elevation_bias = True,
):
  """Returns the quality-filtered monthly GEDI collection and its mosaic.

  No terrain-slope pre-filter is applied: masking footprints by slope would
  truncate the evaluation to near-flat terrain and empty out the steep slope
  stratum. Slope is used only to stratify results downstream, never to
  exclude data.

  Args:
    year: Evaluation year for the monthly GEDI raster collection (defaults to
      2025).
    gedi_max_srtm_diff_m: Maximum allowed difference in meters between GEDI
      lowest mode elevation and SRTM DEM elevation (filters cloud/atmospheric
      scattering anomalies). Defaults to 100.0m.
    nighttime_only: Whether to restrict to nighttime observations
      (solar_elevation < 0).
    power_beams_only: Whether to restrict to full-power beams (5, 6, 8, 11).
    min_sensitivity: Minimum sensitivity threshold (defaults to 0.90).
    no_elevation_bias: Whether to filter out potential ranging error
      (elevation_bias_flag == 0).

  Returns:
    A tuple of (clean_gedi_collection, gedi_mosaic).
  """

  def filter_gedi_quality(image):
    quality = image.select('quality_flag').eq(1)
    degrade = image.select('degrade_flag').eq(0)
    sensitivity = image.select('sensitivity').gt(min_sensitivity)
    mask = quality.And(degrade).And(sensitivity)
    if nighttime_only:
      mask = mask.And(image.select('solar_elevation').lt(0))
    if power_beams_only:
      beam = image.select('beam')
      is_power = beam.eq(5).Or(beam.eq(6)).Or(beam.eq(8)).Or(beam.eq(11))
      mask = mask.And(is_power)
    if no_elevation_bias:
      mask = mask.And(image.select('elevation_bias_flag').eq(0))
    if gedi_max_srtm_diff_m > 0:
      valid_srtm = image.select('digital_elevation_model_srtm').gt(-900)
      cloud_mask = (
          image.select('elev_lowestmode')
          .subtract(image.select('digital_elevation_model_srtm'))
          .abs()
          .lte(gedi_max_srtm_diff_m)
          .And(valid_srtm)
      )
      mask = mask.And(cloud_mask)
    return image.updateMask(mask)

  clean_gedi = (
      ee.ImageCollection(ee_assets_config.GEDI_L2A_MONTHLY_ASSET_ID)
      .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
      .map(filter_gedi_quality)
  )
  gedi_mosaic = clean_gedi.mosaic()
  return clean_gedi, gedi_mosaic


def get_gedi_reference_layers(
    gedi_mosaic,
    egm96,
):
  """Returns geoid-aligned GEDI reference DSM (elev_highestreturn canopy top) and DTM (lowest mode).

  The WorldCover land-cover exclusion is deliberately NOT applied here. It is a
  DTM-metric mask and is applied to ref_dtm/ge_dtm/fabdem together at the DTM
  common-mask stage. Returning an unmasked ref_dtm keeps the DSM above-ground
  vs ground stratification (ref_dsm - ref_dtm) free of DTM-only masking.

  Args:
    gedi_mosaic: GEDI quality-filtered mosaic.
    egm96: EGM96 geoid undulation image.

  Returns:
    A tuple of (ref_dsm_geoid, ref_dtm_geoid) aligned to the EGM96 geoid.
  """
  ref_dsm_geoid = (
      gedi_mosaic.select('elev_highestreturn').subtract(egm96)
  ).rename('ref_dsm')
  ref_dtm_geoid = (
      gedi_mosaic.select('elev_lowestmode').subtract(egm96)
  ).rename('ref_dtm')
  return ref_dsm_geoid, ref_dtm_geoid


def get_gedi_waveform_rh_statistics(
    gedi_mosaic,
):
  """Computes mean and standard deviation across all 100 relative height bands (rh1..rh100).

  Args:
    gedi_mosaic: GEDI quality-filtered mosaic containing rh1 through rh100.

  Returns:
    A tuple of (gedi_mean_rh, gedi_std_rh) images.
  """
  rh_all_bands = [f'rh{i}' for i in range(1, 101)]
  rh_img = gedi_mosaic.select(rh_all_bands)
  mean_rh = rh_img.reduce(ee.Reducer.mean()).rename('gedi_mean_rh')
  std_rh = rh_img.reduce(ee.Reducer.stdDev()).rename('gedi_std_rh')
  return mean_rh, std_rh


def encode_sparse_histograms(
    stats,
    band_names,
    bins_per_meter = 100.0,
):
  """Sparse-encodes `ee.Reducer.fixedHistogram` outputs as compact JSON.

  A fixedHistogram reduction emits, for each band `b`, a key `{b}_histogram`
  holding an Nx2 array of [bin_lower_edge_in_meters, count] rows. This packs
  each one into a JSON string of [[bin_index, count], ...] containing only the
  non-empty bins, where `bin_index = round(bin_lower_edge * bins_per_meter)`.

  The bin index convention here MUST match the decoder in
  `metrics.compute_global_nmad_from_hist_series` (default 1 cm bins, i.e.
  `bin_width=0.01`, which corresponds to `bins_per_meter=100`). Bands with no
  populated bins encode as the string '[]'.

  Args:
    stats: Reducer output dictionary containing the `{band}_histogram` keys.
    band_names: Band names to encode.
    bins_per_meter: Reciprocal of the histogram bin width in meters.

  Returns:
    An ee.Dictionary mapping `{band}_hist` to its JSON-encoded sparse
    histogram string.
  """
  # Scales column 0 (bin lower edge, meters) into an integer bin index and
  # leaves column 1 (the count) untouched.
  scale_mat = ee.Array([[bins_per_meter, 0.0], [0.0, 1.0]])

  def _encode(band_name):
    hist_key = f'{band_name}_histogram'
    hist_val = stats.get(hist_key)

    def _from_nonnull(value):
      arr = ee.Array(value)
      pos_mask = arr.slice(1, 1, 2).gt(0)
      has_pos = pos_mask.reduce(ee.Reducer.anyNonZero(), [0]).get([0, 0])
      return ee.Algorithms.If(
          has_pos,
          ee.String.encodeJSON(
              arr.mask(pos_mask)
              .matrixMultiply(scale_mat)
              .round()
              .toInt()
              .toList()
          ),
          ee.String('[]'),
      )

    return ee.Algorithms.If(
        stats.contains(hist_key),
        ee.Algorithms.If(hist_val, _from_nonnull(hist_val), ee.String('[]')),
        ee.String('[]'),
    )

  return ee.Dictionary.fromLists(
      [f'{b}_hist' for b in band_names],
      [_encode(b) for b in band_names],
  )


def apply_common_mask(images):
  """Applies a common intersection mask across all input images.

  Ensures all models, reference surfaces, and error bands are evaluated
  strictly on identical co-located valid pixels.

  Args:
    images: Dictionary of name -> ee.Image.

  Returns:
    Dictionary of name -> ee.Image, each masked with the intersection mask.
  """
  imgs = list(images.values())
  if not imgs:
    return {}
  common_mask = imgs[0].mask()
  for img in imgs[1:]:
    common_mask = common_mask.And(img.mask())
  return {name: img.updateMask(common_mask) for name, img in images.items()}


def sample_gedi_evaluation_tiles(
    clean_gedi,
    max_samples,
    random_seed = 42,
    exclude_india = True,
    num_points_per_tile = None,
    buffer_radius_m = 750,
):
  """Samples non-overlapping 1.5km x 1.5km evaluation regions from GEDI monthly rasters.

  Args:
    clean_gedi: Quality-filtered GEDI monthly collection.
    max_samples: Maximum number of sample regions to produce.
    random_seed: Random seed for deterministic tile selection and sampling.
    exclude_india: Whether to exclude India from sampling.
    num_points_per_tile: Optional cap on points sampled per 1.5km grid tile.
    buffer_radius_m: Radius in meters to buffer point center into a square tile
      (default 750m for 1.5km x 1.5km).

  Returns:
    ee.FeatureCollection of non-overlapping rectangular tile regions with
    orig_lon/orig_lat properties.
  """
  spatial_tiles_col = clean_gedi
  if exclude_india:
    india_fc = ee.FeatureCollection(
        ee_assets_config.LSIB_SIMPLE_ASSET_ID
    ).filter(ee.Filter.eq('country_na', 'India'))
    india_tile_ids = (
        spatial_tiles_col.filter(ee.Filter.bounds(india_fc.geometry()))
        .aggregate_array('system:index')
        .getInfo()
    )
    spatial_tiles_col = spatial_tiles_col.filter(
        ee.Filter.inList('system:index', india_tile_ids).Not()
    )

  unique_spatial_tiles = (
      spatial_tiles_col.randomColumn('rand', random_seed)
      .sort('rand')
      .limit(1000)
  )
  eff_tiles = unique_spatial_tiles.size()
  pts_per_tile = ee.Number(max_samples).divide(eff_tiles).ceil().max(1)
  if num_points_per_tile is not None:
    pts_per_tile = pts_per_tile.min(num_points_per_tile)

  # Stratify sampling roughly evenly across terrestrial biomes using MODIS
  # MCD12Q1 Annual Biome Classification (LC_Type3).
  # Classes: 1=Grasslands, 2=Shrublands, 3=Broadleaf Croplands, 4=Savannas,
  # 5=Evergreen Broadleaf Forests (Rainforests), 6=Deciduous Broadleaf Forests,
  # 7=Evergreen Needleleaf Forests (Taiga/Conifer), 8=Deciduous Needleleaf
  # Forests, 9=Non-Vegetated Lands (Deserts/Barren), 10=Urban/Built-up.
  biome_img = (
      ee.ImageCollection(ee_assets_config.MODIS_MCD12Q1_ASSET_ID)
      .filterDate('2022-01-01', '2023-01-01')
      .first()
      .select('LC_Type3')
      .rename('biome')
  )
  biome_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  num_biomes = len(biome_classes)
  pts_per_biome = ee.Number(pts_per_tile).divide(num_biomes).ceil().max(1)

  def sample_gedi_tile(img):
    img = ee.Image(img)
    tile_img = img.select(['elev_highestreturn']).addBands(biome_img)
    return tile_img.stratifiedSample(
        numPoints=pts_per_biome,
        classBand='biome',
        region=img.geometry(),
        scale=1500,
        seed=random_seed,
        classValues=biome_classes,
        classPoints=[pts_per_biome] * num_biomes,
        geometries=True,
        dropNulls=True,
        tileScale=8,
    )

  def set_coords(f):
    coords = f.geometry().coordinates()
    return f.set({'orig_lon': coords.get(0), 'orig_lat': coords.get(1)})

  def make_region(feat):
    square = feat.geometry().buffer(buffer_radius_m).bounds()
    return feat.setGeometry(square)

  sample_points_raw = unique_spatial_tiles.map(sample_gedi_tile).flatten()
  points_with_coords = sample_points_raw.map(set_coords)
  return points_with_coords.map(make_region).limit(max_samples)


def generate_plots_and_report(
    df,
    output_dir,
    ref_name = 'GEDI',
    is_lidar = False,
):
  """Generates spatial distribution maps and compiles the summary report."""
  plots_dir = os.path.join(output_dir, 'plots')
  os.makedirs(plots_dir, exist_ok=True)

  print('\nLoading world boundaries for plotting...')
  try:
    url = 'https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson'
    world = gpd.read_file(url)
  except Exception as e:  # pylint: disable=broad-except
    print(
        f'Warning: Could not fetch world boundaries online ({e}). Plotting'
        ' points without background map.'
    )
    world = None

  # Create GeoDataFrame
  gdf = gpd.GeoDataFrame(
      df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326'
  )

  # Custom Error Colormap (Green -> Yellow -> Orange -> Red) for RMSE
  error_colors = ['#96ef87', '#daf92e', '#fffe00', '#fbad02', '#f80703']
  rmse_cmap = LinearSegmentedColormap.from_list('mathworks_error', error_colors)
  rmse_norm = plt.Normalize(vmin=0, vmax=15)

  # Diverging Colormap centered at 0m for Bias
  bias_cmap = 'RdBu_r'
  bias_norm = plt.Normalize(vmin=-20, vmax=20)

  # Definitions of the 12 plots to generate
  plots_config = [
      # 1. RMSE Maps
      (
          'ge_direct_rmse',
          rmse_cmap,
          rmse_norm,
          'GEM-15 DSM v1 RMSE (meters)',
          'map_gem15_rmse.png',
          'RMSE (meters)',
      ),
      (
          'cop_direct_rmse',
          rmse_cmap,
          rmse_norm,
          'Copernicus DEM RMSE (meters)',
          'map_copernicus_rmse.png',
          'RMSE (meters)',
      ),
      (
          'alos_direct_rmse',
          rmse_cmap,
          rmse_norm,
          'ALOS DEM RMSE (meters)',
          'map_alos_rmse.png',
          'RMSE (meters)',
      ),
      (
          'nasa_direct_rmse',
          rmse_cmap,
          rmse_norm,
          'NASADEM RMSE (meters)',
          'map_nasa_rmse.png',
          'RMSE (meters)',
      ),
      # 2. Bias Maps
      (
          'ge_bias',
          bias_cmap,
          bias_norm,
          f'GEM-15 DSM v1 Bias against {ref_name} (meters)',
          'map_gem15_bias.png',
          f'Bias (meters) [Blue = DEM < {ref_name} | Red = DEM > {ref_name}]',
      ),
      (
          'cop_bias',
          bias_cmap,
          bias_norm,
          f'Copernicus DEM Bias against {ref_name} (meters)',
          'map_copernicus_bias.png',
          f'Bias (meters) [Blue = DEM < {ref_name} | Red = DEM > {ref_name}]',
      ),
      (
          'alos_bias',
          bias_cmap,
          bias_norm,
          f'ALOS DEM Bias against {ref_name} (meters)',
          'map_alos_bias.png',
          f'Bias (meters) [Blue = DEM < {ref_name} | Red = DEM > {ref_name}]',
      ),
      (
          'nasa_bias',
          bias_cmap,
          bias_norm,
          f'NASADEM Bias against {ref_name} (meters)',
          'map_nasa_bias.png',
          f'Bias (meters) [Blue = DEM < {ref_name} | Red = DEM > {ref_name}]',
      ),
      # 3. Debiased RMSE Maps
      (
          'ge_debiased_rmse',
          rmse_cmap,
          rmse_norm,
          'GEM-15 DSM v1 Debiased RMSE (meters)',
          'map_gem15_debiased_rmse.png',
          'Debiased RMSE (meters)',
      ),
      (
          'cop_debiased_rmse',
          rmse_cmap,
          rmse_norm,
          'Copernicus DEM Debiased RMSE (meters)',
          'map_copernicus_debiased_rmse.png',
          'Debiased RMSE (meters)',
      ),
      (
          'alos_debiased_rmse',
          rmse_cmap,
          rmse_norm,
          'ALOS DEM Debiased RMSE (meters)',
          'map_alos_debiased_rmse.png',
          'Debiased RMSE (meters)',
      ),
      (
          'nasa_debiased_rmse',
          rmse_cmap,
          rmse_norm,
          'NASADEM Debiased RMSE (meters)',
          'map_nasa_debiased_rmse.png',
          'Debiased RMSE (meters)',
      ),
      # 4. DTM RMSE Maps
      (
          'ge_dtm_direct_rmse',
          rmse_cmap,
          rmse_norm,
          'GEM-15 DTM v1 RMSE (meters)',
          'map_gem15_dtm_rmse.png',
          'RMSE (meters)',
      ),
      (
          'fab_direct_rmse',
          rmse_cmap,
          rmse_norm,
          'FABDEM RMSE (meters)',
          'map_fabdem_rmse.png',
          'RMSE (meters)',
      ),
      # 5. DTM Bias Maps
      (
          'ge_dtm_bias',
          bias_cmap,
          bias_norm,
          f'GEM-15 DTM v1 Bias against {ref_name} (meters)',
          'map_gem15_dtm_bias.png',
          f'Bias (meters) [Blue = DTM < {ref_name} | Red = DTM > {ref_name}]',
      ),
      (
          'fab_bias',
          bias_cmap,
          bias_norm,
          f'FABDEM Bias against {ref_name} (meters)',
          'map_fabdem_bias.png',
          f'Bias (meters) [Blue = DTM < {ref_name} | Red = DTM > {ref_name}]',
      ),
      # 6. DTM Debiased RMSE Maps
      (
          'ge_dtm_debiased_rmse',
          rmse_cmap,
          rmse_norm,
          'GEM-15 DTM v1 Debiased RMSE (meters)',
          'map_gem15_dtm_debiased_rmse.png',
          'Debiased RMSE (meters)',
      ),
      (
          'fab_debiased_rmse',
          rmse_cmap,
          rmse_norm,
          'FABDEM Debiased RMSE (meters)',
          'map_fabdem_debiased_rmse.png',
          'Debiased RMSE (meters)',
      ),
  ]

  for col_name, cmap, norm, title, filename, label in plots_config:
    plot_col = col_name
    if plot_col not in gdf.columns:
      continue
    print(f'Generating spatial plot: {title}...')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    if world is not None:
      world.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.5)

    gdf.plot(
        ax=ax,
        column=plot_col,
        cmap=cmap,
        norm=norm,
        markersize=12,
        alpha=0.8,
        legend=False,
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([-180, 180])
    ax.set_ylim([-60, 85])
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # pylint: disable=protected-access
    cbar = fig.colorbar(
        sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.7
    )
    cbar.set_label(label, fontsize=11, fontweight='bold')

    # Save directly to destination (local)
    dest_out_path = os.path.join(plots_dir, filename)
    if not os.path.exists(plots_dir):
      os.makedirs(plots_dir, exist_ok=True)
    with open(dest_out_path, 'wb') as f:
      plt.savefig(f, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to: {dest_out_path}')

  # Calculate Direct RMSE, Tile-Level Debiased RMSE, NMAD, and Bias
  means = metrics.compute_benchmark_metrics_from_df(df)

  # 3.5 Generate Grouped Bar Chart Plot for DSM Metrics
  print('Generating grouped DSM performance comparison bar chart...')
  metrics_labels = [
      'Bias\n(Overall)',
      'DRMSE\n(Overall)',
      'NMAD\n(Overall)',
      'DRMSE\n(Above Ground)',
      'NMAD\n(Above Ground)',
      'DRMSE\n(Bare Earth)',
      'NMAD\n(Bare Earth)',
  ]

  ge_values = [
      means['ge_bias'],
      means['ge_debiased_rmse'],
      means['ge_nmad'],
      means['ge_debiased_rmse_ag'],
      means['ge_nmad_ag'],
      means['ge_debiased_rmse_g'],
      means['ge_nmad_g'],
  ]

  cop_values = [
      means['cop_bias'],
      means['cop_debiased_rmse'],
      means['cop_nmad'],
      means['cop_debiased_rmse_ag'],
      means['cop_nmad_ag'],
      means['cop_debiased_rmse_g'],
      means['cop_nmad_g'],
  ]

  alos_values = [
      means['alos_bias'],
      means['alos_debiased_rmse'],
      means['alos_nmad'],
      means['alos_debiased_rmse_ag'],
      means['alos_nmad_ag'],
      means['alos_debiased_rmse_g'],
      means['alos_nmad_g'],
  ]

  nasa_values = [
      means['nasa_bias'],
      means['nasa_debiased_rmse'],
      means['nasa_nmad'],
      means['nasa_debiased_rmse_ag'],
      means['nasa_nmad_ag'],
      means['nasa_debiased_rmse_g'],
      means['nasa_nmad_g'],
  ]

  x_positions = [float(i) for i in range(len(metrics_labels))]
  bar_width = 0.2

  fig, ax = plt.subplots(figsize=(15, 8), dpi=150)

  rects1 = ax.bar(
      [pos - 1.5 * bar_width for pos in x_positions],
      ge_values,
      bar_width,
      label='GEM-15 DSM v1',
      color='#ff7f0e',
  )
  rects2 = ax.bar(
      [pos - 0.5 * bar_width for pos in x_positions],
      cop_values,
      bar_width,
      label='Copernicus DEM',
      color='#2ca02c',
  )
  rects3 = ax.bar(
      [pos + 0.5 * bar_width for pos in x_positions],
      alos_values,
      bar_width,
      label='ALOS AW3D30',
      color='#1f77b4',
  )
  rects4 = ax.bar(
      [pos + 1.5 * bar_width for pos in x_positions],
      nasa_values,
      bar_width,
      label='NASADEM',
      color='#d62728',
  )

  ax.set_ylabel('Elevation Difference (meters)', fontsize=12, fontweight='bold')
  ax.set_title(
      f'Global DSM Performance Comparison vs {ref_name}',
      fontsize=16,
      fontweight='bold',
      pad=20,
  )
  ax.set_xticks(x_positions)
  ax.set_xticklabels(metrics_labels, fontsize=9, fontweight='bold')
  ax.legend(fontsize=11)
  ax.grid(True, linestyle='--', alpha=0.3)

  for i, label in enumerate(metrics_labels):
    group_rects = [rects1[i], rects2[i], rects3[i], rects4[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    is_bias = 'Mean Error' in label or 'Bias' in label
    if is_bias:
      best_val = min([abs(h) for h in valid_vals]) if valid_vals else None
    else:
      best_val = min(valid_vals) if valid_vals else None

    for rect, height in zip(group_rects, h_vals):
      if np.isnan(height):
        continue
      va = 'bottom' if height >= 0 else 'top'
      xytext = (0, 3) if height >= 0 else (0, -12)
      test_val = abs(height) if is_bias else height
      is_best = best_val is not None and np.isclose(
          test_val, best_val, atol=1e-4
      )
      fw = 'bold' if is_best else 'normal'
      ax.annotate(
          f'{height:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, height),
          xytext=xytext,
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=8,
          fontweight=fw,
      )

  all_vals = ge_values + cop_values + alos_values + nasa_values
  finite_vals = [v for v in all_vals if np.isfinite(v)]
  if finite_vals:
    ymin, ymax = min(finite_vals), max(finite_vals)
    ax.set_ylim(
        [ymin * 1.25 if ymin < 0 else 0, ymax * 1.15 if ymax > 0 else 1.0]
    )

  chart_filename = 'dem_metrics_comparison_barchart.png'
  dest_chart_path = os.path.join(output_dir, chart_filename)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
  with open(dest_chart_path, 'wb') as f:
    plt.savefig(f, bbox_inches='tight')
  plt.close()
  print(f'Saved DSM comparison bar chart to: {dest_chart_path}')

  # 3.6 Generate Grouped Bar Chart Plot for DTM Metrics
  print('Generating grouped DTM performance comparison bar chart...')
  dtm_metrics_labels = [
      'Bias\n(Overall)',
      'DRMSE\n(Overall)',
      'NMAD\n(Overall)',
      'DRMSE\n(Above Ground)',
      'NMAD\n(Above Ground)',
      'DRMSE\n(Bare Earth)',
      'NMAD\n(Bare Earth)',
  ]

  ge_dtm_values = [
      means['ge_dtm_bias'],
      means['ge_dtm_debiased_rmse'],
      means['ge_dtm_nmad'],
      means['ge_dtm_debiased_rmse_ag'],
      means['ge_dtm_nmad_ag'],
      means['ge_dtm_debiased_rmse_g'],
      means['ge_dtm_nmad_g'],
  ]

  fab_values = [
      means['fab_bias'],
      means['fab_debiased_rmse'],
      means['fab_nmad'],
      means['fab_debiased_rmse_ag'],
      means['fab_nmad_ag'],
      means['fab_debiased_rmse_g'],
      means['fab_nmad_g'],
  ]

  dtm_x_positions = [float(i) for i in range(len(dtm_metrics_labels))]
  dtm_bar_width = 0.3

  fig, ax = plt.subplots(figsize=(13, 7), dpi=150)

  rects_ge_dtm = ax.bar(
      [pos - 0.5 * dtm_bar_width for pos in dtm_x_positions],
      ge_dtm_values,
      dtm_bar_width,
      label='GEM-15 DTM v1',
      color='#ff7f0e',
  )
  rects_fab = ax.bar(
      [pos + 0.5 * dtm_bar_width for pos in dtm_x_positions],
      fab_values,
      dtm_bar_width,
      label='FABDEM',
      color='#2ca02c',
  )

  ax.set_ylabel('Elevation Difference (meters)', fontsize=12, fontweight='bold')
  ax.set_title(
      f'Global DTM Performance Comparison vs {ref_name}',
      fontsize=16,
      fontweight='bold',
      pad=20,
  )
  ax.set_xticks(dtm_x_positions)
  ax.set_xticklabels(dtm_metrics_labels, fontsize=9, fontweight='bold')
  ax.legend(fontsize=11)
  ax.grid(True, linestyle='--', alpha=0.3)

  for i, label in enumerate(dtm_metrics_labels):
    group_rects = [rects_ge_dtm[i], rects_fab[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    is_bias = 'Bias' in label
    if is_bias:
      best_val = min([abs(h) for h in valid_vals]) if valid_vals else None
    else:
      best_val = min(valid_vals) if valid_vals else None

    for rect, height in zip(group_rects, h_vals):
      if np.isnan(height):
        continue
      va = 'bottom' if height >= 0 else 'top'
      xytext = (0, 3) if height >= 0 else (0, -12)
      test_val = abs(height) if is_bias else height
      is_best = best_val is not None and np.isclose(
          test_val, best_val, atol=1e-4
      )
      fw = 'bold' if is_best else 'normal'
      ax.annotate(
          f'{height:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, height),
          xytext=xytext,
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=8,
          fontweight=fw,
      )

  all_dtm_vals = ge_dtm_values + fab_values
  finite_dtm_vals = [v for v in all_dtm_vals if np.isfinite(v)]
  if finite_dtm_vals:
    ymin, ymax = min(finite_dtm_vals), max(finite_dtm_vals)
    ax.set_ylim(
        [ymin * 1.25 if ymin < 0 else 0, ymax * 1.15 if ymax > 0 else 1.0]
    )

  dtm_chart_filename = 'dtm_metrics_comparison_barchart.png'
  dest_dtm_chart_path = os.path.join(output_dir, dtm_chart_filename)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
  with open(dest_dtm_chart_path, 'wb') as f:
    plt.savefig(f, bbox_inches='tight')
  plt.close()
  print(f'Saved DTM comparison bar chart to: {dest_dtm_chart_path}')

  # 3.7 Generate Grouped Bar Chart Plot for Slope-Stratified DSM Metrics
  print('Generating grouped slope-stratified DSM bar chart...')
  dsm_slope_labels = [
      'Bias\n(Flat < 10°)',
      'DRMSE\n(Flat < 10°)',
      'NMAD\n(Flat < 10°)',
      'Bias\n(Steep >= 10°)',
      'DRMSE\n(Steep >= 10°)',
      'NMAD\n(Steep >= 10°)',
  ]
  ge_slope_vals = [
      means.get('ge_bias_flat', 0.0),
      means.get('ge_debiased_rmse_flat', 0.0),
      means.get('ge_nmad_flat', 0.0),
      means.get('ge_bias_steep', 0.0),
      means.get('ge_debiased_rmse_steep', 0.0),
      means.get('ge_nmad_steep', 0.0),
  ]
  cop_slope_vals = [
      means.get('cop_bias_flat', 0.0),
      means.get('cop_debiased_rmse_flat', 0.0),
      means.get('cop_nmad_flat', 0.0),
      means.get('cop_bias_steep', 0.0),
      means.get('cop_debiased_rmse_steep', 0.0),
      means.get('cop_nmad_steep', 0.0),
  ]
  alos_slope_vals = [
      means.get('alos_bias_flat', 0.0),
      means.get('alos_debiased_rmse_flat', 0.0),
      means.get('alos_nmad_flat', 0.0),
      means.get('alos_bias_steep', 0.0),
      means.get('alos_debiased_rmse_steep', 0.0),
      means.get('alos_nmad_steep', 0.0),
  ]
  nasa_slope_vals = [
      means.get('nasa_bias_flat', 0.0),
      means.get('nasa_debiased_rmse_flat', 0.0),
      means.get('nasa_nmad_flat', 0.0),
      means.get('nasa_bias_steep', 0.0),
      means.get('nasa_debiased_rmse_steep', 0.0),
      means.get('nasa_nmad_steep', 0.0),
  ]

  slope_x_positions = [float(i) for i in range(len(dsm_slope_labels))]
  bar_width = 0.2
  fig, ax = plt.subplots(figsize=(15, 8), dpi=150)
  rects_ge = ax.bar(
      [pos - 1.5 * bar_width for pos in slope_x_positions],
      ge_slope_vals,
      bar_width,
      label='GEM-15 DSM v1 (15m)',
      color='#ff7f0e',
  )
  rects_cop = ax.bar(
      [pos - 0.5 * bar_width for pos in slope_x_positions],
      cop_slope_vals,
      bar_width,
      label='Copernicus DEM',
      color='#2ca02c',
  )
  rects_alos = ax.bar(
      [pos + 0.5 * bar_width for pos in slope_x_positions],
      alos_slope_vals,
      bar_width,
      label='ALOS AW3D30',
      color='#1f77b4',
  )
  rects_nasa = ax.bar(
      [pos + 1.5 * bar_width for pos in slope_x_positions],
      nasa_slope_vals,
      bar_width,
      label='NASADEM',
      color='#d62728',
  )
  ax.set_ylabel('Elevation Difference (meters)', fontsize=12, fontweight='bold')
  ax.set_title(
      f'Slope-Stratified DSM Comparison vs {ref_name} (Flat < 10° vs Steep >='
      ' 10°)',
      fontsize=16,
      fontweight='bold',
      pad=20,
  )
  ax.set_xticks(slope_x_positions)
  ax.set_xticklabels(dsm_slope_labels, fontsize=9, fontweight='bold')
  ax.legend(fontsize=11)
  ax.grid(True, linestyle='--', alpha=0.3)

  for i, label in enumerate(dsm_slope_labels):
    group_rects = [rects_ge[i], rects_cop[i], rects_alos[i], rects_nasa[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    is_bias = 'Bias' in label
    if is_bias:
      best_val = min([abs(h) for h in valid_vals]) if valid_vals else None
    else:
      best_val = min(valid_vals) if valid_vals else None
    for rect, height in zip(group_rects, h_vals):
      if np.isnan(height):
        continue
      va = 'bottom' if height >= 0 else 'top'
      xytext = (0, 3) if height >= 0 else (0, -12)
      test_val = abs(height) if is_bias else height
      is_best = best_val is not None and np.isclose(
          test_val, best_val, atol=1e-4
      )
      fw = 'bold' if is_best else 'normal'
      ax.annotate(
          f'{height:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, height),
          xytext=xytext,
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=8,
          fontweight=fw,
      )
  all_slope_dsm_vals = (
      ge_slope_vals + cop_slope_vals + alos_slope_vals + nasa_slope_vals
  )
  finite_slope_dsm_vals = [v for v in all_slope_dsm_vals if np.isfinite(v)]
  if finite_slope_dsm_vals:
    ymin, ymax = min(finite_slope_dsm_vals), max(finite_slope_dsm_vals)
    ax.set_ylim(
        [ymin * 1.25 if ymin < 0 else 0, ymax * 1.15 if ymax > 0 else 1.0]
    )
  dsm_slope_chart_path = os.path.join(
      output_dir, 'dem_slope_stratified_barchart.png'
  )
  with open(dsm_slope_chart_path, 'wb') as f:
    plt.savefig(f, bbox_inches='tight')
  plt.close()
  print(f'Saved slope-stratified DSM bar chart to: {dsm_slope_chart_path}')

  # 3.8 Generate Grouped Bar Chart Plot for Slope-Stratified DTM Metrics
  print('Generating grouped slope-stratified DTM bar chart...')
  dtm_slope_labels = [
      'Bias\n(Flat < 10°)',
      'DRMSE\n(Flat < 10°)',
      'NMAD\n(Flat < 10°)',
      'Bias\n(Steep >= 10°)',
      'DRMSE\n(Steep >= 10°)',
      'NMAD\n(Steep >= 10°)',
  ]
  ge_dtm_slope_vals = [
      means.get('ge_dtm_bias_flat', 0.0),
      means.get('ge_dtm_debiased_rmse_flat', 0.0),
      means.get('ge_dtm_nmad_flat', 0.0),
      means.get('ge_dtm_bias_steep', 0.0),
      means.get('ge_dtm_debiased_rmse_steep', 0.0),
      means.get('ge_dtm_nmad_steep', 0.0),
  ]
  fab_slope_vals = [
      means.get('fab_bias_flat', 0.0),
      means.get('fab_debiased_rmse_flat', 0.0),
      means.get('fab_nmad_flat', 0.0),
      means.get('fab_bias_steep', 0.0),
      means.get('fab_debiased_rmse_steep', 0.0),
      means.get('fab_nmad_steep', 0.0),
  ]
  dtm_slope_x_positions = [float(i) for i in range(len(dtm_slope_labels))]
  dtm_bar_width = 0.3
  fig, ax = plt.subplots(figsize=(13, 7), dpi=150)
  rects_ge_dtm = ax.bar(
      [pos - 0.5 * dtm_bar_width for pos in dtm_slope_x_positions],
      ge_dtm_slope_vals,
      dtm_bar_width,
      label='GEM-15 DTM v1 (15m)',
      color='#ff7f0e',
  )
  rects_fab = ax.bar(
      [pos + 0.5 * dtm_bar_width for pos in dtm_slope_x_positions],
      fab_slope_vals,
      dtm_bar_width,
      label='FABDEM',
      color='#2ca02c',
  )
  ax.set_ylabel('Elevation Difference (meters)', fontsize=12, fontweight='bold')
  ax.set_title(
      f'Slope-Stratified DTM Comparison vs {ref_name} (Flat < 10° vs Steep >='
      ' 10°)',
      fontsize=16,
      fontweight='bold',
      pad=20,
  )
  ax.set_xticks(dtm_slope_x_positions)
  ax.set_xticklabels(dtm_slope_labels, fontsize=9, fontweight='bold')
  ax.legend(fontsize=11)
  ax.grid(True, linestyle='--', alpha=0.3)

  for i, label in enumerate(dtm_slope_labels):
    group_rects = [rects_ge_dtm[i], rects_fab[i]]
    h_vals = [r.get_height() for r in group_rects]
    valid_vals = [h for h in h_vals if not np.isnan(h)]
    is_bias = 'Bias' in label
    if is_bias:
      best_val = min([abs(h) for h in valid_vals]) if valid_vals else None
    else:
      best_val = min(valid_vals) if valid_vals else None
    for rect, height in zip(group_rects, h_vals):
      if np.isnan(height):
        continue
      va = 'bottom' if height >= 0 else 'top'
      xytext = (0, 3) if height >= 0 else (0, -12)
      test_val = abs(height) if is_bias else height
      is_best = best_val is not None and np.isclose(
          test_val, best_val, atol=1e-4
      )
      fw = 'bold' if is_best else 'normal'
      ax.annotate(
          f'{height:.2f}',
          xy=(rect.get_x() + rect.get_width() / 2, height),
          xytext=xytext,
          textcoords='offset points',
          ha='center',
          va=va,
          fontsize=8,
          fontweight=fw,
      )
  all_slope_dtm_vals = ge_dtm_slope_vals + fab_slope_vals
  finite_slope_dtm_vals = [v for v in all_slope_dtm_vals if np.isfinite(v)]
  if finite_slope_dtm_vals:
    ymin, ymax = min(finite_slope_dtm_vals), max(finite_slope_dtm_vals)
    ax.set_ylim(
        [ymin * 1.25 if ymin < 0 else 0, ymax * 1.15 if ymax > 0 else 1.0]
    )
  dtm_slope_chart_path = os.path.join(
      output_dir, 'dtm_slope_stratified_barchart.png'
  )
  with open(dtm_slope_chart_path, 'wb') as f:
    plt.savefig(f, bbox_inches='tight')
  plt.close()
  print(f'Saved slope-stratified DTM bar chart to: {dtm_slope_chart_path}')

  # 4. Generate report.md
  print('\nGenerating summary report (report.md)...')

  best_overall = (
      'GEM-15'
      if means['ge_direct_rmse'] < means['cop_direct_rmse']
      else 'Copernicus'
  )
  best_ag = (
      'GEM-15'
      if means['ge_direct_rmse_ag'] < means['cop_direct_rmse_ag']
      else 'Copernicus'
  )
  best_g = (
      'GEM-15'
      if means['ge_direct_rmse_g'] < means['cop_direct_rmse_g']
      else 'Copernicus'
  )
  best_deb_overall = (
      'GEM-15'
      if means['ge_debiased_rmse'] < means['cop_debiased_rmse']
      else 'Copernicus'
  )
  best_deb_ag = (
      'GEM-15'
      if means['ge_debiased_rmse_ag'] < means['cop_debiased_rmse_ag']
      else 'Copernicus'
  )
  best_deb_g = (
      'GEM-15'
      if means['ge_debiased_rmse_g'] < means['cop_debiased_rmse_g']
      else 'Copernicus'
  )
  findings_text = (
      f'1. **Overall**: Lowest RMSE is **{best_overall}** (GEM-15:'
      f' {means["ge_direct_rmse"]:.2f}m vs Copernicus:'
      f' {means["cop_direct_rmse"]:.2f}m) | Lowest Debiased RMSE is'
      f' **{best_deb_overall}** (GEM-15: {means["ge_debiased_rmse"]:.2f}m vs'
      f' Copernicus: {means["cop_debiased_rmse"]:.2f}m).\n2. **Above Ground'
      f' (Canopy/Buildings, DSM - DTM > 5m)**: Lowest RMSE is **{best_ag}**'
      f' (GEM-15: {means["ge_direct_rmse_ag"]:.2f}m vs Copernicus:'
      f' {means["cop_direct_rmse_ag"]:.2f}m) | Lowest Debiased RMSE is'
      f' **{best_deb_ag}** (GEM-15: {means["ge_debiased_rmse_ag"]:.2f}m vs'
      f' Copernicus: {means["cop_debiased_rmse_ag"]:.2f}m).\n3. **Ground (Bare'
      f' earth, DSM - DTM <= 5m)**: Lowest RMSE is **{best_g}** (GEM-15:'
      f' {means["ge_direct_rmse_g"]:.2f}m vs Copernicus:'
      f' {means["cop_direct_rmse_g"]:.2f}m) | Lowest Debiased RMSE is'
      f' **{best_deb_g}** (GEM-15: {means["ge_debiased_rmse_g"]:.2f}m vs'
      f' Copernicus: {means["cop_debiased_rmse_g"]:.2f}m).'
  )

  if is_lidar:
    report_title = f'# DEM & DTM vs {ref_name} Validation Report'
    intro_text = (
        'This report presents a validation and comparison of the GEM-15'
        f' DSM v1 (15m) and public global DEMs against reference {ref_name}'
        ' Lidar DSM data using tile-level debiased RMSE (1.5km x 1.5km tiles).'
    )
    datum_text = (
        f'To address vertical datum mismatches, the reference {ref_name} Lidar'
        ' values and Copernicus DEM are aligned to the EGM96 geoid using local'
        ' geoid offsets (e.g., OSGM15, CGVD2013, REDNAP) and EGM2008 offsets'
        ' respectively. GEM-15, ALOS, and NASADEM are kept in their'
        ' native EGM96 geoid frames.'
    )
    canopy_text = (
        'Since the reference Lidar DSM represents the top response of the'
        ' canopy and detailed structures, it provides a direct surface'
        ' elevation reference for validating global DSM products.'
    )
  else:
    report_title = '# DEM & DTM vs GEDI Global Validation Report (2025 Monthly)'
    intro_text = (
        'This report presents a validation and comparison of the GEM-15'
        ' DSM v1 (15m resampled to 25m) and public global DEMs against'
        ' spaceborne GEDI Lidar footprints (2025 Monthly Collection) using'
        ' tile-level debiased RMSE (1.5km x 1.5km tiles).'
    )
    datum_text = (
        'To address vertical datum mismatches, the GEDI canopy top'
        ' elevation (elev_highestreturn, native to the WGS84 ellipsoid) is'
        ' aligned to the geoid by subtracting the local EGM96 geoid'
        ' undulations. GEM-15, Copernicus, ALOS, and NASADEM are kept in'
        ' their native geoid frames.'
    )
    canopy_text = (
        'Since GEDI highest returns represent the top of the canopy'
        ' physical surface, a systematic negative bias exists across all DEMs'
        ' in Above Ground areas.'
    )

  table_overall_header = '| DEM Source | Bias (m) | DRMSE (m) | NMAD (m) |'
  table_overall_rows = [
      (
          f"| **GEM-15 DSM v1** | **{means['ge_bias']:.2f}** |"
          f" **{means['ge_debiased_rmse']:.2f}** |"
          f" **{means['ge_nmad']:.2f}** |"
      ),
      (
          f"| Copernicus DEM | {means['cop_bias']:.2f} |"
          f" {means['cop_debiased_rmse']:.2f} | {means['cop_nmad']:.2f} |"
      ),
      (
          f"| ALOS AW3D30 | {means['alos_bias']:.2f} |"
          f" {means['alos_debiased_rmse']:.2f} | {means['alos_nmad']:.2f} |"
      ),
      (
          f"| NASADEM | {means['nasa_bias']:.2f} |"
          f" {means['nasa_debiased_rmse']:.2f} | {means['nasa_nmad']:.2f} |"
      ),
  ]
  table_ag_header = '| DEM Source | Bias (m) | DRMSE (m) | NMAD (m) |'
  table_ag_rows = [
      (
          f"| **GEM-15 DSM v1** | **{means['ge_bias_ag']:.2f}** |"
          f" **{means['ge_debiased_rmse_ag']:.2f}** |"
          f" **{means['ge_nmad_ag']:.2f}** |"
      ),
      (
          f"| Copernicus DEM | {means['cop_bias_ag']:.2f} |"
          f" {means['cop_debiased_rmse_ag']:.2f} | {means['cop_nmad_ag']:.2f} |"
      ),
      (
          f"| ALOS AW3D30 | {means['alos_bias_ag']:.2f} |"
          f" {means['alos_debiased_rmse_ag']:.2f} |"
          f" {means['alos_nmad_ag']:.2f} |"
      ),
      (
          f"| NASADEM | {means['nasa_bias_ag']:.2f} |"
          f" {means['nasa_debiased_rmse_ag']:.2f} |"
          f" {means['nasa_nmad_ag']:.2f} |"
      ),
  ]
  table_g_header = '| DEM Source | Bias (m) | DRMSE (m) | NMAD (m) |'
  table_g_rows = [
      (
          f"| **GEM-15 DSM v1** | **{means['ge_bias_g']:.2f}** |"
          f" **{means['ge_debiased_rmse_g']:.2f}** |"
          f" **{means['ge_nmad_g']:.2f}** |"
      ),
      (
          f"| Copernicus DEM | {means['cop_bias_g']:.2f} |"
          f" {means['cop_debiased_rmse_g']:.2f} | {means['cop_nmad_g']:.2f} |"
      ),
      (
          f"| ALOS AW3D30 | {means['alos_bias_g']:.2f} |"
          f" {means['alos_debiased_rmse_g']:.2f} |"
          f" {means['alos_nmad_g']:.2f} |"
      ),
      (
          f"| NASADEM | {means['nasa_bias_g']:.2f} |"
          f" {means['nasa_debiased_rmse_g']:.2f} |"
          f" {means['nasa_nmad_g']:.2f} |"
      ),
  ]

  report_lines = [
      report_title,
      '',
      intro_text,
      '',
      '## Grouped Performance Charts',
      '### DSM Comparison (GEM-15 DSM v1 vs Global DEMs)',
      '![Grouped DSM Performance Chart](dem_metrics_comparison_barchart.png)',
      '',
      '### DTM Comparison (GEM-15 DTM v1 vs FABDEM)',
      '![Grouped DTM Performance Chart](dtm_metrics_comparison_barchart.png)',
      '',
      '## 1. Methodology',
      f'- **Datum Alignment**: {datum_text}',
      f'- **Canopy Bias Note**: {canopy_text}',
      (
          '- **Evaluation footprint**: The validation was performed across'
          f' {len(df)} sampled regions.'
      ),
      '',
      '## 2. DSM Validation Statistics (By Terrain Subsets)',
      '',
      '### 2.1 Overall DSM Statistics',
      '',
      table_overall_header,
      '| :--- | :---: | :---: | :---: |',
  ]
  report_lines.extend(table_overall_rows)
  report_lines.extend([
      '',
      (
          '### 2.2 Above Ground DSM Statistics (Canopy / Buildings, DSM -'
          ' DTM > 5m)'
      ),
      '',
      table_ag_header,
      '| :--- | :---: | :---: | :---: |',
  ])
  report_lines.extend(table_ag_rows)
  report_lines.extend([
      '',
      '### 2.3 Ground DSM Statistics (Bare Earth, DSM - DTM <= 5m)',
      '',
      table_g_header,
      '| :--- | :---: | :---: | :---: |',
  ])
  report_lines.extend(table_g_rows)
  report_lines.extend([
      '',
      '### DSM Key Findings:',
      findings_text,
  ])

  best_dtm_overall = (
      'GEM-15 DTM'
      if means['ge_dtm_direct_rmse'] < means['fab_direct_rmse']
      else 'FABDEM'
  )
  best_dtm_ag = (
      'GEM-15 DTM'
      if means['ge_dtm_direct_rmse_ag'] < means['fab_direct_rmse_ag']
      else 'FABDEM'
  )
  best_dtm_g = (
      'GEM-15 DTM'
      if means['ge_dtm_direct_rmse_g'] < means['fab_direct_rmse_g']
      else 'FABDEM'
  )
  best_dtm_deb_overall = (
      'GEM-15 DTM'
      if means['ge_dtm_debiased_rmse'] < means['fab_debiased_rmse']
      else 'FABDEM'
  )
  best_dtm_deb_ag = (
      'GEM-15 DTM'
      if means['ge_dtm_debiased_rmse_ag'] < means['fab_debiased_rmse_ag']
      else 'FABDEM'
  )
  best_dtm_deb_g = (
      'GEM-15 DTM'
      if means['ge_dtm_debiased_rmse_g'] < means['fab_debiased_rmse_g']
      else 'FABDEM'
  )
  dtm_findings_text = (
      f'1. **Overall DTM**: Lowest RMSE is **{best_dtm_overall}**'
      f' (GEM-15 DTM: {means["ge_dtm_direct_rmse"]:.2f}m vs FABDEM:'
      f' {means["fab_direct_rmse"]:.2f}m) | Lowest Debiased RMSE is'
      f' **{best_dtm_deb_overall}** (GEM-15 DTM:'
      f' {means["ge_dtm_debiased_rmse"]:.2f}m vs FABDEM:'
      f' {means["fab_debiased_rmse"]:.2f}m).\n'
      '2. **Above Ground Canopy / Structures (DSM - DTM > 5m)**: Lowest'
      f' RMSE is **{best_dtm_ag}** (GEM-15 DTM:'
      f' {means["ge_dtm_direct_rmse_ag"]:.2f}m vs FABDEM:'
      f' {means["fab_direct_rmse_ag"]:.2f}m) | Lowest Debiased RMSE is'
      f' **{best_dtm_deb_ag}** (GEM-15 DTM:'
      f' {means["ge_dtm_debiased_rmse_ag"]:.2f}m vs FABDEM:'
      f' {means["fab_debiased_rmse_ag"]:.2f}m).\n'
      '3. **Bare Earth Ground (DSM - DTM <= 5m)**: Lowest RMSE is'
      f' **{best_dtm_g}** (GEM-15 DTM:'
      f' {means["ge_dtm_direct_rmse_g"]:.2f}m vs FABDEM:'
      f' {means["fab_direct_rmse_g"]:.2f}m) | Lowest Debiased RMSE is'
      f' **{best_dtm_deb_g}** (GEM-15 DTM:'
      f' {means["ge_dtm_debiased_rmse_g"]:.2f}m vs FABDEM:'
      f' {means["fab_debiased_rmse_g"]:.2f}m).'
  )

  table_dtm_overall_rows = [
      (
          f"| **GEM-15 DTM v1** | **{means['ge_dtm_bias']:.2f}** |"
          f" **{means['ge_dtm_debiased_rmse']:.2f}** |"
          f" **{means['ge_dtm_nmad']:.2f}** |"
      ),
      (
          f"| FABDEM | {means['fab_bias']:.2f} |"
          f" {means['fab_debiased_rmse']:.2f} | {means['fab_nmad']:.2f} |"
      ),
  ]
  table_dtm_ag_rows = [
      (
          f"| **GEM-15 DTM v1** | **{means['ge_dtm_bias_ag']:.2f}** |"
          f" **{means['ge_dtm_debiased_rmse_ag']:.2f}** |"
          f" **{means['ge_dtm_nmad_ag']:.2f}** |"
      ),
      (
          f"| FABDEM | {means['fab_bias_ag']:.2f} |"
          f" {means['fab_debiased_rmse_ag']:.2f} | {means['fab_nmad_ag']:.2f} |"
      ),
  ]
  table_dtm_g_rows = [
      (
          f"| **GEM-15 DTM v1** | **{means['ge_dtm_bias_g']:.2f}** |"
          f" **{means['ge_dtm_debiased_rmse_g']:.2f}** |"
          f" **{means['ge_dtm_nmad_g']:.2f}** |"
      ),
      (
          f"| FABDEM | {means['fab_bias_g']:.2f} |"
          f" {means['fab_debiased_rmse_g']:.2f} | {means['fab_nmad_g']:.2f} |"
      ),
  ]

  report_lines.extend([
      '',
      '---',
      '',
      f'## 3. DTM Validation Statistics vs {ref_name} Reference DTM',
      '',
      (
          'This section evaluates bare-earth digital terrain models'
          ' (GEM-15 DTM v1 and FABDEM) directly against reference'
          f' bare-earth DTM data from {ref_name}.'
      ),
      '',
      '### 3.1 Overall DTM Statistics',
      '',
      '| DTM Source | Bias (m) | DRMSE (m) | NMAD (m) |',
      '| :--- | :---: | :---: | :---: |',
  ])
  report_lines.extend(table_dtm_overall_rows)
  report_lines.extend([
      '',
      (
          '### 3.2 Above Ground DTM Statistics (Canopy / Buildings, DSM -'
          ' DTM > 5m)'
      ),
      '',
      '| DTM Source | Bias (m) | DRMSE (m) | NMAD (m) |',
      '| :--- | :---: | :---: | :---: |',
  ])
  report_lines.extend(table_dtm_ag_rows)
  report_lines.extend([
      '',
      '### 3.3 Ground DTM Statistics (Bare Earth, DSM - DTM <= 5m)',
      '',
      '| DTM Source | Bias (m) | DRMSE (m) | NMAD (m) |',
      '| :--- | :---: | :---: | :---: |',
  ])
  report_lines.extend(table_dtm_g_rows)
  report_lines.extend([
      '',
      '### DTM Key Findings:',
      dtm_findings_text,
      '',
      '---',
      '',
      '## 4. Spatial Analysis Maps',
      (
          'The following maps show the spatial distribution of metrics (RMSE,'
          ' Bias, and Debiased RMSE):'
      ),
      '',
      '### DSM RMSE Maps (0m to 15m)',
      '- [GEM-15 DSM RMSE Map](plots/map_gem15_rmse.png)',
      '- [Copernicus DEM RMSE Map](plots/map_copernicus_rmse.png)',
      '- [ALOS AW3D30 RMSE Map](plots/map_alos_rmse.png)',
      '- [NASADEM RMSE Map](plots/map_nasa_rmse.png)',
      '',
      '### DSM Bias Maps (-20m to 20m)',
      '- [GEM-15 DSM Bias Map](plots/map_gem15_bias.png)',
      '- [Copernicus DEM Bias Map](plots/map_copernicus_bias.png)',
      '- [ALOS AW3D30 Bias Map](plots/map_alos_bias.png)',
      '- [NASADEM Bias Map](plots/map_nasa_bias.png)',
      '',
      '### DSM Debiased RMSE Maps (0m to 15m)',
      (
          '- [GEM-15 DSM Debiased RMSE'
          ' Map](plots/map_gem15_debiased_rmse.png)'
      ),
      (
          '- [Copernicus DEM Debiased RMSE'
          ' Map](plots/map_copernicus_debiased_rmse.png)'
      ),
      '- [ALOS AW3D30 Debiased RMSE Map](plots/map_alos_debiased_rmse.png)',
      '- [NASADEM Debiased RMSE Map](plots/map_nasa_debiased_rmse.png)',
      '',
      '### DTM RMSE Maps (0m to 15m)',
      '- [GEM-15 DTM RMSE Map](plots/map_gem15_dtm_rmse.png)',
      '- [FABDEM RMSE Map](plots/map_fabdem_rmse.png)',
      '',
      '### DTM Bias Maps (-20m to 20m)',
      '- [GEM-15 DTM Bias Map](plots/map_gem15_dtm_bias.png)',
      '- [FABDEM Bias Map](plots/map_fabdem_bias.png)',
      '',
      '### DTM Debiased RMSE Maps (0m to 15m)',
      (
          '- [GEM-15 DTM Debiased RMSE'
          ' Map](plots/map_gem15_dtm_debiased_rmse.png)'
      ),
      '- [FABDEM Debiased RMSE Map](plots/map_fabdem_debiased_rmse.png)',
  ])

  report_content = '\n'.join(report_lines) + '\n'

  report_path = os.path.join(output_dir, 'report.md')
  with open(report_path, 'w') as f:
    f.write(report_content)
  print(f'Saved summary report to: {report_path}')

  # Save summary_metrics.csv
  summary_record: dict[str, Any] = {
      'ref_name': ref_name,
      'is_lidar': is_lidar,
      'num_samples': len(df),
  }
  summary_record.update(means)
  summary_df = pd.DataFrame([summary_record])
  summary_csv_path = os.path.join(output_dir, 'summary_metrics.csv')
  with open(summary_csv_path, 'w') as f:
    summary_df.to_csv(f, index=False)
  print(f'Saved summary metrics to: {summary_csv_path}')


def get_utm_crs(lon, lat):
  """Returns the EPSG code for the local UTM zone at (lon, lat)."""
  zone = int(np.floor((float(lon) + 180.0) / 6.0)) + 1
  zone = max(1, min(60, zone))
  prefix = 32600 if float(lat) >= 0.0 else 32700
  return f'EPSG:{prefix + zone}'


def add_rgb_scalebar(
    ax,
    width_m,
    fontsize = 9.2,
    bar_m = None,
    loc = 'bottom_right',
):
  """Draws a Google Maps / Earth Engine style scalebar on an RGB axes panel.

  Renders a slightly translucent white background pill in the bottom-right (or
  bottom-left) corner with the distance label (e.g. '100 m') on the left and a
  U-shaped scale bracket ('|_____|') of exact proportional width on the right.

  Args:
    ax: Matplotlib Axes containing the RGB image.
    width_m: Physical ground width (in meters) across the full horizontal span
      of the RGB panel.
    fontsize: Font size for the scalebar distance label.
    bar_m: Optional explicit scalebar length in meters.
    loc: Corner placement ('bottom_right' or 'bottom_left').
  """
  if width_m <= 0:
    return
  if bar_m is None:
    candidates = [20, 50, 100, 200, 250, 500, 1000]
    bar_m = min(candidates, key=lambda c: abs((c / width_m) - 0.24))
  bar_frac = float(bar_m) / float(width_m)
  label_str = f'{bar_m} m'
  fig = ax.figure
  ax_w_pt = 200.0
  if fig is not None:
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if bbox.width > 0 and bbox.height > 0:
      ax_w_pt = min(bbox.width, bbox.height) * 72.0
  text_w_est = (len(label_str) * 0.65 * fontsize) / ax_w_pt
  pad_left_x = 13.0 / ax_w_pt
  pad_right_x = 6.0 / ax_w_pt
  gap_x = 4.0 / ax_w_pt

  if loc == 'bottom_left':
    x_box_left = 0.020
    x_text_right = x_box_left + pad_left_x + text_w_est
    x_bar_left = x_text_right + gap_x
    x_bar_right = x_bar_left + bar_frac
    x_box_right = x_bar_right + pad_right_x
  else:
    x_bar_right = 0.962
    x_bar_left = x_bar_right - bar_frac
    x_text_right = x_bar_left - gap_x
    x_box_left = max(0.015, x_text_right - text_w_est - pad_left_x)
    x_box_right = 0.962 + pad_right_x
  y_box_bot = 0.022
  y_box_top = 0.102

  bg_patch = mpatches.FancyBboxPatch(
      (x_box_left, y_box_bot),
      x_box_right - x_box_left,
      y_box_top - y_box_bot,
      boxstyle='round,pad=0.0,rounding_size=0.014',
      facecolor='white',
      edgecolor='#b0b5b9',
      linewidth=0.6,
      alpha=0.84,
      transform=ax.transAxes,
      zorder=18,
  )
  ax.add_patch(bg_patch)

  ax.text(
      x_text_right,
      0.5 * (y_box_bot + y_box_top),
      label_str,
      transform=ax.transAxes,
      fontsize=fontsize,
      fontweight='bold',
      color='#1a1a1a',
      ha='right',
      va='center',
      zorder=20,
  )

  y_bar = y_box_bot + 0.025
  y_tick = y_box_bot + 0.058
  xs = [x_bar_left, x_bar_left, x_bar_right, x_bar_right]
  ys = [y_tick, y_bar, y_bar, y_tick]
  ax.plot(
      xs,
      ys,
      transform=ax.transAxes,
      color='white',
      linewidth=2.8,
      solid_capstyle='butt',
      solid_joinstyle='miter',
      zorder=19,
  )
  ax.plot(
      xs,
      ys,
      transform=ax.transAxes,
      color='#1a1a1a',
      linewidth=1.6,
      solid_capstyle='butt',
      solid_joinstyle='miter',
      zorder=20,
  )
