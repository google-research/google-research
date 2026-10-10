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

"""Generate Supplementary Paginated Multi-Dataset LiDAR Visual Comparison Figures.

Renders 4 paginated figures (3 datasets per page, 2 rows per dataset) showing
1 representative sample location from each of the 12 Aerial & Drone LiDAR
validation datasets:
- Row 1 (Optical + DSM Group):
  [1] GT RGB (PNeo 30cm)
  [2] LiDAR DSM GT (Native 1m)
  [3] LiDAR DSM GT (Resampled 15m)
  [4] GEM-15 DSM (15m)
  [5] Copernicus GLO-30 DSM (30m)
  [6] ALOS AW3D30 DSM (30m)
  [7] NASADEM DSM (30m)
- Row 2 (Site Metadata/Colorbar + DTM Group, vertically aligned under Row 1):
  [1] Site Info & Shared Elevation Colorbar
  [2] LiDAR DTM GT (Native 1m)
  [3] LiDAR DTM GT (Resampled 15m)
  [4] GEM-15 DTM (15m)
  [5] FABDEM DTM (30m)
"""

from collections.abc import Sequence
import concurrent.futures
import io
import os
import textwrap
from typing import Any
import urllib.request
from absl import app
from absl import flags
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ee
from gem_15 import ee_assets_config
from gem_15 import lidar_configs
from gem_15 import utils
from gem_15.plots import benchmark_data

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'output_dir',
    f'{benchmark_data.DEFAULT_VALIDATION_DIR}/plots',
    'Output directory for generated supplementary example pages.',
)
flags.DEFINE_string(
    'ee_project',
    None,
    'Google Cloud project ID for Earth Engine.',
)
flags.DEFINE_string(
    'ee_service_account',
    None,
    'Service account principal for Earth Engine authentication.',
)

TURBO_HEX = [
    '#fadd67',
    '#b8b257',
    '#7ca88f',
    '#4aa4ca',
    '#3869d1',
    '#6f5bc8',
    '#b862b0',
    '#bc3a6b',
    '#cb3438',
    '#f08429',
    '#fbd227',
]
CMAP_PAPER = LinearSegmentedColormap.from_list('paper_turbo', TURBO_HEX, N=256)

SAMPLE_SITES: list[dict[str, Any]] = [
    # Page 1: Drone LiDAR Benchmarks (3 datasets)
    {
        'key': 'global_tech_mahindra',
        'title': 'Global Drone LiDAR (Tech Mahindra)',
        'subtitle': 'Selangor, Malaysia (3.006°N, 101.538°E)',
        'category': 'Drone LiDAR',
        'lon': 101.538077,
        'lat': 3.006389,
        'buffer_m': 160,
    },
    {
        'key': 'india_tech_mahindra',
        'title': 'India Drone LiDAR (Tech Mahindra)',
        'subtitle': 'Karnataka, India (15.158°N, 76.925°E)',
        'category': 'Drone LiDAR',
        'lon': 76.925441,
        'lat': 15.158447,
        'buffer_m': 160,
    },
    {
        'key': '3dtrees',
        'title': '3D Trees Drone LiDAR',
        'subtitle': 'Goč Forest Reserve, Serbia (43.545°N, 20.766°E)',
        'category': 'Drone LiDAR',
        'lon': 20.765771,
        'lat': 43.544890,
        'buffer_m': 105,
    },
    # Page 2: National Aerial LiDAR I (3 datasets)
    {
        'key': 'neon',
        'title': 'US NEON Aerial LiDAR',
        'subtitle': 'Lenoir Landing, Alabama, USA (31.881°N, 88.198°W)',
        'category': 'National Aerial LiDAR',
        'lon': -88.198225,
        'lat': 31.881079,
        'buffer_m': 180,
    },
    {
        'key': 'new_zealand',
        'title': 'New Zealand Aerial LiDAR (LINZ)',
        'subtitle': 'Vancouver Cres, Christchurch, NZ (43.522°S, 172.690°E)',
        'category': 'National Aerial LiDAR',
        'lon': 172.6903,
        'lat': -43.52187,
        'buffer_m': 280,
        'vmin': 0.0,
        'vmax': 48.0,
    },
    {
        'key': 'uk_ea',
        'title': 'UK England Aerial LiDAR (Environment Agency)',
        'subtitle': 'South Devon, England, UK (50.464°N, 3.549°W)',
        'category': 'National Aerial LiDAR',
        'lon': -3.548789,
        'lat': 50.463881,
        'buffer_m': 175,
    },
    # Page 3: National Aerial LiDAR II (3 datasets)
    {
        'key': 'canada',
        'title': 'Canada Aerial LiDAR (HRDEM)',
        'subtitle': 'British Columbia, Canada (50.524°N, 126.193°W)',
        'category': 'National Aerial LiDAR',
        'lon': -126.192508,
        'lat': 50.523846,
        'buffer_m': 180,
    },
    {
        'key': 'spain',
        'title': 'Spain Aerial LiDAR (PNOA)',
        'subtitle': 'Badajoz, Extremadura, Spain (38.912°N, 7.053°W)',
        'category': 'National Aerial LiDAR',
        'lon': -7.053301,
        'lat': 38.911645,
        'buffer_m': 180,
    },
    {
        'key': 'brazil',
        'title': 'Brazil Aerial LiDAR (Sustainable Landscapes)',
        'subtitle': 'Manaus, Amazonas, Brazil (2.945°S, 59.924°W)',
        'category': 'National Aerial LiDAR',
        'lon': -59.924479,
        'lat': -2.945239,
        'buffer_m': 160,
    },
    # Page 4: National Aerial LiDAR III (3 datasets)
    {
        'key': 'indonesia',
        'title': 'Indonesia Aerial LiDAR (Kalimantan)',
        'subtitle': 'West Kalimantan, Indonesia (1.430°S, 111.087°E)',
        'category': 'National Aerial LiDAR',
        'lon': 111.087352,
        'lat': -1.430275,
        'buffer_m': 160,
    },
    {
        'key': 'mozambique',
        'title': 'Mozambique Aerial LiDAR (CMS Zambezi)',
        'subtitle': 'Zambezi Delta, Mozambique (18.793°S, 36.232°E)',
        'category': 'National Aerial LiDAR',
        'lon': 36.231862,
        'lat': -18.792708,
        'buffer_m': 160,
    },
    {
        'key': 'us_tropical',
        'title': 'US Tropical Aerial LiDAR (USGS 3DEP)',
        'subtitle': (
            'Everglades National Park, Florida, USA (25.174°N, 80.915°W)'
        ),
        'category': 'National Aerial LiDAR',
        'lon': -80.914514,
        'lat': 25.173826,
        'buffer_m': 150,
    },
]


def _fetch_numpy_grid(
    img,
    band,
    box,
    grid_dim = 160,
):
  """Fetches a 2D float numpy array for an Earth Engine Image band over box."""
  url = img.select(band).getDownloadURL({
      'region': box,
      'dimensions': f'{grid_dim}x{grid_dim}',
      'format': 'NPY',
  })
  with urllib.request.urlopen(url, timeout=60) as resp:
    raw = resp.read()
  arr = np.load(io.BytesIO(raw))
  if arr.dtype.names:
    arr = arr[band].astype(np.float64)
  else:
    arr = arr.astype(np.float64)
  arr[arr <= -999.0] = np.nan
  arr[arr == 0.0] = np.nan
  return arr


def _fetch_pneo_rgb(box, dims = 360):
  """Fetches and auto-stretches PNeo 30cm RGB imagery over box."""
  pneo_col = ee.ImageCollection(ee_assets_config.HIGH_RES_OPTICAL_RGB_ASSET_ID)
  filtered = pneo_col.filterBounds(box)
  img = filtered.mosaic().select(['R', 'G', 'B'])
  stats = img.reduceRegion(
      reducer=ee.Reducer.percentile([2, 98]),
      geometry=box,
      scale=1,
      maxPixels=1e7,
      bestEffort=True,
  ).getInfo()
  r_min = float(stats.get('R_p2') or 15.0)
  g_min = float(stats.get('G_p2') or 15.0)
  b_min = float(stats.get('B_p2') or 15.0)
  r_max = float(stats.get('R_p98') or 160.0)
  g_max = float(stats.get('G_p98') or 160.0)
  b_max = float(stats.get('B_p98') or 160.0)
  vmin = min(r_min, g_min, b_min)
  vmax = max(r_max, g_max, b_max)
  if vmax <= vmin + 5:
    vmin, vmax = 10.0, 180.0
  vis = img.visualize(
      bands=['R', 'G', 'B'],
      min=vmin,
      max=vmax,
      gamma=1.15,
  )
  url = vis.getThumbURL(
      {'dimensions': f'{dims}x{dims}', 'region': box, 'format': 'png'}
  )
  with urllib.request.urlopen(url, timeout=60) as resp:
    pil_img = Image.open(io.BytesIO(resp.read())).convert('RGB')
  return np.asarray(pil_img)


def _prepare_lidar_images(
    dataset_key,
    box,
    egm96,
    utm_crs,
):
  """Returns (native_dsm, native_dtm, dsm_15m, dtm_15m) aligned to EGM96."""
  cfg = lidar_configs.LIDAR_DATASET_CONFIGS[dataset_key]
  asset_id = cfg['asset_id']
  asset_info = ee.data.getAsset(asset_id)
  asset_type = asset_info.get('type')

  col = None
  if asset_type == 'IMAGE_COLLECTION':
    col = ee.ImageCollection(asset_id).filterBounds(box.buffer(250))
    if cfg['date_range']:
      col = col.filterDate(*cfg['date_range'])
    if cfg['collection_filter']:
      col = cfg['collection_filter'](col)
    if cfg['image_preprocess_fn']:
      col = col.map(cfg['image_preprocess_fn'])
    first_img = ee.Image(col.first())
    native_proj = first_img.select(0).projection()
    lidar_img = col.mosaic()
  else:
    lidar_img = ee.Image(asset_id)
    native_proj = lidar_img.select(cfg['dsm_band']).projection()

  if cfg['geoid_asset'] is None:
    local_geoid = ee.Image.constant(0)
  else:
    local_geoid = (
        ee.Image(cfg['geoid_asset'])
        .select(cfg['geoid_band'])
        .resample('bilinear')
    )
    if cfg['geoid_scale'] != 1.0:
      local_geoid = local_geoid.multiply(cfg['geoid_scale'])

  raw_dsm = lidar_img.select(cfg['dsm_band']).toDouble()
  raw_dtm = lidar_img.select(cfg['dtm_band']).toDouble()
  valid_mask = raw_dtm.neq(0).And(raw_dsm.neq(0))
  native_dsm = (
      raw_dsm.updateMask(valid_mask)
      .add(local_geoid)
      .subtract(egm96)
      .rename('dsm_nat')
  )
  native_dtm = (
      raw_dtm.updateMask(valid_mask)
      .add(local_geoid)
      .subtract(egm96)
      .rename('dtm_nat')
  )

  if col is not None:

    def _downsample_tile(img):
      dsm = (
          img.select(cfg['dsm_band'])
          .toDouble()
          .updateMask(img.select(cfg['dtm_band']).neq(0))
          .add(local_geoid)
          .subtract(egm96)
      )
      dtm = (
          img.select(cfg['dtm_band'])
          .toDouble()
          .updateMask(img.select(cfg['dtm_band']).neq(0))
          .add(local_geoid)
          .subtract(egm96)
      )
      dsm_15 = dsm.reduceResolution(
          reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
      ).reproject(crs=utm_crs, scale=15)
      dtm_15 = dtm.reduceResolution(
          reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
      ).reproject(crs=utm_crs, scale=15)
      return dsm_15.rename('dsm_15m').addBands(dtm_15.rename('dtm_15m'))

    ds_col = col.map(_downsample_tile).mosaic()
    dsm_15m = ds_col.select('dsm_15m')
    dtm_15m = ds_col.select('dtm_15m')
  else:
    dsm_15m = (
        native_dsm.setDefaultProjection(native_proj)
        .reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
        )
        .reproject(crs=utm_crs, scale=15)
        .rename('dsm_15m')
    )
    dtm_15m = (
        native_dtm.setDefaultProjection(native_proj)
        .reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=2048, bestEffort=True
        )
        .reproject(crs=utm_crs, scale=15)
        .rename('dtm_15m')
    )

  return native_dsm, native_dtm, dsm_15m, dtm_15m


def _render_shaded_relief(
    arr,
    vmin,
    vmax,
):
  """Renders an elevation array using the clean paper Turbo colormap."""
  nan_mask = np.isnan(arr)
  fill_val = float(np.nanmedian(arr)) if not np.all(nan_mask) else vmin
  arr_filled = np.where(nan_mask, fill_val, arr)
  normed = np.clip((arr_filled - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
  rgb = CMAP_PAPER(normed)[Ellipsis, :3].copy()
  if np.any(nan_mask):
    rgb[nan_mask] = [0.94, 0.94, 0.94]
  return rgb


def fetch_site_bundle(site):
  """Fetches all layers (PNeo RGB, DSMs, DTMs, and GEM-15 percentiles) for a single site."""
  print(f"Fetching layers for [{site['title']}] ({site['subtitle']})...")
  pt = ee.Geometry.Point([site['lon'], site['lat']])
  box = pt.buffer(site['buffer_m']).bounds()
  utm_crs = utils.get_utm_crs(site['lon'], site['lat'])

  gem15 = utils.get_gem15_layers(resample_bilinear=False)
  ref_dems = utils.get_global_reference_dems(resample_bilinear=False)
  egm96 = ref_dems['egm96'].resample('bilinear')

  native_dsm, native_dtm, dsm_15m, dtm_15m = _prepare_lidar_images(
      site['key'], box, egm96, utm_crs
  )

  # Reproject 15m and 30m layers onto their true local UTM resolution grids so
  # pixel blockiness (15m vs 30m) matches true ground meters at all latitudes.
  ge_dsm = gem15['dsm'].reproject(crs=utm_crs, scale=15).rename('ge_dsm')
  ge_p95 = gem15['p95'].reproject(crs=utm_crs, scale=15).rename('ge_p95')
  ge_p5 = gem15['p5'].reproject(crs=utm_crs, scale=15).rename('ge_p5')
  ge_dtm = gem15['dtm'].reproject(crs=utm_crs, scale=15).rename('ge_dtm')
  cop_dsm = (
      ref_dems['copernicus'].reproject(crs=utm_crs, scale=30).rename('cop_dsm')
  )
  alos_dsm = (
      ref_dems['alos'].reproject(crs=utm_crs, scale=30).rename('alos_dsm')
  )
  nasa_dsm = (
      ref_dems['nasa'].reproject(crs=utm_crs, scale=30).rename('nasa_dsm')
  )
  fab_dtm = (
      ref_dems['fabdem'].reproject(crs=utm_crs, scale=30).rename('fab_dtm')
  )

  rgb_arr = _fetch_pneo_rgb(box, dims=400)
  arr_dsm_nat = _fetch_numpy_grid(native_dsm, 'dsm_nat', box, grid_dim=320)
  arr_dtm_nat = _fetch_numpy_grid(native_dtm, 'dtm_nat', box, grid_dim=320)
  arr_dsm_15 = _fetch_numpy_grid(dsm_15m, 'dsm_15m', box, grid_dim=200)
  arr_dtm_15 = _fetch_numpy_grid(dtm_15m, 'dtm_15m', box, grid_dim=200)
  arr_ge_dsm = _fetch_numpy_grid(ge_dsm, 'ge_dsm', box, grid_dim=200)
  arr_ge_p95 = _fetch_numpy_grid(ge_p95, 'ge_p95', box, grid_dim=200)
  arr_ge_p5 = _fetch_numpy_grid(ge_p5, 'ge_p5', box, grid_dim=200)
  arr_ge_dtm = _fetch_numpy_grid(ge_dtm, 'ge_dtm', box, grid_dim=200)
  arr_cop_dsm = _fetch_numpy_grid(cop_dsm, 'cop_dsm', box, grid_dim=200)
  arr_alos_dsm = _fetch_numpy_grid(alos_dsm, 'alos_dsm', box, grid_dim=200)
  arr_nasa_dsm = _fetch_numpy_grid(nasa_dsm, 'nasa_dsm', box, grid_dim=200)
  arr_fab_dtm = _fetch_numpy_grid(fab_dtm, 'fab_dtm', box, grid_dim=200)

  # Align local patch vertical offset (debias) to the LiDAR GT mean so all
  # layers share the exact same [vmin, vmax] elevation scale bar cleanly.
  ref_dsm_mean = float(np.nanmean(arr_dsm_15))
  ref_dtm_mean = float(np.nanmean(arr_dtm_15))
  ge_dsm_shift = ref_dsm_mean - float(np.nanmean(arr_ge_dsm))
  arr_ge_dsm += ge_dsm_shift
  arr_ge_p95 += ge_dsm_shift
  for arr in [arr_cop_dsm, arr_alos_dsm, arr_nasa_dsm]:
    if not np.all(np.isnan(arr)):
      arr += ref_dsm_mean - float(np.nanmean(arr))
  ge_dtm_shift = ref_dtm_mean - float(np.nanmean(arr_ge_dtm))
  arr_ge_dtm += ge_dtm_shift
  arr_ge_p5 += ge_dtm_shift
  for arr in [arr_fab_dtm]:
    if not np.all(np.isnan(arr)):
      arr += ref_dtm_mean - float(np.nanmean(arr))

  if site.get('vmin') is not None and site.get('vmax') is not None:
    vmin = float(site['vmin'])
    vmax = float(site['vmax'])
  else:
    valid_vals = np.concatenate([
        arr_dtm_15[~np.isnan(arr_dtm_15)],
        arr_dsm_15[~np.isnan(arr_dsm_15)],
        arr_ge_dsm[~np.isnan(arr_ge_dsm)],
    ])
    vmin = float(np.percentile(valid_vals, 2)) - 1.5
    vmax = float(np.percentile(valid_vals, 98)) + 1.5
    if vmax - vmin < 8.0:
      mid = 0.5 * (vmin + vmax)
      vmin, vmax = mid - 4.0, mid + 4.0

  return {
      'site': site,
      'vmin': vmin,
      'vmax': vmax,
      'rgb': rgb_arr,
      'dsm_nat': _render_shaded_relief(arr_dsm_nat, vmin, vmax),
      'dsm_15': _render_shaded_relief(arr_dsm_15, vmin, vmax),
      'ge_dsm': _render_shaded_relief(arr_ge_dsm, vmin, vmax),
      'ge_p95': _render_shaded_relief(arr_ge_p95, vmin, vmax),
      'ge_p5': _render_shaded_relief(arr_ge_p5, vmin, vmax),
      'cop_dsm': _render_shaded_relief(arr_cop_dsm, vmin, vmax),
      'alos_dsm': _render_shaded_relief(arr_alos_dsm, vmin, vmax),
      'nasa_dsm': _render_shaded_relief(arr_nasa_dsm, vmin, vmax),
      'dtm_nat': _render_shaded_relief(arr_dtm_nat, vmin, vmax),
      'dtm_15': _render_shaded_relief(arr_dtm_15, vmin, vmax),
      'ge_dtm': _render_shaded_relief(arr_ge_dtm, vmin, vmax),
      'fab_dtm': _render_shaded_relief(arr_fab_dtm, vmin, vmax),
  }


def render_page(
    bundles,
    page_idx,
    page_title,
    out_path,
):
  """Renders 3 LiDAR datasets stacked vertically (2 rows × 7 columns per dataset)."""
  fig = plt.figure(figsize=(22.5, 19.2), dpi=160, facecolor='white')
  outer_gs = gridspec.GridSpec(
      3,
      1,
      figure=fig,
      hspace=0.09,
      left=0.012,
      right=0.988,
      top=0.94,
      bottom=0.012,
  )

  fig.suptitle(
      'Supplementary Qualitative Comparison across Aerial & Drone LiDAR'
      f' Benchmarks — Page {page_idx}/4: {page_title}',
      fontsize=19,
      fontweight='bold',
      y=0.982,
      color='#111111',
  )

  for ds_idx, bundle in enumerate(bundles):
    site = bundle['site']
    vmin, vmax = bundle['vmin'], bundle['vmax']
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        7,
        subplot_spec=outer_gs[ds_idx],
        wspace=0.025,
        hspace=0.11,
    )

    panel_letter = chr(65 + (page_idx - 1) * 3 + ds_idx)

    # Row 1: GT RGB + 6 DSM Layers
    row1_panels = [
        ('GT RGB (PNeo 30cm)', bundle['rgb'], '#222222', 'normal'),
        ('LiDAR DSM GT (Native)', bundle['dsm_nat'], '#222222', 'normal'),
        ('LiDAR DSM GT (15m)', bundle['dsm_15'], '#222222', 'normal'),
        ('GEM-15 DSM (15m)', bundle['ge_dsm'], '#d95f02', 'bold'),
        ('Copernicus DSM (30m)', bundle['cop_dsm'], '#222222', 'normal'),
        ('ALOS AW3D30 DSM (30m)', bundle['alos_dsm'], '#222222', 'normal'),
        ('NASADEM DSM (30m)', bundle['nasa_dsm'], '#222222', 'normal'),
    ]
    for col_idx, (title, img_arr, title_color, title_weight) in enumerate(
        row1_panels
    ):
      ax = fig.add_subplot(inner_gs[0, col_idx])
      ax.imshow(img_arr)
      if col_idx == 0:
        utils.add_rgb_scalebar(ax, 2.0 * float(site['buffer_m']), fontsize=8.8)
      ax.set_title(
          title,
          fontsize=12,
          fontweight=title_weight,
          color=title_color,
          pad=3.5,
      )
      ax.set_xticks([])
      ax.set_yticks([])
      for spine in ax.spines.values():
        spine.set_edgecolor('#d95f02' if 'GEM-15' in title else '#888888')
        spine.set_linewidth(2.0 if 'GEM-15' in title else 0.8)

    # Row 2: Col 1-4 under DSM (Native DTM, 15m DTM, GEM-15 DTM, FABDEM DTM)
    row2_panels = [
        (1, 'LiDAR DTM GT (Native)', bundle['dtm_nat'], '#222222', 'normal'),
        (2, 'LiDAR DTM GT (15m)', bundle['dtm_15'], '#222222', 'normal'),
        (3, 'GEM-15 DTM (15m)', bundle['ge_dtm'], '#d95f02', 'bold'),
        (4, 'FABDEM DTM (30m)', bundle['fab_dtm'], '#222222', 'normal'),
    ]
    for col_idx, title, img_arr, title_color, title_weight in row2_panels:
      ax = fig.add_subplot(inner_gs[1, col_idx])
      ax.imshow(img_arr)
      ax.set_title(
          title,
          fontsize=12,
          fontweight=title_weight,
          color=title_color,
          pad=3.5,
      )
      ax.set_xticks([])
      ax.set_yticks([])
      for spine in ax.spines.values():
        spine.set_edgecolor('#d95f02' if 'GEM-15' in title else '#888888')
        spine.set_linewidth(2.0 if 'GEM-15' in title else 0.8)

    # Row 2 Col 0: Clean borderless metadata (Dataset name + Lat/Lon only)
    ax_info = fig.add_subplot(inner_gs[1, 0])
    ax_info.axis('off')
    lat_val, lon_val = float(site['lat']), float(site['lon'])
    lat_str = (
        f'{abs(lat_val):.3f}°N' if lat_val >= 0 else f'{abs(lat_val):.3f}°S'
    )
    lon_str = (
        f'{abs(lon_val):.3f}°E' if lon_val >= 0 else f'{abs(lon_val):.3f}°W'
    )
    loc_name = site['subtitle'].split('(')[0].strip()
    wrapped_title = textwrap.fill(f"({panel_letter}) {site['title']}", width=24)
    ax_info.text(
        0.5,
        0.54,
        wrapped_title,
        transform=ax_info.transAxes,
        fontsize=12.5,
        fontweight='bold',
        color='#111111',
        va='bottom',
        ha='center',
        linespacing=1.25,
    )
    ax_info.text(
        0.5,
        0.48,
        f'{loc_name}\n{lat_str}, {lon_str}',
        transform=ax_info.transAxes,
        fontsize=11.5,
        fontweight='normal',
        color='#333333',
        va='top',
        ha='center',
        linespacing=1.35,
    )

    # Row 2 Col 5-6: Shared Elevation Colorbar
    ax_cbar_host = fig.add_subplot(inner_gs[1, 5:7])
    ax_cbar_host.axis('off')
    cbar_ax = ax_cbar_host.inset_axes([0.06, 0.48, 0.88, 0.20])
    sm = plt.cm.ScalarMappable(
        cmap=CMAP_PAPER, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm._A = []  # pylint: disable=protected-access
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(
        'Orthometric Elevation (m, EGM96)',
        fontsize=12,
        fontweight='normal',
        labelpad=5,
    )
    cbar.ax.tick_params(labelsize=11)

  parent_dir = os.path.dirname(out_path)
  if parent_dir and not os.path.exists(parent_dir):
    os.makedirs(parent_dir, exist_ok=True)
  buf = io.BytesIO()
  fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
  plt.close(fig)
  with open(out_path, 'wb') as f:
    f.write(buf.getvalue())
  print(f'Saved paginated supplementary figure: {out_path}')


def render_single_vancouver_showcase(
    bundle, out_dir
):
  """Renders standalone 2-row figures for Vancouver Cres, Christchurch, NZ."""
  site = bundle['site']
  vmin, vmax = bundle['vmin'], bundle['vmax']
  lat_val, lon_val = float(site['lat']), float(site['lon'])
  lat_str = f'{abs(lat_val):.3f}°N' if lat_val >= 0 else f'{abs(lat_val):.3f}°S'
  lon_str = f'{abs(lon_val):.3f}°E' if lon_val >= 0 else f'{abs(lon_val):.3f}°W'
  loc_name = site['subtitle'].split('(')[0].strip()

  # 1. Standalone 2-row x 7-col (Supplementary LiDAR comparison format)
  fig1 = plt.figure(figsize=(22.5, 6.8), dpi=180, facecolor='white')
  gs1 = gridspec.GridSpec(
      2,
      7,
      figure=fig1,
      wspace=0.025,
      hspace=0.11,
      left=0.012,
      right=0.988,
      top=0.92,
      bottom=0.03,
  )
  row1_panels = [
      ('GT RGB (PNeo 30cm)', bundle['rgb'], '#222222', 'normal'),
      ('LiDAR DSM GT (1m)', bundle['dsm_nat'], '#222222', 'normal'),
      ('LiDAR DSM GT (15m)', bundle['dsm_15'], '#222222', 'normal'),
      ('GEM-15 DSM (15m)', bundle['ge_dsm'], '#d95f02', 'bold'),
      ('Copernicus DSM (30m)', bundle['cop_dsm'], '#222222', 'normal'),
      ('ALOS AW3D30 DSM (30m)', bundle['alos_dsm'], '#222222', 'normal'),
      ('NASADEM DSM (30m)', bundle['nasa_dsm'], '#222222', 'normal'),
  ]
  for col_idx, (title, img_arr, title_color, title_weight) in enumerate(
      row1_panels
  ):
    ax = fig1.add_subplot(gs1[0, col_idx])
    ax.imshow(img_arr)
    if col_idx == 0:
      utils.add_rgb_scalebar(ax, 2.0 * float(site['buffer_m']), fontsize=9.0)
    ax.set_title(
        title,
        fontsize=12.5,
        fontweight=title_weight,
        color=title_color,
        pad=4.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_edgecolor('#d95f02' if 'GEM-15' in title else '#888888')
      spine.set_linewidth(2.2 if 'GEM-15' in title else 0.8)

  row2_panels = [
      (1, 'LiDAR DTM GT (1m)', bundle['dtm_nat'], '#222222', 'normal'),
      (2, 'LiDAR DTM GT (15m)', bundle['dtm_15'], '#222222', 'normal'),
      (3, 'GEM-15 DTM (15m)', bundle['ge_dtm'], '#d95f02', 'bold'),
      (4, 'FABDEM DTM (30m)', bundle['fab_dtm'], '#222222', 'normal'),
  ]
  for col_idx, title, img_arr, title_color, title_weight in row2_panels:
    ax = fig1.add_subplot(gs1[1, col_idx])
    ax.imshow(img_arr)
    ax.set_title(
        title,
        fontsize=12.5,
        fontweight=title_weight,
        color=title_color,
        pad=4.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_edgecolor('#d95f02' if 'GEM-15' in title else '#888888')
      spine.set_linewidth(2.2 if 'GEM-15' in title else 0.8)

  ax_info1 = fig1.add_subplot(gs1[1, 0])
  ax_info1.axis('off')
  wrapped_title1 = textwrap.fill(site['title'], width=22)
  ax_info1.text(
      0.5,
      0.55,
      wrapped_title1,
      transform=ax_info1.transAxes,
      fontsize=13,
      fontweight='bold',
      color='#111111',
      va='bottom',
      ha='center',
      linespacing=1.25,
  )
  ax_info1.text(
      0.5,
      0.48,
      f'{loc_name}\n{lat_str}, {lon_str}',
      transform=ax_info1.transAxes,
      fontsize=11.5,
      fontweight='normal',
      color='#333333',
      va='top',
      ha='center',
      linespacing=1.35,
  )

  ax_cbar_host1 = fig1.add_subplot(gs1[1, 5:7])
  ax_cbar_host1.axis('off')
  cbar_ax1 = ax_cbar_host1.inset_axes([0.06, 0.48, 0.88, 0.20])
  sm1 = plt.cm.ScalarMappable(
      cmap=CMAP_PAPER, norm=plt.Normalize(vmin=vmin, vmax=vmax)
  )
  sm1._A = []  # pylint: disable=protected-access
  cbar1 = fig1.colorbar(sm1, cax=cbar_ax1, orientation='horizontal')
  cbar1.set_label(
      'Orthometric Elevation (m, EGM96)',
      fontsize=12,
      fontweight='normal',
      labelpad=5,
  )
  cbar1.ax.tick_params(labelsize=11)

  out_path1 = os.path.join(
      out_dir, 'vancouver_cres_new_zealand_lidar_comparison.png'
  )
  buf1 = io.BytesIO()
  fig1.savefig(buf1, format='png', dpi=180, bbox_inches='tight')
  plt.close(fig1)
  with open(out_path1, 'wb') as f:
    f.write(buf1.getvalue())
  print(f'Saved Vancouver Cres 7-col LiDAR comparison: {out_path1}')

  # 2. Standalone 2-row x 6-col (Showcase format with GEM-15 P95 & P5)
  fig2 = plt.figure(figsize=(19.8, 6.9), dpi=180, facecolor='white')
  gs2 = gridspec.GridSpec(
      2,
      6,
      figure=fig2,
      wspace=0.024,
      hspace=0.11,
      left=0.012,
      right=0.988,
      top=0.92,
      bottom=0.03,
  )
  sc_row1 = [
      ('Pléiades Neo RGB (30cm)', bundle['rgb'], '#222222', 'normal'),
      ('LiDAR DSM GT (1m)', bundle['dsm_nat'], '#222222', 'normal'),
      ('GEM-15 DSM (15m)', bundle['ge_dsm'], '#d95f02', 'bold'),
      ('GEM-15 P95 (15m)', bundle['ge_p95'], '#d95f02', 'bold'),
      ('GEM-15 P5 (15m)', bundle['ge_p5'], '#d95f02', 'bold'),
      ('GEM-15 DTM (15m)', bundle['ge_dtm'], '#d95f02', 'bold'),
  ]
  for col_idx, (title, img_arr, title_color, title_weight) in enumerate(
      sc_row1
  ):
    ax = fig2.add_subplot(gs2[0, col_idx])
    ax.imshow(img_arr)
    if col_idx == 0:
      utils.add_rgb_scalebar(ax, 2.0 * float(site['buffer_m']), fontsize=9.2)
    ax.set_title(
        title,
        fontsize=12.5,
        fontweight=title_weight,
        color=title_color,
        pad=4.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_edgecolor('#d95f02' if 'GEM-15' in title else '#888888')
      spine.set_linewidth(2.2 if 'GEM-15' in title else 0.8)

  sc_row2 = [
      (1, 'LiDAR DTM GT (1m)', bundle['dtm_nat'], '#222222', 'normal'),
      (2, 'Copernicus DSM (30m)', bundle['cop_dsm'], '#222222', 'normal'),
      (3, 'ALOS AW3D30 DSM (30m)', bundle['alos_dsm'], '#222222', 'normal'),
      (4, 'NASADEM DSM (30m)', bundle['nasa_dsm'], '#222222', 'normal'),
      (5, 'FABDEM DTM (30m)', bundle['fab_dtm'], '#222222', 'normal'),
  ]
  for col_idx, title, img_arr, title_color, title_weight in sc_row2:
    ax = fig2.add_subplot(gs2[1, col_idx])
    ax.imshow(img_arr)
    ax.set_title(
        title,
        fontsize=12.5,
        fontweight=title_weight,
        color=title_color,
        pad=4.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_edgecolor('#d95f02' if 'GEM-15' in title else '#888888')
      spine.set_linewidth(2.2 if 'GEM-15' in title else 0.8)

  ax_info2 = fig2.add_subplot(gs2[1, 0])
  ax_info2.axis('off')
  ax_info2.text(
      0.5,
      0.74,
      wrapped_title1,
      transform=ax_info2.transAxes,
      fontsize=12.5,
      fontweight='bold',
      color='#111111',
      va='bottom',
      ha='center',
      linespacing=1.22,
  )
  ax_info2.text(
      0.5,
      0.70,
      f'{loc_name}\n{lat_str}, {lon_str}',
      transform=ax_info2.transAxes,
      fontsize=11,
      fontweight='normal',
      color='#333333',
      va='top',
      ha='center',
      linespacing=1.30,
  )
  cbar_ax2 = ax_info2.inset_axes([0.06, 0.24, 0.88, 0.11])
  sm2 = plt.cm.ScalarMappable(
      cmap=CMAP_PAPER, norm=plt.Normalize(vmin=vmin, vmax=vmax)
  )
  sm2._A = []  # pylint: disable=protected-access
  cbar2 = fig2.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
  cbar2.set_label(
      'Orthometric Elevation (m, EGM96)',
      fontsize=10.5,
      fontweight='normal',
      labelpad=3,
  )
  cbar2.ax.tick_params(labelsize=9.5)

  out_path2 = os.path.join(
      out_dir, 'vancouver_cres_new_zealand_with_percentiles.png'
  )
  buf2 = io.BytesIO()
  fig2.savefig(buf2, format='png', dpi=180, bbox_inches='tight')
  plt.close(fig2)
  with open(out_path2, 'wb') as f:
    f.write(buf2.getvalue())
  print(f'Saved Vancouver Cres 6-col Showcase (with P95/P5): {out_path2}')


def main(argv):
  del argv  # Unused.
  utils.authenticate_earth_engine(
      ee_service_account=FLAGS.ee_service_account,
      project=FLAGS.ee_project,
      opt_url='https://earthengine-highvolume.googleapis.com',
  )

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    all_bundles = list(executor.map(fetch_site_bundle, SAMPLE_SITES))

  # Render standalone 2-row figures for Vancouver Cres, New Zealand (index 4)
  render_single_vancouver_showcase(all_bundles[4], FLAGS.output_dir)

  pages = [
      (
          1,
          'Drone LiDAR Benchmarks (Global, India, 3D Trees)',
          all_bundles[0:3],
          'supplementary_lidar_examples_page1_drone.png',
      ),
      (
          2,
          'National Aerial LiDAR I (US NEON, New Zealand, UK England)',
          all_bundles[3:6],
          'supplementary_lidar_examples_page2_aerial_1.png',
      ),
      (
          3,
          'National Aerial LiDAR II (Canada, Spain, Brazil)',
          all_bundles[6:9],
          'supplementary_lidar_examples_page3_aerial_2.png',
      ),
      (
          4,
          'National Aerial LiDAR III (Indonesia, Mozambique, US Tropical)',
          all_bundles[9:12],
          'supplementary_lidar_examples_page4_aerial_3.png',
      ),
  ]

  for page_idx, page_title, page_bundles, fname in pages:
    out_path = os.path.join(FLAGS.output_dir, fname)
    render_page(page_bundles, page_idx, page_title, out_path)


if __name__ == '__main__':
  app.run(main)
