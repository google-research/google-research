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

"""Generate 3-Sample Qualitative Showcase Figure for GEM-15 vs Global DEMs.

Renders 3 showcase locations stacked vertically (2 rows x 6 columns per sample)
matching the clean supplementary figure styling:
- Row 1 (Optical + Full GEM-15 Suite, 6 columns):
  [1] Pléiades Neo RGB (30cm)
  [2] GEM-15 DSM (15m)
  [3] GEM-15 P95 (15m)
  [4] GEM-15 P5 (15m)
  [5] GEM-15 DTM (15m)
  [6] GEM-15 Heightmap (DSM - DTM)
- Row 2 (Clean Metadata & Colorbars + Global 30m Benchmarks, 6 columns):
  [1] Site Name, Lat/Lon & Elevation / Heightmap Colorbars
  [2] Copernicus DSM (30m)
  [3] ALOS AW3D30 DSM (30m)
  [4] NASADEM DSM (30m)
  [5] FABDEM DTM (30m)
  [6] Copernicus Heightmap (DSM - FABDEM)
"""

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
from gem_15 import utils
from gem_15.plots import benchmark_data

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'output_dir',
    f'{benchmark_data.DEFAULT_VALIDATION_DIR}/plots',
    'Output directory for generated showcase figure.',
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

HEIGHT_HEX = [
    '#ffffff',
    '#e5f5e0',
    '#a1d99b',
    '#41ab5d',
    '#238b45',
    '#00441b',
]
CMAP_HEIGHT = LinearSegmentedColormap.from_list(
    'white_to_green', HEIGHT_HEX, N=256
)

SHOWCASE_SAMPLES: list[dict[str, Any]] = [
    {
        'id': 'sample1_forest',
        'title': 'Forest Canopy & Valley Clearing',
        'location': 'Schwarzwald, Germany',
        'lon': 8.238,
        'lat': 48.748,
        'half_size_deg': 0.0050,
        'buffer_m': 400.0,
        'vmin': 160.0,
        'vmax': 305.0,
        'hmin': 0.0,
        'hmax': 50.0,
        'rgb_min': 20,
        'rgb_max': 140,
    },
    {
        'id': 'sample2_urban',
        'title': 'High-Density Urban Morphology',
        'location': 'Eixample, Barcelona, Spain',
        'lon': 2.165,
        'lat': 41.395,
        'half_size_deg': 0.0045,
        'buffer_m': 400.0,
        'vmin': 25.0,
        'vmax': 75.0,
        'hmin': 0.0,
        'hmax': 50.0,
        'rgb_min': 20,
        'rgb_max': 180,
    },
    {
        'id': 'sample3_mine',
        'title': 'Open-Cut Mine & Terraces',
        'location': 'Kalgoorlie Super Pit, Australia',
        'lon': 121.510,
        'lat': -30.760,
        'half_size_deg': 0.0050,
        'buffer_m': 500.0,
        'vmin': 362.0,
        'vmax': 478.0,
        'hmin': 0.0,
        'hmax': 50.0,
        'rgb_min': 20,
        'rgb_max': 180,
    },
]


def _fetch_numpy_grid(
    img, band, box, grid_dim = 240
):
  """Samples an Earth Engine image band onto a 2D numpy array."""
  url = img.select([band]).getDownloadURL({
      'region': box,
      'dimensions': f'{grid_dim}x{grid_dim}',
      'format': 'NPY',
  })
  raw = urllib.request.urlopen(url, timeout=60).read()
  structured = np.load(io.BytesIO(raw))
  arr = np.array(structured[band], dtype=np.float64)
  arr[arr <= -999.0] = np.nan
  return arr


def _fetch_pneo_rgb(
    box, rgb_min, rgb_max, dims = 512
):
  """Fetches 30cm Pléiades Neo RGB thumbnail."""
  pneo_col = (
      ee.ImageCollection(ee_assets_config.HIGH_RES_OPTICAL_RGB_ASSET_ID)
      .filterBounds(box)
      .select(['R', 'G', 'B'])
  )
  pneo_mosaic = pneo_col.mosaic()
  vis = pneo_mosaic.visualize(
      bands=['R', 'G', 'B'],
      min=rgb_min,
      max=rgb_max,
      gamma=1.12,
  )
  url = vis.getThumbURL({
      'dimensions': f'{dims}x{dims}',
      'region': box,
      'format': 'png',
  })
  raw = urllib.request.urlopen(url, timeout=60).read()
  return np.array(Image.open(io.BytesIO(raw)).convert('RGB'))


def _render_shaded_relief(
    arr, vmin, vmax
):
  """Renders 2D elevation array with the paper Turbo colormap."""
  nan_mask = np.isnan(arr)
  fill_val = float(np.nanmedian(arr)) if not np.all(nan_mask) else vmin
  arr_filled = np.where(nan_mask, fill_val, arr)

  norm = np.clip((arr_filled - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
  rgb = CMAP_PAPER(norm)[Ellipsis, :3].copy()
  if np.any(nan_mask):
    rgb[nan_mask] = [0.94, 0.94, 0.94]
  return rgb


def _render_heightmap(arr, hmin, hmax):
  """Renders above-ground heightmap (0m = pure white, hmax = dark green)."""
  nan_mask = np.isnan(arr)
  arr_clean = np.where(nan_mask, 0.0, np.maximum(arr, 0.0))
  norm = np.clip((arr_clean - hmin) / max(hmax - hmin, 1e-6), 0.0, 1.0)
  rgb = CMAP_HEIGHT(norm)[Ellipsis, :3].copy()
  if np.any(nan_mask):
    rgb[nan_mask] = [0.96, 0.96, 0.96]
  return rgb


def fetch_sample_bundle(sample):
  """Fetches all 11 panels (PNeo RGB, 5 GEM-15 layers, 5 Benchmark layers)."""
  print(f"Fetching layers for [{sample['title']}] ({sample['location']})...")
  lon, lat = float(sample['lon']), float(sample['lat'])
  buf_m = float(sample.get('buffer_m', 400.0))
  box = ee.Geometry.Point([lon, lat]).buffer(buf_m).bounds()
  utm_crs = utils.get_utm_crs(lon, lat)

  gem15 = utils.get_gem15_layers(resample_bilinear=False)
  ref_dems = utils.get_global_reference_dems(resample_bilinear=False)

  # Preserve native 15m vs 30m pixel resolution in true ground meters (UTM)
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

  rgb_arr = _fetch_pneo_rgb(box, sample['rgb_min'], sample['rgb_max'], dims=512)
  arr_ge_dsm = _fetch_numpy_grid(ge_dsm, 'ge_dsm', box, grid_dim=240)
  arr_ge_p95 = _fetch_numpy_grid(ge_p95, 'ge_p95', box, grid_dim=240)
  arr_ge_p5 = _fetch_numpy_grid(ge_p5, 'ge_p5', box, grid_dim=240)
  arr_ge_dtm = _fetch_numpy_grid(ge_dtm, 'ge_dtm', box, grid_dim=240)

  arr_cop_dsm = _fetch_numpy_grid(cop_dsm, 'cop_dsm', box, grid_dim=240)
  arr_alos_dsm = _fetch_numpy_grid(alos_dsm, 'alos_dsm', box, grid_dim=240)
  arr_nasa_dsm = _fetch_numpy_grid(nasa_dsm, 'nasa_dsm', box, grid_dim=240)
  arr_fab_dtm = _fetch_numpy_grid(fab_dtm, 'fab_dtm', box, grid_dim=240)

  # Compute above-ground heightmaps
  arr_ge_h = np.maximum(arr_ge_dsm - arr_ge_dtm, 0.0)
  arr_cop_h = np.maximum(arr_cop_dsm - arr_fab_dtm, 0.0)

  # Align global 30m benchmarks to the GEM-15 local vertical datum so all
  # elevation panels share the exact same [vmin, vmax] colorbar cleanly, while
  # keeping GEM-15 (P95, DSM, P5, DTM) untouched relative to each other.
  ge_dtm_mean = float(np.nanmean(arr_ge_dtm))
  ge_dsm_mean = float(np.nanmean(arr_ge_dsm))
  if not np.all(np.isnan(arr_fab_dtm)):
    fab_offset = ge_dtm_mean - float(np.nanmean(arr_fab_dtm))
    arr_fab_dtm += fab_offset
    arr_cop_dsm += fab_offset
  for arr in [arr_alos_dsm, arr_nasa_dsm]:
    if not np.all(np.isnan(arr)):
      arr += ge_dsm_mean - float(np.nanmean(arr))

  if sample['vmin'] is not None and sample['vmax'] is not None:
    vmin, vmax = float(sample['vmin']), float(sample['vmax'])
  else:
    vmin = float(np.nanpercentile(arr_ge_dsm, 2))
    vmax = float(np.nanpercentile(arr_ge_dsm, 98))
  hmin, hmax = float(sample['hmin']), float(sample['hmax'])

  return {
      'sample': sample,
      'vmin': vmin,
      'vmax': vmax,
      'hmin': hmin,
      'hmax': hmax,
      'rgb': rgb_arr,
      'ge_dsm': _render_shaded_relief(arr_ge_dsm, vmin, vmax),
      'ge_p95': _render_shaded_relief(arr_ge_p95, vmin, vmax),
      'ge_p5': _render_shaded_relief(arr_ge_p5, vmin, vmax),
      'ge_dtm': _render_shaded_relief(arr_ge_dtm, vmin, vmax),
      'ge_h': _render_heightmap(arr_ge_h, hmin, hmax),
      'cop_dsm': _render_shaded_relief(arr_cop_dsm, vmin, vmax),
      'alos_dsm': _render_shaded_relief(arr_alos_dsm, vmin, vmax),
      'nasa_dsm': _render_shaded_relief(arr_nasa_dsm, vmin, vmax),
      'fab_dtm': _render_shaded_relief(arr_fab_dtm, vmin, vmax),
      'cop_h': _render_heightmap(arr_cop_h, hmin, hmax),
  }


def render_showcase_figure(
    bundles, out_path
):
  """Renders the 3-sample qualitative comparison figure (2 rows x 6 cols per sample)."""
  fig = plt.figure(figsize=(21.5, 19.2), dpi=180, facecolor='white')
  outer_gs = gridspec.GridSpec(
      3,
      1,
      figure=fig,
      hspace=0.10,
      left=0.012,
      right=0.988,
      top=0.965,
      bottom=0.012,
  )

  row_bottom_axes = []
  row_top_axes = []

  for s_idx, bundle in enumerate(bundles):
    sample = bundle['sample']
    vmin, vmax = bundle['vmin'], bundle['vmax']
    hmin, hmax = bundle['hmin'], bundle['hmax']
    panel_letter = chr(65 + s_idx)

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        6,
        subplot_spec=outer_gs[s_idx],
        wspace=0.025,
        hspace=0.11,
    )

    # Row 1: Optical + 5 GEM-15 Layers (DSM, P95, P5, DTM, Heightmap)
    row1_panels = [
        ('Pléiades Neo RGB (30cm)', bundle['rgb'], '#222222', 'normal', False),
        ('GEM-15 DSM (15m)', bundle['ge_dsm'], '#d95f02', 'bold', True),
        ('GEM-15 P95 (15m)', bundle['ge_p95'], '#d95f02', 'bold', True),
        ('GEM-15 P5 (15m)', bundle['ge_p5'], '#d95f02', 'bold', True),
        ('GEM-15 DTM (15m)', bundle['ge_dtm'], '#d95f02', 'bold', True),
        (
            'GEM-15 Heightmap (DSM − DTM)',
            bundle['ge_h'],
            '#d95f02',
            'bold',
            True,
        ),
    ]
    for col_idx, (
        title,
        img_arr,
        title_color,
        title_weight,
        is_gem,
    ) in enumerate(row1_panels):
      ax = fig.add_subplot(inner_gs[0, col_idx])
      ax.imshow(img_arr)
      if col_idx == 0:
        row_top_axes.append(ax)
        width_m = 2.0 * float(sample.get('buffer_m', 400.0))
        utils.add_rgb_scalebar(ax, width_m, fontsize=9.0)
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
        spine.set_edgecolor('#d95f02' if is_gem else '#888888')
        spine.set_linewidth(2.0 if is_gem else 0.8)

    # Row 2 Col 0: Clean borderless metadata + Elevation & Heightmap Colorbars
    ax_info = fig.add_subplot(inner_gs[1, 0])
    row_bottom_axes.append(ax_info)
    ax_info.axis('off')
    lat_val, lon_val = float(sample['lat']), float(sample['lon'])
    lat_str = (
        f'{abs(lat_val):.3f}°N' if lat_val >= 0 else f'{abs(lat_val):.3f}°S'
    )
    lon_str = (
        f'{abs(lon_val):.3f}°E' if lon_val >= 0 else f'{abs(lon_val):.3f}°W'
    )
    wrapped_title = textwrap.fill(
        f"({panel_letter}) {sample['title']}", width=24
    )
    ax_info.text(
        0.5,
        0.95,
        wrapped_title,
        transform=ax_info.transAxes,
        fontsize=12.5,
        fontweight='bold',
        color='#111111',
        va='top',
        ha='center',
        linespacing=1.22,
    )
    ax_info.text(
        0.5,
        0.72,
        f"{sample['location']}\n{lat_str}, {lon_str}",
        transform=ax_info.transAxes,
        fontsize=11.5,
        fontweight='normal',
        color='#333333',
        va='top',
        ha='center',
        linespacing=1.30,
    )

    # Per-site Orthometric Elevation colorbar inside Col 0
    cbar_elev_ax = ax_info.inset_axes([0.08, 0.38, 0.84, 0.085])
    sm_elev = plt.cm.ScalarMappable(
        cmap=CMAP_PAPER, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm_elev._A = []  # pylint: disable=protected-access
    cbar_elev = fig.colorbar(
        sm_elev, cax=cbar_elev_ax, orientation='horizontal'
    )
    cbar_elev.set_label(
        'Orthometric Elevation (m, EGM96)',
        fontsize=10.5,
        fontweight='normal',
        labelpad=3,
    )
    cbar_elev.ax.tick_params(labelsize=9.5)

    # Height Above Ground colorbar inside Col 0
    cbar_h_ax = ax_info.inset_axes([0.08, 0.08, 0.84, 0.085])
    sm_h = plt.cm.ScalarMappable(
        cmap=CMAP_HEIGHT, norm=plt.Normalize(vmin=hmin, vmax=hmax)
    )
    sm_h._A = []  # pylint: disable=protected-access
    cbar_h = fig.colorbar(sm_h, cax=cbar_h_ax, orientation='horizontal')
    cbar_h.set_label(
        'Height Above Ground (m)',
        fontsize=10.5,
        fontweight='normal',
        labelpad=3,
    )
    cbar_h.ax.tick_params(labelsize=9.5)

    # Row 2 Cols 1-5: Global 30m Benchmarks aligned beneath Row 1
    row2_panels = [
        (1, 'Copernicus DSM (30m)', bundle['cop_dsm']),
        (2, 'ALOS AW3D30 DSM (30m)', bundle['alos_dsm']),
        (3, 'NASADEM DSM (30m)', bundle['nasa_dsm']),
        (4, 'FABDEM DTM (30m)', bundle['fab_dtm']),
        (5, 'Copernicus Heightmap (30m)', bundle['cop_h']),
    ]
    for col_idx, title, img_arr in row2_panels:
      ax = fig.add_subplot(inner_gs[1, col_idx])
      ax.imshow(img_arr)
      ax.set_title(
          title,
          fontsize=12,
          fontweight='normal',
          color='#222222',
          pad=3.5,
      )
      ax.set_xticks([])
      ax.set_yticks([])
      for spine in ax.spines.values():
        spine.set_edgecolor('#888888')
        spine.set_linewidth(0.8)

  # Draw subtle dashed horizontal dividers between the 3 samples
  fig.canvas.draw()
  for idx in range(2):
    y_bot = row_bottom_axes[idx].get_position().y0
    y_top = row_top_axes[idx + 1].get_position().y1
    y_mid = 0.5 * (y_bot + y_top) + 0.004
    fig.add_artist(
        plt.Line2D(
            [0.015, 0.985],
            [y_mid, y_mid],
            color='#90a4ae',
            linestyle='--',
            linewidth=1.2,
            transform=fig.transFigure,
        )
    )

  parent_dir = os.path.dirname(out_path)
  if parent_dir and not os.path.exists(parent_dir):
    os.makedirs(parent_dir, exist_ok=True)
  buf = io.BytesIO()
  fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
  plt.close(fig)
  with open(out_path, 'wb') as f:
    f.write(buf.getvalue())
  print(f'Saved qualitative GEM-15 showcase figure: {out_path}')


VANCOUVER_SAMPLE: dict[str, Any] = {
    'id': 'sample4_vancouver_nz',
    'title': 'Suburban Shelterbelt & Residential',
    'location': 'Vancouver Cres, New Zealand',
    'lon': 172.6931,
    'lat': -43.5226,
    'buffer_m': 450.0,
    'rect': [172.6871, -43.5268, 172.6992, -43.5184],
    'vmin': 10.0,
    'vmax': 50.0,
    'hmin': 0.0,
    'hmax': 35.0,
    'rgb_min': 15,
    'rgb_max': 185,
}


def _get_ee_thumb(
    ee_img,
    geom,
    vmin,
    vmax,
    palette,
    dims = '700x700',
):
  """Fetches a direct Earth Engine rendered PNG thumbnail for an elevation layer."""
  vis = ee_img.visualize(min=vmin, max=vmax, palette=palette)
  url = vis.getThumbURL({'dimensions': dims, 'region': geom, 'format': 'png'})
  raw = urllib.request.urlopen(url, timeout=60).read()
  return np.array(Image.open(io.BytesIO(raw)).convert('RGB'))


# 9 strictly evenly-spaced stops across [0m, 50m] (every 6.25m):
NZ_EVEN_HEX = [
    '#f7f2a6',  # 0.00m (yellow)
    '#65c48c',  # 6.25m (green-teal)
    '#2585d6',  # 12.50m (cobalt blue)
    '#4f3ac8',  # 18.75m (deep indigo)
    '#8e249d',  # 25.00m (purple-magenta)
    '#cc2148',  # 31.25m (crimson red)
    '#eb561e',  # 37.50m (bright orange-red)
    '#fba922',  # 43.75m (amber-gold)
    '#fee038',  # 50.00m (bright yellow-gold)
]
CMAP_NZ_EVEN = LinearSegmentedColormap.from_list('nz_even', NZ_EVEN_HEX, N=256)

# Full-spectrum perceptual Turbo with compressed low-end stops on [10m, 50m]
NZ_TURBO_FULL_HEX = [
    '#30123b',  # 10.0m
    '#4145ab',  # 12.7m
    '#4675ed',  # 15.3m (bare ground / streets / GEM-15 DTM)
    '#2db7f5',  # 18.0m (lower house roofs)
    '#1ae4b6',  # 20.7m (suburban house roofs)
    '#52f667',  # 23.3m (tall roofs / FABDEM tree remnant)
    '#a4fc3c',  # 26.0m
    '#d9ef34',  # 28.7m
    '#f9c932',  # 31.3m
    '#fe9b2d',  # 34.0m
    '#f36517',  # 36.7m
    '#df3807',  # 39.3m
    '#c11c02',  # 42.0m
    '#9e0c01',  # 44.7m
    '#7a0403',  # 47.3m
    '#540202',  # 50.0m
]
CMAP_NZ_TURBO_FULL = LinearSegmentedColormap.from_list(
    'nz_turbo_full', NZ_TURBO_FULL_HEX, N=256
)


def fetch_vancouver_no_lidar_bundle(
    sample,
    palette_hex = None,
    cmap_obj = CMAP_NZ_EVEN,
):
  """Fetches all layers directly from Earth Engine via getThumbURL on [10, 50]."""
  if palette_hex is None:
    palette_hex = NZ_EVEN_HEX
  print(
      'Fetching layers directly from Earth Engine for'
      f" [{sample['location']}]..."
  )
  lon_val, lat_val = float(sample['lon']), float(sample['lat'])
  buf_m = float(sample.get('buffer_m', 450.0))
  box = ee.Geometry.Point([lon_val, lat_val]).buffer(buf_m).bounds()
  utm_crs = utils.get_utm_crs(lon_val, lat_val)
  vmin, vmax = float(sample['vmin']), float(sample['vmax'])
  hmin, hmax = float(sample['hmin']), float(sample['hmax'])

  gem15 = ee.ImageCollection(ee_assets_config.GEM15_ASSET_ID)
  egm96 = ee.Image(ee_assets_config.GEOID_EGM96_ASSET_ID).select('b1')
  egm2008 = ee.Image(ee_assets_config.GEOID_EGM2008_ASSET_ID).select('b1')
  nz_col = ee.ImageCollection(ee_assets_config.LIDAR_NEW_ZEALAND_ASSET_ID)
  nz_img = nz_col.mosaic()
  nz_geoid = ee.Image(ee_assets_config.GEOID_NZ_NZVD2016_ASSET_ID).select('b1')

  # Native rasters referenced to WGS84 ellipsoidal height (+egm96/+egm2008/nz)
  # and reprojected to local UTM 15m/30m grids so pixel blocks match ground
  # meters.
  lidar_dsm = nz_img.select('dsm').add(nz_geoid)
  lidar_dtm = nz_img.select('dem').add(nz_geoid)
  gem_dsm = (
      gem15.select('dsm').mosaic().add(egm96).reproject(crs=utm_crs, scale=15)
  )
  gem_p95 = (
      gem15.select('dsm_percentile_95')
      .mosaic()
      .add(egm96)
      .reproject(crs=utm_crs, scale=15)
  )
  gem_p5 = (
      gem15.select('dsm_percentile_5')
      .mosaic()
      .add(egm96)
      .reproject(crs=utm_crs, scale=15)
  )
  gem_dtm = (
      gem15.select('dtm').mosaic().add(egm96).reproject(crs=utm_crs, scale=15)
  )

  cop_dem = (
      ee.ImageCollection(ee_assets_config.COPERNICUS_GLO30_ASSET_ID)
      .mosaic()
      .select('DEM')
      .add(egm2008)
      .reproject(crs=utm_crs, scale=30)
  )
  alos_dem = (
      ee.ImageCollection(ee_assets_config.ALOS_AW3D30_ASSET_ID)
      .select('DSM')
      .mosaic()
      .add(egm96)
      .reproject(crs=utm_crs, scale=30)
  )
  nasa_dem = (
      ee.Image(ee_assets_config.NASADEM_ASSET_ID)
      .select('elevation')
      .add(egm96)
      .reproject(crs=utm_crs, scale=30)
  )
  fab_dem = (
      ee.ImageCollection(ee_assets_config.FABDEM_ASSET_ID)
      .mosaic()
      .select('b1')
      .add(egm2008)
      .reproject(crs=utm_crs, scale=30)
  )

  gem_h = (
      gem_dsm.subtract(gem_dtm).clamp(0, hmax).reproject(crs=utm_crs, scale=15)
  )
  cop_h = (
      cop_dem.subtract(fab_dem).clamp(0, hmax).reproject(crs=utm_crs, scale=30)
  )

  rgb_arr = _fetch_pneo_rgb(box, sample['rgb_min'], sample['rgb_max'], dims=700)

  return {
      'sample': sample,
      'vmin': vmin,
      'vmax': vmax,
      'hmin': hmin,
      'hmax': hmax,
      'cmap': cmap_obj,
      'rgb': rgb_arr,
      'lidar_dsm': _get_ee_thumb(lidar_dsm, box, vmin, vmax, palette_hex),
      'lidar_dtm': _get_ee_thumb(lidar_dtm, box, vmin, vmax, palette_hex),
      'ge_dsm': _get_ee_thumb(gem_dsm, box, vmin, vmax, palette_hex),
      'ge_p95': _get_ee_thumb(gem_p95, box, vmin, vmax, palette_hex),
      'ge_p5': _get_ee_thumb(gem_p5, box, vmin, vmax, palette_hex),
      'ge_dtm': _get_ee_thumb(gem_dtm, box, vmin, vmax, palette_hex),
      'ge_h': _get_ee_thumb(gem_h, box, hmin, hmax, HEIGHT_HEX),
      'cop_dsm': _get_ee_thumb(cop_dem, box, vmin, vmax, palette_hex),
      'alos_dsm': _get_ee_thumb(alos_dem, box, vmin, vmax, palette_hex),
      'nasa_dsm': _get_ee_thumb(nasa_dem, box, vmin, vmax, palette_hex),
      'fab_dtm': _get_ee_thumb(fab_dem, box, vmin, vmax, palette_hex),
      'cop_h': _get_ee_thumb(cop_h, box, hmin, hmax, HEIGHT_HEX),
  }


def render_single_vancouver_no_lidar(
    bundle, out_dir
):
  """Renders standalone 2-row figures for Vancouver Cres, NZ without LiDAR."""
  sample = bundle['sample']
  vmin, vmax = bundle['vmin'], bundle['vmax']
  lat_val, lon_val = float(sample['lat']), float(sample['lon'])
  lat_str = f'{abs(lat_val):.3f}°N' if lat_val >= 0 else f'{abs(lat_val):.3f}°S'
  lon_str = f'{abs(lon_val):.3f}°E' if lon_val >= 0 else f'{abs(lon_val):.3f}°W'

  vancouver_width_m = 2.0 * float(sample.get('buffer_m', 450.0))

  # Variant A: 2 rows x 5 cols (Row 1: RGB + 4 GEM-15; Row 2: Cbar + Baselines)
  fig5 = plt.figure(figsize=(17.8, 6.9), dpi=180, facecolor='white')
  gs5 = gridspec.GridSpec(
      2,
      5,
      figure=fig5,
      wspace=0.025,
      hspace=0.11,
      left=0.012,
      right=0.988,
      top=0.92,
      bottom=0.03,
  )
  r1_5col = [
      ('Pléiades Neo RGB (30cm)', bundle['rgb'], '#222222', 'normal', False),
      ('GEM-15 DSM (15m)', bundle['ge_dsm'], '#d95f02', 'bold', True),
      ('GEM-15 P95 (15m)', bundle['ge_p95'], '#d95f02', 'bold', True),
      ('GEM-15 P5 (15m)', bundle['ge_p5'], '#d95f02', 'bold', True),
      ('GEM-15 DTM (15m)', bundle['ge_dtm'], '#d95f02', 'bold', True),
  ]
  for col_idx, (title, img_arr, title_color, title_weight, is_gem) in enumerate(
      r1_5col
  ):
    ax = fig5.add_subplot(gs5[0, col_idx])
    ax.imshow(img_arr)
    if col_idx == 0:
      utils.add_rgb_scalebar(ax, vancouver_width_m, fontsize=9.2)
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
      spine.set_edgecolor('#d95f02' if is_gem else '#888888')
      spine.set_linewidth(2.2 if is_gem else 0.8)

  r2_5col = [
      (1, 'Copernicus DSM (30m)', bundle['cop_dsm']),
      (2, 'ALOS AW3D30 DSM (30m)', bundle['alos_dsm']),
      (3, 'NASADEM DSM (30m)', bundle['nasa_dsm']),
      (4, 'FABDEM DTM (30m)', bundle['fab_dtm']),
  ]
  for col_idx, title, img_arr in r2_5col:
    ax = fig5.add_subplot(gs5[1, col_idx])
    ax.imshow(img_arr)
    ax.set_title(
        title,
        fontsize=12.5,
        fontweight='normal',
        color='#222222',
        pad=4.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_edgecolor('#888888')
      spine.set_linewidth(0.8)

  ax_info5 = fig5.add_subplot(gs5[1, 0])
  ax_info5.axis('off')
  ax_info5.text(
      0.5,
      0.72,
      sample['location'],
      transform=ax_info5.transAxes,
      fontsize=13,
      fontweight='bold',
      color='#111111',
      va='bottom',
      ha='center',
  )
  ax_info5.text(
      0.5,
      0.67,
      f'{lat_str}, {lon_str}',
      transform=ax_info5.transAxes,
      fontsize=11.5,
      fontweight='normal',
      color='#333333',
      va='top',
      ha='center',
  )
  cbar_ax5 = ax_info5.inset_axes([0.06, 0.25, 0.88, 0.11])
  sm5 = plt.cm.ScalarMappable(
      cmap=bundle.get('cmap', CMAP_NZ_EVEN),
      norm=plt.Normalize(vmin=vmin, vmax=vmax),
  )
  sm5._A = []  # pylint: disable=protected-access
  cbar5 = fig5.colorbar(sm5, cax=cbar_ax5, orientation='horizontal')
  cbar5.set_ticks([10, 20, 30, 40, 50])
  cbar5.set_label(
      'Orthometric Elevation (m, EGM96)',
      fontsize=11,
      fontweight='normal',
      labelpad=3,
  )
  cbar5.ax.tick_params(labelsize=10)

  out_path5 = os.path.join(
      out_dir,
      bundle.get(
          'out_filename', 'vancouver_cres_new_zealand_no_lidar_5col.png'
      ),
  )
  buf5 = io.BytesIO()
  fig5.savefig(buf5, format='png', dpi=180, bbox_inches='tight')
  plt.close(fig5)
  with open(out_path5, 'wb') as f:
    f.write(buf5.getvalue())
  print(f'Saved Vancouver Cres 5-col (no LiDAR): {out_path5}')

  # Variant B: 2 rows x 6 cols WITH 1m LiDAR DSM + DTM
  fig6 = plt.figure(figsize=(21.4, 6.9), dpi=180, facecolor='white')
  gs6 = gridspec.GridSpec(
      2,
      6,
      figure=fig6,
      wspace=0.025,
      hspace=0.11,
      left=0.012,
      right=0.988,
      top=0.92,
      bottom=0.03,
  )
  r1_6col = [
      ('Pléiades Neo RGB (30cm)', bundle['rgb'], '#222222', 'normal', False),
      ('LiDAR DSM GT (1m)', bundle['lidar_dsm'], '#222222', 'normal', False),
      ('GEM-15 DSM (15m)', bundle['ge_dsm'], '#d95f02', 'bold', True),
      ('GEM-15 P95 (15m)', bundle['ge_p95'], '#d95f02', 'bold', True),
      ('GEM-15 P5 (15m)', bundle['ge_p5'], '#d95f02', 'bold', True),
      ('GEM-15 DTM (15m)', bundle['ge_dtm'], '#d95f02', 'bold', True),
  ]
  for col_idx, (title, img_arr, title_color, title_weight, is_gem) in enumerate(
      r1_6col
  ):
    ax = fig6.add_subplot(gs6[0, col_idx])
    ax.imshow(img_arr)
    if col_idx == 0:
      utils.add_rgb_scalebar(ax, vancouver_width_m, fontsize=9.2)
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
      spine.set_edgecolor('#d95f02' if is_gem else '#888888')
      spine.set_linewidth(2.2 if is_gem else 0.8)

  r2_6col = [
      (1, 'LiDAR DTM GT (1m)', bundle['lidar_dtm']),
      (2, 'Copernicus DSM (30m)', bundle['cop_dsm']),
      (3, 'ALOS AW3D30 DSM (30m)', bundle['alos_dsm']),
      (4, 'NASADEM DSM (30m)', bundle['nasa_dsm']),
      (5, 'FABDEM DTM (30m)', bundle['fab_dtm']),
  ]
  for col_idx, title, img_arr in r2_6col:
    ax = fig6.add_subplot(gs6[1, col_idx])
    ax.imshow(img_arr)
    ax.set_title(
        title,
        fontsize=12.5,
        fontweight='normal',
        color='#222222',
        pad=4.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_edgecolor('#888888')
      spine.set_linewidth(0.8)

  ax_info6 = fig6.add_subplot(gs6[1, 0])
  ax_info6.axis('off')
  ax_info6.text(
      0.5,
      0.72,
      sample['location'],
      transform=ax_info6.transAxes,
      fontsize=13,
      fontweight='bold',
      color='#111111',
      va='bottom',
      ha='center',
  )
  ax_info6.text(
      0.5,
      0.67,
      f'{lat_str}, {lon_str}',
      transform=ax_info6.transAxes,
      fontsize=11.5,
      fontweight='normal',
      color='#333333',
      va='top',
      ha='center',
  )
  cbar_elev_ax6 = ax_info6.inset_axes([0.06, 0.25, 0.88, 0.11])
  sm_elev6 = plt.cm.ScalarMappable(
      cmap=bundle.get('cmap', CMAP_NZ_EVEN),
      norm=plt.Normalize(vmin=vmin, vmax=vmax),
  )
  sm_elev6._A = []  # pylint: disable=protected-access
  cbar_elev6 = fig6.colorbar(
      sm_elev6, cax=cbar_elev_ax6, orientation='horizontal'
  )
  cbar_elev6.set_ticks([10, 20, 30, 40, 50])
  cbar_elev6.set_label(
      'Orthometric Elevation (m, EGM96)',
      fontsize=11,
      fontweight='normal',
      labelpad=3,
  )
  cbar_elev6.ax.tick_params(labelsize=10)

  out_path6 = os.path.join(
      out_dir, 'vancouver_cres_new_zealand_with_lidar_6col.png'
  )
  buf6 = io.BytesIO()
  fig6.savefig(buf6, format='png', dpi=180, bbox_inches='tight')
  plt.close(fig6)
  with open(out_path6, 'wb') as f:
    f.write(buf6.getvalue())
  print(f'Saved Vancouver Cres 6-col (with 1m LiDAR DSM + DTM): {out_path6}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  utils.authenticate_earth_engine(
      ee_service_account=FLAGS.ee_service_account,
      project=FLAGS.ee_project,
  )

  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    showcase_bundles = list(executor.map(fetch_sample_bundle, SHOWCASE_SAMPLES))
  render_showcase_figure(
      showcase_bundles,
      os.path.join(FLAGS.output_dir, 'qualitative_gem15_showcase_3samples.png'),
  )

  vancouver_bundle = fetch_vancouver_no_lidar_bundle(
      VANCOUVER_SAMPLE, NZ_EVEN_HEX, CMAP_NZ_EVEN
  )
  vancouver_bundle['out_filename'] = (
      'vancouver_cres_new_zealand_no_lidar_5col.png'
  )
  render_single_vancouver_no_lidar(vancouver_bundle, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
