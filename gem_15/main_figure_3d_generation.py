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

"""Generates the seamless 3D Digital Twin paper headline figures for GEM-15.

Renders an oblique 3D perspective comparison over the 1.08 km x 1.08 km
Downtown Mandan Civic & Commercial District (US NEON 2025 1m LiDAR, site NOGP):
  - High-Res LiDAR (1m, with RGB): 60cm USDA NAIP optical imagery draped in 3D
    over the 1m US NEON Aerial LiDAR DSM.
  - High-Res LiDAR DSM (1m): 1m US NEON Aerial LiDAR DSM colored purely by the
    DSM elevation colormap + 3D directional hillshading.
  - LiDAR DSM (15m Downsampled): 1m US NEON LiDAR block-averaged onto the exact
    15m x 15m GEM-15 UTM grid.
  - GEM-15 DSM (15m, Ours): 15m GEM-15 Standard DSM (Nearest-Neighbor) in 3D.
  - GEM-15 P95 DSM (15m, Ours): 15m GEM-15 95th-Percentile DSM in 3D.
  - Copernicus GLO-30 (30m), ALOS AW3D30 (30m), NASADEM (30m): Public 30m DEMs
    aligned to the exact 30m x 30m (2x2 of 15m GEM-15) UTM grid so that 1 30m
    pixel equals exactly 4 (2x2) 15m GEM-15 pixels.
"""

import io
import os
import urllib.request

from absl import app
from absl import flags
import google.auth
import google.auth.impersonated_credentials
import matplotlib
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import scipy.ndimage as ndi

import ee
from gem_15 import ee_assets_config

matplotlib.use('Agg')

try:
  import pyvista as pv  # pylint: disable=g-import-not-at-top

  pv.OFF_SCREEN = True
  _HAS_PYVISTA = True
except ImportError:
  pv = None
  _HAS_PYVISTA = False

flags.DEFINE_string(
    'output_dir',
    '/tmp',
    'Directory where the generated 3D headline figures will be saved.',
)
flags.DEFINE_string(
    'cache_dir',
    '/tmp',
    'Directory for caching downloaded 1m Earth Engine NPY arrays.',
)
flags.DEFINE_string(
    'ee_project',
    None,
    'Google Cloud project ID used for Earth Engine initialization.',
)
flags.DEFINE_string(
    'ee_service_account',
    None,
    'Service account principal to impersonate when initializing Earth Engine.',
)

CAM_LARGE = (555.0, 495.0, 3.0, 475.0, 28.0, 35.0)


def get_custom_dsm_cmap():
  """Returns the custom terrain-to-elevation colormap for the 3D scene."""
  c_ground = np.array([239, 234, 222]) / 255.0
  c_low = np.array([218, 224, 200]) / 255.0
  c_teal = np.array([62, 166, 159]) / 255.0
  c_amber = np.array([238, 162, 54]) / 255.0
  c_coral = np.array([224, 90, 71]) / 255.0
  c_crimson = np.array([158, 27, 68]) / 255.0
  c_plum = np.array([88, 18, 76]) / 255.0
  colors = [
      (0.00, c_ground),
      (0.10, c_low),
      (0.24, c_teal),
      (0.45, c_amber),
      (0.68, c_coral),
      (0.88, c_crimson),
      (1.00, c_plum),
  ]
  return LinearSegmentedColormap.from_list('CustomModernLUT', colors, N=512)


PURE_DSM_CMAP = get_custom_dsm_cmap()


def init_earth_engine():
  """Initializes Earth Engine with default or impersonated credentials."""
  if not flags.FLAGS.ee_project:
    raise ValueError(
        'Earth Engine Cloud project (--ee_project) must be specified.'
    )
  if flags.FLAGS.ee_service_account:
    base_creds, _ = google.auth.default()
    impersonated = google.auth.impersonated_credentials.Credentials(
        source_credentials=base_creds,
        target_principal=flags.FLAGS.ee_service_account,
        target_scopes=[
            'https://www.googleapis.com/auth/earthengine',
            'https://www.googleapis.com/auth/cloud-platform',
        ],
    )
    ee.Initialize(credentials=impersonated, project=flags.FLAGS.ee_project)
  else:
    ee.Initialize(project=flags.FLAGS.ee_project)


def extract_1m_npy(
    img,
    name,
    grid_box,
    proj_1m,
    cache_dir,
):
  """Extracts or loads a cached 1m-grid NPY array over the 1080m x 1080m box."""
  cache_path = os.path.join(cache_dir, f'large_mandan_{name}.npy')
  if os.path.exists(cache_path):
    try:
      arr = np.load(cache_path)
      if arr.shape[0] > 1000:
        print(f'Loaded cached {name}: {arr.shape}')
        return arr
    except Exception:  # pylint: disable=broad-except
      pass
  print(f'Downloading 1080m x 1080m 1m NPY for {name}...')
  url = img.reproject(proj_1m).getDownloadURL({
      'region': grid_box,
      'scale': 1.0,
      'crs': 'EPSG:32614',
      'format': 'NPY',
  })
  req = urllib.request.urlopen(url)
  structured = np.load(io.BytesIO(req.read()))
  arr = structured['val'].astype(np.float32)
  np.save(cache_path, arr)
  return arr


def block_reduce_to_grid(
    arr_1m,
    step,
    ox,
    oy,
    mode = 'mean',
):
  """Block-reduces a 1m array onto an exact (step x step) UTM grid."""
  h_dim, w_dim = arr_1m.shape
  out = np.copy(arr_1m)
  ys = [0] + list(range(oy if oy > 0 else step, h_dim, step)) + [h_dim]
  xs = [0] + list(range(ox if ox > 0 else step, w_dim, step)) + [w_dim]
  for i in range(len(ys) - 1):
    y1, y2 = ys[i], ys[i + 1]
    if y2 <= y1:
      continue
    for j in range(len(xs) - 1):
      x1, x2 = xs[j], xs[j + 1]
      if x2 <= x1:
        continue
      patch = arr_1m[y1:y2, x1:x2]
      if mode == 'mean':
        out[y1:y2, x1:x2] = np.mean(patch)
      elif mode == 'p95':
        out[y1:y2, x1:x2] = np.percentile(patch, 95)
  return out


def prep_grid_dsm(
    arr,
    street_mask,
    vp,
):
  """Aligns vertical datum to the shared bare-street level so all 3D tiles sit at the exact same Z elevation."""
  datum = float(np.median(arr[vp][street_mask]))
  rel = np.clip(arr - datum, 0.0, 15.5)
  return ndi.gaussian_filter(rel, sigma=0.35)


def compute_hillshade_hi(
    z_val,
    x0 = 0,
    y0 = 0,
    grid_step = None,
):
  """Computes directional hillshade + optional native pixel-grid seam cue."""
  h_dim, w_dim = z_val.shape
  dz_dx = np.gradient(z_val, 1.0, axis=1)
  dz_dy = np.gradient(z_val, 1.0, axis=0)
  slope = np.arctan(np.hypot(dz_dx, dz_dy))
  aspect = np.arctan2(dz_dy, -dz_dx)
  az, alt = np.radians(315.0), np.radians(38.0)
  hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(
      az - aspect
  )
  ao = 1.0 / (1.0 + 0.065 * (np.hypot(dz_dx, dz_dy) ** 1.25))
  hs_norm = np.clip((0.56 + 0.64 * hs) * (0.75 + 0.25 * ao), 0.34, 1.18)

  if grid_step is not None:
    seam = np.ones_like(hs_norm)
    yy, xx = np.meshgrid(np.arange(h_dim), np.arange(w_dim), indexing='ij')
    on_edge = (((xx - x0) % grid_step) == 0) | (((yy - y0) % grid_step) == 0)
    seam[on_edge] = 0.915
    hs_norm = hs_norm * seam

  return (
      np.array(
          Image.fromarray((hs_norm * 200).astype(np.uint8)).resize(
              (2600, 2600), Image.BILINEAR
          )
      ).astype(np.float32)
      / 200.0
  )


def render_scene_large(
    z_dsm,
    x_grid,
    y_grid,
    m_opt_hi,
    x0 = 0,
    y0 = 0,
    grid_step = None,
    mode = 'pure_dsm_color',
    cam_params = CAM_LARGE,
    resample_override = None,
):
  """Renders a 3D oblique view of the DSM surface with a strictly locked camera."""
  h_dim, w_dim = z_dsm.shape
  z_val = z_dsm * 1.30
  hs_hi = compute_hillshade_hi(z_val, x0=x0, y0=y0, grid_step=grid_step)

  if mode == 'optical_3d_lidar':
    tex_rgb = np.clip(m_opt_hi * hs_hi[:, :, None], 0, 255).astype(np.uint8)
  else:
    h_norm = np.clip(z_dsm / 12.0, 0.0, 1.0)
    resample_mode = (
        resample_override
        if resample_override is not None
        else (Image.NEAREST if grid_step is not None else Image.BILINEAR)
    )
    h_hi = (
        np.array(
            Image.fromarray((h_norm * 255).astype(np.uint8)).resize(
                (2600, 2600), resample_mode
            )
        ).astype(np.float32)
        / 255.0
    )
    color_lut = (PURE_DSM_CMAP(h_hi)[:, :, :3] * 255.0).astype(np.float32)
    tex_rgb = np.clip(color_lut * hs_hi[:, :, None], 0, 255).astype(np.uint8)

  if _HAS_PYVISTA:
    tex = pv.numpy_to_texture(tex_rgb[::-1, :, :])
    tex.interpolate = True
    grid = pv.StructuredGrid(x_grid, y_grid, z_val)
    grid.texture_map_to_plane(
        origin=[0, 0, 0],
        point_u=[w_dim - 1, 0, 0],
        point_v=[0, h_dim - 1, 0],
        inplace=True,
    )
    p = pv.Plotter(off_screen=True, window_size=[1350, 1020])
    p.set_background('#1e222a')
    p.add_mesh(
        grid,
        texture=tex,
        smooth_shading=True,
        ambient=0.52,
        diffuse=0.65,
        specular=0.08,
        reset_camera=False,
    )
    cx_c, cy_c, cz_c, dist, elev_deg, azim_deg = cam_params
    elev_rad, azim_rad = np.radians(elev_deg), np.radians(azim_deg)
    p.camera.position = (
        cx_c - dist * np.cos(elev_rad) * np.sin(azim_rad),
        cy_c - dist * np.cos(elev_rad) * np.cos(azim_rad),
        cz_c + dist * np.sin(elev_rad),
    )
    p.camera.focal_point = (cx_c, cy_c, cz_c)
    p.camera.up = (0.0, 0.0, 1.0)
    p.camera.view_angle = 30.0 / 1.18
    p.camera.clipping_range = (10.0, 2500.0)
    img = p.screenshot(return_img=True)
    p.close()
    return img

  fig = plt.figure(figsize=(13.5, 10.2), dpi=100, facecolor='#1e222a')
  ax = fig.add_subplot(111, projection='3d', facecolor='#1e222a')
  tex_small = (
      np.array(
          Image.fromarray(tex_rgb).resize((w_dim, h_dim), Image.BILINEAR)
      ).astype(np.float32)
      / 255.0
  )
  ax.plot_surface(
      x_grid[200:800, 200:850],
      y_grid[200:800, 200:850],
      z_val[200:800, 200:850],
      facecolors=tex_small[200:800, 200:850],
      rstride=2,
      cstride=2,
      shade=False,
      antialiased=False,
  )
  ax.set_xlim(200, 850)
  ax.set_ylim(200, 800)
  ax.set_zlim(0.0, 20.15)
  ax.view_init(elev=28.0, azim=-125.0)
  ax.set_box_aspect((1.0, 1.0, 0.08))
  ax.set_axis_off()
  buf = io.BytesIO()
  plt.savefig(buf, format='png', pad_inches=0)
  plt.close(fig)
  buf.seek(0)
  return np.array(Image.open(buf).convert('RGB'))


def compose_mosaic(
    imgs,
    labels,
    badge_styles,
    rows,
    cols,
    figsize,
    out_png,
    out_jpg,
):
  """Composes a multi-panel 3D comparison figure."""
  fig, axes = plt.subplots(
      rows, cols, figsize=figsize, dpi=160, facecolor='#FFFFFF'
  )
  fig.subplots_adjust(
      left=0.008,
      right=0.992,
      top=0.988,
      bottom=0.072,
      wspace=0.010,
      hspace=0.014,
  )

  for idx, ax in enumerate(axes.flat):
    ax.imshow(imgs[idx])
    ax.axis('off')

    border_col, badge_bg, badge_fg = badge_styles[idx]
    lw = (
        2.4
        if ('GEM-15' in labels[idx])
        else (2.0 if 'LiDAR' in labels[idx] else 1.1)
    )
    frame = patches.Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=border_col,
        linewidth=lw,
        clip_on=False,
        zorder=6,
    )
    ax.add_patch(frame)

    ax.text(
        0.50,
        0.045,
        labels[idx],
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=14.5 if cols == 4 else 16.5,
        fontweight='bold',
        color=badge_fg,
        bbox=dict(
            boxstyle='round,pad=0.46,rounding_size=0.52',
            facecolor=badge_bg,
            edgecolor=border_col,
            linewidth=1.4,
            alpha=0.92,
        ),
        zorder=10,
    )

  cbar_ax = fig.add_axes([0.28, 0.033, 0.44, 0.015])
  norm = matplotlib.colors.Normalize(vmin=502.0, vmax=514.0)
  cb = matplotlib.colorbar.ColorbarBase(
      cbar_ax, cmap=PURE_DSM_CMAP, norm=norm, orientation='horizontal'
  )
  cb.set_label(
      'Elevation (m ASL)',
      fontsize=15.5,
      fontweight='bold',
      color='#1E293B',
      labelpad=6,
  )
  cb.ax.tick_params(labelsize=13.0, colors='#334155')

  buf_png = io.BytesIO()
  plt.savefig(buf_png, format='png', dpi=160, facecolor='#FFFFFF')
  plt.close(fig)
  buf_png.seek(0)
  with open(out_png, 'wb') as f_png:
    f_png.write(buf_png.getvalue())

  buf_png.seek(0)
  img_pil = Image.open(buf_png).convert('RGB')
  buf_jpg = io.BytesIO()
  img_pil.save(buf_jpg, format='JPEG', quality=95)
  with open(out_jpg, 'wb') as f_jpg:
    f_jpg.write(buf_jpg.getvalue())
  print(f'Saved seamless paper figure: {out_png}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  init_earth_engine()
  cache_dir = flags.FLAGS.cache_dir
  out_dir = flags.FLAGS.output_dir
  os.makedirs(out_dir, exist_ok=True)

  lon, lat = -100.8895, 46.8272
  proj_utm = ee.Projection('EPSG:32614')
  coords = (
      ee.Geometry.Point([lon, lat])
      .transform(proj_utm, 1.0)
      .coordinates()
      .getInfo()
  )
  cx, cy = coords[0], coords[1]
  half_size = 540.0
  grid_box = ee.Geometry.Rectangle(
      [cx - half_size, cy - half_size, cx + half_size, cy + half_size],
      proj_utm,
      False,
  )
  proj_1m = proj_utm.atScale(1.0)

  egm96 = ee.Image(ee_assets_config.GEOID_EGM96_ASSET_ID).select('b1')
  egm2008 = ee.Image(ee_assets_config.GEOID_EGM2008_ASSET_ID).select('b1')

  neon_col = (
      ee.ImageCollection(ee_assets_config.LIDAR_US_NEON_ASSET_ID)
      .filterDate('2025-01-01', '2026-01-01')
      .filter(ee.Filter.eq('NEON_SITE', 'NOGP'))
      .filterBounds(grid_box)
  )
  neon_dsm = (
      neon_col.mosaic().select('DSM').add(egm2008).subtract(egm96).rename('val')
  )

  gem15_col = ee.ImageCollection(ee_assets_config.GEM15_ASSET_ID).filterBounds(
      grid_box
  )
  gem_dsm_nn = gem15_col.select('dsm').mosaic().rename('val')
  gem_p95_nn = gem15_col.select('dsm_percentile_95').mosaic().rename('val')
  gem_dtm_nn = gem15_col.select('dtm').mosaic().rename('val')

  cop_col = ee.ImageCollection(
      ee_assets_config.COPERNICUS_GLO30_ASSET_ID
  ).filterBounds(grid_box)
  cop_dsm_nn = (
      cop_col.select('DEM').mosaic().add(egm2008).subtract(egm96).rename('val')
  )
  alos_col = ee.ImageCollection(
      ee_assets_config.ALOS_AW3D30_ASSET_ID
  ).filterBounds(grid_box)
  alos_dsm_nn = alos_col.select('DSM').mosaic().rename('val')
  nasa_dsm_nn = (
      ee.Image(ee_assets_config.NASADEM_ASSET_ID)
      .select('elevation')
      .rename('val')
  )

  l_dsm = extract_1m_npy(neon_dsm, 'lidar_dsm', grid_box, proj_1m, cache_dir)[
      ::-1, :
  ]
  g_dsm_nn = extract_1m_npy(
      gem_dsm_nn, 'gem_dsm_nn', grid_box, proj_1m, cache_dir
  )[::-1, :]
  g_p95_nn = extract_1m_npy(
      gem_p95_nn, 'gem_p95_nn', grid_box, proj_1m, cache_dir
  )[::-1, :]
  g_dtm_nn = extract_1m_npy(
      gem_dtm_nn, 'gem_dtm_nn', grid_box, proj_1m, cache_dir
  )[::-1, :]
  c_dsm_raw = extract_1m_npy(
      cop_dsm_nn, 'cop_dsm_nn', grid_box, proj_1m, cache_dir
  )[::-1, :]
  a_dsm_raw = extract_1m_npy(
      alos_dsm_nn, 'alos_dsm_nn', grid_box, proj_1m, cache_dir
  )[::-1, :]
  n_dsm_raw = extract_1m_npy(
      nasa_dsm_nn, 'nasa_dsm_nn', grid_box, proj_1m, cache_dir
  )[::-1, :]

  dx_edges = np.where(np.abs(np.diff(g_dsm_nn[500, :])) > 1e-4)[0] + 1
  dy_edges = np.where(np.abs(np.diff(g_dsm_nn[:, 500])) > 1e-4)[0] + 1
  x0 = int(np.median(dx_edges % 15))
  y0 = int(np.median(dy_edges % 15))

  # Downsample 1m LiDAR to exact 15m x 15m GEM-15 UTM grid, and align 30m DEMs
  # to exact 30m x 30m (2x2 of 15m GEM-15) UTM grid.
  l_dsm_15m = block_reduce_to_grid(l_dsm, 15, x0, y0, mode='mean')
  c_dsm_30m = block_reduce_to_grid(c_dsm_raw, 30, x0, y0, mode='mean')
  a_dsm_30m = block_reduce_to_grid(a_dsm_raw, 30, x0, y0, mode='mean')
  n_dsm_30m = block_reduce_to_grid(n_dsm_raw, 30, x0, y0, mode='mean')

  opt_cache = os.path.join(cache_dir, 'large_mandan_opt.npy')
  if os.path.exists(opt_cache):
    m_opt = np.load(opt_cache)
  else:
    naip = (
        ee.ImageCollection(ee_assets_config.USDA_NAIP_ASSET_ID)
        .filterBounds(grid_box)
        .sort('system:time_start', False)
        .first()
        .select(['R', 'G', 'B'])
    )
    url_opt = naip.getDownloadURL({
        'region': grid_box,
        'scale': 0.6,
        'crs': 'EPSG:32614',
        'format': 'NPY',
    })
    req_opt = urllib.request.urlopen(url_opt)
    s_opt = np.load(io.BytesIO(req_opt.read()))
    m_opt = np.stack([s_opt['R'], s_opt['G'], s_opt['B']], axis=-1).astype(
        np.uint8
    )[::-1, :, :]
    m_opt = ndi.shift(m_opt, (2, -3, 0), order=3, mode='nearest')
    np.save(opt_cache, m_opt)

  h_dim, w_dim = l_dsm.shape
  x_arr = np.arange(w_dim, dtype=np.float32)
  y_arr = np.arange(h_dim, dtype=np.float32)
  x_grid, y_grid = np.meshgrid(x_arr, y_arr)

  mp1, mp99 = np.percentile(m_opt, 1), np.percentile(m_opt, 99)
  m_opt_norm = (
      np.clip(
          (m_opt.astype(np.float32) - mp1) / max(mp99 - mp1, 1e-3), 0.0, 1.0
      )
      ** 0.92
  )
  m_opt_hi = np.array(
      ImageEnhance.Sharpness(
          Image.fromarray((m_opt_norm * 255).astype(np.uint8)).resize(
              (2600, 2600), Image.LANCZOS
          )
      ).enhance(1.35)
  ).astype(np.float32)

  vp = (slice(220, 740), slice(220, 820))
  lid_datum = float(np.percentile(l_dsm[vp], 15))
  street_mask = (l_dsm[vp] - lid_datum) < 0.6
  z_lid_1m = np.clip(l_dsm - lid_datum, 0.0, 15.5)
  z_lid_1m = ndi.gaussian_filter(ndi.median_filter(z_lid_1m, size=3), sigma=0.6)

  # 15m downsampled LiDAR uses the exact same lid_datum as 1m LiDAR:
  z_lid_15m = ndi.gaussian_filter(
      np.clip(l_dsm_15m - lid_datum, 0.0, 15.5), sigma=0.35
  )
  # GEM-15 DSM, P95, and DTM share the exact same GEM-15 bare-street datum:
  gem_datum = float(np.median(g_dtm_nn[vp][street_mask]))
  z_gem_15m = ndi.gaussian_filter(
      np.clip(g_dsm_nn - gem_datum, 0.0, 15.5), sigma=0.35
  )
  z_gem_p95_15m = ndi.gaussian_filter(
      np.clip(g_p95_nn - gem_datum, 0.0, 15.5), sigma=0.35
  )
  z_gem_dtm_15m = ndi.gaussian_filter(
      np.clip(g_dtm_nn - gem_datum, 0.0, 15.5), sigma=0.35
  )
  z_cop_30m = prep_grid_dsm(c_dsm_30m, street_mask, vp)
  z_alos_30m = prep_grid_dsm(a_dsm_30m, street_mask, vp)
  z_nasa_30m = prep_grid_dsm(n_dsm_30m, street_mask, vp)

  img_opt_1m = render_scene_large(
      z_lid_1m,
      x_grid,
      y_grid,
      m_opt_hi,
      x0=x0,
      y0=y0,
      grid_step=15,
      mode='optical_3d_lidar',
      resample_override=Image.BILINEAR,
  )
  img_lid_1m = render_scene_large(
      z_lid_1m,
      x_grid,
      y_grid,
      m_opt_hi,
      x0=x0,
      y0=y0,
      grid_step=15,
      mode='pure_dsm_color',
      resample_override=Image.BILINEAR,
  )
  img_lid_15m = render_scene_large(
      z_lid_15m, x_grid, y_grid, m_opt_hi, x0=x0, y0=y0, grid_step=15
  )
  img_gem_15m = render_scene_large(
      z_gem_15m, x_grid, y_grid, m_opt_hi, x0=x0, y0=y0, grid_step=15
  )
  img_gem_p95_15m = render_scene_large(
      z_gem_p95_15m, x_grid, y_grid, m_opt_hi, x0=x0, y0=y0, grid_step=15
  )
  img_gem_dtm_15m = render_scene_large(
      z_gem_dtm_15m, x_grid, y_grid, m_opt_hi, x0=x0, y0=y0, grid_step=15
  )
  img_cop_30m = render_scene_large(
      z_cop_30m, x_grid, y_grid, m_opt_hi, x0=x0, y0=y0, grid_step=30
  )
  img_alos_30m = render_scene_large(
      z_alos_30m, x_grid, y_grid, m_opt_hi, x0=x0, y0=y0, grid_step=30
  )
  img_nasa_30m = render_scene_large(
      z_nasa_30m, x_grid, y_grid, m_opt_hi, x0=x0, y0=y0, grid_step=30
  )

  style_ref = ('#475569', '#0F172A', '#F8FAFC')
  style_lid = ('#0D9488', '#0F292A', '#99F6E4')
  style_ours = ('#D97706', '#2E1805', '#FDE68A')
  style_pub = ('#64748B', '#1E293B', '#E2E8F0')

  compose_mosaic(
      imgs=[
          img_opt_1m,
          img_lid_1m,
          img_lid_15m,
          img_gem_dtm_15m,
          img_gem_p95_15m,
          img_gem_15m,
          img_nasa_30m,
          img_alos_30m,
          img_cop_30m,
      ],
      labels=[
          '(a) High-Res LiDAR (1m, with RGB)',
          '(b) High-Res LiDAR DSM (1m)',
          '(c) LiDAR DSM (15m Downsampled)',
          '(d) GEM-15 DTM (15m, Ours)',
          '(e) GEM-15 P95 DSM (15m, Ours)',
          '(f) GEM-15 DSM (15m, Ours)',
          '(g) NASADEM (30m)',
          '(h) ALOS AW3D30 (30m)',
          '(i) Copernicus GLO-30 (30m)',
      ],
      badge_styles=[
          style_ref,
          style_lid,
          style_lid,
          style_ours,
          style_ours,
          style_ours,
          style_pub,
          style_pub,
          style_pub,
      ],
      rows=3,
      cols=3,
      figsize=(22.0, 16.8),
      out_png=os.path.join(out_dir, 'main_figure_3d_9tile_3x3_v13.png'),
      out_jpg=os.path.join(out_dir, 'main_figure_3d_9tile_3x3_v13.jpg'),
  )


if __name__ == '__main__':
  app.run(main)
