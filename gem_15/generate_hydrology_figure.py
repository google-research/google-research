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

"""Generates hydrology and stream pathway figure for GEM-15."""

from collections.abc import Sequence
import heapq
import io
import os
import urllib.request

from absl import app
from absl import flags
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import splev
from scipy.interpolate import splprep

import ee
from gem_15 import ee_assets_config
from gem_15 import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "ee_project",
    None,
    "Google Cloud project ID for Earth Engine.",
    required=True,
)
flags.DEFINE_string(
    "ee_service_account",
    None,
    "Optional service account principal for Earth Engine.",
)
flags.DEFINE_string(
    "output_dir",
    "/tmp",
    "Directory where output figures will be saved.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # 1. Earth Engine Authentication
  utils.authenticate_earth_engine(
      project=FLAGS.ee_project, ee_service_account=FLAGS.ee_service_account
  )

  # Colormap
  turbo_hex = [
      "#fadd67",
      "#b8b257",
      "#7ca88f",
      "#4aa4ca",
      "#3869d1",
      "#6f5bc8",
      "#b862b0",
      "#bc3a6b",
      "#cb3438",
      "#f08429",
      "#fbd227",
  ]
  cmap_paper = LinearSegmentedColormap.from_list(
      "paper_turbo", turbo_hex, N=256
  )

  # 2. Datasets
  gem15 = ee.ImageCollection(ee_assets_config.GEM15_ASSET_ID)
  gem_dsm = gem15.select("dsm").mosaic()
  gem_dtm = gem15.select("dtm").mosaic()

  egm96 = ee.Image(ee_assets_config.GEOID_EGM96_ASSET_ID).select("b1")
  egm2008 = ee.Image(ee_assets_config.GEOID_EGM2008_ASSET_ID).select("b1")

  cop_raw = (
      ee.ImageCollection(ee_assets_config.COPERNICUS_GLO30_ASSET_ID)
      .mosaic()
      .select("DEM")
  )
  cop_dem = cop_raw.add(egm2008).subtract(egm96)

  fab_raw = (
      ee.ImageCollection(ee_assets_config.FABDEM_ASSET_ID).mosaic().select("b1")
  )
  fab_dem = fab_raw.add(egm2008).subtract(egm96)

  osgm15 = ee.Image(ee_assets_config.GEOID_UK_OSGM15_ASSET_ID).select("b1")
  ea_img = ee.Image(ee_assets_config.LIDAR_UK_EA_ASSET_ID)
  ea_dtm = ea_img.select("dtm").add(osgm15).subtract(egm96)

  pneo_col = ee.ImageCollection(ee_assets_config.HIGH_RES_OPTICAL_RGB_ASSET_ID)

  center_lon, center_lat = -2.983, 52.131
  utm_crs = utils.get_utm_crs(center_lon, center_lat)
  buf = 340
  box = ee.Geometry.Point([center_lon, center_lat]).buffer(buf).bounds()
  coords = box.coordinates().getInfo()[0]
  w, s, e, n = coords[0][0], coords[0][1], coords[2][0], coords[2][1]

  gem_dsm = gem_dsm.reproject(crs=utm_crs, scale=15)
  gem_dtm = gem_dtm.reproject(crs=utm_crs, scale=15)
  cop_dem = cop_dem.reproject(crs=utm_crs, scale=30)
  fab_dem = fab_dem.reproject(crs=utm_crs, scale=30)

  # 3. Hydro-enforced thalweg routing on 2.5m LiDAR grid
  dtm_res = ea_dtm.reproject(crs="EPSG:4326", scale=2.5)
  grid = dtm_res.sampleRectangle(region=box, defaultValue=-9999).getInfo()[
      "properties"
  ]["dtm"]
  lidar = np.array(grid, dtype=np.float32)
  lidar[lidar == -9999] = np.nan
  h_rows, w_cols = lidar.shape

  def dijkstra_path(r1, c1, r2, c2, grid_z):
    """Computes least-cost drainage pathway using Dijkstra's algorithm."""
    dist = np.full((h_rows, w_cols), np.inf)
    parent = {}
    dist[r1, c1] = 0.0
    pq = [(0.0, r1, c1)]
    while pq:
      dist_val, row_idx, col_idx = heapq.heappop(pq)
      if dist_val > dist[row_idx, col_idx]:
        continue
      if (row_idx, col_idx) == (r2, c2):
        break
      for dr, dc in [
          (-1, 0),
          (1, 0),
          (0, -1),
          (0, 1),
          (-1, -1),
          (-1, 1),
          (1, -1),
          (1, 1),
      ]:
        nr, nc = row_idx + dr, col_idx + dc
        if (
            0 <= nr < h_rows
            and 0 <= nc < w_cols
            and not np.isnan(grid_z[nr, nc])
        ):
          dz = grid_z[nr, nc] - grid_z[row_idx, col_idx]
          geom_d = np.sqrt(dr * dr + dc * dc)
          cost = geom_d * (
              1.0 + (grid_z[nr, nc] - 62.0) * 0.5 + max(0.0, dz) * 1000.0
          )
          if dist[row_idx, col_idx] + cost < dist[nr, nc]:
            dist[nr, nc] = dist[row_idx, col_idx] + cost
            parent[(nr, nc)] = (row_idx, col_idx)
            heapq.heappush(pq, (dist[nr, nc], nr, nc))
    curr = (r2, c2)
    path = [curr]
    while curr in parent:
      curr = parent[curr]
      path.append(curr)
    path.reverse()
    return path

  p_top = dijkstra_path(0, 137, 70, 225, lidar)
  culvert_steps = 15
  p_culvert = [
      (
          int(round(70 + (81 - 70) * i / culvert_steps)),
          int(round(225 + (308 - 225) * i / culvert_steps)),
      )
      for i in range(1, culvert_steps)
  ]
  p_bot = dijkstra_path(81, 308, 272, 251, lidar)
  high_prob_path = p_top + p_culvert + p_bot

  step = max(1, len(high_prob_path) // 35)
  key_coords = []
  for r, c in high_prob_path[::step]:
    lo = w + (c / (w_cols - 1)) * (e - w)
    la = n - (r / (h_rows - 1)) * (n - s)
    key_coords.append([lo, la])
  if high_prob_path[-1] not in high_prob_path[::step]:
    r, c = high_prob_path[-1]
    key_coords.append(
        [w + (c / (w_cols - 1)) * (e - w), n - (r / (h_rows - 1)) * (n - s)]
    )

  key_coords = np.array(key_coords)
  tck, _ = splprep([key_coords[:, 0], key_coords[:, 1]], s=0.000000005)

  # Tiny margin (~13 px inside borders) so circle markers are completely
  # inside the frame without touching borders.
  u_fine = np.linspace(0.045, 0.955, 120)
  lons_flow, lats_flow = splev(u_fine, tck)

  # 4. Visualizations
  disp_min = 60.0
  disp_max = 95.0
  vis_dem = {"min": disp_min, "max": disp_max, "palette": turbo_hex}

  def fetch_thumb(ee_img, vis, dimensions="700x700"):
    url = ee_img.getThumbURL(
        {**vis, "dimensions": dimensions, "region": box, "format": "png"}
    )
    with urllib.request.urlopen(url) as resp:
      return np.array(Image.open(io.BytesIO(resp.read())))

  # Base Optical RGB
  pneo_img = pneo_col.filterBounds(box).sort("CLOUD_COVER").first()
  vis_pneo = pneo_img.visualize(bands=["R", "G", "B"], min=35, max=210)
  url_rgb = vis_pneo.getThumbURL(
      {"dimensions": "700x700", "region": box, "format": "png"}
  )
  with urllib.request.urlopen(url_rgb) as resp:
    rgb_base = np.array(Image.open(io.BytesIO(resp.read())))

  # Fetch all DSMs and DTMs
  cop_arr = fetch_thumb(cop_dem, vis_dem)
  gem_dsm_arr = fetch_thumb(gem_dsm, vis_dem)
  ea_arr = fetch_thumb(ea_dtm, vis_dem)
  fab_arr = fetch_thumb(fab_dem, vis_dem)
  gem_dtm_arr = fetch_thumb(gem_dtm, vis_dem)

  # Sample along flowline for DTM profiles
  pts = [
      ee.Feature(ee.Geometry.Point([lo, la]), {"idx": i})
      for i, (lo, la) in enumerate(zip(lons_flow, lats_flow))
  ]
  fc = ee.FeatureCollection(pts)
  stack = ee.Image.cat([
      ea_dtm.rename("ea_dtm"),
      fab_dem.rename("fab_dem"),
      gem_dtm.rename("gem_dtm"),
  ])
  samples = stack.sampleRegions(
      collection=fc, scale=2, geometries=False
  ).getInfo()["features"]
  samples.sort(key=lambda f: f["properties"]["idx"])

  def haversine(lon1, lat1, lon2, lat2):
    """Calculates great-circle distance between two geographic coordinates."""
    earth_radius_m = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    )
    return 2 * earth_radius_m * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

  seg_dists = [0.0]
  for i in range(1, len(lons_flow)):
    d = haversine(
        lons_flow[i - 1], lats_flow[i - 1], lons_flow[i], lats_flow[i]
    )
    seg_dists.append(seg_dists[-1] + d)
  dists = np.array(seg_dists)

  ea_prof = np.array([f["properties"].get("ea_dtm", np.nan) for f in samples])
  fab_prof = np.array([f["properties"].get("fab_dem", np.nan) for f in samples])
  gem_dtm_prof = np.array(
      [f["properties"].get("gem_dtm", np.nan) for f in samples]
  )

  # Pixel coordinates for flowline
  scale_x = 700.0 / (w_cols - 1)
  scale_y = 700.0 / (h_rows - 1)
  px_flow = []
  for lo, la in zip(lons_flow, lats_flow):
    c_f = (lo - w) / (e - w) * (w_cols - 1) * scale_x
    r_f = (n - la) / (n - s) * (h_rows - 1) * scale_y
    px_flow.append([c_f, r_f])
  px_flow = np.array(px_flow)

  # 5. Build Figure: 2x3 grid on left, 25% taller plot on right
  fig = plt.figure(figsize=(19, 9.0), dpi=300)

  gs_left = fig.add_gridspec(
      2,
      3,
      left=0.03,
      right=0.60,
      bottom=0.10,
      top=0.90,
      hspace=0.18,
      wspace=0.06,
  )

  gs_right = fig.add_gridspec(
      1,
      1,
      left=0.635,
      right=0.98,
      bottom=0.22,
      top=0.72,
  )

  # Panels arranged with all DSMs and DTMs
  panels = [
      (rgb_base, "(a) Optical (30 cm Pléiades Neo)", False),
      (cop_arr, "(b) Copernicus GLO-30 DSM (30 m)", True),
      (gem_dsm_arr, "(c) GEM-15 v1 DSM (15 m)", True),
      (ea_arr, "(d) UK EA 1 m Airborne LiDAR DTM", True),
      (fab_arr, "(e) FABDEM DTM (30 m)", True),
      (gem_dtm_arr, "(f) GEM-15 v1 DTM (15 m)", True),
  ]

  for idx, (img, title, is_dem) in enumerate(panels):
    r, c = idx // 3, idx % 3
    ax = fig.add_subplot(gs_left[r, c])
    ax.set_aspect("equal")
    ax.imshow(img)
    if not is_dem:
      utils.add_rgb_scalebar(ax, 680.0, fontsize=8.8, loc="bottom_left")
      # Stream path in vibrant blue on Optical
      ax.plot(
          px_flow[:, 0], px_flow[:, 1], color="#023e8a", linewidth=3.6, zorder=8
      )
      ax.plot(
          px_flow[:, 0], px_flow[:, 1], color="#0096c7", linewidth=2.2, zorder=9
      )

      # Source A and Outlet B Markers (clearly inside image frame)
      ax.scatter(
          px_flow[0, 0],
          px_flow[0, 1],
          color="white",
          edgecolor="#023e8a",
          s=85,
          linewidth=1.5,
          zorder=10,
      )
      ax.text(
          px_flow[0, 0] + 50,
          px_flow[0, 1],
          "A (Source)",
          color="white",
          fontsize=8.8,
          fontweight="bold",
          ha="center",
          va="center",
          bbox=dict(facecolor="#023e8a", alpha=0.85, edgecolor="none", pad=2.0),
          zorder=11,
      )
      ax.scatter(
          px_flow[-1, 0],
          px_flow[-1, 1],
          color="white",
          edgecolor="#023e8a",
          s=85,
          linewidth=1.5,
          zorder=10,
      )
      ax.text(
          px_flow[-1, 0] + 82,
          px_flow[-1, 1],
          "B (Outlet)",
          color="white",
          fontsize=8.8,
          fontweight="bold",
          ha="center",
          va="center",
          bbox=dict(facecolor="#023e8a", alpha=0.85, edgecolor="none", pad=2.0),
          zorder=11,
      )
    else:
      # Highlight path with white color on all DEM panels
      ax.plot(
          px_flow[:, 0],
          px_flow[:, 1],
          color="#000000",
          linewidth=3.4,
          alpha=0.7,
          zorder=7,
      )
      ax.plot(
          px_flow[:, 0], px_flow[:, 1], color="#ffffff", linewidth=2.2, zorder=8
      )

      # Markers A and B (clearly inside image frame)
      ax.text(
          px_flow[0, 0],
          px_flow[0, 1],
          "A",
          color="black",
          fontsize=8.5,
          fontweight="bold",
          ha="center",
          va="center",
          bbox=dict(
              boxstyle="circle,pad=0.22", fc="#ffffff", ec="#000000", lw=1.1
          ),
          zorder=10,
      )
      ax.text(
          px_flow[-1, 0],
          px_flow[-1, 1],
          "B",
          color="black",
          fontsize=8.5,
          fontweight="bold",
          ha="center",
          va="center",
          bbox=dict(
              boxstyle="circle,pad=0.22", fc="#ffffff", ec="#000000", lw=1.1
          ),
          zorder=10,
      )

    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_color("#666666")
      spine.set_linewidth(0.8)

  # Right: Profile ONLY showing DTM curves (height increased by 25%)
  ax_prof = fig.add_subplot(gs_right[0, 0])
  ax_prof.plot(
      dists,
      ea_prof,
      label="Reference: UK EA 1 m LiDAR DTM",
      color="#1a1a1a",
      linewidth=2.8,
      zorder=5,
  )
  ax_prof.plot(
      dists,
      fab_prof,
      label="FABDEM DTM (30 m)",
      color="#e66101",
      linewidth=2.2,
      linestyle="--",
      zorder=3,
  )
  ax_prof.plot(
      dists,
      gem_dtm_prof,
      label="GEM-15 v1 DTM (15 m)",
      color="#0077b6",
      linewidth=2.6,
      zorder=4,
  )

  ax_prof.set_xlabel(
      "Downstream Distance along Stream Thalweg A $\\rightarrow$ B (m)",
      fontsize=11.5,
      fontweight="bold",
      labelpad=8,
  )
  ax_prof.set_ylabel(
      "Elevation (m, EGM96)", fontsize=11.5, fontweight="bold", labelpad=8
  )
  ax_prof.set_title(
      "Cross-Section A–B: Hydrodynamic Stream Thalweg Elevation",
      fontsize=12.5,
      fontweight="bold",
      pad=10,
  )
  ax_prof.grid(True, linestyle=":", alpha=0.6)
  ax_prof.legend(
      loc="upper right",
      frameon=True,
      facecolor="white",
      edgecolor="#cccccc",
      fontsize=10.5,
  )
  ax_prof.set_ylim(61.5, 78.0)
  ax_prof.tick_params(labelsize=10.5)

  # Shared Colorbar below DEM panels using exact Paper Turbo Colormap
  cbar_ax = fig.add_axes([0.08, 0.035, 0.45, 0.022])
  sm = plt.cm.ScalarMappable(
      cmap=cmap_paper, norm=plt.Normalize(vmin=disp_min, vmax=disp_max)
  )
  sm.set_array([])
  cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
  cbar.set_label(
      "Elevation (m, EGM96 Geoid)", fontsize=10, fontweight="bold", labelpad=4
  )
  cbar.set_ticks([60, 70, 80, 90, 95])
  cbar.ax.tick_params(labelsize=9)

  plt.suptitle(
      "Environmental Application 1: Hydrology — Bare-Earth DTM Stream Routing &"
      " Hydro-Enforcement (Eardisley, UK)",
      fontsize=14,
      fontweight="bold",
      y=0.97,
  )

  os.makedirs(FLAGS.output_dir, exist_ok=True)
  out_png = os.path.join(
      FLAGS.output_dir, "figure_supp_hydrology_clean_margins.png"
  )
  out_jpg = os.path.join(
      FLAGS.output_dir, "figure_supp_hydrology_clean_margins.jpg"
  )
  plt.savefig(out_png, dpi=300, bbox_inches="tight")
  plt.close()

  im = Image.open(out_png).convert("RGB")
  im.save(out_jpg, "JPEG", quality=88, optimize=True)
  print(f"Saved {out_png} and {out_jpg} successfully.")


if __name__ == "__main__":
  app.run(main)
