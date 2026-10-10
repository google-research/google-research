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

"""Generates urban morphology figure for GEM-15."""

from collections.abc import Sequence
import io
import os
import urllib.request

from absl import app
from absl import flags
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

  gem15 = ee.ImageCollection(ee_assets_config.GEM15_ASSET_ID)
  gem_dsm = gem15.select("dsm").mosaic()
  gem_p95 = gem15.select("dsm_percentile_95").mosaic()
  gem_p5 = gem15.select("dsm_percentile_5").mosaic()

  cop_raw = (
      ee.ImageCollection(ee_assets_config.COPERNICUS_GLO30_ASSET_ID)
      .mosaic()
      .select("DEM")
  )
  egm96 = ee.Image(ee_assets_config.GEOID_EGM96_ASSET_ID).select("b1")
  egm2008 = ee.Image(ee_assets_config.GEOID_EGM2008_ASSET_ID).select("b1")
  cop_dem = cop_raw.add(egm2008).subtract(egm96)

  nz_col = ee.ImageCollection(ee_assets_config.LIDAR_NEW_ZEALAND_ASSET_ID)
  nz_img = nz_col.mosaic()
  nz_geoid = ee.Image(ee_assets_config.GEOID_NZ_NZVD2016_ASSET_ID).select("b1")
  lidar_dsm = nz_img.select("dsm").add(nz_geoid).subtract(egm96)

  pneo_col = ee.ImageCollection(ee_assets_config.HIGH_RES_OPTICAL_RGB_ASSET_ID)

  center_u = [172.6903, -43.52187]
  utm_crs_u = utils.get_utm_crs(center_u[0], center_u[1])
  box_u = ee.Geometry.Point(center_u).buffer(250).bounds()

  gem_dsm = gem_dsm.reproject(crs=utm_crs_u, scale=15)
  gem_p95 = gem_p95.reproject(crs=utm_crs_u, scale=15)
  gem_p5 = gem_p5.reproject(crs=utm_crs_u, scale=15)
  cop_dem = cop_dem.reproject(crs=utm_crs_u, scale=30)

  p_start = [172.6888, -43.5220]
  p_end = [172.6920, -43.5220]
  line_geom = ee.Geometry.LineString([p_start, p_end])

  # Elevation display range cycling at 45m
  vmin_u = 5.0
  vmax_u = 45.0
  print(f"Urban elevation display range: [{vmin_u:.1f} m, {vmax_u:.1f} m]")

  def get_thumb(ee_img, geom, vmin, vmax, palette, dims="700x700"):
    vis = ee_img.visualize(min=vmin, max=vmax, palette=palette)
    url = vis.getThumbURL({"dimensions": dims, "region": geom, "format": "png"})
    with urllib.request.urlopen(url) as req:
      return Image.open(io.BytesIO(req.read()))

  # 1. PNeo RGB with line
  pneo_img = pneo_col.filterBounds(box_u).first()
  vis_pneo = pneo_img.visualize(
      bands=["R", "G", "B"], min=20, max=190, gamma=1.2
  )
  line_mask = ee.Image().byte().paint(line_geom, 1, 3)
  pneo_with_transect = vis_pneo.blend(line_mask.visualize(palette=["#ff1744"]))
  url_rgb = pneo_with_transect.getThumbURL(
      {"dimensions": "700x700", "region": box_u, "format": "png"}
  )
  with urllib.request.urlopen(url_rgb) as req:
    img_rgb_u = Image.open(io.BytesIO(req.read()))

  # 2. DEMs with unified turbo palette
  img_cop_u = get_thumb(cop_dem, box_u, vmin_u, vmax_u, turbo_hex)
  img_dsm_u = get_thumb(gem_dsm, box_u, vmin_u, vmax_u, turbo_hex)
  img_lidar_u = get_thumb(lidar_dsm, box_u, vmin_u, vmax_u, turbo_hex)
  img_p5_u = get_thumb(gem_p5, box_u, vmin_u, vmax_u, turbo_hex)
  img_p95_u = get_thumb(gem_p95, box_u, vmin_u, vmax_u, turbo_hex)

  # 3. Sample along transect
  num_pts = 95
  pts_u = [
      [
          p_start[0] + (p_end[0] - p_start[0]) * i / (num_pts - 1),
          p_start[1] + (p_end[1] - p_start[1]) * i / (num_pts - 1),
      ]
      for i in range(num_pts)
  ]
  total_dist_m = 260.0
  fc_u = ee.FeatureCollection([
      ee.Feature(
          ee.Geometry.Point(p), {"dist_m": i * (total_dist_m / (num_pts - 1))}
      )
      for i, p in enumerate(pts_u)
  ])

  stack_u = ee.Image.cat([
      gem_dsm.rename("dsm"),
      gem_p95.rename("p95"),
      cop_dem.rename("cop"),
      lidar_dsm.rename("lidar_dsm"),
  ])
  samples_u = stack_u.reduceRegions(
      collection=fc_u, reducer=ee.Reducer.first(), scale=1
  ).getInfo()["features"]
  val_u = [
      s["properties"]
      for s in samples_u
      if all(
          s["properties"].get(k) is not None
          for k in ["dsm", "p95", "cop", "lidar_dsm"]
      )
  ]

  dist_u = np.array([p["dist_m"] for p in val_u])
  dsm_u_pts = np.array([p["dsm"] for p in val_u])
  p95_u_pts = np.array([p["p95"] for p in val_u])
  cop_u_pts = np.array([p["cop"] for p in val_u])
  lidar_dsm_pts = np.array([p["lidar_dsm"] for p in val_u])

  # Pixel coordinate calculation for Point A and B
  box_info = box_u.getInfo()["coordinates"][0]
  lons = [c[0] for c in box_info]
  lats = [c[1] for c in box_info]
  min_lon, max_lon = min(lons), max(lons)
  min_lat, max_lat = min(lats), max(lats)

  def to_px(lon, lat, size = 700):
    x = (lon - min_lon) / (max_lon - min_lon) * size
    y = (max_lat - lat) / (max_lat - min_lat) * size
    return x, y

  xa, ya = to_px(p_start[0], p_start[1])
  xb, yb = to_px(p_end[0], p_end[1])

  fig = plt.figure(figsize=(19, 9.0), dpi=300)

  gs_left = fig.add_gridspec(
      2,
      3,
      left=0.03,
      right=0.60,
      bottom=0.10,
      top=0.90,
      hspace=0.14,
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

  # Panels arranged with GLO-30 and GEM-15 DSM horizontally adjacent in top row
  panels = [
      (img_rgb_u, "(a) Optical (30 cm Pléiades Neo)", False),
      (img_cop_u, "(b) Copernicus GLO-30 DSM (30 m)", True),
      (img_dsm_u, "(c) GEM-15 v1 DSM (15 m)", True),
      (img_lidar_u, "(d) LINZ 1 m Airborne LiDAR DSM", True),
      (img_p5_u, r"(e) GEM-15 v1 $P_5$ (Ground, 15 m)", True),
      (img_p95_u, r"(f) GEM-15 v1 $P_{95}$ (Rooftops, 15 m)", True),
  ]

  for idx, (img, title, is_dem) in enumerate(panels):
    ax = fig.add_subplot(gs_left[idx // 3, idx % 3])
    ax.set_aspect("equal")
    ax.imshow(img)
    if not is_dem:
      utils.add_rgb_scalebar(ax, 500.0, fontsize=8.8)
      ax.text(
          xa,
          ya,
          "A",
          color="black",
          fontsize=10,
          fontweight="bold",
          ha="center",
          va="center",
          bbox=dict(
              boxstyle="circle,pad=0.25", fc="#ffffff", ec="#000000", lw=1.2
          ),
      )
      ax.text(
          xb,
          yb,
          "B",
          color="black",
          fontsize=10,
          fontweight="bold",
          ha="center",
          va="center",
          bbox=dict(
              boxstyle="circle,pad=0.25", fc="#ffffff", ec="#000000", lw=1.2
          ),
      )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_color("#666666")
      spine.set_linewidth(0.8)

  # Right: Profile (Height reduced to 2/3rd)
  ax_p = fig.add_subplot(gs_right[0, 0])
  ax_p.plot(
      dist_u,
      lidar_dsm_pts,
      label="LINZ 1 m Airborne LiDAR DSM (Reference)",
      color="#252525",
      linewidth=1.8,
      alpha=0.9,
      linestyle=":",
  )
  ax_p.plot(
      dist_u,
      cop_u_pts,
      label="Copernicus GLO-30 DSM (30 m)",
      color="#7570b3",
      linewidth=2.6,
      linestyle="--",
  )
  ax_p.plot(
      dist_u,
      p95_u_pts,
      label=r"GEM-15 $P_{95}$ (Rooftop Ridges)",
      color="#d95f02",
      linewidth=2.4,
  )
  ax_p.plot(
      dist_u,
      dsm_u_pts,
      label="GEM-15 v1 DSM Mean",
      color="#e7298a",
      linewidth=2.2,
  )

  ax_p.legend(
      loc="upper left",
      frameon=True,
      facecolor="white",
      framealpha=0.95,
      edgecolor="#cccccc",
      fontsize=9.5,
  )
  ax_p.set_xlabel(
      "Distance along Transect A–B (meters)",
      fontsize=11,
      fontweight="bold",
      labelpad=8,
  )
  ax_p.set_ylabel(
      "Elevation above EGM96 Geoid (m)",
      fontsize=11,
      fontweight="bold",
      labelpad=8,
  )
  ax_p.set_title(
      "Cross-Section A–B: Resolving Building Surfaces vs. LiDAR Reference",
      fontsize=12,
      fontweight="bold",
      pad=10,
  )
  ax_p.grid(True, linestyle=":", alpha=0.6)
  ax_p.set_xlim(min(dist_u), max(dist_u))
  ax_p.set_ylim(4.5, 18.5)
  ax_p.tick_params(labelsize=10)

  # Shared Colorbar with exact paper Turbo Colormap cycling at 45m
  cbar_ax = fig.add_axes([0.08, 0.035, 0.45, 0.022])
  sm = plt.cm.ScalarMappable(
      cmap=cmap_paper, norm=plt.Normalize(vmin=vmin_u, vmax=vmax_u)
  )
  sm.set_array([])
  cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
  cbar.set_label(
      "Elevation (m, EGM96 Geoid)", fontsize=10, fontweight="bold", labelpad=4
  )
  cbar.set_ticks([5, 15, 25, 35, 45])
  cbar.ax.tick_params(labelsize=9)

  plt.suptitle(
      "Environmental Application 3: Urban Morphology — Rooftop Forms vs. Street"
      " Grade (Christchurch, NZ)",
      fontsize=14,
      fontweight="bold",
      y=0.97,
  )

  os.makedirs(FLAGS.output_dir, exist_ok=True)
  out_png = os.path.join(FLAGS.output_dir, "figure_supp_urban_45m.png")
  out_jpg = os.path.join(FLAGS.output_dir, "figure_supp_urban_45m.jpg")
  plt.savefig(out_png, dpi=300, bbox_inches="tight")
  plt.close()

  im = Image.open(out_png).convert("RGB")
  im.save(out_jpg, "JPEG", quality=88, optimize=True)
  print(f"Saved {out_png} and {out_jpg}")


if __name__ == "__main__":
  app.run(main)
