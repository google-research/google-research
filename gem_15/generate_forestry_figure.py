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

"""Generates forestry canopy height figure for GEM-15."""

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

  # 1. Earth Engine Authentication
  utils.authenticate_earth_engine(
      project=FLAGS.ee_project, ee_service_account=FLAGS.ee_service_account
  )

  # 2. Colormap
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

  # 3. Datasets
  gem15 = ee.ImageCollection(ee_assets_config.GEM15_ASSET_ID)
  gem_dsm = gem15.select("dsm").mosaic()
  gem_p95 = gem15.select("dsm_percentile_95").mosaic()

  cop_raw = (
      ee.ImageCollection(ee_assets_config.COPERNICUS_GLO30_ASSET_ID)
      .mosaic()
      .select("DEM")
  )
  egm96 = ee.Image(ee_assets_config.GEOID_EGM96_ASSET_ID).select("b1")
  egm2008 = ee.Image(ee_assets_config.GEOID_EGM2008_ASSET_ID).select("b1")
  cop_dem = cop_raw.add(egm2008).subtract(egm96)

  alos_raw = (
      ee.ImageCollection(ee_assets_config.ALOS_AW3D30_ASSET_ID)
      .mosaic()
      .select("DSM")
  )
  nasadem = ee.Image(ee_assets_config.NASADEM_ASSET_ID).select("elevation")

  pneo_col = ee.ImageCollection(ee_assets_config.HIGH_RES_OPTICAL_RGB_ASSET_ID)

  # 4. Region: Sinop, Mato Grosso, Brazil
  center_f = [-55.480, -11.648]
  utm_crs_f = utils.get_utm_crs(center_f[0], center_f[1])
  box_f = ee.Geometry.Point(center_f).buffer(400).bounds()

  gem_dsm = gem_dsm.reproject(crs=utm_crs_f, scale=15)
  gem_p95 = gem_p95.reproject(crs=utm_crs_f, scale=15)
  cop_dem = cop_dem.reproject(crs=utm_crs_f, scale=30)
  alos_raw = alos_raw.reproject(crs=utm_crs_f, scale=30)
  nasadem = nasadem.reproject(crs=utm_crs_f, scale=30)

  p_start = [-55.47979592, -11.64816327]  # Rainforest
  p_end = [-55.47773217, -11.64644348]  # Field
  line_geom = ee.Geometry.LineString([p_start, p_end])

  # Elevation display range in Sinop: 345 m to 385 m
  disp_min = 345.0
  disp_max = 385.0
  print(f"Elevation display range: [{disp_min:.1f} m, {disp_max:.1f} m]")

  def get_thumb(ee_img, geom, vmin, vmax, palette, dims="700x700"):
    vis = ee_img.visualize(min=vmin, max=vmax, palette=palette)
    url = vis.getThumbURL({"dimensions": dims, "region": geom, "format": "png"})
    with urllib.request.urlopen(url) as req:
      return Image.open(io.BytesIO(req.read()))

  # Fetch PNeo 30 cm Optical with Transect Line
  pneo_img = pneo_col.filterBounds(box_f).first()
  vis_pneo = pneo_img.visualize(
      bands=["R", "G", "B"], min=15, max=140, gamma=1.2
  )
  line_mask = ee.Image().byte().paint(line_geom, 1, 3)
  pneo_with_transect = vis_pneo.blend(line_mask.visualize(palette=["#ff1744"]))
  url_rgb = pneo_with_transect.getThumbURL(
      {"dimensions": "700x700", "region": box_f, "format": "png"}
  )
  with urllib.request.urlopen(url_rgb) as req:
    img_rgb = Image.open(io.BytesIO(req.read()))

  # Fetch DEM Thumbnails with unified Turbo colormap
  img_gem_dsm = get_thumb(gem_dsm, box_f, disp_min, disp_max, turbo_hex)
  img_cop = get_thumb(cop_dem, box_f, disp_min, disp_max, turbo_hex)
  img_alos = get_thumb(alos_raw, box_f, disp_min, disp_max, turbo_hex)
  img_nasadem = get_thumb(nasadem, box_f, disp_min, disp_max, turbo_hex)
  img_gem_p95 = get_thumb(gem_p95, box_f, disp_min, disp_max, turbo_hex)

  # Sample along Transect A-B
  num_pts = 95
  total_dist_m = 280.0
  pts_f = [
      [
          p_start[0] + (p_end[0] - p_start[0]) * i / (num_pts - 1),
          p_start[1] + (p_end[1] - p_start[1]) * i / (num_pts - 1),
      ]
      for i in range(num_pts)
  ]
  fc_f = ee.FeatureCollection([
      ee.Feature(
          ee.Geometry.Point(p), {"dist_m": i * (total_dist_m / (num_pts - 1))}
      )
      for i, p in enumerate(pts_f)
  ])

  stack_f = ee.Image.cat([
      gem_dsm.rename("dsm"),
      gem_p95.rename("p95"),
      cop_dem.rename("cop"),
      alos_raw.rename("alos"),
      nasadem.rename("nasadem"),
  ])

  samples_f = stack_f.reduceRegions(
      collection=fc_f, reducer=ee.Reducer.first(), scale=1
  ).getInfo()["features"]
  val_f = [
      s["properties"]
      for s in samples_f
      if all(
          s["properties"].get(k) is not None
          for k in ["dsm", "p95", "cop", "alos", "nasadem"]
      )
  ]

  dist_f = np.array([p["dist_m"] for p in val_f])
  dsm_f_pts = np.array([p["dsm"] for p in val_f])
  p95_f_pts = np.array([p["p95"] for p in val_f])
  cop_f_pts = np.array([p["cop"] for p in val_f])
  alos_f_pts = np.array([p["alos"] for p in val_f])
  nasadem_f_pts = np.array([p["nasadem"] for p in val_f])

  # Transition distance
  trans_idx = np.where(dsm_f_pts < 360.0)[0][0]
  trans_dist = dist_f[trans_idx]

  # Pixel coordinate calculation for Point A and B
  box_info = box_f.getInfo()["coordinates"][0]
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

  # Layout: 19 x 9.0 inches matching all supplementary figures
  fig = plt.figure(figsize=(19, 9.0), dpi=300)

  # Left 2x3 grid
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

  # Right profile: increased height by 25% (bottom=0.22, top=0.72) matching
  # Figures S1 & S3
  gs_right = fig.add_gridspec(
      1,
      1,
      left=0.635,
      right=0.98,
      bottom=0.22,
      top=0.72,
  )

  # Order: (a) Optical, (b) GEM DSM next to RGB, (c) Copernicus 3rd, (d) ALOS,
  # (e) NASADEM, (f) GEM P95
  panels = [
      (img_rgb, "(a) Optical (30 cm Pléiades Neo)", False),
      (img_gem_dsm, "(b) GEM-15 v1 DSM (15 m)", True),
      (img_cop, "(c) Copernicus GLO-30 DSM (30 m)", True),
      (img_alos, "(d) ALOS AW3D30 (30 m)", True),
      (img_nasadem, "(e) NASADEM HGT (30 m)", True),
      (img_gem_p95, r"(f) GEM-15 $P_{95}$ (Tree Crowns, 15 m)", True),
  ]

  for idx, (img, title, is_dem) in enumerate(panels):
    ax = fig.add_subplot(gs_left[idx // 3, idx % 3])
    ax.set_aspect("equal")
    ax.imshow(img)
    if not is_dem:
      utils.add_rgb_scalebar(ax, 800.0, fontsize=8.8)
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

  # Right: Profile (+25% height)
  ax_p = fig.add_subplot(gs_right[0, 0])
  ax_p.plot(
      dist_f,
      nasadem_f_pts,
      label="NASADEM HGT (30 m)",
      color="#8c564b",
      linewidth=2.0,
      linestyle="-.",
      zorder=3,
  )
  ax_p.plot(
      dist_f,
      alos_f_pts,
      label="ALOS AW3D30 (30 m)",
      color="#17becf",
      linewidth=2.0,
      linestyle="--",
      zorder=4,
  )
  ax_p.plot(
      dist_f,
      cop_f_pts,
      label="Copernicus GLO-30 DSM (30 m)",
      color="#7570b3",
      linewidth=2.4,
      linestyle="--",
      zorder=5,
  )
  ax_p.plot(
      dist_f,
      dsm_f_pts,
      label="GEM-15 v1 DSM Mean (15 m)",
      color="#e41a1c",
      linewidth=2.8,
      zorder=6,
  )
  ax_p.plot(
      dist_f,
      p95_f_pts,
      label=r"GEM-15 $P_{95}$ (Tree Crowns, 15 m)",
      color="#2ca02c",
      linewidth=2.4,
      linestyle="-",
      zorder=7,
  )

  ax_p.axvline(
      x=trans_dist,
      color="#b2182b",
      linestyle=":",
      linewidth=2.0,
      label="Stand Boundary",
  )
  ax_p.axvspan(min(dist_f), trans_dist, color="#2ca25f", alpha=0.06)
  ax_p.axvspan(trans_dist, max(dist_f), color="#fdbf6f", alpha=0.06)

  ax_p.text(
      trans_dist * 0.5,
      338,
      "PRIMARY RAINFOREST",
      fontsize=10,
      fontweight="bold",
      color="#1b7837",
      ha="center",
      bbox=dict(boxstyle="round,pad=0.25", fc="#e5f5e0", ec="#2ca25f", lw=1),
  )
  ax_p.text(
      (trans_dist + max(dist_f)) * 0.5,
      338,
      "AGRICULTURAL FIELD",
      fontsize=10,
      fontweight="bold",
      color="#b35806",
      ha="center",
      bbox=dict(boxstyle="round,pad=0.25", fc="#fff7bc", ec="#fe9929", lw=1),
  )

  ax_p.legend(
      loc="upper right",
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
      "Cross-Section A–B: Resolving Sharp Canopy Cliff vs. Spaceborne DEM"
      " Smearing",
      fontsize=12,
      fontweight="bold",
      pad=10,
  )
  ax_p.grid(True, linestyle=":", alpha=0.6)
  ax_p.set_xlim(min(dist_f), max(dist_f))
  ax_p.set_ylim(335, 395)
  ax_p.tick_params(labelsize=10)

  # Shared Colorbar with exact paper Turbo Colormap below DEM panels
  cbar_ax = fig.add_axes([0.08, 0.035, 0.45, 0.022])
  sm = plt.cm.ScalarMappable(
      cmap=cmap_paper, norm=plt.Normalize(vmin=disp_min, vmax=disp_max)
  )
  sm.set_array([])
  cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
  cbar.set_label(
      "Elevation (m, EGM96 Geoid)", fontsize=10, fontweight="bold", labelpad=4
  )
  cbar.ax.tick_params(labelsize=9)

  plt.suptitle(
      "Environmental Application 2: Forestry & Canopy Height — Resolving Canopy"
      " Step at Deforestation Boundary (Sinop, Brazil)",
      fontsize=14,
      fontweight="bold",
      y=0.97,
  )

  os.makedirs(FLAGS.output_dir, exist_ok=True)
  out_png = os.path.join(
      FLAGS.output_dir, "figure_supp_forestry_sinop_perfect.png"
  )
  out_jpg = os.path.join(
      FLAGS.output_dir, "figure_supp_forestry_sinop_perfect.jpg"
  )
  plt.savefig(out_png, dpi=300, bbox_inches="tight")
  plt.close()

  im = Image.open(out_png).convert("RGB")
  im.save(out_jpg, "JPEG", quality=88, optimize=True)
  print(f"Saved {out_png} and {out_jpg} successfully.")


if __name__ == "__main__":
  app.run(main)
