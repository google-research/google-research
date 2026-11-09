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

r"""Minimal Earth Engine Python example for loading and plotting GEM-15.

Example usage:
  earthengine authenticate
  python3 examples/ee_example.py \
      --ee_project=your-cloud-project-id \
      --bbox=-122.440,37.748,-122.400,37.778 \
      --out_prefix=gem15_ee_sf
"""

from collections.abc import Sequence
import io
import urllib.request

from absl import app
from absl import flags
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import ee

GEM15_ASSET_ID = 'projects/climate-and-sustainability/assets/gem-15_v1'
BAND_NAMES = (
    'dsm',
    'dtm',
    'dsm_percentile_5',
    'dsm_percentile_50',
    'dsm_percentile_95',
    'dsm_std',
)

FLAGS = flags.FLAGS
flags.DEFINE_string('ee_project', None, 'Google Cloud project ID for EE.')
flags.DEFINE_list(
    'bbox',
    ['-122.440', '37.748', '-122.400', '37.778'],
    'WGS84 bbox as west,south,east,north.',
)
flags.DEFINE_string('out_prefix', 'gem15_ee_sf', 'Output file prefix.')
flags.DEFINE_float('scale_m', 15.0, 'Scale in meters for stats and GeoTIFF.')


def _fetch_thumb(
    img: ee.Image,
    region: ee.Geometry,
    vmin: float,
    vmax: float,
    cmap_name: str,
) -> np.ndarray:
  """Fetches a visualized PNG thumbnail from Earth Engine."""
  palette = [
      colors.to_hex(plt.get_cmap(cmap_name)(i / 15.0)) for i in range(16)
  ]
  url = img.visualize(min=vmin, max=vmax, palette=palette).getThumbURL({
      'region': region,
      'dimensions': 512,
      'format': 'png',
  })
  with urllib.request.urlopen(url, timeout=60) as resp:
    with Image.open(io.BytesIO(resp.read())) as im:
      return np.asarray(im.convert('RGB'))


def main(argv: Sequence[str]) -> None:
  del argv
  if FLAGS.ee_project:
    ee.Initialize(project=FLAGS.ee_project)
  else:
    ee.Initialize()

  bbox = [float(v) for v in FLAGS.bbox]
  region = ee.Geometry.Rectangle(bbox, proj='EPSG:4326', geodesic=False)
  img = ee.ImageCollection(GEM15_ASSET_ID).filterBounds(region).mosaic()
  ndsm = img.select('dsm').subtract(img.select('dtm')).rename('ndsm')

  reducer = ee.Reducer.minMax().combine(
      ee.Reducer.mean(), sharedInputs=True
  ).combine(ee.Reducer.percentile([2, 98]), sharedInputs=True)
  stats = (
      img.select(['dsm', 'dtm']).addBands(ndsm).reduceRegion(
          reducer=reducer,
          geometry=region,
          scale=FLAGS.scale_m,
          bestEffort=True,
      ).getInfo()
      or {}
  )
  for b in ('dsm', 'dtm', 'ndsm'):
    print(
        f"{b:5s}: min={stats.get(f'{b}_min')} mean={stats.get(f'{b}_mean')}"
        f" max={stats.get(f'{b}_max')}"
    )

  vmin = float(stats.get('dsm_p2') or 0.0)
  vmax = max(vmin + 1.0, float(stats.get('dsm_p98') or 200.0))
  hmax = max(1.0, float(stats.get('ndsm_p98') or 30.0))

  png_path = f'{FLAGS.out_prefix}.png'
  fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
  for ax, band_img, title, cmap_name, lo, hi in [
      (axes[0], img.select('dsm'), 'DSM (m)', 'terrain', vmin, vmax),
      (axes[1], img.select('dtm'), 'DTM (m)', 'terrain', vmin, vmax),
      (axes[2], ndsm, 'Height: DSM - DTM (m)', 'YlGn', 0.0, hmax),
  ]:
    ax.imshow(_fetch_thumb(band_img, region, lo, hi, cmap_name))
    ax.set_title(title)
    ax.axis('off')
    mappable = plt.cm.ScalarMappable(
        norm=colors.Normalize(vmin=lo, vmax=hi), cmap=cmap_name
    )
    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
  fig.savefig(png_path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved {png_path}')

  tif_path = f'{FLAGS.out_prefix}.tif'
  try:
    url = img.select(list(BAND_NAMES)).getDownloadURL({
        'region': region,
        'scale': FLAGS.scale_m,
        'format': 'GEO_TIFF',
    })
    with (
        urllib.request.urlopen(url, timeout=120) as resp,
        open(tif_path, 'wb') as f,
    ):
      f.write(resp.read())
    print(f'Saved {tif_path}')
  except ee.EEException as exc:
    print(f'Skipped GeoTIFF download ({exc}).')


if __name__ == '__main__':
  app.run(main)
