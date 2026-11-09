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

r"""Minimal STAC example for reading and plotting GEM-15 COGs over a bbox.

Example usage:
  python3 examples/stac_example.py \
      --bbox -122.440 37.748 -122.400 37.778 \
      --out_prefix gem15_stac_sf
"""

import argparse
import json
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pystac
import rasterio


DEFAULT_CATALOG = (
    'https://storage.googleapis.com/climate-and-sustainability/'
    'gem-15/stac/catalog.json'
)
DEFAULT_BBOX = (-122.440, 37.748, -122.400, 37.778)


def load_items(
    catalog_url, bbox
):
  """Streams STAC items from items.ndjson and filters by bbox."""
  ndjson_url = (
      catalog_url
      if catalog_url.endswith('.ndjson')
      else urllib.parse.urljoin(catalog_url, 'items.ndjson')
  )
  west, south, east, north = bbox
  items: list[pystac.Item] = []

  stream = (
      urllib.request.urlopen(ndjson_url, timeout=60)
      if urllib.parse.urlparse(ndjson_url).scheme in ('http', 'https')
      else open(ndjson_url, 'rb')
  )
  with stream:
    for line in stream:
      if not line.strip():
        continue
      record = json.loads(line)
      b = record.get('bbox')
      if b and b[0] < east and b[2] > west and b[1] < north and b[3] > south:
        items.append(pystac.Item.from_dict(record))

  if not items:
    raise RuntimeError(f'No GEM-15 STAC items found for bbox={bbox}.')
  return items


def main():
  parser = argparse.ArgumentParser(description='Read GEM-15 COGs via STAC.')
  parser.add_argument(
      '--bbox', nargs=4, type=float, default=list(DEFAULT_BBOX),
      metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
  )
  parser.add_argument('--catalog', default=DEFAULT_CATALOG)
  parser.add_argument('--out_prefix', default='gem15_stac_sf')
  args = parser.parse_args()
  bbox = tuple(args.bbox)

  items = load_items(args.catalog, bbox)
  print(f'Loaded {len(items)} STAC item(s): {[i.id for i in items]}')

  ds = odc.stac.load(
      items,
      bbox=bbox,
      crs='utm',
      resolution=15,
      resampling='bilinear',
      groupby=lambda *_: 0,
  ).isel(time=0)
  dsm_bands = [
      b['name'] for b in items[0].assets['dsm'].extra_fields['raster:bands']
  ]
  ds = ds.rename({f'dsm.{i + 1}': name for i, name in enumerate(dsm_bands)})

  dsm = ds['dsm'].to_numpy()
  dtm = ds['dtm'].to_numpy()
  ndsm = dsm - dtm
  for name, arr in [('dsm', dsm), ('dtm', dtm), ('ndsm', ndsm)]:
    print(
        f'{name:5s}: min={np.nanmin(arr):.2f}m mean={np.nanmean(arr):.2f}m'
        f' max={np.nanmax(arr):.2f}m'
    )

  tif_path, png_path = f'{args.out_prefix}.tif', f'{args.out_prefix}.png'
  band_order = (
      'dsm',
      'dtm',
      'dsm_percentile_5',
      'dsm_percentile_50',
      'dsm_percentile_95',
      'dsm_std',
  )
  with rasterio.open(
      tif_path, 'w', driver='GTiff', dtype='float32', nodata=np.nan,
      compress='deflate', count=len(band_order), crs=ds.odc.crs.to_wkt(),
      transform=ds.odc.transform, height=dsm.shape[0], width=dsm.shape[1],
  ) as dst:
    for idx, name in enumerate(band_order, start=1):
      dst.write(ds[name].to_numpy(), idx)
      dst.set_band_description(idx, name)

  elev = np.concatenate([dsm[np.isfinite(dsm)], dtm[np.isfinite(dtm)]])
  vmin, vmax = np.percentile(elev, [2, 98])
  hmax = max(1.0, float(np.nanpercentile(ndsm, 98)))

  fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
  for ax, arr, title, cmap, lo, hi in [
      (axes[0], dsm, 'DSM (m)', 'terrain', vmin, vmax),
      (axes[1], dtm, 'DTM (m)', 'terrain', vmin, vmax),
      (axes[2], ndsm, 'Height: DSM - DTM (m)', 'YlGn', 0.0, hmax),
  ]:
    im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
  fig.savefig(png_path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved {tif_path} and {png_path}')


if __name__ == '__main__':
  main()
