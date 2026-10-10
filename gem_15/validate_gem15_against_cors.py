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

"""GEM-15 International GNSS CORS Survey Validation Engine (2025).

This script evaluates GEM-15 DSM v1 (15m) and public global DEMs
(Copernicus GLO-30, ALOS AW3D30, and NASADEM) against the International
GNSS CORS Network (Nevada Geodetic Laboratory / IGS / EUREF / GEONET /
AuScope / GeoNet NZ / SIRGAS / AFREF) for 2025 across all 6 inhabited
continents, saving per-station and continent-stratified CSV results to disk.
"""

from collections.abc import Sequence
import concurrent.futures
import os
from typing import Any
from absl import app
from absl import flags
import pandas as pd
import ee
from gem_15 import ee_assets_config
from gem_15 import metrics
from gem_15 import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir',
    None,
    'The local directory to save output CSV and reports.',
)
flags.DEFINE_integer(
    'year',
    2025,
    'Observation year to evaluate (default: 2025).',
)
flags.DEFINE_string(
    'cors_asset_id',
    ee_assets_config.CORS_STATIONS_ASSET_ID,
    'Earth Engine FeatureCollection asset for International GNSS CORS 2025'
    ' stations.',
)
flags.DEFINE_integer(
    'limit_points',
    0,
    'Limit the evaluation per year to a random subset of this size (set <= 0'
    ' for no limit, evaluating all global stations).',
)
flags.DEFINE_bool(
    'resample_bilinear',
    True,
    'Whether to resample raster DEMs using bilinear interpolation.',
)
flags.DEFINE_integer(
    'random_seed',
    42,
    'Random seed for deterministic sampling.',
)
flags.DEFINE_integer(
    'num_workers',
    12,
    'Number of parallel worker threads for evaluating global station batches.',
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

CONTINENT_ORDER = [
    'Global (All)',
    'North America',
    'Europe',
    'Asia',
    'Oceania',
    'South America',
    'Africa',
]


def compute_subset_metrics(df_sub):
  """Computes exact Bias, Direct RMSE, Debiased RMSE, and Median NMAD using metrics.py."""
  return metrics.compute_point_metrics_from_df(df_sub)


def load_global_cors_stations_df(year):
  """Loads International GNSS CORS stations directly from the Earth Engine asset."""
  del year  # Asset is pre-filtered to 2025 active stations
  print(
      'Loading International GNSS CORS asset directly from Earth Engine:'
      f' {FLAGS.cors_asset_id}'
  )
  fc = ee.FeatureCollection(FLAGS.cors_asset_id)
  total_sz = int(utils.call_ee_with_retries(fc.size()) or 0)
  rows = []
  for offset in range(0, total_sz, 5000):
    feats = utils.call_ee_with_retries(fc.toList(5000, offset)) or []
    for feat in feats:
      p = feat['properties']
      c = feat.get('geometry', {}).get('coordinates', [0.0, 0.0])
      rows.append({
          'station_id': str(p.get('station_id', '')),
          'lon': float(c[0]),
          'lat': float(c[1]),
          'ortho_height': float(p['ortho_height']),
          'continent': str(p['continent']),
      })
  return pd.DataFrame(rows)


def run_validation_for_year(year):
  """Runs the global point-to-pixel validation across all International CORS stations."""
  print('\n=========================================================')
  print(f'Running International GNSS CORS Validation for Year: {year}')
  print('=========================================================')

  stations_df = load_global_cors_stations_df(year)
  if FLAGS.limit_points > 0 and len(stations_df) > FLAGS.limit_points:
    stations_df = stations_df.sample(
        n=FLAGS.limit_points, random_state=FLAGS.random_seed
    )
  stations_df = stations_df.sort_values(
      ['continent', 'lon', 'lat']
  ).reset_index(drop=True)

  print(
      'Total International GNSS CORS stations to evaluate:'
      f' {len(stations_df):,}'
  )

  gem15_layers = utils.get_gem15_layers(
      resample_bilinear=FLAGS.resample_bilinear
  )
  ge_dsm_masked = gem15_layers['dsm'].updateMask(gem15_layers['dsm'].neq(0))
  ref_dems = utils.get_global_reference_dems(
      resample_bilinear=FLAGS.resample_bilinear
  )
  dem_image = ee.Image.cat([
      ge_dsm_masked.rename('ge'),
      ref_dems['copernicus'],
      ref_dems['alos'],
      ref_dems['nasa'],
  ])

  batch_size = 400
  chunks = [
      stations_df.iloc[i : i + batch_size]
      for i in range(0, len(stations_df), batch_size)
  ]

  def _eval_chunk(cdf):
    ee_feats = []
    for _, r in cdf.iterrows():
      lat = float(r['lat'])
      lon = float(r['lon'])
      cont = str(r['continent'])
      ee_feats.append(
          ee.Feature(
              ee.Geometry.Point([lon, lat]),
              {
                  'station_id': str(r['station_id']),
                  'lon': lon,
                  'lat': lat,
                  'ref_elevation': float(r['ortho_height']),
                  'continent': cont,
              },
          )
      )
    fc = ee.FeatureCollection(ee_feats)
    sampled = dem_image.sampleRegions(
        collection=fc,
        properties=['station_id', 'lon', 'lat', 'ref_elevation', 'continent'],
        scale=1,
        geometries=False,
    )
    valid = sampled.filter(
        ee.Filter.notNull(['ge', 'copernicus', 'alos', 'nasa'])
    )
    try:
      info = utils.call_ee_with_retries(valid.toList(len(cdf))) or []
    except ee.EEException as e:
      if len(cdf) <= 25:
        raise RuntimeError(
            f'Failed to evaluate sub-batch of {len(cdf)} CORS stations: {e}'
        ) from e
      mid = len(cdf) // 2
      return _eval_chunk(cdf.iloc[:mid]) + _eval_chunk(cdf.iloc[mid:])
    out = []
    for feat in info:
      props = feat['properties']
      ref_val = float(props['ref_elevation'])
      ge_val = float(props['ge'])
      cop_val = float(props['copernicus'])
      alos_val = float(props['alos'])
      nasa_val = float(props['nasa'])
      out.append({
          'station_id': str(props.get('station_id', '')),
          'lon': float(props['lon']),
          'lat': float(props['lat']),
          'continent': str(props.get('continent', '')),
          'ref_elevation': ref_val,
          'ge': ge_val,
          'cop': cop_val,
          'alos': alos_val,
          'nasa': nasa_val,
      })
    return out

  records: list[dict[str, Any]] = []
  print(
      f'Evaluating {len(chunks)} batches across {FLAGS.num_workers} parallel'
      ' workers...'
  )
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=FLAGS.num_workers
  ) as executor:
    for batch_res in executor.map(_eval_chunk, chunks):
      records.extend(batch_res)

  df = pd.DataFrame(records)
  print(
      f'Successfully compiled {len(df):,} valid International GNSS CORS'
      f' records for year {year}.'
  )
  return df


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not FLAGS.output_dir:
    raise ValueError('Flag --output_dir must be specified.')

  os.makedirs(FLAGS.output_dir, exist_ok=True)
  utils.authenticate_earth_engine(
      ee_service_account=FLAGS.ee_service_account,
      project=FLAGS.ee_project,
      opt_url='https://earthengine-highvolume.googleapis.com',
  )

  target_year = FLAGS.year

  df = run_validation_for_year(target_year)
  if df.empty:
    raise ValueError(f'No valid GNSS CORS stations returned for {target_year}.')

  # Save per-station results CSV (both results.csv and
  # results_{target_year}.csv)
  results_csv_path = os.path.join(FLAGS.output_dir, 'results.csv')
  with open(results_csv_path, 'w') as f:
    df.to_csv(f, index=False)
  year_csv_path = os.path.join(FLAGS.output_dir, f'results_{target_year}.csv')
  with open(year_csv_path, 'w') as f:
    df.to_csv(f, index=False)
  print(f'Saved compiled station CSV results to: {results_csv_path}')

  # Compute Continent-Stratified Summary Table
  cont_rows = []
  global_metrics = compute_subset_metrics(df)
  global_metrics['continent'] = 'Global (All)'
  cont_rows.append(global_metrics)

  for cont in CONTINENT_ORDER[1:]:
    sub_df = df[df['continent'] == cont]
    if not sub_df.empty:
      m = compute_subset_metrics(sub_df)
      m['continent'] = cont
      cont_rows.append(m)

  cont_df = pd.DataFrame(cont_rows)
  cont_csv_path = os.path.join(
      FLAGS.output_dir, f'continent_summary_{target_year}.csv'
  )
  with open(cont_csv_path, 'w') as f:
    cont_df.to_csv(f, index=False)
  print(f'Saved continent-stratified summary CSV to: {cont_csv_path}')

  # Also save timeseries_results.csv for compatibility with runner checks
  ts_df = pd.DataFrame([{
      'year': target_year,
      'count': global_metrics['count'],
      'ge_direct_rmse': global_metrics['ge_direct_rmse'],
      'ge_bias': global_metrics['ge_bias'],
      'ge_debiased_rmse': global_metrics['ge_debiased_rmse'],
      'ge_nmad': global_metrics['ge_nmad'],
      'cop_direct_rmse': global_metrics['cop_direct_rmse'],
      'cop_bias': global_metrics['cop_bias'],
      'cop_debiased_rmse': global_metrics['cop_debiased_rmse'],
      'cop_nmad': global_metrics['cop_nmad'],
      'alos_direct_rmse': global_metrics['alos_direct_rmse'],
      'alos_bias': global_metrics['alos_bias'],
      'alos_debiased_rmse': global_metrics['alos_debiased_rmse'],
      'alos_nmad': global_metrics['alos_nmad'],
      'nasa_direct_rmse': global_metrics['nasa_direct_rmse'],
      'nasa_bias': global_metrics['nasa_bias'],
      'nasa_debiased_rmse': global_metrics['nasa_debiased_rmse'],
      'nasa_nmad': global_metrics['nasa_nmad'],
  }])
  ts_csv_path = os.path.join(FLAGS.output_dir, 'timeseries_results.csv')
  with open(ts_csv_path, 'w') as f:
    ts_df.to_csv(f, index=False)

  print('\nInternational GNSS CORS validation completed successfully!')


if __name__ == '__main__':
  app.run(main)
