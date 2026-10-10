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

"""Configuration definitions for national and drone airborne LiDAR datasets."""

from typing import Any, Dict
import ee
from gem_15 import ee_assets_config

LIDAR_DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'uk_ea': {
        'asset_id': ee_assets_config.LIDAR_UK_EA_ASSET_ID,
        'match_keys': ['england', 'uk_ea'],
        'display_name': 'England Lidar',
        'dsm_band': 'dsm_first',
        'dtm_band': 'dtm',
        'geoid_asset': ee_assets_config.GEOID_UK_OSGM15_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 1.0,
        'buffer_radius': 750,
        'date_range': ('2020-01-01', '2027-01-01'),
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': None,
        'per_image_downsample': False,
        'description': (
            'Using local geoid model OSGM15 (UK). 2022 National Terrain Model'
            ' release.'
        ),
    },
    'canada': {
        'asset_id': ee_assets_config.LIDAR_CANADA_ASSET_ID,
        'match_keys': ['canada'],
        'display_name': 'Canada Lidar',
        'dsm_band': 'dsm',
        'dtm_band': 'dtm',
        'geoid_asset': ee_assets_config.GEOID_CANADA_CGVD2013_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 0.001,
        'buffer_radius': 750,
        'date_range': ('2020-01-01', '2027-01-01'),
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': None,
        'per_image_downsample': False,
        'description': (
            'Using local geoid model CGVD2013 (Canada) with scale factor 0.001'
            ' (2020+ surveys).'
        ),
    },
    'spain': {
        'asset_id': ee_assets_config.LIDAR_SPAIN_ASSET_ID,
        'match_keys': ['spain'],
        'display_name': 'Spain Lidar',
        'dsm_band': 'dsm',
        'dtm_band': 'dtm',
        'geoid_asset': ee_assets_config.GEOID_SPAIN_REDNAP_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 1.0,
        'buffer_radius': 750,
        'date_range': ('2018-01-01', '2027-01-01'),
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': None,
        'per_image_downsample': False,
        'description': 'Using local geoid model REDNAP (Spain, 2018+ surveys).',
    },
    'new_zealand': {
        'asset_id': ee_assets_config.LIDAR_NEW_ZEALAND_ASSET_ID,
        'match_keys': ['new_zealand'],
        'display_name': 'New Zealand Lidar',
        'dsm_band': 'dsm',
        'dtm_band': 'dem',
        'geoid_asset': ee_assets_config.GEOID_NZ_NZVD2016_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 1.0,
        'buffer_radius': 750,
        'date_range': ('2024-01-01', '2027-01-01'),
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': None,
        'per_image_downsample': False,
        'description': (
            'Using local geoid model NZVD2016 (New Zealand, 2024+ surveys).'
        ),
    },
    'mozambique': {
        'asset_id': ee_assets_config.LIDAR_MOZAMBIQUE_ASSET_ID,
        'match_keys': ['mozambique'],
        'display_name': 'Mozambique Lidar',
        'dsm_band': 'dsm',
        'dtm_band': 'dtm',
        'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 1.0,
        'buffer_radius': 750,
        'date_range': None,
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': None,
        'per_image_downsample': False,
        'description': (
            'Using EGM2008 geoid as local approximation for Mozambique (CMS'
            ' Zambezi).'
        ),
    },
    'indonesia': {
        'asset_id': ee_assets_config.LIDAR_INDONESIA_ASSET_ID,
        'match_keys': ['indonesia'],
        'display_name': 'Indonesia Lidar',
        'dsm_band': 'DSM',
        'dtm_band': 'DTM',
        'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 1.0,
        'buffer_radius': 750,
        'date_range': None,
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': None,
        'per_image_downsample': False,
        'description': (
            'Using EGM2008 geoid as local approximation for Indonesia (DGN95).'
        ),
    },
    'brazil': {
        'asset_id': ee_assets_config.LIDAR_BRAZIL_ASSET_ID,
        'match_keys': ['brazil'],
        'display_name': 'Brazil Lidar',
        'dsm_band': 'DSM',
        'dtm_band': 'DTM',
        'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 1.0,
        'buffer_radius': 750,
        'date_range': None,
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': 50,
        'per_image_downsample': False,
        'description': (
            'Using EGM2008 geoid as local approximation for Brazil'
            ' (SIRGAS2000).'
        ),
    },
    'us_tropical': {
        'asset_id': ee_assets_config.LIDAR_US_TROPICAL_ASSET_ID,
        'match_keys': ['us_tropical'],
        'display_name': 'US Tropical Lidar',
        'dsm_band': 'dsm',
        'dtm_band': 'DTM',
        'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 1.0,
        'buffer_radius': 750,
        'date_range': None,
        'collection_filter': lambda col: col.filter(
            ee.Filter.stringContains('system:index', '2020')
        ),
        # In US Tropical LiDAR (Everglades / Keys / Puerto Rico), 'DTM' is
        # bare-earth terrain elevation (m) and 'CHM' is canopy height (m), so
        # DSM = DTM + CHM. Valid range filtering (0 < DTM < 100 m and
        # 0 <= CHM <= 60 m) excludes coastal water/nodata fringes and
        # atmospheric LiDAR return outliers.
        'image_preprocess_fn': lambda img: (
            img.addBands(
                img.select('DTM').add(img.select('CHM')).rename('dsm')
            ).updateMask(
                img.select('DTM')
                .gt(0)
                .And(img.select('DTM').lt(100))
                .And(img.select('CHM').gte(0))
                .And(img.select('CHM').lte(60))
            )
        ),
        'pts_per_tile_cap': None,
        'per_image_downsample': False,
        'description': (
            'Using EGM2008 geoid as local approximation for US Tropical'
            ' (NAVD88).'
        ),
    },
    'india_tech_mahindra': {
        'asset_id': ee_assets_config.LIDAR_INDIA_DRONE_ASSET_ID,
        'match_keys': ['india_tech_mahindra'],
        'display_name': 'India Tech Mahindra',
        'dsm_band': 'dsm',
        'dtm_band': 'dtm',
        'geoid_asset': None,
        'geoid_band': None,
        'geoid_scale': 0.0,
        'buffer_radius': 750,
        'date_range': None,
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': 3,
        'per_image_downsample': False,
        'description': (
            'Using WGS84 Ellipsoid (local_geoid=0) for Tech Mahindra'
            ' (converting raw ellipsoidal heights h -> H_EGM96 via h -'
            ' N_EGM96).'
        ),
    },
    'global_tech_mahindra': {
        'asset_id': ee_assets_config.LIDAR_GLOBAL_DRONE_ASSET_ID,
        'match_keys': ['global_tech_mahindra', 'tech_mahindra'],
        'display_name': 'Global Tech Mahindra',
        'dsm_band': 'dsm',
        'dtm_band': 'dtm',
        'geoid_asset': None,
        'geoid_band': None,
        'geoid_scale': 0.0,
        'buffer_radius': 750,
        'date_range': None,
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': 3,
        'per_image_downsample': False,
        'description': (
            'Using WGS84 Ellipsoid (local_geoid=0) for Tech Mahindra'
            ' (converting raw ellipsoidal heights h -> H_EGM96 via h -'
            ' N_EGM96).'
        ),
    },
    '3dtrees': {
        'asset_id': ee_assets_config.LIDAR_3DTREES_DRONE_ASSET_ID,
        'match_keys': ['3dtrees', 'drone_lidar'],
        'display_name': '3D Trees Drone Lidar',
        'dsm_band': 'dsm',
        'dtm_band': 'dtm',
        'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 1.0,
        'buffer_radius': 750,
        'date_range': None,
        'collection_filter': lambda col: col.filterBounds(
            ee.Geometry.BBox(-180, -60, 180, 60)
        ),
        # In 3D Trees Drone LiDAR assets (38 bands, 0-indexed):
        #   - Band 3 (4th band, 'dtm'): Digital Terrain Model elevation (m)
        #   - Band 4 (5th band, 'chm'): Canopy Height Model (m), DSM = dtm + chm
        #   - Band 37 (38th band, 'mask'): Valid footprint mask (1 = valid)
        # Quality filtering retains valid footprints (Band 37 == 1) with
        # physical canopy heights (0 <= Band 4 <= 60 m) and non-constant
        # terrain models.
        'image_preprocess_fn': lambda img: ee.Image.cat([
            img.select([3], ['dtm'])
            .toDouble()
            .add(img.select([4], ['chm']).toDouble())
            .updateMask(
                img.select([37])
                .eq(1)
                .And(img.select([4]).gte(0))
                .And(img.select([4]).lte(60))
                # Some tiles have the same value for the entire DTM band, these
                # are some interpolated tiles which we should skip in our
                # evaluation.
                .And(
                    img.select([3])
                    .reduceNeighborhood(
                        ee.Reducer.stdDev(), ee.Kernel.square(3)
                    )
                    .gt(1e-5)
                )
            )
            .rename('dsm'),
            img.select([3], ['dtm'])
            .toDouble()
            .updateMask(
                img.select([37])
                .eq(1)
                .And(img.select([4]).gte(0))
                .And(img.select([4]).lte(60))
                # Some tiles have the same value for the entire DTM band, these
                # are some interpolated tiles which we should skip in our
                # evaluation.
                .And(
                    img.select([3])
                    .reduceNeighborhood(
                        ee.Reducer.stdDev(), ee.Kernel.square(3)
                    )
                    .gt(1e-5)
                )
            )
            .rename('dtm'),
        ]),
        'pts_per_tile_cap': 3,
        'per_image_downsample': False,
        'description': (
            'Using EGM2008 geoid for 3D Trees Drone Lidar (Alaska / Boreal).'
        ),
    },
    'neon': {
        'asset_id': ee_assets_config.LIDAR_US_NEON_ASSET_ID,
        'match_keys': ['neon'],
        'display_name': 'US NEON Aerial Lidar',
        'dsm_band': 'DSM',
        'dtm_band': 'DTM',
        'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
        'geoid_band': 'b1',
        'geoid_scale': 1.0,
        'buffer_radius': 750,
        'date_range': ('2022-01-01', '2026-01-01'),
        'collection_filter': None,
        'image_preprocess_fn': None,
        'pts_per_tile_cap': None,
        'per_image_downsample': True,
        'description': (
            'Using local geoid model EGM2008 for US NEON 1m DSM/DTM (NAVD88,'
            ' 2022-2025 surveys).'
        ),
    },
}

# US NEON Yearly Aerial LiDAR Datasets (2013–2025)
for _year in range(2013, 2026):
  LIDAR_DATASET_CONFIGS[f'neon_{_year}'] = {
      'asset_id': ee_assets_config.LIDAR_US_NEON_ASSET_ID,
      'match_keys': [f'neon_{_year}', f'neon{_year}'],
      'display_name': f'US NEON Aerial Lidar ({_year})',
      'dsm_band': 'DSM',
      'dtm_band': 'DTM',
      'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
      'geoid_band': 'b1',
      'geoid_scale': 1.0,
      'buffer_radius': 750,
      'date_range': (f'{_year}-01-01', f'{_year + 1}-01-01'),
      'collection_filter': None,
      'image_preprocess_fn': None,
      'pts_per_tile_cap': None,
      'per_image_downsample': True,
      'description': (
          f'Using local geoid model EGM2008 for US NEON 1m DSM/DTM ({_year}'
          ' surveys).'
      ),
  }

# US NEON Constant Cohort Series 0: 3 California Sierra Sites (2013–2018)
NEON_SERIES_0_SITES = ['SJER', 'SOAP', 'TEAK']

# US NEON Constant Cohort Series 1: 14 Common Sites (2016–2018)
NEON_SERIES_1_SITES = [
    'BART',
    'CLBJ',
    'DELA',
    'DSNY',
    'GRSM',
    'HARV',
    'JERC',
    'KONZ',
    'LENO',
    'OAES',
    'ORNL',
    'OSBS',
    'TALL',
    'UKFS',
]

# US NEON Constant Cohort Series 2: 12 Common Sites (2019–2025)
NEON_SERIES_2_SITES = [
    'ABBY',
    'BONA',
    'CLBJ',
    'DEJU',
    'DSNY',
    'HEAL',
    'NOGP',
    'OAES',
    'OSBS',
    'SCBI',
    'WOOD',
    'WREF',
]


def _make_neon_site_filter(sites_list):
  return lambda col: col.filter(ee.Filter.inList('NEON_SITE', sites_list))


for _year in [2013, 2017, 2018]:
  LIDAR_DATASET_CONFIGS[f'neon_series_0_{_year}'] = {
      'asset_id': ee_assets_config.LIDAR_US_NEON_ASSET_ID,
      'match_keys': [f'neon_series_0_{_year}', f'neon_s0_{_year}'],
      'display_name': f'US NEON Series 0 ({_year})',
      'dsm_band': 'DSM',
      'dtm_band': 'DTM',
      'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
      'geoid_band': 'b1',
      'geoid_scale': 1.0,
      'buffer_radius': 750,
      'date_range': (f'{_year}-01-01', f'{_year + 1}-01-01'),
      'collection_filter': _make_neon_site_filter(NEON_SERIES_0_SITES),
      'image_preprocess_fn': None,
      'pts_per_tile_cap': None,
      'per_image_downsample': True,
      'description': (
          f'Using local geoid model EGM2008 for US NEON 1m DSM/DTM ({_year}'
          ' Series 0: 3 California Sierra sites).'
      ),
  }

for _year in [2016, 2017, 2018]:
  LIDAR_DATASET_CONFIGS[f'neon_series_1_{_year}'] = {
      'asset_id': ee_assets_config.LIDAR_US_NEON_ASSET_ID,
      'match_keys': [f'neon_series_1_{_year}', f'neon_s1_{_year}'],
      'display_name': f'US NEON Series 1 ({_year})',
      'dsm_band': 'DSM',
      'dtm_band': 'DTM',
      'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
      'geoid_band': 'b1',
      'geoid_scale': 1.0,
      'buffer_radius': 750,
      'date_range': (f'{_year}-01-01', f'{_year + 1}-01-01'),
      'collection_filter': _make_neon_site_filter(NEON_SERIES_1_SITES),
      'image_preprocess_fn': None,
      'pts_per_tile_cap': None,
      'per_image_downsample': True,
      'description': (
          f'Using local geoid model EGM2008 for US NEON 1m DSM/DTM ({_year}'
          ' Series 1: 14 constant core sites).'
      ),
  }

for _year in [2019, 2021, 2023, 2025]:
  LIDAR_DATASET_CONFIGS[f'neon_series_2_{_year}'] = {
      'asset_id': ee_assets_config.LIDAR_US_NEON_ASSET_ID,
      'match_keys': [f'neon_series_2_{_year}', f'neon_s2_{_year}'],
      'display_name': f'US NEON Series 2 ({_year})',
      'dsm_band': 'DSM',
      'dtm_band': 'DTM',
      'geoid_asset': ee_assets_config.GEOID_EGM2008_ASSET_ID,
      'geoid_band': 'b1',
      'geoid_scale': 1.0,
      'buffer_radius': 750,
      'date_range': (f'{_year}-01-01', f'{_year + 1}-01-01'),
      'collection_filter': _make_neon_site_filter(NEON_SERIES_2_SITES),
      'image_preprocess_fn': None,
      'pts_per_tile_cap': None,
      'per_image_downsample': True,
      'description': (
          f'Using local geoid model EGM2008 for US NEON 1m DSM/DTM ({_year}'
          ' Series 2: 12 constant core sites).'
      ),
  }


def get_lidar_config(key_or_path):
  """Retrieves configuration parameters for a given LiDAR dataset key or path."""
  if not key_or_path:
    raise ValueError('LiDAR dataset key or path must be specified.')
  key_lower = key_or_path.lower()
  if key_lower in LIDAR_DATASET_CONFIGS:
    return LIDAR_DATASET_CONFIGS[key_lower]
  for _, cfg in LIDAR_DATASET_CONFIGS.items():
    if key_lower == cfg['asset_id'].lower() or any(
        key_lower == k.lower() for k in cfg['match_keys']
    ):
      return cfg
  raise ValueError(
      f'Unsupported or unconfigured LiDAR dataset key/path: "{key_or_path}". '
      f'Supported keys: {list(LIDAR_DATASET_CONFIGS.keys())}'
  )
