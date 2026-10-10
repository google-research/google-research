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

"""Centralized Earth Engine asset configuration for GEM-15 benchmarking.

Public global elevation models and spaceborne benchmarks point to their public
Earth Engine Catalog asset IDs. For local or custom aerial/drone LiDAR
collections, vertical datum geoid offset grids, CORS station tables, and
high-resolution optical imagery, replace the `projects/YOUR_PROJECT/assets/...`
placeholders with the Earth Engine asset paths in your Cloud project.
"""

# ==============================================================================
# 1. GEM-15 & Public Global DEM / Reference Catalog Assets
# ==============================================================================
GEM15_ASSET_ID = 'projects/climate-and-sustainability/assets/gem-15_v1'
COPERNICUS_GLO30_ASSET_ID = 'COPERNICUS/DEM/GLO30_2024_1'
ALOS_AW3D30_ASSET_ID = 'JAXA/ALOS/AW3D30/V4_1'
NASADEM_ASSET_ID = 'NASA/NASADEM_HGT/001'
FABDEM_ASSET_ID = 'projects/sat-io/open-datasets/FABDEM'
SRTM_ASSET_ID = 'USGS/SRTMGL1_003'

GEDI_L2A_MONTHLY_ASSET_ID = 'LARSE/GEDI/GEDI02_A_002_MONTHLY'
ESA_WORLDCOVER_ASSET_ID = 'ESA/WorldCover/v200'
MODIS_MCD12Q1_ASSET_ID = 'MODIS/061/MCD12Q1'
LSIB_SIMPLE_ASSET_ID = 'USDOS/LSIB_SIMPLE/2017'
USDA_NAIP_ASSET_ID = 'USDA/NAIP/DOQQ'

# ==============================================================================
# 2. Geoid Undulation & Vertical Datum Offset Grids (Placeholders)
# ==============================================================================
GEOID_EGM96_ASSET_ID = 'projects/YOUR_PROJECT/assets/geoid_offsets/egm96'
GEOID_EGM2008_ASSET_ID = 'projects/YOUR_PROJECT/assets/geoid_offsets/egm2008'
GEOID_UK_OSGM15_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/geoid_offsets/uk_os_OSGM15_GB'
)
GEOID_CANADA_CGVD2013_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/geoid_offsets/ca_nrc_CGG2013an83'
)
GEOID_SPAIN_REDNAP_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/geoid_offsets/es_ign_egm08-rednap'
)
GEOID_NZ_NZVD2016_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/geoid_offsets/nz_linz_nzgeoid2016'
)

# ==============================================================================
# 3. Aerial & Drone LiDAR Reference Datasets (Placeholders)
# ==============================================================================
LIDAR_UK_EA_ASSET_ID = 'UK/EA/ENGLAND_1M_TERRAIN/2022'
LIDAR_CANADA_ASSET_ID = 'projects/YOUR_PROJECT/assets/lidar/lidar_canada_hrdem'
LIDAR_SPAIN_ASSET_ID = 'projects/YOUR_PROJECT/assets/lidar/lidar_spain'
LIDAR_NEW_ZEALAND_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/lidar/lidar_new_zealand'
)
LIDAR_MOZAMBIQUE_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/lidar/lidar_mozambique'
)
LIDAR_INDONESIA_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/lidar/lidar_indonesia_kalimantan'
)
LIDAR_BRAZIL_ASSET_ID = 'projects/YOUR_PROJECT/assets/lidar/lidar_brazil'
LIDAR_US_TROPICAL_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/lidar/lidar_us_tropical'
)
LIDAR_INDIA_DRONE_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/lidar/india_tech_mahindra'
)
LIDAR_GLOBAL_DRONE_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/lidar/global_tech_mahindra'
)
LIDAR_3DTREES_DRONE_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/3dtrees/drone_lidar'
)
LIDAR_US_NEON_ASSET_ID = 'projects/neon-prod-earthengine/assets/DEM/001'

# ==============================================================================
# 4. Geodetic GNSS / CORS Stations & High-Resolution Optical RGB (Placeholders)
# ==============================================================================
CORS_STATIONS_ASSET_ID = 'projects/YOUR_PROJECT/assets/ngl_global_cors_2025'
HIGH_RES_OPTICAL_RGB_ASSET_ID = (
    'projects/YOUR_PROJECT/assets/high_res_optical_rgb'
)
