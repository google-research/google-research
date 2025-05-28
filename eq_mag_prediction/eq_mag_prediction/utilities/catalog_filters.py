# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""A module containing ways to filter the catalog."""

from typing import Callable, Optional, Tuple

import geopandas as gpd
import gin
import numpy as np
import pandas as pd
import shapely

from eq_mag_prediction.utilities import geometry
from eq_mag_prediction.utilities import time_conversions

EarthquakeCriterionType = Callable[[pd.DataFrame], np.ndarray]


def is_in_rectangle_utm(
    catalog,
    center,
    x_side_meters,
    y_side_meters,
):
  """Returns a mask indicating whether each earthquake is in a rectangle."""
  return (
      (catalog.x_utm < center[0] + x_side_meters / 2)
      & (catalog.x_utm >= center[0] - x_side_meters / 2)
      & (catalog.y_utm < center[1] + y_side_meters / 2)
      & (catalog.y_utm >= center[1] - y_side_meters / 2)
  )


def is_in_rectangle(
    catalog,
    longitude_range,
    latitude_range,
):
  """Returns a mask indicating whether each earthquake is in a rectangle."""
  from_lng, to_lng = longitude_range
  from_lat, to_lat = latitude_range
  if to_lng < 0 and from_lng > 0:
    return is_in_rectangle(
        catalog, (-180, to_lng), latitude_range
    ) | is_in_rectangle(catalog, (from_lng, 180), latitude_range)
  assert to_lat > from_lat
  return (
      (catalog.longitude < to_lng)
      & (catalog.longitude >= from_lng)
      & (catalog.latitude < to_lat)
      & (catalog.latitude >= from_lat)
  )


def is_in_polygon(
    catalog,
    longitude_coords,
    latitude_coords,
):
  """Returns a mask indicating whether each earthquake is in a polygon."""
  assert len(longitude_coords) == len(latitude_coords)
  if len(longitude_coords) == 2:
    return is_in_rectangle(catalog, longitude_coords, latitude_coords)
  gdf = gpd.GeoDataFrame({
      'geometry': [
          shapely.geometry.Point(r.longitude, r.latitude)
          for r in catalog.itertuples()
      ]
  })
  polygon = shapely.geometry.Polygon(zip(longitude_coords, latitude_coords))
  return gdf.within(polygon).values


def limit_catalog_to_square(
    catalog, center, side_deg
):
  """Returns a sub-catalog limited to a square, given in lng/lat coordinates."""
  return catalog[
      is_in_rectangle(
          catalog,
          (center.lng - side_deg / 2, center.lng + side_deg / 2),
          (center.lat - side_deg / 2, center.lat + side_deg / 2),
      )
  ]


@gin.configurable(denylist=['catalog'])
def earthquake_criterion(
    catalog,
    *,
    longitude_range = (140, 141),
    latitude_range = (36.5, 37.5),
    start_timestamp = time_conversions.datetime_japan_to_time(2011, 4),
    end_timestamp = None,
    max_depth = 40,
    min_magnitude = 0.5,
):
  """Returns a mask for earthquakes, according to common criteria.

  The default values that are provided match the Iwaki preset.
  Note that downstream the earthquakes are binned by location, but filtering
  by some square at this stage can improve performance.

  Args:
    catalog: The catalog of earthquakes.
    longitude_range: The range of the longitude (x) side of the rectangle in
      which earthquakes are kept.
    latitude_range: The range of the latitude (y) side of the rectangle in which
      earthquakes are kept.
    start_timestamp: Only keep earthquakes starting with this timestamp (seconds
      since  Epoch).
    end_timestamp: (optional) Only keep earthquakes ending before ot up to this
      timestamp (seconds since  Epoch). Defaults to the maximal time in the
      catalog.
    max_depth: Maximal depth of earthquakes to keep.
    min_magnitude: Minimal magnitude of earthquakes to keep.

  Returns:
    A boolean numpy array, True where the corresponding earthquake in the
    catalog matches all of the criteria above.
  """

  end_timestamp = catalog.time.max() if end_timestamp is None else end_timestamp
  return (
      (catalog.depth <= max_depth)
      & (catalog.magnitude >= min_magnitude)
      & (catalog.time >= start_timestamp)
      & (catalog.time <= end_timestamp)
      & is_in_polygon(catalog, longitude_range, latitude_range)
  )


@gin.configurable
def return_entire_catalog_criterion(catalog):
  """An earthquake identity criterion, returns the entire catalog."""
  return np.full(catalog.shape[:1], True)
