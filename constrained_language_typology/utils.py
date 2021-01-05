# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Miscellaneous utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

# pylint: disable=superfluous-parens


def haversine_distance(origin, destination):
  """Calculates the Haversine distance.

  Args:
    origin: (tuple) A latitude/longitude pair for origin point.
    destination: (tuple) A latitude/longitude pair for destination.

  Returns:
    Distance in km (floating point value).

  Examples
  --------
  >>> munich = (48.1372, 11.5756)
  >>> berlin = (52.5186, 13.4083)
  >>> round(haversine_distance(munich, berlin), 1)
  504.2

  >>> new_york_city = (40.712777777778, -74.005833333333)  # NYC
  >>> round(haversine_distance(berlin, new_york_city), 1)
  6385.3
  """
  lat1, lon1 = origin
  lat2, lon2 = destination
  if not (-90.0 <= lat1 <= 90):
    raise ValueError("lat1={:2.2f}, but must be in [-90,+90]".format(lat1))
  if not (-90.0 <= lat2 <= 90):
    raise ValueError("lat2={:2.2f}, but must be in [-90,+90]".format(lat2))
  if not (-180.0 <= lon1 <= 180):
    raise ValueError("lon1={:2.2f}, but must be in [-180,+180]".format(lon1))
  if not (-180.0 <= lon2 <= 180):
    raise ValueError("lon2={:2.2f}, but must be in [-180,+180]".format(lon2))
  radius = 6371  # km.

  dlat = math.radians(lat2 - lat1)
  dlon = math.radians(lon2 - lon1)
  a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * (
      math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2))
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  d = radius * c
  return d


def n_vector(lat, lon):
  """Converts lat/long to n-vector 3D Cartesian representation."""
  # Convert to radians.
  if not (-90.0 <= lat <= 90):
    raise ValueError("lat={:2.2f}, but must be in [-90,+90]".format(lat))
  rad_lat = math.radians(lat)
  if not (-180.0 <= lon <= 180):
    raise ValueError("lon={:2.2f}, but must be in [-180,+180]".format(lon))
  rad_lon = math.radians(lon)

  x = math.cos(rad_lat) * math.cos(rad_lon)
  y = math.cos(rad_lat) * math.sin(rad_lon)
  z = math.sin(rad_lat)
  return x, y, z


def locations_centroid(locations):
  """Computes the centroid of a set of latitude/longitude tuples.

  Args:
    locations: (list) A list of latitude/longitude tuples.

  Returns:
    Centroid point as a latitude/longitude tuple.

  Example:
  --------
  >>> munich = (48.1372, 11.5756)
  >>> berlin = (52.5186, 13.4083)
  >>> the_hague = (52.0705, 4.3007)
  >>> locations_centroid([munich, berlin, the_hague])
  (50.97369621759347, 9.802006312301206)  # ==> Am Mühlrain, 36179 Bebra.
  """
  # Cartesian coordinates.
  x = 0.0
  y = 0.0
  z = 0.0
  for lat, lon in locations:
    # Convert individual lat/long pair to Cartesian coordinates.
    d_x, d_y, d_z = n_vector(lat, lon)

    # Accumulate the average.
    x += d_x
    y += d_y
    z += d_z

  # Compute final average.
  x /= len(locations)
  y /= len(locations)
  z /= len(locations)

  # Convert the Cartesian average back to latitude and longitude (in radians).
  lon = math.atan2(y, x)
  hyp = math.sqrt(x * x + y * y)
  lat = math.atan2(z, hyp)

  # Back to degrees.
  return (math.degrees(lat), math.degrees(lon))
