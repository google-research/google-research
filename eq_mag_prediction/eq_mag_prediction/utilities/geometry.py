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

"""Functions that have to do with the spatial properties of GETAS."""

import dataclasses
import functools
import itertools
import math
from typing import List, Sequence, Tuple

import gin
import numpy as np
import pandas as pd
import pyproj

from eq_mag_prediction.utilities import data_utils

_NUMERICAL_ERROR = 1e-10


def _deg2lng(degrees):
  if degrees > 180:
    return degrees - 360
  return degrees


class _LongitudeRange:
  """Represents a longitude range."""

  def __init__(self, min_lng, max_lng):
    self.min_lng = min_lng
    self.max_lng = max_lng
    self.includes_antimeridian = max_lng < min_lng

  def midpoint(self):
    if not self.includes_antimeridian:
      return (self.min_lng + self.max_lng) / 2
    return _deg2lng((self.min_lng + self.max_lng) / 2 + 180)

  def random_in_range(self):
    if not self.includes_antimeridian:
      return np.random.uniform(low=self.min_lng, high=self.max_lng)
    return _deg2lng(
        np.random.uniform(low=self.min_lng, high=self.max_lng + 360)
    )

  def int_sized_arange(self, delta_deg, digits):
    int_delta = int(np.round(delta_deg * (10**digits)))
    max_lng = self.max_lng + 360 if self.includes_antimeridian else self.max_lng
    int_lng_range = (
        int(np.round(self.min_lng * (10**digits))),
        int(np.round(max_lng * (10**digits))),
    )
    return np.arange(*int_lng_range, int_delta)


@gin.configurable
@dataclasses.dataclass(frozen=True)
class Point:
  """Specifies a point in longitude-latitude coordinates."""

  lng: float
  lat: float

  def __str__(self):
    return f'{self.lng}_{self.lat}'


@gin.configurable
@dataclasses.dataclass(frozen=True)
class Rectangle:
  """Specifies a rectangle in longitude-latitude coordinates."""

  min_lng: float
  max_lng: float
  min_lat: float
  max_lat: float

  def __post_init__(self):
    """Validates the input."""
    assert -180 < self.min_lng <= 180
    assert -180 < self.max_lng <= 180
    assert -90 <= self.min_lat < self.max_lat <= 90

  @classmethod
  @gin.configurable('Rectangle.from_coordinate_ranges')
  def from_coordinate_ranges(
      cls,
      longitude_range,
      latitude_range,
  ):
    """Initializes from longitude and latitude ranges."""
    return Rectangle(
        min_lng=longitude_range[0],
        max_lng=longitude_range[1],
        min_lat=latitude_range[0],
        max_lat=latitude_range[1],
    )

  def utm_center(
      self, projection = data_utils.japan_projection()
  ):
    """Returns the center of the rectangle in UTM coordinates."""
    return projection(
        _LongitudeRange(self.min_lng, self.max_lng).midpoint(),
        (self.min_lat + self.max_lat) / 2,
    )

  @functools.cached_property
  def center(self):
    """Returns the center of the rectangle in longitude-latitude coordinates."""
    return Point(
        lng=_LongitudeRange(self.min_lng, self.max_lng).midpoint(),
        lat=(self.min_lat + self.max_lat) / 2,
    )

  def y_centroid(self):
    """Calculates the y-centroid of the rectangle.

    This is a generic implementation that could be extended to general polygons.
    It currently only works if all of the coordinates are positive.

    Returns:
      The y-coordinate of the centroid of the rectangle.
    """
    points = [
        (self.min_lng, self.min_lat),
        (self.max_lng, self.min_lat),
        (self.max_lng, self.max_lat),
        (self.min_lng, self.max_lat),
    ]
    points.append(points[0])

    area = 0
    ymoment = 0

    for i in range(len(points) - 1):
      triangle_area = (
          points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1]
      )
      area += triangle_area
      ymoment += (points[i][1] + points[i + 1][1]) * triangle_area

    area /= 2
    return ymoment / (6 * area)

  def project_longitude(self):
    """Projects the rectangle longitude with respect to itself."""
    return Rectangle(
        project_longitude(self.min_lng, self),
        project_longitude(self.max_lng, self),
        self.min_lat,
        self.max_lat,
    )

  def sample_random_location(
      self, projection = data_utils.japan_projection()
  ):
    """Uniformly samples a location inside the Rectangle."""
    sample = (
        _LongitudeRange(self.min_lng, self.max_lng).random_in_range(),
        np.random.uniform(low=self.min_lat, high=self.max_lat),
    )
    return projection(*sample)


@dataclasses.dataclass(frozen=True)
class Rectangles:
  """A container for a grid of rectangles, with useful methods.

  Attributes:
    rectangles: The list of rectangles.
    side_deg: The length of the side of the rectangles of the grid (in degrees).
    length: The number of rectangles in the grid.
  """

  rectangles: Sequence[Sequence[Rectangle]]
  side_deg: float
  length: int

  @classmethod
  @gin.configurable(denylist=['cls'])
  def init_grid(
      cls,
      longitude_range = (141, 145),
      latitude_range = (36, 42),
      side_deg = 10e3,
  ):
    """Calculates evenly spaced rectangles.

    Args:
      longitude_range: The range of the longitude (x) side of the rectangle in
        which earthquakes are kept.
      latitude_range: The range of the latitude (y) side of the rectangle in
        which earthquakes are kept.
      side_deg: The length of the side of the rectangles of the grid in degrees.

    Returns:
      A list of rectangles for the requested grid.
    """
    if (
        np.round(side_deg, 2) != side_deg
        or np.round(longitude_range[0], 2) != longitude_range[0]
        or np.round(longitude_range[1], 2) != longitude_range[1]
        or np.round(latitude_range[0], 2) != latitude_range[0]
        or np.round(latitude_range[1], 2) != latitude_range[1]
    ):
      raise ValueError('The grid precision is up to 2 significant digits')

    int_side_deg = int(np.round(side_deg * 100))
    int_lat_range = (
        int(np.round(latitude_range[0] * 100)),
        int(np.round(latitude_range[1] * 100)),
    )

    min_lng_points = _LongitudeRange(*longitude_range).int_sized_arange(
        delta_deg=side_deg, digits=2
    )
    min_lat_points = np.arange(*int_lat_range, int_side_deg)

    rectangles = []
    for lng in min_lng_points:
      row = []
      for lat in min_lat_points:
        row.append(
            Rectangle(
                _deg2lng(lng / 100),
                _deg2lng((lng + int_side_deg) / 100),
                lat / 100,
                (lat + int_side_deg) / 100,
            )
        )
      rectangles.append(row)

    return Rectangles(
        rectangles, side_deg, sum([len(row) for row in rectangles])
    )

  def index_of(self, location):
    """Returns the index of the bin that contains the input location.

    Args:
      location: In UTM coordinates.

    Returns:
      The index of the bin that contains that location.
    """
    # Currently the location is in UTM, because the evaluation space-times are
    # in UTM. This can cause some issues due to curvature. In particular, it is
    # hard to exclude the endge of the rectangle. It is probably not critical to
    # the performance of the model - most locations will not be on the edge of a
    # bin. I'll fix this in a separate CL.
    lng, lat = utm_to_lng_lat(*location)
    flattened_rectangles = list(itertools.chain(*self.rectangles))
    for i, rectangle in enumerate(flattened_rectangles):
      if rectangle.min_lng <= lng <= rectangle.max_lng:
        if rectangle.min_lat <= lat <= rectangle.max_lat:
          return i
    raise ValueError(
        f'Location {location} - ({lng}, {lat}) is not in any of the bins {self}'
    )

  def to_centers(self):
    """Finds the centers of all rectangles in longitude-latitude coordinates."""
    return [[rectangle.center for rectangle in row] for row in self.rectangles]


def rotation_matrix_about_axis(angle, axis):
  """Generate a 3D rotation matrix of a given angle about a given axis.

  This function uses to matrix form of the Rodrigues' rotation formula:
  https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
  https://mathworld.wolfram.com/RodriguesRotationFormula.html

  Args:
    angle: The angle of rotation in radians
    axis: A 1D, 3-element numpy array defining the axis of rotation

  Returns:
    A 3x3 numpy array which describes the rotation
      of angle about axis.
  """
  # Verify u is a len=3, 1D vector
  assert all([(d in (3, 1)) for d in axis.shape])
  # If axis is the zeros vector, return identity (no rotation) instead of null:
  if np.allclose(np.zeros_like(axis), axis, rtol=0, atol=_NUMERICAL_ERROR):
    return np.eye(3)
  axis = axis.ravel()
  axis = axis / np.linalg.norm(axis)

  u_vector_product_matrix = np.array([
      [0, -axis[2], axis[1]],
      [axis[2], 0, -axis[0]],
      [-axis[1], axis[0], 0],
  ])
  u_outer_prod = axis.reshape((-1, 1)) * axis.reshape((1, -1))
  rotation_matrix_about_u_by_t = (
      np.cos(angle) * np.eye(3)
      + np.sin(angle) * u_vector_product_matrix
      + (1 - np.cos(angle)) * u_outer_prod
  )
  return rotation_matrix_about_u_by_t


def rotation_matrix_between_vectors_in_3d(
    source_vector, target_vector
):
  """Returns the roation matrix that aligns the source to the target vector."""
  # If requested rotatian is from vector to itself, return identity:
  source_vector_normalized = _normalize_vector(source_vector)[0].ravel()
  target_vector_normalized = _normalize_vector(target_vector)[0].ravel()
  dot_product = np.dot(source_vector_normalized, target_vector_normalized)
  parallel_vectors = np.allclose(
      np.abs(dot_product), 1, rtol=0, atol=_NUMERICAL_ERROR
  )
  if parallel_vectors:
    return np.sign(dot_product) * np.eye(3)
  if np.allclose(
      source_vector_normalized,
      target_vector_normalized,
      rtol=0,
      atol=_NUMERICAL_ERROR,
  ):
    return np.eye(3)
  rotation_axis, rotation_angle = get_rotation_axis_and_angle(
      source_vector, target_vector
  )
  return rotation_matrix_about_axis(rotation_angle, rotation_axis)


def get_rotation_axis_and_angle(
    source_vector, target_vector
):
  """Calculates a relevant rotation axis and angle between two 3d vectors."""
  source_vector_normalized = _normalize_vector(source_vector)[0].ravel()
  target_vector_normalized = _normalize_vector(target_vector)[0].ravel()
  rotation_angle = np.arccos(
      np.dot(source_vector_normalized, target_vector_normalized)
  )
  rotation_axis, _ = _normalize_vector(
      np.cross(source_vector_normalized, target_vector_normalized)
  )
  return rotation_axis, rotation_angle


def _normalize_vector(v):
  vector_norm = np.linalg.norm(v)
  normalized_vector = np.array(v).ravel()[:, None] / vector_norm
  return normalized_vector, vector_norm


def utm_to_lng_lat(x, y):
  """Converts UTM coordinates back to longitude and latitude."""
  lng, lat = data_utils.PROJECTIONS['japan'](x, y, inverse=True)
  lng_deg, lat_deg = math.floor(lng), math.floor(lat)
  # Converting back to minutes, because then we can limit to the precision of
  # the catalog - 2 decimal points. This helps deal with points near the
  # boundary of rectangles.
  lng_minutes = np.round((lng - lng_deg) * 60, 2)
  lat_minutes = np.round((lat - lat_deg) * 60, 2)
  return lng_deg + lng_minutes / 60, lat_deg + lat_minutes / 60


@gin.configurable
def init_rectangles_grid(
    longitude_range,
    latitude_range,
    side_deg,
):
  """A thin wrapper around Rectangles.init_grid, for scoped binding."""
  return Rectangles.init_grid(longitude_range, latitude_range, side_deg)


def project_longitude(longitude, rectangle):
  """Projects a longitude with respect to a rectangle."""
  return longitude * np.cos(rectangle.y_centroid() * np.pi / 180)


def inverse_project_longitude(longitude, rectangle):
  """Inverts the projection of the longitude with respect to a rectangle."""
  return longitude / np.cos(rectangle.y_centroid() * np.pi / 180)


def project_longitude_catalog(
    catalog, rectangle
):
  """Projects the longitude column with respect to a rectangle."""
  projected = catalog.copy()
  projected['projected_longitude'] = project_longitude(
      catalog['longitude'].values, rectangle
  )
  projected.drop(labels=['longitude'], axis='columns', inplace=True)
  return projected


def inverse_project_longitude_catalog(
    projected, rectangle
):
  """Inverts the projection of the longitude with respect to a rectangle."""
  catalog = projected.copy()
  catalog['longitude'] = inverse_project_longitude(
      projected['projected_longitude'].values, rectangle
  )
  catalog.drop(labels=['projected_longitude'], axis='columns', inplace=True)
  return catalog
