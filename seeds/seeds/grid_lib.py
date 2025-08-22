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

"""Base classes and utility functions for grids on the sphere."""

from __future__ import annotations

import abc
from collections.abc import Callable
import dataclasses
import functools
import math

import numpy as np
import scipy.spatial as ss
import xarray as xr


def xyz_to_lonlat(points):
  """Maps 3D Cartesian coordinates to spherical longitude-latitude coordinates.

  Notes:
    - The origin is mapped to (0, 0) by convention.
    - The range for the latitude is -90 (south pole) to 90 (north pole) and that
      for the longitude is 0 to 360.

  Args:
    points: An array of the shape (..., 3) for the coordinates of the points in
      the Cartesian coordinates in 3D.

  Returns:
    A 2-tuple of (latitude, longitude) for the mapped input points on the
    sphere.
  """
  if points.shape[-1] != 3:
    raise ValueError('This only works for 3D Cartesian coordinates.')
  rxy = np.hypot(points[Ellipsis, 0], points[Ellipsis, 1])
  lat = np.arctan2(points[Ellipsis, 2], rxy) / np.pi * 180.0
  lon = np.arctan2(points[:, 1], points[:, 0]) / np.pi * 180.0 % 360.0
  return lon, lat


def lonlat_to_xyz(lon, lat):
  """Maps longitude-latitude coordinates to 3D Cartesian coordinates.

  Args:
    lon: An array of longitudes in degrees.
    lat: An array of latitudes in degrees.

  Returns:
    An array with the last dimension for the 3 Cartesian coordinates.

  Note:
    `xyz_to_lonlat` is the inverse of this function but not vice versa: the
    former maps the entire Euclidean space (including the origin by convention)
    to the unit sphere, while `lonlat_to_xyz` is an embedding of the sphere into
    the Euclidean space.
  """
  lon_ = lon / 180 * np.pi
  lat_ = lat / 180 * np.pi
  return np.array(
      [np.cos(lon_) * np.cos(lat_), np.sin(lon_) * np.cos(lat_), np.sin(lat_)]
  ).T


class GridSphere(abc.ABC):
  """Base class for grids on the sphere."""

  @property
  @abc.abstractmethod
  def grid_points(self):
    """Returns the grid points in longitudes and latitudes."""

  @property
  @abc.abstractmethod
  def num_points(self):
    """Returns the total number of grid points."""

  @classmethod
  @abc.abstractmethod
  def on(cls, data):
    """Infers a grid from data whose last axis matches its grid points."""

  @abc.abstractmethod
  def to(self, other):
    """Returns an interpolator to the other grid."""

  @property
  def data_shape(self):
    """Returns the shape of data on this grid."""
    return (self.num_points,)

  @property
  @abc.abstractmethod
  def angular_resolution(self):
    """Returns the approximate angular resolution of this grid."""

  def plot_gridder(self):
    """Makes a regridder to Equirectangular at a similar angular resolution.

    Returns:
      The regridder function which takes an np.array on this grid to a DataArray
      on the Equirectangular. This is mostly used for plotting.
    """
    # Make an interpolator to the standard lat-lon grid.
    n = int(np.ceil(180.0 / self.angular_resolution))
    grid = Equirectangular.from_shape((n + 1, n * 2))
    interpolator = self.to(grid)

    def gridder(data, dataarray=True):
      """Regrids data to Equirectangular.

      Args:
        data: Input data whose last dimensions matches the points on a grid.
        dataarray: If True, returns a DataArray with metadata. If False, returns
          the regridded numpy array.

      Returns:
        Regridded data.
      """
      interpolated = interpolator(data)
      if dataarray:
        return xr.DataArray(
            interpolated,
            coords={'latitude': grid.latitude, 'longitude': grid.longitude},
        )
      return interpolated

    return gridder


@dataclasses.dataclass
class Interpolator(abc.ABC):
  """Interpolator from the source grid to the target grid.

  Attributes:
    source: The source grid.
    target: The target grid.
  """

  source: GridSphere
  target: GridSphere

  @abc.abstractmethod
  def __call__(self, data):
    """Returns the interpolated data."""


@dataclasses.dataclass
class IDWInterpolator(Interpolator):
  """Inverse distance weighting interpolator.

  Reference: https://en.wikipedia.org/wiki/Inverse_distance_weighting

  This can be used to interpolate between arbitrary grids on the sphere using
  the Euclidean distance in the ambient space.

  A natural alternative is to use the geodesic distance on the sphere. Recall
  that for two points on the sphere at half central angle θ, the Euclidean
  distance is 2sin(θ) while the spherical distance is 2θ. Since sin is monotonic
  on [0,π/2], the Euclidean neighbors are the same as the spherical neighbors.
  Since θ≥sin(θ), the Euclidean IDW is slightly smoother than the spherical IDW.

  Attributes:
    source: The source grid.
    target: The target grid.
    neighbors: The number of closets neighbors to use.
  """

  neighbors: int
  _neighbor_indices: np.ndarray = dataclasses.field(init=False, repr=False)
  _weights: np.ndarray = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    # Embed points on the sphere into the ambient Euclidean space.
    source_points = lonlat_to_xyz(*self.source.grid_points)
    target_points = lonlat_to_xyz(*self.target.grid_points)
    # Compute the closest neighbors and the inverse distance weight.
    tree = ss.KDTree(source_points)
    distance, self._neighbor_indices = tree.query(target_points, self.neighbors)
    inverse_distance = 1 / (distance + np.finfo(target_points.dtype).eps)
    self._weights = inverse_distance / inverse_distance.sum(axis=1)[:, None]

  def __call__(self, data):
    """Returns the interpolated data."""
    source_shape = self.source.data_shape
    source_ndim = len(source_shape)
    if data.shape[-source_ndim:] != source_shape:
      raise ValueError(
          f'The trailing dimensions of data {data.shape[-source_ndim:]} do'
          f' not match the source grid data shape {source_shape}.'
      )
    leading_dims = data.shape[:-source_ndim]
    result = np.sum(data[Ellipsis, self._neighbor_indices] * self._weights, axis=-1)
    target_shape = self.target.data_shape
    target_ndim = len(target_shape)
    if result.shape[-target_ndim:] != target_shape:
      result = result.reshape(leading_dims + target_shape)
    return result


@dataclasses.dataclass(frozen=True)
class CubedSphere(GridSphere):
  """Equi-angular cubedsphere grid for the sphere.

  Attributes:
    size: The number of grid points on each cube edge. The angular resolution is
      (90 / size).

  Reference: https://geos-chem.seas.harvard.edu/cubesphere_comparison.
  """

  size: int

  @functools.cached_property
  def grid_points(self):
    """Returns the grid points in longitudes and latitudes."""
    # Make the equi-angular grid for the 90 degrees arc.
    gap = np.pi / 2 / self.size
    edge_grid_in_radians = np.linspace(
        -np.pi / 4 + gap / 2, np.pi / 4 - gap / 2, self.size
    )
    # Project the grid to a straight edge between -1 to 1.
    edge_grid = (np.tan(edge_grid_in_radians) + 1) / 2
    # Make the reference square.
    x, y = np.meshgrid(edge_grid, edge_grid)
    ref = np.vstack([x.flatten(), y.flatten()]).T
    # Map the reference square to each face of the cube.
    sq2 = np.sqrt(2)
    corners = np.array([
        [[sq2, 0, -1], [0, sq2, -1], [sq2, 0, 1]],
        [[0, sq2, -1], [-sq2, 0, -1], [0, sq2, 1]],
        [[-sq2, 0, -1], [0, -sq2, -1], [-sq2, 0, 1]],
        [[0, -sq2, -1], [sq2, 0, -1], [0, -sq2, 1]],
        [[-sq2, 0, 1], [0, -sq2, 1], [0, sq2, 1]],
        [[0, sq2, -1], [sq2, 0, -1], [-sq2, 0, -1]],
    ]).transpose(0, 2, 1)
    grid_points = np.einsum(
        'ij,klj->ikl', ref, corners[:, :, 1:] - corners[:, :, 0, None]
    )
    grid_points += corners[:, :, 0]
    grid_points = grid_points.transpose(1, 0, 2).reshape(-1, 3)
    # Gnomonic projection of the points on the cube to the sphere.
    return xyz_to_lonlat(grid_points)

  @property
  def num_points(self):
    """Returns the total number of grid points."""
    return int(6 * self.size**2)

  @property
  def angular_resolution(self):
    """Returns the angular resolution of this grid."""
    return 90.0 / self.size

  @classmethod
  def on(cls, data):
    """Infers a grid from data whose last axis matches its grid points."""
    num_points = data.shape[-1]
    size = math.isqrt(num_points // 6)
    result = cls(size)
    if num_points != result.num_points:
      raise ValueError(f'No cubedsphere has {num_points} grid points.')
    return result

  def to(self, other):
    """Returns an interpolator to the other grid."""
    return IDWInterpolator(self, other, neighbors=4)

  def to_faces(self, data):
    """Reshapes the data from (..., num_grid_points) to (..., 6, size, size)."""
    return data.reshape(data.shape[:-1] + (6, self.size, self.size))


@dataclasses.dataclass(frozen=True)
class Equirectangular(GridSphere):
  """Rectangular longitude-latitude grid for the sphere.

  Reference: https://en.wikipedia.org/wiki/Equirectangular_projection

  Attributes:
    longitude: The longitude nodes.
    latitude: The latitude nodes.  The ratio of len(longitude) and len(latitude)
      should be around 2. The angular resolution is 360 / len(longitude).
  """

  longitude: np.ndarray
  latitude: np.ndarray

  @functools.cached_property
  def grid_points(self):
    """Returns the grid points in longitudes and latitudes."""
    lon, lat = np.meshgrid(self.longitude, self.latitude)
    return lon.flatten(), lat.flatten()

  @functools.cached_property
  def num_points(self):
    """Returns the total number of grid points."""
    return len(self.longitude) * len(self.latitude)

  @property
  def angular_resolution(self):
    """Returns the angular resolution of this grid."""
    return np.median(np.diff(self.longitude))

  @property
  def data_shape(self):
    """Returns the shape of data on this grid."""
    return len(self.latitude), len(self.longitude)

  @classmethod
  def from_shape(cls, shape):
    """Makes a standard Equirectangular from the shape of associated data.

    The standard grid has 2n nodes for longitudes in [0, 360) and n+1 nodes for
    latitudes in [-90, 90].

    Args:
      shape: The data shape (num_latitudes, num_longitudes).

    Returns:
      The standard grid matching the shape.
    """
    num_lats, num_lons = shape
    if num_lons % 2 == 0 and num_lats == num_lons // 2 + 1:
      return cls(
          longitude=np.linspace(0, 360, num_lons, endpoint=False),
          latitude=np.linspace(-90, 90, num_lats),
      )
    raise ValueError('The given shape is not for a standard rectangular grid.')

  @classmethod
  def on(cls, data):
    """Infers a grid from data whose last axis matches its grid points."""
    if isinstance(data, xr.DataArray):
      return cls(longitude=data.longitude.data, latitude=data.latitude.data)
    else:
      return cls.from_shape(data.shape[-2:])
    raise ValueError('Cannot infer the grid of the data')

  def to(self, other):
    """Returns an interpolator to the other grid."""
    return BilinearInterpolator(self, other)


_BilinearInterpolatorCache = list[tuple[np.ndarray, np.ndarray, float]]


def _compute_bilinear_interpolator_cache(
    source_x,
    source_y,
    target_x,
    target_y,
):
  """Prepares cached data for bilinear interpolation.

  The bilinear interpolation between two fixed grids is a sparse matrix vector
  product, because the value at an output grid point is a weighted sum of
  the values at the 4 corner vertices of the square the point is in. This
  function computes the indicial map from the source to the target and the
  corresponding weights. Given the data here _bilinear_interpolate evaluates
  the sparse matrix vector product.

  Args:
    source_x: The source grid x coordinates.
    source_y: The source grid y coordinates.
    target_x: The x coordinates of the target points.
    target_y: The y coordinates of the target points.  The format for the source
      and the target are different. If interpolating onto the source grid, we
      would have target_x, target_y = np.meshgrid(source_x, source_y).

  Returns:
    Cached data for evaluating _eval_bilinear_interpolate() on the given grids.
  """
  if (
      target_x.min() < source_x.min()
      or target_y.min() < source_y.min()
      or target_x.max() > source_x.max()
      or target_y.max() > source_y.max()
  ):
    raise ValueError('Some target points are outside of the source grid.')
  # Identify the squares the target points are in.
  index_x = np.digitize(target_x, source_x, right=True)
  index_y = np.digitize(target_y, source_y, right=True)
  # Compute the local coordinates of the points in their corresponding squares.
  local_x = (target_x - source_x[index_x - 1]) / (
      source_x[index_x] - source_x[index_x - 1]
  )
  local_y = (target_y - source_y[index_y - 1]) / (
      source_y[index_y] - source_y[index_y - 1]
  )
  # Save the indexers and the weights
  cache = []
  cache.append((index_y - 1, index_x - 1, (1 - local_x) * (1 - local_y)))
  cache.append((index_y - 1, index_x, local_x * (1 - local_y)))
  cache.append((index_y, index_x - 1, local_y * (1 - local_x)))
  cache.append((index_y, index_x, local_x * local_y))
  return cache


def _eval_bilinear_interpolate(
    cache, data
):
  """Computes the bilinear interpolation of the data using cached grid data."""
  result = 0
  for index_y, index_x, weights in cache:
    result += data[Ellipsis, index_y, index_x] * weights
  return np.asarray(result)


@dataclasses.dataclass
class BilinearInterpolator(Interpolator):
  """Bilinear interpolator for rectangular lon-lat grid on the sphere.

  Attributes:
    source: The source grid.
    target: The target grid.
  """

  _cache: _BilinearInterpolatorCache = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    if not isinstance(self.source, Equirectangular):
      raise ValueError('This only supports interpolation from Equirectangular.')
    source_x = self.source.longitude
    num_longitudes = len(source_x)
    # Wrap the longitude around.
    source_x = np.concatenate([source_x, [360.0]])
    source_y = self.source.latitude
    cache = _compute_bilinear_interpolator_cache(
        source_x, source_y, *self.target.grid_points
    )
    # Wrap the mapped longitude index around.
    self._cache = [
        (iy, ix % num_longitudes, weights) for iy, ix, weights in cache
    ]

  def __call__(self, data):
    """Returns the interpolated data."""
    return _eval_bilinear_interpolate(self._cache, data)
