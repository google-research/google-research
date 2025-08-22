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

"""Geometry utilities."""

from typing import Union

import chex
from internal import rigid_body
from internal import spin_math
import jax
from jax import numpy as jnp
from jax import random
import numpy as onp
import optax


_ArrayType = Union[onp.ndarray, jnp.ndarray]


def line_distance(point1, dir1, point2,
                  dir2):
  """Compute the distance between two lines in 3D.

  Note that this is the distance between lines and not line segments or rays;
  i.e., it does not consider endpoints and will compute the distance assuming
  the line extends infinitely in both directions.

  Args:
    point1: (3,) a point on the first line.
    dir1: (3,) the direction vector of the first line.
    point2: (3,) a point on the second line.
    dir2: (3,) the direction vector of the second line.

  Returns:
    The distance between the two lines.
  """
  is_parallel = are_lines_parallel(dir1, dir2)
  skew_dist = skew_line_distance(point1, dir1, point2, dir2)
  parallel_dist = line_to_point_distance(point1, dir1, point2)

  return jnp.where(is_parallel, parallel_dist, skew_dist)


def skew_line_closest_points(point1, dir1,
                             point2,
                             dir2):
  """Compute the shortest distance between two skew lines.

  See:
    https://en.wikipedia.org/wiki/Skew_lines#Nearest_points


  Args:
    point1: a point on the first line.
    dir1: the direction vector of the first line.
    point2: a point on the second line.
    dir2: the direction vector of the second line.

  Returns:
    The distance between the two skew lines.
  """
  # Make sure direction vectors are unit.
  dir1 = spin_math.normalize(dir1)
  dir2 = spin_math.normalize(dir2)

  # The vector perpendicular to both lines.
  n = jnp.cross(dir1, dir2)

  # Compute the point on line 1 nearest to line 2.
  n2 = jnp.cross(dir2, n)
  c1 = point1 + jnp.dot(point2 - point1, n2) / jnp.dot(dir1, n2) * dir1

  # Compute the point on line 2 nearest to line 1.
  n1 = jnp.cross(dir1, n)
  c2 = point2 + jnp.dot(point1 - point2, n1) / jnp.dot(dir2, n1) * dir2

  return c1, c2  # pytype: disable=bad-return-type  # jax-ndarray


def skew_line_distance(point1, dir1,
                       point2, dir2):
  """Compute the shortest distance between two skew lines.

  Args:
    point1: a point on the first line.
    dir1: the direction vector of the first line.
    point2: a point on the second line.
    dir2: the direction vector of the second line.

  Returns:
    The distance between the two skew lines.
  """
  c1, c2 = skew_line_closest_points(point1, dir1, point2, dir2)
  return jnp.linalg.norm(c1 - c2)


def line_closest_point(line_point, line_dir,
                       query_point):
  """Return the closest point on the line to a point.

  Args:
    line_point: a point on the line.
    line_dir: the direction vector of the line.
    query_point: the query point.

  Returns:
    The closest point on the line to the query point.
  """
  # Make sure direction vector is unit.
  line_dir = spin_math.normalize(line_dir)
  # Find the point along the line that is closest.
  t = jnp.dot(query_point - line_point, line_dir)
  return line_point + t * line_dir


def line_to_point_distance(line_point, line_dir,
                           query_point):
  """Return the distance from point to a line.

  Args:
    line_point: a point on the line.
    line_dir: the direction vector of the line.
    query_point: the point to compute the distance to.

  Returns:
    The closest distance between the line and the point.
  """
  closest_point = line_closest_point(line_point, line_dir, query_point)
  return jnp.linalg.norm(query_point - closest_point)


def ray_sphere_intersection(origin,
                            direction,
                            radius = 1.0):
  """Computes the intersecting point between a ray and a sphere.

  Variables use notation from Wikipedia:
    u: direction of ray
    o: origin of ray

  References:
    https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

  Args:
    origin: The origin of the ray.
    direction: The direction of the ray.
    radius: The radius of the sphere.

  Returns:
    The intersecting point on the sphere.
  """
  u_dot_o = jnp.sum(direction * origin, axis=-1, keepdims=True)
  nabla = u_dot_o**2 - (jnp.linalg.norm(origin, keepdims=True)**2 - radius**2)
  # This this is a ray and not a line, we only need to consider the case where
  # nabla is positive.
  distance = -u_dot_o + jnp.sqrt(nabla)
  return origin + distance * direction


def are_lines_parallel(dir1, dir2):
  eps = jnp.finfo(jnp.float32).eps
  dir1 = spin_math.normalize(dir1)
  dir2 = spin_math.normalize(dir2)
  return jnp.dot(dir1, dir2) >= 1.0 - eps  # pytype: disable=bad-return-type  # jnp-type


def spherical_equirectangular_grid(
    height,
    width,
    min_elevation = 0,
    max_elevation = jnp.pi,
    min_azimuth = 0,
    max_azimuth = 2 * jnp.pi):
  """Creates an equirectangular grid (panorama) in spherical coordinates.

  Args:
    height: The height of the output grid.
    width: The width of the output grid.
    min_elevation: The minimum value for the elevation.
    max_elevation: The maximum value for the elevation.
    min_azimuth: The minimum value for the azimuth.
    max_azimuth: The maximum value for the azimuth.

  Returns:
    elevations: (height, width) An array containing the elevations.
    azimuths: (height, width) An array containing the azimuths.
  """
  elevations = jnp.linspace(min_elevation, max_elevation, height)
  # Prevent duplicate sample since 0 and 2*pi are the same azimuth.
  azimuths = jnp.linspace(min_azimuth, max_azimuth, width, endpoint=False)
  azimuths, elevations = jnp.meshgrid(azimuths, elevations)
  return elevations, azimuths  # pytype: disable=bad-return-type  # jax-ndarray


def spherical_to_cartesian(
    r,
    theta,
    phi,
):
  """Converts spherical to cartesian coordinates.

  For more details see cartesian_to_spherical below.
  Args:
    r: (..., 1) Radius of spherical coordinate.
    theta: (..., 1) Elevation of spherical coordinate.
    phi: (..., 1) Azimuth of spherical coordinate.

  Returns:
    Cartesian coordinates of shape (..., 3) defined by x, y, z.
  """
  x = r * jnp.sin(theta) * jnp.cos(phi)
  y = r * jnp.sin(theta) * jnp.sin(phi)
  z = r * jnp.cos(theta)

  return jnp.stack([x, y, z], axis=-1)


def cartesian_to_spherical(
    cartesian_vector,
    eps = onp.float32(onp.finfo(onp.float32).tiny)
):
  """Converts cartesian to spherical coordinates.

  Uses a right-handed coordinate system where z is up and y is right. The
  spherical coordinates are defined by radius (r), inclination (theta)
  ∈ [0, π]) from fixed zenit direction (z) and azimuth (phi) ∈ [0, 2π]) from
  x-axis to y-axis.

  We are using the phyiscal coordinate system as described here:
  https://en.wikipedia.org/wiki/Spherical_coordinate_system.

  Args:
    cartesian_vector: (..., 3) Cartesian coordinates defined by (x, y, z).
    eps: Epsilon used for safe_acos.

  Returns:
    Spherical coordinates as tuple of r, elevation (theta), azimuth (phi).
  """
  x = cartesian_vector[Ellipsis, 0]
  y = cartesian_vector[Ellipsis, 1]
  z = cartesian_vector[Ellipsis, 2]

  r = optax.safe_norm(cartesian_vector, min_norm=eps, axis=-1)
  theta = spin_math.safe_acos(z / r)
  phi = jnp.arctan2(y, x)
  return r, theta, phi  # pytype: disable=bad-return-type  # jax-ndarray


def sample_random_points_on_sphere(key, num_points,
                                   min_radius,
                                   max_radius):
  """Sample points uniformly on sphere with random radius within bounds.

  Args:
    key: Seed for random sampling.
    num_points: Number of points to sample.
    min_radius: Minimum euclidean distance of point from center of sphere.
    max_radius: Maximum euclidean distance of point from center of sphere.

  Returns:
    Array of uniform points (N, 3) on sphere with random radius.
  """

  key1, key2, _ = random.split(key, 3)

  random_radii = random.uniform(
      key1, (num_points, 1), minval=min_radius, maxval=max_radius)
  v = spin_math.normalize(random.normal(key2, (num_points, 3)))

  return v * random_radii  # pytype: disable=bad-return-type  # jax-ndarray


def sample_points_evenly_on_sphere(num_points,):
  """Deterministically sample points on a sphere that are evenly distributed.

  Uses a generalization of the sunflower spiral to sample points that are
  distibuted evenly on a sphere.

  References:
    http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069
    https://mathoverflow.net/questions/24850/is-there-a-generalisation-of-the-sunflower-spiral-to-higher-dimensions
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075

  Args:
    num_points: The number of points to sample.

  Returns:
    (num_points, 3) The sampled points.
  """
  golden_ratio = (1 + 5**0.5) / 2
  indices = jnp.arange(0, num_points, dtype=jnp.float32) + 0.5
  azimuths = jnp.pi * 2 * golden_ratio * indices
  elevations = jnp.arccos(1 - 2 * indices / num_points)
  points = spherical_to_cartesian(1.0, elevations, azimuths)  # pytype: disable=wrong-arg-types  # jax-ndarray
  return points


def is_point_in_convex_hull(point,
                            hull_normals,
                            hull_offsets,
                            padding = 0.0):
  """Computes whether the given points are inside or outside a convex hull.

  The convex hull is defined using the normals and offsets of a facet.
  If the dot product between a point and a normal is less than the offset, then
  it is on the inner side of that facet. If this is true for all facets, then
  the point is inside the convex hull.

  References:
    http://www.qhull.org/html/index.htm
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html

  Args:
    point: (..., D) An array containing the points to test.
    hull_normals: (F, D) The normals of the facets of the convex hull.
    hull_offsets: (F, D) The offsets of the facets of the convex hull.
    padding: A number to pad the convex hull by. A positive value expands the
      convex hull while a negative number shrinks it.

  Returns:
    A boolean array of shape (...,) that is True if a point is inside the hull
    and False otherwise.
  """
  input_shape = point.shape[:-1]
  point = point.reshape(-1, point.shape[-1])
  dots = hull_normals @ point.T
  mask = (dots <= -hull_offsets[:, None] + padding).all(axis=0)
  return mask.reshape(input_shape)


def cosine_to_deg(array):
  """Converts cosine angle to degrees.

  Args:
    array: containing cosine angles (e.g. result of dot product).

  Returns:
    array with angles as degrees.
  """
  return jnp.degrees(jnp.arccos(array.clip(-1, 1)))


# TODO(phenzler): Convert this to xnp once we have a more solid code base that
# supports xnp.
def onp_cosine_to_deg(array):
  """Converts cosine angle to degrees.

  Args:
    array: containing cosine angles (e.g. result of dot product).

  Returns:
    array with angles as degrees.
  """
  return onp.degrees(onp.arccos(array.clip(-1, 1)))


def rotation_distance(rotation_mat1,
                      rotation_mat2):
  """Computes the angle between two rotation matrices in degrees.

  Args:
    rotation_mat1: (3, 3) The first batch of rotation matrix.
    rotation_mat2: (3, 3) The second batch of rotation matrix.

  Returns:
    The angle in degrees between 0 and 180.
  """
  axis_angle1 = rigid_body.log_so3(rotation_mat1)
  axis_angle2 = rigid_body.log_so3(rotation_mat2)
  orientation_error_deg = jnp.degrees(
      jnp.linalg.norm(axis_angle1 - axis_angle2, axis=-1))
  return jnp.where(
      orientation_error_deg < 180,
      orientation_error_deg,  # pytype: disable=bad-return-type  # jnp-type
      360 - orientation_error_deg)


def compute_bbox_from_xyza(
    xyza,
    padding,
    alpha_threshold = 0.99,
):
  """Computes a bounding box given an xyza array.

  Args:
    xyza: An array of shape (..., 4) containing the XYZ coordinates in the first
      three channels and an alpha value in the last.
    padding: A padding value to be added to all sides.
    alpha_threshold: The threshold at which to binarize the alpha into a mask.

  Returns:
    A bounding box of shape (2, 3) containing (min_coords, max_coords).
  """
  padding = onp.array(padding)
  xyz = xyza[Ellipsis, :3]
  alpha = xyza[Ellipsis, 3]
  mask = alpha > alpha_threshold
  xyz = xyz[mask]
  xyz = xyz.reshape(-1, 3)
  min_coord = xyz.min(axis=0) - padding
  max_coord = xyz.max(axis=0) + padding
  return onp.stack([min_coord, max_coord], axis=0)