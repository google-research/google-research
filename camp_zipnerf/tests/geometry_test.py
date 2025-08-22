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

"""Tests for math_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from internal import geometry
from jax import random
import numpy as np


class GeometryTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('non_unit_same', np.array([1, 2, 3]), np.array([1, 2, 3])),
      ('non_unit_diff', np.array([2, 2, 2]), np.array([1, 1, 1])),
      ('unit_same', np.array([1, 0, 0]), np.array([1, 0, 0])),
      ('unit_diff', np.array([1, 0, 0]), np.array([2, 0, 0])),
  )
  def test_are_lines_parallel_parallel(self, d1, d2):
    self.assertTrue(geometry.are_lines_parallel(d1, d2))

  @parameterized.named_parameters(
      ('skew', np.array([1, 0, 0]), np.array([1, 1, 1])),
      ('perpendicular', np.array([1, 0, 0]), np.array([0, 1, 0])),
  )
  def test_are_lines_parallel_not_parallel(self, d1, d2):
    self.assertFalse(geometry.are_lines_parallel(d1, d2))

  @parameterized.named_parameters(
      dict(
          testcase_name='identical',
          p1=np.array([1, 1, 1]),
          d1=np.array([3, 2, 1]),
          p2=np.array([1, 1, 1]),
          d2=np.array([3, 2, 1]),
          dist=0.0,
      ),
      dict(
          testcase_name='parallel',
          p1=np.array([0, 0, 0]),
          d1=np.array([1, 0, 0]),
          p2=np.array([0, 1, 0]),
          d2=np.array([1, 0, 0]),
          dist=1.0,
      ),
      dict(
          testcase_name='skew',
          p1=np.array([0, 0, 0]),
          d1=np.array([1, 0, 1]),
          p2=np.array([0, 3, 0]),
          d2=np.array([1, 0, 0]),
          dist=3.0,
      ),
      dict(
          testcase_name='intersect',
          p1=np.array([0, 0, 1]),
          d1=np.array([0, 0, -1]),
          p2=np.array([-1, 0, 0]),
          d2=np.array([1, 0, 0]),
          dist=0.0,
      ),
  )
  def test_line_distance(self, p1, d1, p2, d2, dist):
    pred_dist = geometry.line_distance(p1, d1, p2, d2)
    np.testing.assert_almost_equal(pred_dist, dist)

  @parameterized.named_parameters(
      dict(
          testcase_name='y_axis',
          p=np.array([0.0, 0.0, 0.0]),
          d=np.array([0.0, 1.0, 0.0]),
          query_point=np.array([2.0, 2.0, 2.0]),
          closest_point=np.array([0.0, 2.0, 0.0]),
      ),
      dict(
          testcase_name='point_on_line',
          p=np.array([0.0, 0.0, 0.0]),
          d=np.array([1.0, 1.0, 1.0]),
          query_point=np.array([2.0, 2.0, 2.0]),
          closest_point=np.array([2.0, 2.0, 2.0]),
      ),
      dict(
          testcase_name='perpendicular',
          p=np.array([0.0, 0.0, 0.0]),
          d=np.array([1.0, 1.0, 0.0]),
          query_point=np.array([-1.0, 1.0, 0.0]),
          closest_point=np.array([0.0, 0.0, 0.0]),
      ),
  )
  def test_line_closest_point(self, p, d, query_point, closest_point):
    pred_closest_point = geometry.line_closest_point(p, d, query_point)
    np.testing.assert_array_almost_equal(pred_closest_point, closest_point)

  @parameterized.named_parameters(
      ('x', np.array([1.0, 0, 0])),
      ('y', np.array([0, 1.0, 0])),
      ('z', np.array([0, 0, 1.0])),
      ('-x', np.array([-1, 0, 0.0])),
      ('-y', np.array([0, -1, 0.0])),
      ('-z', np.array([0, 0, -1.0])),
      ('0', np.array([0, 0, 0.0])),
  )
  def test_coordinate_transform_round_trip(self, x):
    r, theta, phi = geometry.cartesian_to_spherical(x)
    x_hat = geometry.spherical_to_cartesian(r, theta, phi)
    np.testing.assert_array_almost_equal(x_hat, x, decimal=3)

  @parameterized.named_parameters(
      dict(
          testcase_name='right',
          cartesian=np.array([0.0, 3.0, 0.0]),
          r=3,
          theta=np.pi / 2,
          phi=np.pi / 2,
      ),
      dict(
          testcase_name='up',
          cartesian=np.array([0.0, 0.0, 2.0]),
          r=2,
          theta=0,
          phi=0,
      ),
      dict(
          testcase_name='front',
          cartesian=np.array([1.0, 0.0, 0.0]),
          r=1,
          theta=np.pi / 2,
          phi=0,
      ),
  )
  def test_coordinate_transform(self, cartesian, r, theta, phi):
    r_hat, theta_hat, phi_hat = geometry.cartesian_to_spherical(cartesian)

    np.testing.assert_almost_equal(r, r_hat, decimal=3)
    np.testing.assert_almost_equal(phi, phi_hat, decimal=3)
    np.testing.assert_almost_equal(theta, theta_hat, decimal=3)

    cartesian_hat = geometry.spherical_to_cartesian(r, theta, phi)
    np.testing.assert_array_almost_equal(cartesian, cartesian_hat, decimal=3)

  @parameterized.named_parameters(
      dict(
          testcase_name='monte_carlo',
          num_cameras=int(1e6),
          min_radius=0.7,
          max_radius=1.5,
      )
  )
  def test_sphere_point_sampling(self, num_cameras, min_radius, max_radius):
    rng = random.PRNGKey(42)

    points = geometry.sample_random_points_on_sphere(
        rng, num_cameras, min_radius, max_radius
    )
    np.testing.assert_array_almost_equal(
        np.mean(points, axis=0), np.zeros(3), decimal=3
    )


if __name__ == '__main__':
  absltest.main()