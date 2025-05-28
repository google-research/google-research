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

"""Tests for geometry."""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.utilities import data_utils
from eq_mag_prediction.utilities import geometry

NUMERICAL_ERROR = 1e-10  # a threshold for anumerical error


class RectangleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Excluding antimeridian',
          min_lng=141,
          max_lng=145,
          min_lat=36,
          max_lat=42,
          expected=(143, 39),
      ),
      dict(
          testcase_name='Including antimeridian',
          min_lng=160,
          max_lng=-140,
          min_lat=-20,
          max_lat=-10,
          expected=(-170, -15),
      ),
  )
  def test_center(self, min_lng, max_lng, min_lat, max_lat, expected):
    rectangle = geometry.Rectangle(min_lng, max_lng, min_lat, max_lat)

    self.assertAlmostEqual(rectangle.center, geometry.Point(*expected))

  def test_y_centroid(self):
    rectangle = geometry.Rectangle(141, 145, 36, 42)

    self.assertAlmostEqual(rectangle.y_centroid(), 39)

  def test_project_longitude(self):
    rectangle = geometry.Rectangle(141, 145, 36, 42)

    self.assertAlmostEqual(
        geometry.project_longitude(141, rectangle), 109.57758, 5
    )

  def test_inverse_project_longitude(self):
    rectangle = geometry.Rectangle(141, 145, 36, 42)

    self.assertAlmostEqual(
        geometry.inverse_project_longitude(
            geometry.project_longitude(143, rectangle), rectangle
        ),
        143,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='Excluding antimeridian',
          min_lng=141,
          max_lng=145,
          min_lat=36,
          max_lat=42,
      ),
      dict(
          testcase_name='Including antimeridian',
          min_lng=160,
          max_lng=-140,
          min_lat=-20,
          max_lat=-10,
      ),
  )
  def test_sample_random_location_produces_samples_within_bounds(
      self, min_lng, max_lng, min_lat, max_lat
  ):
    rectangle = geometry.Rectangle(min_lng, max_lng, min_lat, max_lat)

    for _ in range(100):
      sample = rectangle.sample_random_location()
      lng, lat = geometry.utm_to_lng_lat(*sample)

      self.assertBetween(lat, min_lat, max_lat)
      if min_lng < max_lng:
        self.assertBetween(lng, min_lng, max_lng)
      else:
        self.assertTrue((min_lng < lng <= 180) or (-180 < lng < max_lng))


class RectanglesTest(parameterized.TestCase):

  def test_init_grid(self):
    locations = geometry.Rectangles.init_grid(
        longitude_range=(140, 141), latitude_range=(37.5, 39), side_deg=0.5
    )
    self.assertEqual(locations.side_deg, 0.5)
    self.assertSequenceEqual(
        locations.rectangles,
        [
            [
                geometry.Rectangle(140, 140.5, 37.5, 38),
                geometry.Rectangle(140, 140.5, 38, 38.5),
                geometry.Rectangle(140, 140.5, 38.5, 39),
            ],
            [
                geometry.Rectangle(140.5, 141, 37.5, 38),
                geometry.Rectangle(140.5, 141, 38, 38.5),
                geometry.Rectangle(140.5, 141, 38.5, 39),
            ],
        ],
    )
    self.assertEqual(locations.length, 6)

  def test_init_grid_includes_antimeridian(self):
    locations = geometry.Rectangles.init_grid(
        longitude_range=(178, -179), latitude_range=(37, 39), side_deg=1
    )
    self.assertEqual(locations.side_deg, 1)
    self.assertSequenceEqual(
        locations.rectangles,
        [
            [
                geometry.Rectangle(178, 179, 37, 38),
                geometry.Rectangle(178, 179, 38, 39),
            ],
            [
                geometry.Rectangle(179, 180, 37, 38),
                geometry.Rectangle(179, 180, 38, 39),
            ],
            [
                geometry.Rectangle(180, -179, 37, 38),
                geometry.Rectangle(180, -179, 38, 39),
            ],
        ],
    )
    self.assertEqual(locations.length, 6)

  @parameterized.named_parameters(
      dict(
          testcase_name='(140, 38)',
          location=(412503.0, 4205532.6),
          expected_index=0,
      ),
      dict(
          testcase_name='(140.8, 38.1)',
          location=(482766.3, 4216175.2),
          expected_index=4,
      ),
      dict(
          testcase_name='(140.9, 38.51)',
          location=(491583.4, 4261654.1),
          expected_index=5,
      ),
  )
  def test_index_of(self, location, expected_index):
    locations = geometry.Rectangles.init_grid(
        longitude_range=(140, 141), latitude_range=(37.5, 39), side_deg=0.5
    )

    self.assertEqual(locations.index_of(location), expected_index)

  def test_index_of_outside_of_range(self):
    location = data_utils.PROJECTIONS['japan'](137, 38)
    locations = geometry.Rectangles.init_grid(
        longitude_range=(140, 141), latitude_range=(37.5, 39), side_deg=0.5
    )

    with self.assertRaisesRegex(ValueError, 'Location .* is not in any .*'):
      _ = locations.index_of(location)

  def test_to_centers(self):
    locations = geometry.Rectangles.init_grid(
        longitude_range=(140, 141), latitude_range=(37.5, 39), side_deg=0.5
    )

    centers = locations.to_centers()
    expected_centers = [
        [
            geometry.Point(140.25, 37.75),
            geometry.Point(140.25, 38.25),
            geometry.Point(140.25, 38.75),
        ],
        [
            geometry.Point(140.75, 37.75),
            geometry.Point(140.75, 38.25),
            geometry.Point(140.75, 38.75),
        ],
    ]

    self.assertSequenceAlmostEqual(centers, expected_centers)

  def test_utm_to_lng_lat(self):
    lngs = np.random.uniform(low=140, high=145, size=100)
    lats = np.random.uniform(low=30, high=40, size=100)
    for lng, lat in zip(lngs, lats):
      x, y = data_utils.PROJECTIONS['japan'](lng, lat)
      lng_converted, lat_converted = geometry.utm_to_lng_lat(x, y)

      self.assertAlmostEqual(lng_converted, lng, 2)
      self.assertAlmostEqual(lat_converted, lat, 2)


class GeometryTest(parameterized.TestCase):
  rnd_seed = np.random.RandomState(seed=1905)

  @parameterized.named_parameters(
      dict(
          testcase_name='x around z to y',
          vec_2_rotate=[1, 0, 0],
          angle=np.pi / 2,
          rot_axis=[0, 0, 1],
          expected_vector=[0, 1, 0],
      ),
      dict(
          testcase_name='x around z in 45deg',
          vec_2_rotate=[1, 0, 0],
          angle=np.pi / 4,
          rot_axis=[0, 0, 1],
          expected_vector=[1 / np.sqrt(2), 1 / np.sqrt(2), 0],
      ),
      dict(
          testcase_name='x around y to z',
          vec_2_rotate=[1, 0, 0],
          angle=np.pi / 2,
          rot_axis=[0, 1, 0],
          expected_vector=[0, 0, -1],
      ),
      dict(
          testcase_name='x around y to z, length 0.5',
          vec_2_rotate=[0.5, 0, 0],
          angle=np.pi / 2,
          rot_axis=[0, 1, 0],
          expected_vector=[0, 0, -0.5],
      ),
      dict(
          testcase_name='y around x to z',
          vec_2_rotate=[0, 1, 0],
          angle=np.pi / 2,
          rot_axis=[1, 0, 0],
          expected_vector=[0, 0, 1],
      ),
      dict(
          testcase_name='180deg of z around vector between x-z to x',
          vec_2_rotate=[0, 0, 1],
          angle=np.pi,
          rot_axis=[1, 0, 1],
          expected_vector=[1, 0, 0],
      ),
      dict(
          testcase_name='x no rotation around arbitrary axis',
          vec_2_rotate=[1, 0, 0],
          angle=0,
          rot_axis=rnd_seed.uniform(size=(3,)),
          expected_vector=[1, 0, 0],
      ),
  )
  def test_rotation_matrix_about_axis(
      self, vec_2_rotate, angle, rot_axis, expected_vector
  ):
    """Verify rotation matrix is computed correctly in some known and interpetable cases.

    Args:
      vec_2_rotate: An np.ndarray or list of len=3. The vector to be rotated.
      angle: The angle to rotate the vector. In radians.
      rot_axis: The rotation axis. Same format as vec_2_rotate.
      expected_vector: The expected vector after rotation. Same format as
        vec_2_rotate.
    """
    # ensure a column vector
    vec_2_rotate = np.array(vec_2_rotate).ravel()[:, None]
    expected_vector = np.array(expected_vector).ravel()[:, None]
    rot_axis = np.array(rot_axis).ravel()
    rot_mat = geometry.rotation_matrix_about_axis(angle, rot_axis)
    rotated_vector = np.matmul(rot_mat, vec_2_rotate)
    np.testing.assert_allclose(
        expected_vector, rotated_vector, rtol=0, atol=NUMERICAL_ERROR
    )

  def test_verify_basic_rot_matrix_properties(self):
    rnd_seed = np.random.RandomState(seed=1905)
    rot_axis = rnd_seed.uniform(size=(3,))
    # with angle 0 for rotation matrix should be identity
    rot_mat_0 = geometry.rotation_matrix_about_axis(0, rot_axis)
    np.testing.assert_allclose(
        np.eye(3), rot_mat_0, rtol=0, atol=NUMERICAL_ERROR
    )

    # verify that the rotation matrix R holds: R(a+b)=R(a)R(b)
    # create to angles in the range [1e-2, pi]
    angle1 = rnd_seed.rand(1) * (np.pi - 1e-2) + 1e-2
    angle2 = rnd_seed.rand(1) * (np.pi - 1e-2) + 1e-2
    rot_mat_sum = geometry.rotation_matrix_about_axis(angle1 + angle2, rot_axis)
    rot_mat_1 = geometry.rotation_matrix_about_axis(angle1, rot_axis)
    rot_mat_2 = geometry.rotation_matrix_about_axis(angle2, rot_axis)
    rot_mat_multip = np.matmul(rot_mat_1, rot_mat_2)
    np.testing.assert_allclose(
        rot_mat_multip, rot_mat_sum, rtol=0, atol=NUMERICAL_ERROR
    )

  def test_zeros_vector_as_axis_behavior(self):
    rnd_seed = np.random.RandomState(seed=1905)
    # create an angle in the range [1e-2, pi]
    rotation_angle = rnd_seed.rand(1) * (np.pi - 1e-2) + 1e-2
    vector_2_rotate = rnd_seed.rand(3)
    zeros_vector = np.zeros(3)
    rot_mat = geometry.rotation_matrix_about_axis(rotation_angle, zeros_vector)
    rotated_vector = np.matmul(rot_mat, vector_2_rotate)
    np.testing.assert_allclose(
        vector_2_rotate, rotated_vector, rtol=0, atol=NUMERICAL_ERROR
    )
    expected_rotation = np.eye(3)
    np.testing.assert_allclose(
        rot_mat, expected_rotation, rtol=0, atol=NUMERICAL_ERROR
    )

  def test_rotation_matrix_between_vectors_in_3d(self):
    number_of_tests = 10
    rnd_seed = np.random.RandomState(seed=1905)
    v1 = rnd_seed.rand(number_of_tests, 3)
    v2 = rnd_seed.rand(number_of_tests, 3)
    for test_number in range(number_of_tests):
      rotation_matrix = geometry.rotation_matrix_between_vectors_in_3d(
          v1[test_number], v2[test_number]
      )

      # 1st direction
      rotated_v1 = np.matmul(rotation_matrix, v1[test_number])
      normalized_rotated_v1, _ = geometry._normalize_vector(rotated_v1)
      normalized_v2, v2_size = geometry._normalize_vector(v2[test_number])
      np.testing.assert_allclose(
          normalized_rotated_v1.ravel(),
          normalized_v2.ravel(),
          rtol=0,
          atol=NUMERICAL_ERROR,
      )
      np.testing.assert_allclose(
          normalized_rotated_v1.ravel() * v2_size,
          v2[test_number],
          rtol=0,
          atol=NUMERICAL_ERROR,
      )

      # 2nd direction
      rotated_v2 = np.matmul(np.linalg.inv(rotation_matrix), v2[test_number])
      normalized_rotated_v2, _ = geometry._normalize_vector(rotated_v2)
      normalized_v1, v1_size = geometry._normalize_vector(v1[test_number])
      np.testing.assert_allclose(
          normalized_rotated_v2.ravel(),
          normalized_v1.ravel(),
          rtol=0,
          atol=NUMERICAL_ERROR,
      )
      np.testing.assert_allclose(
          normalized_rotated_v2.ravel() * v1_size,
          v1[test_number],
          rtol=0,
          atol=NUMERICAL_ERROR,
      )

  @parameterized.named_parameters(
      dict(testcase_name='similar and parallel', prefactor=1.0),
      dict(testcase_name='similar and anti-parallel', prefactor=-1.0),
  )
  def test_rotation_between_parallel_vectors(self, prefactor):
    rnd_seed = np.random.RandomState(seed=1905)
    n_rotations = 5
    # generate vectors
    source_vector = rnd_seed.uniform(size=(n_rotations, 3))
    # generate noise
    noise_for_axes = (NUMERICAL_ERROR * 1e-2) * rnd_seed.uniform(
        size=(n_rotations, 3)
    )
    noisy_target_vector = prefactor * source_vector + noise_for_axes
    for ax in range(n_rotations):
      resulting_rotation = geometry.rotation_matrix_between_vectors_in_3d(
          source_vector[ax], noisy_target_vector[ax]
      )
      resulting_target_vector = np.matmul(resulting_rotation, source_vector[ax])
      np.testing.assert_allclose(
          prefactor * source_vector[ax],
          resulting_target_vector,
          rtol=0,
          atol=NUMERICAL_ERROR,
      )


if __name__ == '__main__':
  absltest.main()
