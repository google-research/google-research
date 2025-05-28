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

"""Tests for earthquake properties."""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.seismology import earthquake_properties
from eq_mag_prediction.utilities import geometry

_NUMERICAL_ERR = 1e-8


class EarthquakePropertiesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no rotation at all',
          strike=0.0,
          rake=0.0,
          dip=0.0,
          expected_normal=np.array([0, 0, -1]),
          expected_strike=np.array([1, 0, 0]),
          expected_rake=np.array([1, 0, 0]),
      ),
      dict(
          testcase_name='dip 0 strike 0 rake 90deg',
          strike=0.0,
          rake=np.deg2rad(90),
          dip=0.0,
          expected_normal=np.array([0, 0, -1]),
          expected_strike=np.array([1, 0, 0]),
          expected_rake=np.array([0, -1, 0]),
      ),
      dict(
          testcase_name='dip 0 strike 0 rake 270deg',
          strike=0.0,
          rake=np.deg2rad(270),
          dip=0.0,
          expected_normal=np.array([0, 0, -1]),
          expected_strike=np.array([1, 0, 0]),
          expected_rake=np.array([0, 1, 0]),
      ),
      dict(
          testcase_name='dip 45deg strike 0 rake 0',
          strike=0,
          rake=0,
          dip=np.deg2rad(45.0),
          expected_normal=np.array([0, 2**-0.5, -(2**-0.5)]),
          expected_strike=np.array([1, 0, 0]),
          expected_rake=np.array([1, 0, 0]),
      ),
      dict(
          testcase_name='dip 90deg strike 45 rake 0',
          strike=np.deg2rad(45.0),
          rake=0,
          dip=np.deg2rad(90.0),
          expected_normal=np.array([2**-0.5, 2**-0.5, 0]),
          expected_strike=np.array([2**-0.5, -(2**-0.5), 0]),
          expected_rake=np.array([2**-0.5, -(2**-0.5), 0]),
      ),
  )
  def test_moment_vectors_from_angles(
      self, strike, rake, dip, expected_normal, expected_strike, expected_rake
  ):
    eq_vectors = earthquake_properties.moment_vectors_from_angles(
        strike, rake, dip
    )
    result_normal_vector, result_strike_vector, result_rake_vector = eq_vectors
    np.testing.assert_allclose(
        result_normal_vector, expected_normal, rtol=0, atol=_NUMERICAL_ERR
    )
    np.testing.assert_allclose(
        result_strike_vector, expected_strike, rtol=0, atol=_NUMERICAL_ERR
    )
    np.testing.assert_allclose(
        result_rake_vector, expected_rake, rtol=0, atol=_NUMERICAL_ERR
    )

  def test_moment_vectors_directionality(self):
    def _angle_between_vectors(vector1, vector2):
      return np.arccos(np.dot(vector1, vector2))

    n_tests = 10
    angles = self._creat_multiple_random_angles(n_tests)
    (strike, rake, dip) = angles
    for ii in range(n_tests):
      vectors = earthquake_properties.moment_vectors_from_angles(
          strike[ii], rake[ii], dip[ii]
      )
      normal_vector, strike_vector, rake_vector = vectors
      # Normal vector should be orthogonal to strike and rake:
      np.testing.assert_allclose(
          np.pi / 2,
          _angle_between_vectors(normal_vector, rake_vector),
          rtol=0,
          atol=_NUMERICAL_ERR,
      )
      np.testing.assert_allclose(
          np.pi / 2,
          _angle_between_vectors(normal_vector, strike_vector),
          rtol=0,
          atol=_NUMERICAL_ERR,
      )
      # Strike should be orthogonal to e3:
      np.testing.assert_allclose(
          np.pi / 2,
          _angle_between_vectors(np.array([0, 0, 1]), strike_vector),
          rtol=0,
          atol=_NUMERICAL_ERR,
      )

  def test_rotation_from_x1x2x3_to_ned_proeprties(self):
    """Verifies the rotation matrix holds known properties of rotations."""
    n_tests = 10
    angles = self._creat_multiple_random_angles(n_tests)
    (strikes, rakes, dips) = angles
    for i in range(n_tests):
      rotation = earthquake_properties._rotation_from_x1x2x3_to_ned(
          strikes[i], rakes[i], dips[i]
      )
      #  Test if rotation is symmetric
      np.testing.assert_allclose(
          np.matmul(np.linalg.inv(rotation), rotation),
          np.eye(3),
          rtol=0,
          atol=_NUMERICAL_ERR,
      )
      # Test of determinant =+/-1
      np.testing.assert_allclose(
          np.linalg.det(rotation), 1, rtol=0, atol=_NUMERICAL_ERR
      )

  def test_rotation_from_x1x2x3_to_ned(self):
    """Verify the resulting rotation matrix coincides with expected cases."""
    n_tests = 10
    angles = self._creat_multiple_random_angles(n_tests)
    (strikes, rakes, dips) = angles
    # Only dip as nonzero - expected a rotation about the x axis
    strike_i, rake_i = 0, 0
    for i in range(n_tests):
      rotation = earthquake_properties._rotation_from_x1x2x3_to_ned(
          strike_i, rake_i, dips[i]
      )
      expected_rotation = geometry.rotation_matrix_about_axis(
          dips[i], np.array([1, 0, 0])
      )
      np.testing.assert_allclose(
          rotation, expected_rotation, rtol=0, atol=_NUMERICAL_ERR
      )

    # Only dip as zero - expected a rotation about the z axis of strike+rake
    dip_i = 0
    for i in range(n_tests):
      rotation = earthquake_properties._rotation_from_x1x2x3_to_ned(
          strikes[i], rakes[i], dip_i
      )
      expected_rotation = geometry.rotation_matrix_about_axis(
          (strikes[i] + rakes[i]), np.array([0, 0, -1])
      )
      np.testing.assert_allclose(
          rotation, expected_rotation, rtol=0, atol=_NUMERICAL_ERR
      )

  def _creat_multiple_random_angles(self, n_tests):
    def _stretch_01_to_range(numbers, target_range):
      delta = np.max(target_range) - np.min(target_range)
      return (numbers * delta) + np.min(target_range)

    rnd_seed = np.random.RandomState(seed=1905)
    strike_range = (0, 360)
    rake_range = (-90, 90)
    dip_range = (0, 180)
    strike = np.deg2rad(
        _stretch_01_to_range(rnd_seed.rand(n_tests), strike_range)
    )
    rake = np.deg2rad(_stretch_01_to_range(rnd_seed.rand(n_tests), rake_range))
    dip = np.deg2rad(_stretch_01_to_range(rnd_seed.rand(n_tests), dip_range))
    return (strike, rake, dip)


if __name__ == '__main__':
  absltest.main()
