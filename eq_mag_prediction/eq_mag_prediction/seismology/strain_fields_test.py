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

"""Tests for strain fields."""

import itertools

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.seismology import strain_fields

# Aa threshold for determining bearable numerical error. Set below the typical
# strain expected in earthquakes ~1e-6 (Shearer Intro. to Seis.)
_STRAIN_ASSYMETRY_THRESHOLD = 1e-8


class StrainFieldsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='u11_with_moment_3',
          strain_function=strain_fields._double_couple_u11,
          moment=3,
      ),
      dict(
          testcase_name='u12_with_moment_3',
          strain_function=strain_fields._double_couple_u12,
          moment=3,
      ),
      dict(
          testcase_name='u13_with_moment_3',
          strain_function=strain_fields._double_couple_u13,
          moment=3,
      ),
      dict(
          testcase_name='u22_with_moment_3',
          strain_function=strain_fields._double_couple_u22,
          moment=3,
      ),
      dict(
          testcase_name='u23_with_moment_3',
          strain_function=strain_fields._double_couple_u23,
          moment=3,
      ),
      dict(
          testcase_name='u33_with_moment_3',
          strain_function=strain_fields._double_couple_u33,
          moment=3,
      ),
  )
  def test_double_couple_strain_fields_symmetry(self, strain_function, moment):
    """Verifies the resulting strain fields are symmetric to rotation of pi."""
    # mock coordinates:
    coors = np.meshgrid(
        np.linspace(-20, 20, 150),
        np.linspace(-20, 20, 150),
        np.linspace(-20, 20, 150),
    )
    u_ij = strain_function(coors, moment)
    # ensure numerical error is well under expected strain
    self.assertTrue(
        np.allclose(
            u_ij, np.rot90(u_ij, 2), atol=_STRAIN_ASSYMETRY_THRESHOLD, rtol=0
        )
    )

  def test_ensure_symmetry_of_strain_tensor(self):
    coors = np.meshgrid(
        np.linspace(-20, 20, 150),
        np.linspace(-20, 20, 150),
        np.linspace(-20, 20, 150),
    )
    strain_tensor = strain_fields.double_couple_strain_tensor(coors, moment=4)
    for i, j in itertools.combinations_with_replacement(range(3), 2):
      np.testing.assert_array_equal(strain_tensor[i, j], strain_tensor[j, i])

  def test_elastic_constants_relations(self):
    elastic_gamma_from_wave_speed = (
        1 - (strain_fields._C_S / strain_fields._C_P) ** 2
    )
    elastic_gamma_from_lame_constants = strain_fields.elastic_gamma(
        strain_fields._ELASTIC_LAMBDA_DEFAULT, strain_fields._ELASTIC_MU_DEFAULT
    )
    np.testing.assert_allclose(
        elastic_gamma_from_wave_speed, elastic_gamma_from_lame_constants
    )

  def test_youngs_modulus_order_of_magnitude(self):
    # Young's modulus and shear modulus (aka mu) are expected to be in the same
    # order of magnitude.
    elastic_moduli_ratio = (
        strain_fields.youngs_modulus() / strain_fields._ELASTIC_MU_DEFAULT
    )
    self.assertLessEqual(elastic_moduli_ratio, 10)
    self.assertGreaterEqual(elastic_moduli_ratio, 0.1)


if __name__ == '__main__':
  absltest.main()
