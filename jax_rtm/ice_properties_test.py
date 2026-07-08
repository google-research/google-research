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

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Unit tests for ice single-scattering properties, instrumented for units."""

# pylint: disable=g-import-not-at-top

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_rtm import ice_properties
from jax_rtm import pquant

# Instrument the ice_properties module and the test module itself
pquant.instrument_module(ice_properties)
pquant.instrument_module(sys.modules[__name__])


class IcePropertiesTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    ice_properties._ensure_loaded()
    # Inject Area dimensions into cext in the database slices using pquant DSL
    for w in ice_properties._SLICES_DROXTAL:
      omega0, cext, leg = ice_properties._SLICES_DROXTAL[w]
      ice_properties._SLICES_DROXTAL[w] = (
          omega0,
          pquant.Area(cext),
          leg,
      )
    for w in ice_properties._SLICES_COLUMN:
      omega0, cext, leg = ice_properties._SLICES_COLUMN[w]
      ice_properties._SLICES_COLUMN[w] = (
          omega0,
          pquant.Area(cext),
          leg,
      )

  def test_get_ice_properties_dimensions(self):
    wavelength = 0.47
    # r_eff wrapped in Length dimensions using the pquant DSL
    r_eff = pquant.Length(10.0)

    omega0, cext, legendre_coeffs = ice_properties.get_ice_properties_droxtal(
        wavelength, r_eff
    )

    # Verify dimensions using the clean factory .dimensions properties
    self.assertEqual(
        omega0.dimensions, pquant.Dimensionless.dimensions
    )  # Dimensionless albedo
    self.assertEqual(
        cext.dimensions, pquant.Area.dimensions
    )  # Area extinction cross-section
    self.assertEqual(
        legendre_coeffs.dimensions, pquant.Dimensionless.dimensions
    )  # Dimensionless Legendre coefficients

  def test_get_ice_properties(self):
    wavelength = 0.47
    r_eff = 10.0

    # Execute (should run natively and return Quantity objects due to database
    # injection)
    omega0, cext, legendre_coeffs = ice_properties.get_ice_properties_droxtal(
        wavelength, r_eff
    )

    self.assertGreater(omega0, 0.9)
    self.assertLess(omega0, 1.0)
    self.assertTrue(jnp.isfinite(cext))
    self.assertGreater(cext, 0.0)
    self.assertEqual(legendre_coeffs.shape, (128,))

  def test_get_ice_properties_single_column(self):
    wavelength = 0.47
    r_eff = 10.0
    omega0, cext, legendre_coeffs = (
        ice_properties.get_ice_properties_single_column(wavelength, r_eff)
    )
    self.assertGreater(omega0, 0.9)
    self.assertLess(omega0, 1.0)
    self.assertTrue(jnp.isfinite(cext))
    self.assertGreater(cext, 0.0)
    self.assertEqual(legendre_coeffs.shape, (128,))

  def test_differentiability(self):
    # Differentiability test (uses raw floats, runs native)
    def cost_function(r_eff):
      omega0, _, _ = ice_properties.get_ice_properties_droxtal(0.47, r_eff)
      return omega0

    grad_val = jax.grad(cost_function)(10.0)
    print(f"Gradient of omega0 w.r.t r_eff at 10.0: {grad_val}")
    self.assertTrue(jnp.isfinite(grad_val))

  def test_arbitrary_wavelength_2d_interpolation(self):
    # Wavelength is outside precomputed 1D slices.
    # This triggers 2D grid interpolation.
    wavelength = 11.0

    r_eff = 30.0
    omega0_drox, cext_drox, leg_drox = (
        ice_properties.get_ice_properties_droxtal(wavelength, r_eff)
    )
    self.assertGreater(float(omega0_drox), 0.0)
    self.assertLess(float(omega0_drox), 1.0)
    self.assertTrue(jnp.isfinite(cext_drox))
    self.assertEqual(leg_drox.shape, (128,))

    omega0_col, cext_col, leg_col = (
        ice_properties.get_ice_properties_single_column(wavelength, r_eff)
    )
    self.assertGreater(float(omega0_col), 0.0)
    self.assertLess(float(omega0_col), 1.0)
    self.assertTrue(jnp.isfinite(cext_col))
    self.assertEqual(leg_col.shape, (128,))


if __name__ == "__main__":
  absltest.main()
