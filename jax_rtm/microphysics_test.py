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

"""Unit tests for microphysics, instrumented for dimensional analysis."""

# pylint: disable=invalid-name,g-import-not-at-top

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl.testing import absltest
import jax
import jax.numpy as jnp

from jax_rtm import ice_properties
from jax_rtm import microphysics
from jax_rtm import pquant

# Instrument the microphysics module and the test module itself
pquant.instrument_module(microphysics)
pquant.instrument_module(sys.modules[__name__])


class MicrophysicsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize dummy profiles
    self.t_prof = jnp.array([280.0, 260.0, 240.0, 220.0, 200.0])
    self.rho_air = jnp.array([1.2, 1.0, 0.8, 0.6, 0.4])
    self.dz = jnp.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

    # Ensure ice properties are loaded, and dynamically size our dummy cext_grid
    ice_properties._ensure_loaded()  # pylint: disable=protected-access
    self.cext_grid = jnp.ones(len(ice_properties._SIZES)) * 15.0
    self.sizes_grid = jnp.array(ice_properties._SIZES)

    # Fetch default production float constants to wrap dynamically
    prod = microphysics.MicrophysicsParams()

    # Construct a complete dimensional constants package for testing by wrapping
    # the production defaults dynamically. This ensures that if the production
    # values ever change, the unit tests do not break or require updates.
    self.dim_constants = microphysics.MicrophysicsParams(
        # Liquid Constants
        rho_w=pquant.Density(prod.rho_w),
        k_val=pquant.Dimensionless(prod.k_val),
        nc_ocean=pquant.NumberConcentration(prod.nc_ocean),
        nc_land=pquant.NumberConcentration(prod.nc_land),
        r_ref=pquant.Length(prod.r_ref),
        # Ice Constants
        rho_ice=pquant.Density(prod.rho_ice),
        snow_scatter_factor=pquant.MassExtinction(prod.snow_scatter_factor),
        iwp_conv_thresh=pquant.IWP(prod.iwp_conv_thresh),
        sigmoid_scale=pquant.InverseIWP(prod.sigmoid_scale),
        wyser_offset=pquant.Dimensionless(prod.wyser_offset),
        wyser_slope=pquant.TemperaturePower(prod.wyser_slope, -1.5),
        # Calibrated Empirical Parameters (Dimensionless multipliers)
        cext_water_scale=pquant.Dimensionless(prod.cext_water_scale),
        ciwc_multiplier=pquant.Dimensionless(prod.ciwc_multiplier),
        cswc_multiplier=pquant.Dimensionless(prod.cswc_multiplier),
        r_eff_multiplier=pquant.Dimensionless(prod.r_eff_multiplier),
        ica_ice_scale=pquant.Dimensionless(prod.ica_ice_scale),
        ica_liq_scale=pquant.Dimensionless(prod.ica_liq_scale),
        # Wyser Polynomial Coefficients (Length)
        wyser_coeff_0=pquant.Length(prod.wyser_coeff_0),
        wyser_coeff_1=pquant.Length(prod.wyser_coeff_1),
        wyser_coeff_2=pquant.Length(prod.wyser_coeff_2),
        wyser_coeff_3=pquant.Length(prod.wyser_coeff_3),
    )

  def test_liquid_properties(self):
    clwc = jnp.array([1e-4, 1e-5, 0.0, 0.0, 0.0])
    crwc = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    is_land = 0.0

    # Test execution (uses default raw float constants natively)
    tau_water, r_eff_water = microphysics.compute_liquid_properties(
        clwc,
        crwc,
        self.rho_air,
        self.dz,
        wavelength=8.4,
        is_land=is_land,
    )

    # Check sizes and shapes
    self.assertEqual(tau_water.shape, clwc.shape)
    self.assertEqual(r_eff_water.shape, clwc.shape)

    # Verify physical boundary clipping
    self.assertTrue(jnp.all(r_eff_water >= 3.75))
    self.assertTrue(jnp.all(r_eff_water <= 21.0))

    # Test differentiability
    def loss_fn(c):
      tau, _ = microphysics.compute_liquid_properties(
          c,
          crwc,
          self.rho_air,
          self.dz,
          wavelength=8.4,
          is_land=is_land,
      )
      return jnp.sum(tau)

    grad = jax.grad(loss_fn)(clwc)
    self.assertFalse(jnp.any(jnp.isnan(grad)))

  def test_ice_properties(self):
    ciwc = jnp.array([0.0, 1e-6, 1e-5, 1e-4, 0.0])
    cswc = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Test execution (uses default raw float constants natively)
    r_eff, tau_ice = microphysics.compute_ice_properties(
        ciwc=ciwc,
        cswc=cswc,
        T_prof=self.t_prof,
        rho_air=self.rho_air,
        dz=self.dz,
        cext_grid=self.cext_grid,
        sizes_grid=self.sizes_grid,
        ice_spectral_shift_factor=1.145,
    )

    # Shape check
    self.assertEqual(r_eff.shape, ciwc.shape)
    self.assertEqual(tau_ice.shape, ciwc.shape)

    # Physical range check
    self.assertTrue(jnp.all(r_eff >= 5.5))

    # Test differentiability
    def loss_fn(c):
      _, tau = microphysics.compute_ice_properties(
          ciwc=c,
          cswc=cswc,
          T_prof=self.t_prof,
          rho_air=self.rho_air,
          dz=self.dz,
          cext_grid=self.cext_grid,
          sizes_grid=self.sizes_grid,
          ice_spectral_shift_factor=1.145,
      )
      return jnp.sum(tau)

    grad = jax.grad(loss_fn)(ciwc)
    self.assertFalse(jnp.any(jnp.isnan(grad)))

  def test_ocean_directional_emissivity(self):
    em_nadir = jnp.array([0.98, 0.99, 0.97])
    mu_view = 0.5
    wind_speed = 5.0

    em = microphysics.compute_ocean_directional_emissivity(
        em_nadir, mu_view, wind_speed=wind_speed
    )
    self.assertEqual(em.shape, em_nadir.shape)
    self.assertTrue(jnp.all(em >= 0.0))
    self.assertTrue(jnp.all(em <= 1.0))

  def test_liquid_properties_dimensions(self):
    # Inputs wrapped in physical dimensions using the pquant DSL
    clwc = pquant.Dimensionless(jnp.array([1e-4, 1e-5, 0.0, 0.0, 0.0]))
    crwc = pquant.Dimensionless(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    rho_air = pquant.Density(self.rho_air)
    dz = pquant.Length(self.dz)
    is_land = pquant.Dimensionless(0.0)

    # Execute with explicit physical constants injected
    tau_water, r_eff_water = microphysics.compute_liquid_properties(
        clwc,
        crwc,
        rho_air,
        dz,
        wavelength=8.4,
        is_land=is_land,
        params=self.dim_constants,
    )

    # Verify output dimensions using the clean factory .dimensions properties
    self.assertEqual(
        tau_water.dimensions, pquant.Dimensionless.dimensions
    )  # Dimensionless optical depth
    self.assertEqual(
        r_eff_water.dimensions, pquant.Length.dimensions
    )  # Length effective radius

  def test_ice_properties_dimensions(self):
    # Inputs wrapped in physical dimensions using the pquant DSL
    ciwc = pquant.Dimensionless(jnp.array([0.0, 1e-6, 1e-5, 1e-4, 0.0]))
    cswc = pquant.Dimensionless(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    rho_air = pquant.Density(self.rho_air)
    dz = pquant.Length(self.dz)
    T_prof = pquant.Temperature(self.t_prof)

    # Grid inputs from database wrapped in physical dimensions
    cext_grid = pquant.Area(self.cext_grid)
    sizes_grid = pquant.Length(self.sizes_grid)

    # Execute with explicit physical constants injected
    r_eff, tau_ice = microphysics.compute_ice_properties(
        ciwc=ciwc,
        cswc=cswc,
        T_prof=T_prof,
        rho_air=rho_air,
        dz=dz,
        cext_grid=cext_grid,
        sizes_grid=sizes_grid,
        ice_spectral_shift_factor=pquant.Dimensionless(1.145),
        params=self.dim_constants,
    )

    # Verify output dimensions using the clean factory .dimensions properties
    self.assertEqual(
        r_eff.dimensions, pquant.Length.dimensions
    )  # Length effective radius
    self.assertEqual(
        tau_ice.dimensions, pquant.Dimensionless.dimensions
    )  # Dimensionless optical depth

  def test_microphysics_params_pytree(self):
    params = microphysics.MicrophysicsParams(rho_w=1005.0)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(restored.rho_w, 1005.0)
    self.assertEqual(restored, params)

  def test_apply_thermodynamic_phase_mask(self):
    ciwc_raw = jnp.array([1e-4, 1e-4, 1e-4])
    clwc_raw = jnp.array([1e-4, 1e-4, 1e-4])
    t_prof = jnp.array([280.0, 258.15, 230.0])
    ciwc_phys, clwc_phys = microphysics.apply_thermodynamic_phase_mask(
        ciwc_raw, clwc_raw, t_prof
    )
    self.assertAlmostEqual(float(ciwc_phys[0]), 0.0, places=4)
    self.assertAlmostEqual(float(clwc_phys[0]), 1e-4, places=4)
    self.assertAlmostEqual(float(ciwc_phys[1]), 0.5e-4, places=6)
    self.assertAlmostEqual(float(clwc_phys[1]), 0.5e-4, places=6)
    self.assertAlmostEqual(float(ciwc_phys[2]), 1e-4, places=4)
    self.assertAlmostEqual(float(clwc_phys[2]), 0.0, places=4)


if __name__ == "__main__":
  absltest.main()
