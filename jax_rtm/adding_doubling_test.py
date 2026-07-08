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

"""Unit tests for JAX adding-doubling implementation."""

# pylint: disable=invalid-name,g-import-not-at-top

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_rtm import adding_doubling
from jax_rtm import ice_properties
from jax_rtm import kokhanovsky_data
import numpy as np

jax.config.update("jax_enable_x64", True)


class AddingDoublingTest(absltest.TestCase):

  def test_adding_step_zero_reflection(self):
    # If layer 1 has zero reflection and identity transmission,
    # the result should be layer 2's properties.
    n = 4
    r1 = jnp.zeros((n, n))
    t1 = jnp.eye(n)
    r2 = jnp.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6],
    ])
    t2 = jnp.array([
        [0.9, 0.8, 0.7, 0.6],
        [0.5, 0.4, 0.3, 0.2],
        [0.1, 0.0, 0.1, 0.2],
        [0.3, 0.4, 0.5, 0.6],
    ])

    j_up1 = jnp.zeros(n)
    j_down1 = jnp.zeros(n)
    e1 = 1.0
    j_up2 = jnp.zeros(n)
    j_down2 = jnp.zeros(n)
    e2 = 1.0

    r12, t12, _, _, _ = adding_doubling.adding_step(
        r1, t1, j_up1, j_down1, e1, r2, t2, j_up2, j_down2, e2
    )

    np.testing.assert_allclose(r12, r2)
    np.testing.assert_allclose(t12, t2)

  def test_adding_step_identity_transmission(self):
    # If layer 2 has zero reflection and identity transmission,
    # the result should be layer 1's properties.
    n = 4
    r1 = jnp.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6],
    ])
    t1 = jnp.array([
        [0.9, 0.8, 0.7, 0.6],
        [0.5, 0.4, 0.3, 0.2],
        [0.1, 0.0, 0.1, 0.2],
        [0.3, 0.4, 0.5, 0.6],
    ])
    r2 = jnp.zeros((n, n))
    t2 = jnp.eye(n)

    j_up1 = jnp.zeros(n)
    j_down1 = jnp.zeros(n)
    e1 = 1.0
    j_up2 = jnp.zeros(n)
    j_down2 = jnp.zeros(n)
    e2 = 1.0

    r12, t12, _, _, _ = adding_doubling.adding_step(
        r1, t1, j_up1, j_down1, e1, r2, t2, j_up2, j_down2, e2
    )

    np.testing.assert_allclose(r12, r1)
    np.testing.assert_allclose(t12, t1)

  def test_doubling_step_zero_steps(self):
    n = 4
    r = jnp.eye(n) * 0.1
    t = jnp.eye(n) * 0.9
    j_up = jnp.zeros(n)
    j_down = jnp.zeros(n)
    e = 1.0

    r_thick, t_thick, _, _, _ = adding_doubling.doubling_step(
        r, t, j_up, j_down, e, 0
    )

    np.testing.assert_allclose(r_thick, r)
    np.testing.assert_allclose(t_thick, t)

  def test_rayleigh_scattering_simple(self):
    # A simple test to check if ADM runs with a Rayleigh-like phase matrix.
    # We use N=2 streams and 2 Stokes parameters (I, Q).
    mu = jnp.array([0.5, 1.0])
    delta_tau = 0.01

    # Helper to compute Rayleigh phase matrix for m=0
    def get_P(mu):
      N = len(mu)
      P = np.zeros((2 * N, 2 * N))
      for i in range(N):
        for j in range(N):
          m1 = mu[i]
          m2 = mu[j]
          p11 = 0.75 * (
              1.0 + m1**2 * m2**2 + 0.5 * (1.0 - m1**2) * (1.0 - m2**2)
          )
          p12 = 0.75 * (0.5 * (1.0 - m1**2) * (1.0 - m2**2))
          p21 = p12
          p22 = p12

          P[2 * i, 2 * j] = p11
          P[2 * i, 2 * j + 1] = p12
          P[2 * i + 1, 2 * j] = p21
          P[2 * i + 1, 2 * j + 1] = p22
      return jnp.array(P)

    P = get_P(mu)

    # Construct initial R and T for a thin layer
    N = len(mu)
    R = np.zeros((2 * N, 2 * N))
    T = np.zeros((2 * N, 2 * N))

    for i in range(N):
      for j in range(N):
        factor = delta_tau / (4.0 * mu[i])
        R[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = (
            factor * P[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
        )
        T[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = (
            factor * P[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
        )
        if i == j:
          T[2 * i, 2 * j] += jnp.exp(-delta_tau / mu[i])
          T[2 * i + 1, 2 * j + 1] += jnp.exp(-delta_tau / mu[i])

    R = jnp.array(R)
    T = jnp.array(T)

    # Run doubling
    j_up = jnp.zeros(2 * N)
    j_down = jnp.zeros(2 * N)
    e = 1.0
    r_thick, t_thick, _, _, _ = adding_doubling.doubling_step(
        R, T, j_up, j_down, e, 5
    )  # 32 times thicker

    # Basic checks
    self.assertEqual(r_thick.shape, (4, 4))
    self.assertEqual(t_thick.shape, (4, 4))
    self.assertTrue(jnp.all(r_thick >= 0))
    self.assertTrue(jnp.all(t_thick >= 0))
    self.assertFalse(np.allclose(r_thick, R))
    self.assertFalse(np.allclose(t_thick, T))

  def test_identity_case_tau_zero(self):
    # Test that tau=0 results in R=0 and T=I.
    mu = jnp.array([0.5, 1.0])
    delta_tau = 0.0

    def get_P(mu):
      N = len(mu)
      P = np.zeros((2 * N, 2 * N))
      for i in range(N):
        for j in range(N):
          m1 = mu[i]
          m2 = mu[j]
          p11 = 0.75 * (
              1.0 + m1**2 * m2**2 + 0.5 * (1.0 - m1**2) * (1.0 - m2**2)
          )
          p12 = 0.75 * (0.5 * (1.0 - m1**2) * (1.0 - m2**2))
          p21 = p12
          p22 = p12

          P[2 * i, 2 * j] = p11
          P[2 * i, 2 * j + 1] = p12
          P[2 * i + 1, 2 * j] = p21
          P[2 * i + 1, 2 * j + 1] = p22
      return jnp.array(P)

    P = get_P(mu)
    N = len(mu)
    R = np.zeros((2 * N, 2 * N))
    T = np.zeros((2 * N, 2 * N))

    for i in range(N):
      for j in range(N):
        factor = delta_tau / (4.0 * mu[i])
        R[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = (
            factor * P[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
        )
        T[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = (
            factor * P[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
        )
        if i == j:
          T[2 * i, 2 * j] += jnp.exp(-delta_tau / mu[i])
          T[2 * i + 1, 2 * j + 1] += jnp.exp(-delta_tau / mu[i])

    R = jnp.array(R)
    T = jnp.array(T)

    j_up = jnp.zeros(2 * N)
    j_down = jnp.zeros(2 * N)
    e = 1.0
    r_thick, t_thick, _, _, _ = adding_doubling.doubling_step(
        R, T, j_up, j_down, e, 5
    )

    np.testing.assert_allclose(r_thick, 0.0, atol=1e-7)
    np.testing.assert_allclose(t_thick, jnp.eye(2 * N), atol=1e-7)

  def test_rayleigh_benchmarks(self):
    # Test against Kokhanovsky Rayleigh benchmarks.
    # We compare the azimuthally averaged component (m=0).

    tau = 0.3262
    omega = 1.0
    n_streams = 128
    num_doubling_steps = 20
    mu_0 = 0.5

    _, _, j_up_thick, _, _, mu, _ = adding_doubling.solve_layer(
        n_streams, tau, omega, num_doubling_steps, mu_0
    )

    # Validation benchmarks from Kokhanovsky et al. (2010),
    # https://doi.org/10.1016/j.jqsrt.2010.03.005
    # Book: Benchmark results in vector radiative transfer
    data = kokhanovsky_data.RAYLEIGH

    # Compare for each VZA in data
    for vza, phi_data in data.items():
      # Compute azimuthally averaged benchmark values
      i_0 = (
          phi_data[0]["IR"] + phi_data[180]["IR"] + 2 * phi_data[90]["IR"]
      ) / 4.0

      # Interpolate solver result to mu_vza
      mu_vza = jnp.cos(jnp.deg2rad(vza))

      # j_up_thick is the diffuse intensity at the top of the atmosphere
      # It has shape (2*N,), we need the I component (even indices)
      i_components = j_up_thick[0::2]

      val = jnp.interp(mu_vza, mu, i_components)

      # Solver (I/mu0) matches the benchmark Reflection Function
      val_mu0 = val / mu_0

      print(f"VZA: {vza}, Benchmark I0: {i_0}, Solver: {val_mu0}")

      # Assert relative error is less than 2.5%
      np.testing.assert_allclose(val_mu0, i_0, rtol=0.025)

  def test_ice_energy_conservation(self):
    wavelength = 0.47
    r_eff = 10.0
    n_streams = 16
    num_doubling_steps = 25
    mu_0 = 0.5

    omega0, _, legendre_coeffs = ice_properties.get_ice_properties_droxtal(
        wavelength, r_eff
    )

    # FORCE beta_0 to 1.0 to ensure energy conservation
    legendre_coeffs = legendre_coeffs.at[0].set(1.0)

    tau = 2.0
    tau_star, _, legendre_coeffs_star = adding_doubling.deltam_truncate(
        tau, omega0, legendre_coeffs, n_streams
    )

    mu, w = adding_doubling.get_quadrature(n_streams)

    (
        P_diffuse_forward,
        P_diffuse_backward,
        P_direct_forward,
        P_direct_backward,
    ) = adding_doubling.legendre_phase_matrix(legendre_coeffs_star, mu, w, mu_0)

    # For conservative scattering test, we should use omega = 1.0

    omega_star_cons = 1.0

    _, _, j_up_thick, j_down_thick, e_thick, _, _ = (
        adding_doubling.solve_layer_generic(
            n_streams,
            tau_star,
            omega_star_cons,
            P_diffuse_forward,
            P_diffuse_backward,
            P_direct_forward,
            P_direct_backward,
            num_doubling_steps,
            mu_0,
        )
    )

    i_up = j_up_thick[::2]
    i_down = j_down_thick[::2]

    reflected_flux = jnp.sum(2.0 * mu * w * i_up)
    transmitted_flux = jnp.sum(2.0 * mu * w * i_down) + mu_0 * e_thick

    total_flux = reflected_flux + transmitted_flux
    print(f"Total flux: {total_flux}, expected: {mu_0}")

    self.assertAlmostEqual(float(total_flux), float(mu_0), places=3)

  def test_kirchhoff_law(self):
    n_streams = 16
    tau = 2.0
    omega = 0.5
    num_doubling_steps = 20
    mu_0 = 0.5
    temperature = 250.0
    wavelength = 10.0

    mu, w = adding_doubling.get_quadrature(n_streams)
    beta_coeffs = jnp.zeros(2 * n_streams)
    beta_coeffs = beta_coeffs.at[0].set(1.0)
    P_diffuse_forward, P_diffuse_backward, _, _ = (
        adding_doubling.legendre_phase_matrix(beta_coeffs, mu, w, mu_0)
    )

    # Zero out direct beam phase functions to isolate thermal emission
    P_direct_forward = jnp.zeros(2 * n_streams)
    P_direct_backward = jnp.zeros(2 * n_streams)

    r_thick, t_thick, j_up_thick, _, _, _, _ = (
        adding_doubling.solve_layer_generic(
            n_streams,
            tau,
            omega,
            P_diffuse_forward,
            P_diffuse_backward,
            P_direct_forward,
            P_direct_backward,
            num_doubling_steps,
            mu_0,
            temperature,
            wavelength,
            is_thermal=True,
        )
    )

    i_up = j_up_thick[::2]

    b_lambda = adding_doubling.planck_function(wavelength, temperature)

    # Revert to simple sum (assuming weights are already in matrices)
    r_flux = jnp.sum(r_thick[::2, ::2], axis=1)
    t_flux = jnp.sum(t_thick[::2, ::2], axis=1)

    expected_emission = (1.0 - r_flux - t_flux) * b_lambda

    print(f"i_up: {i_up}")
    print(f"expected_emission: {expected_emission}")

    self.assertTrue(jnp.allclose(i_up, expected_emission, rtol=1e-3))

  def test_solve_layer_generic_batched(self):
    n_streams = 4
    num_pixels = 5
    num_doubling_steps = 10
    mu_0 = 0.5

    key = jax.random.key(0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    tau = jax.random.uniform(k1, (num_pixels,)) * 2.0
    omega = jax.random.uniform(k2, (num_pixels,)) * 0.9 + 0.05

    p_df = jax.random.uniform(k3, (num_pixels, 2 * n_streams, 2 * n_streams))
    p_db = jax.random.uniform(k4, (num_pixels, 2 * n_streams, 2 * n_streams))
    p_sf = jax.random.uniform(k5, (num_pixels, 2 * n_streams))
    p_sb = jax.random.uniform(k6, (num_pixels, 2 * n_streams))

    # Make diagonally dominant to ensure stable inversion in test
    p_df = p_df + jnp.eye(2 * n_streams)[jnp.newaxis, :, :] * 5.0
    p_db = p_db + jnp.eye(2 * n_streams)[jnp.newaxis, :, :] * 5.0

    # Baseline: vmap over solve_layer_generic
    def solve_single(t, o, p_df_val, p_db_val, p_sf_val, p_sb_val):
      return adding_doubling.solve_layer_generic(
          n_streams,
          t,
          o,
          p_df_val,
          p_db_val,
          p_sf_val,
          p_sb_val,
          num_doubling_steps,
          mu_0,
      )

    vmap_solve = jax.vmap(solve_single, in_axes=(0, 0, 0, 0, 0, 0))

    r_vmap, t_vmap, j_up_vmap, j_down_vmap, e_vmap, _, _ = vmap_solve(
        tau, omega, p_df, p_db, p_sf, p_sb
    )

    # Batched version
    r_batch, t_batch, j_up_batch, j_down_batch, e_batch, _, _ = (
        adding_doubling.solve_layer_generic_batched(
            n_streams,
            tau,
            omega,
            p_df,
            p_db,
            p_sf,
            p_sb,
            num_doubling_steps,
            mu_0,
        )
    )

    # Compare
    np.testing.assert_allclose(r_batch, r_vmap, atol=1e-5)
    np.testing.assert_allclose(t_batch, t_vmap, atol=1e-5)
    np.testing.assert_allclose(j_up_batch, j_up_vmap, atol=1e-5)
    np.testing.assert_allclose(j_down_batch, j_down_vmap, atol=1e-5)
    np.testing.assert_allclose(e_batch, e_vmap, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
