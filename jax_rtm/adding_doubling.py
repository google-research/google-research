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
# pylint: skip-file
# -*- coding:utf-8 -*-

"""Polarized Adding-Doubling Method in JAX."""

import jax
import jax.numpy as jnp
import numpy as np
from . import ice_properties
from . import satellite_config


@jax.jit
def planck_function(wavelength_um, temperature_k):
  """Calculates Planck radiance in W * m^-2 * sr^-1 * um^-1.

  Args:
    wavelength_um: Wavelength in microns.
    temperature_k: Temperature in Kelvin.
  """
  h = 6.62607015e-34  # J s
  c = 2.99792458e8  # m/s
  k_b = 1.380649e-23  # J/K

  c1 = 2.0 * h * c**2 * 1e24
  c2 = h * c / (k_b * 1e-6)

  return (
      c1
      / (wavelength_um**5)
      * 1.0
      / (jnp.exp(c2 / (wavelength_um * temperature_k)) - 1.0)
  )


@jax.jit
def batch_interp_1d(x, xp, fp_matrix):
  """Vectorized 1D linear interpolation across an array of features in parallel."""
  if fp_matrix.ndim == 1:
    fp_matrix = jnp.tile(fp_matrix[:, None], (1, len(xp)))
  idx = jnp.searchsorted(xp, x, side='right')
  idx = jnp.clip(idx, 1, len(xp) - 1)
  x0 = xp[idx - 1]
  x1 = xp[idx]
  y0 = fp_matrix[:, idx - 1]
  y1 = fp_matrix[:, idx]
  weight = (x - x0) / (x1 - x0)
  val = y0 + weight * (y1 - y0)
  val = jnp.where(x < xp[0], fp_matrix[:, 0], val)
  val = jnp.where(x > xp[-1], fp_matrix[:, -1], val)
  return val


@jax.jit
def legendre_hg(g, max_degree=128):
  """Generates Legendre expansion coefficients for Henyey-Greenstein phase function.

  beta_k = g^k
  """
  k = jnp.arange(max_degree)
  return g**k


def adding_step(r1, t1, j_up1, j_down1, e1, r2, t2, j_up2, j_down2, e2):
  e1 = jnp.asarray(e1)
  e2 = jnp.asarray(e2)

  # 2. Source Term Preparation
  n_blocks = e1.shape[0] if len(e1.shape) > 0 else 1
  if n_blocks > 1:
    block_size = r1.shape[0] // n_blocks
    e1_expanded = jnp.repeat(e1, block_size)
  else:
    e1_expanded = e1

  # Y is the total diffuse downwelling radiation between the layers
  source_for_y = j_down1 + jnp.matmul(r1, j_up2) * e1_expanded

  source_for_y_col = source_for_y[Ellipsis, jnp.newaxis]
  rhs_stacked = jnp.concatenate([t1, source_for_y_col], axis=-1)

  # Richardson solver: (I - r1*r2) * sol = rhs_stacked
  A = jnp.matmul(r1, r2)

  # Unrolled Richardson iteration (2 steps)
  sol = rhs_stacked
  sol = rhs_stacked + jnp.matmul(A, sol)
  sol = rhs_stacked + jnp.matmul(A, sol)

  # Unpack solutions
  d = sol[Ellipsis, :-1]
  y = sol[Ellipsis, -1]

  # Standard R and T updates
  u = jnp.matmul(r2, d)
  r12 = r1 + jnp.matmul(t1, u)
  t12 = jnp.matmul(t2, d)

  # X is the total diffuse upwelling radiation between the layers
  x = j_up2 * e1_expanded + jnp.matmul(r2, y)

  # Emerging diffuse radiation
  j_up12 = j_up1 + jnp.matmul(t1, x)
  j_down12 = j_down2 * e1_expanded + jnp.matmul(t2, y)

  # Combined direct beam attenuation
  e12 = e1 * e2

  return r12, t12, j_up12, j_down12, e12


def doubling_step(r, t, j_up, j_down, e, num_steps):
  def scan_body(carry, _):
    curr_r, curr_t, curr_j_up, curr_j_down, curr_e = carry
    new_carry = adding_step(
        curr_r,
        curr_t,
        curr_j_up,
        curr_j_down,
        curr_e,
        curr_r,
        curr_t,
        curr_j_up,
        curr_j_down,
        curr_e,
    )
    return new_carry, None

  final_state, _ = jax.lax.scan(
      scan_body, (r, t, j_up, j_down, e), None, length=num_steps
  )
  return final_state


def get_quadrature(n_streams):
  """Get Gauss-Legendre quadrature points and weights for [0, 1]."""
  mu, w = np.polynomial.legendre.leggauss(n_streams)
  # Scale to [0, 1]
  mu_half = 0.5 * (mu + 1.0)
  w_half = 0.5 * w
  return jnp.array(mu_half), jnp.array(w_half)


def rayleigh_phase_matrix(mu):
  """Compute Rayleigh phase matrix for m=0 for given streams (I and Q)."""
  n = len(mu)
  P = np.zeros((2 * n, 2 * n))
  for i in range(n):
    for j in range(n):
      m1 = mu[i]
      m2 = mu[j]
      p11 = 0.75 * (1.0 + m1**2 * m2**2 + 0.5 * (1.0 - m1**2) * (1.0 - m2**2))
      p12 = 0.75 * (m1**2 - 0.5 * (1.0 - m1**2) * (1.0 - m2**2))
      p21 = 0.75 * (m2**2 - 0.5 * (1.0 - m1**2) * (1.0 - m2**2))
      p22 = 0.75 * (0.5 * (1.0 - m1**2) * (1.0 - m2**2))

      P[2 * i, 2 * j] = p11
      P[2 * i, 2 * j + 1] = p12
      P[2 * i + 1, 2 * j] = p21
      P[2 * i + 1, 2 * j + 1] = p22
  return jnp.array(P)


def compute_legendre_matrix(mu, max_degree):
  """Compute Legendre polynomials at points mu up to max_degree.

  Args:
    mu: Points at which to evaluate (shape (N,)).
    max_degree: Maximum degree (usually 2N - 1).

  Returns:
    legendre_mat: Matrix of shape (max_degree + 1, N) where legendre_mat[k, i] =
    P_k(mu[i]).
  """
  n_points = mu.shape[0]
  legendre_mat = [jnp.ones(n_points), mu]

  for k in range(1, max_degree):
    pk1 = ((2 * k + 1) * mu * legendre_mat[k] - k * legendre_mat[k - 1]) / (
        k + 1
    )
    legendre_mat.append(pk1)

  return jnp.stack(legendre_mat)


def legendre_phase_matrix(beta_coeffs, mu, w, mu_0):
  """Compute azimuthally averaged phase matrix from Legendre coefficients."""
  n_streams_val = mu.shape[0]
  K = beta_coeffs.shape[0]  # This will now correctly be 2N
  legendre_mat = compute_legendre_matrix(mu, K - 1)

  k_arr = jnp.arange(K)
  L_k = (2 * k_arr + 1) * beta_coeffs

  # FORWARD Diffuse Matrix (for Transmission)
  p11_forward_diffuse = jnp.matmul(
      legendre_mat.T, L_k[:, jnp.newaxis] * legendre_mat
  )

  # BACKWARD Diffuse Matrix (for Reflection) - note the (-1)**k
  L_k_backward = L_k * (-1.0) ** k_arr
  p11_backward_diffuse = jnp.matmul(
      legendre_mat.T, L_k_backward[:, jnp.newaxis] * legendre_mat
  )

  p_diffuse_forward = jnp.zeros((2 * n_streams_val, 2 * n_streams_val))
  p_diffuse_forward = p_diffuse_forward.at[::2, ::2].set(p11_forward_diffuse)

  p_diffuse_backward = jnp.zeros((2 * n_streams_val, 2 * n_streams_val))
  p_diffuse_backward = p_diffuse_backward.at[::2, ::2].set(p11_backward_diffuse)

  # Direct Beam Vectors
  a_0 = compute_legendre_matrix(jnp.array([mu_0]), K - 1)[:, 0]
  p11_forward_dir = jnp.matmul(legendre_mat.T, L_k * a_0)
  p11_backward_dir = jnp.matmul(legendre_mat.T, L_k * a_0 * (-1.0) ** k_arr)

  p_direct_forward = jnp.zeros(2 * n_streams_val)
  p_direct_forward = p_direct_forward.at[::2].set(p11_forward_dir)

  p_direct_backward = jnp.zeros(2 * n_streams_val)
  p_direct_backward = p_direct_backward.at[::2].set(p11_backward_dir)

  return (
      p_diffuse_forward,
      p_diffuse_backward,
      p_direct_forward,
      p_direct_backward,
  )


# Delta-M Truncation method for scaling highly asymmetric phase functions
# Citation: Wiscombe (1977), https://doi.org/10.1175/1520-0469(1977)034<1408:TDMRYA>2.0.CO;2
def deltam_truncate(tau, omega, beta_coeffs, n_streams):
  """Apply Delta-M truncation to optical properties."""
  # f is the 2N-th coefficient (e.g., index 32 for n_streams=16)
  f = beta_coeffs[2 * n_streams]

  tau_star = tau * (1.0 - omega * f)
  omega_star = omega * (1.0 - f) / (1.0 - omega * f)

  # Scale and truncate the Legendre coefficients to exactly length 2N
  beta_coeffs_star = (beta_coeffs[: 2 * n_streams] - f) / (1.0 - f)

  return tau_star, omega_star, beta_coeffs_star


def init_thin_layer(
    mu,
    w,
    P_diffuse_forward,
    P_diffuse_backward,
    P_direct_forward,
    P_direct_backward,
    tau,
    omega,
    mu_0,
    temperature=None,
    wavelength=None,
    is_thermal=False,
):
  """Compute initial R, T, and source terms for a thin layer."""
  n = len(mu)

  # DIFFUSE-TO-DIFFUSE SCATTERING
  factor_diffuse = tau * omega / (2.0 * mu)  # shape (n,)
  factor_reshaped = factor_diffuse[
      :, jnp.newaxis, jnp.newaxis, jnp.newaxis
  ]  # shape (n, 1, 1, 1)
  w_reshaped = w[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]  # shape (1, 1, n, 1)

  P_diff_back_reshaped = P_diffuse_backward.reshape(n, 2, n, 2)
  P_diff_forw_reshaped = P_diffuse_forward.reshape(n, 2, n, 2)

  R_blocks = factor_reshaped * P_diff_back_reshaped * w_reshaped
  T_blocks = factor_reshaped * P_diff_forw_reshaped * w_reshaped

  R = R_blocks.reshape(2 * n, 2 * n)
  T = T_blocks.reshape(2 * n, 2 * n)

  # Unscattered diffuse transmission
  exp_term = jnp.exp(-tau / mu)  # shape (n,)
  exp_term_repeated = jnp.repeat(exp_term, 2)  # shape (2n,)
  T = T + jnp.diag(exp_term_repeated)

  # DIRECT-TO-DIFFUSE SCATTERING
  factor_direct = (tau * omega) / (4.0 * mu)  # shape (n,)
  factor_direct_repeated = jnp.repeat(factor_direct, 2)  # shape (2n,)

  j_up = factor_direct_repeated * P_direct_backward
  j_down = factor_direct_repeated * P_direct_forward

  # Thermal emission
  if temperature is not None and wavelength is not None:
    B_lambda = planck_function(wavelength, temperature)
    thermal_glow = (
        (1.0 - omega) * B_lambda * (1.0 - jnp.exp(-tau / mu))
    )  # shape (n,)
    j_up = j_up.at[::2].add(thermal_glow)
    j_down = j_down.at[::2].add(thermal_glow)

  if is_thermal:
    e = 1.0
  else:
    e = jnp.exp(-tau / mu_0)

  return R, T, j_up, j_down, e


def solve_layer_generic(
    n_streams,
    tau,
    omega,
    P_diffuse_forward,
    P_diffuse_backward,
    P_direct_forward,
    P_direct_backward,
    num_doubling_steps,
    mu_0=0.5,
    temperature=None,
    wavelength=None,
    is_thermal=False,
):
  """Solve radiative transfer for a single layer using ADM with arbitrary phase function."""
  mu, w = get_quadrature(n_streams)

  # Initial thin layer optical depth
  delta_tau = tau / (2**num_doubling_steps)

  R, T, j_up, j_down, e = init_thin_layer(
      mu,
      w,
      P_diffuse_forward,
      P_diffuse_backward,
      P_direct_forward,
      P_direct_backward,
      delta_tau,
      omega,
      mu_0,
      temperature,
      wavelength,
      is_thermal,
  )

  r_thick, t_thick, j_up_thick, j_down_thick, e_thick = doubling_step(
      R, T, j_up, j_down, e, num_doubling_steps
  )

  return r_thick, t_thick, j_up_thick, j_down_thick, e_thick, mu, w


def solve_layer(n_streams, tau, omega, num_doubling_steps, mu_0=0.5):
  """Solve radiative transfer for a single layer using ADM (Rayleigh)."""
  mu, w = get_quadrature(n_streams)
  P = rayleigh_phase_matrix(mu)

  # For Rayleigh, direct beam phase is same as diffuse phase evaluated at mu_i and mu_0.
  n = len(mu)
  p_direct = np.zeros(2 * n)
  for i in range(n):
    m1 = mu[i]
    m2 = mu_0
    p11 = 0.75 * (1.0 + m1**2 * m2**2 + 0.5 * (1.0 - m1**2) * (1.0 - m2**2))
    p12 = 0.75 * (m1**2 - 0.5 * (1.0 - m1**2) * (1.0 - m2**2))
    p_direct[2 * i] = p11
    p_direct[2 * i + 1] = p12

  p_direct = jnp.array(p_direct)

  return solve_layer_generic(
      n_streams, tau, omega, P, P, p_direct, p_direct, num_doubling_steps, mu_0
  )


def solve_layer_generic_batched(
    n_streams,
    tau,
    omega,
    P_diffuse_forward,
    P_diffuse_backward,
    P_direct_forward,
    P_direct_backward,
    num_doubling_steps,
    mu_0=0.5,
    temperature=None,
    wavelength=None,
    is_thermal=False,
):
  """Solve radiative transfer for a single layer using ADM, batched over pixels."""
  mu, w = get_quadrature(n_streams)

  # Initial thin layer optical depth
  delta_tau = tau / (2**num_doubling_steps)

  # Vmap init_thin_layer over pixels
  vmap_init = jax.vmap(
      init_thin_layer,
      in_axes=(
          None,
          None,
          0,
          0,
          0,
          0,
          0,
          0,
          None,
          0 if temperature is not None else None,
          None,
          None,
      ),
  )

  R_batch, T_batch, j_up_batch, j_down_batch, e_batch = vmap_init(
      mu,
      w,
      P_diffuse_forward,
      P_diffuse_backward,
      P_direct_forward,
      P_direct_backward,
      delta_tau,
      omega,
      mu_0,
      temperature,
      wavelength,
      is_thermal,
  )

  num_pixels = tau.shape[0]
  n = 2 * n_streams

  # Convert R_batch, T_batch to block-diagonal
  R_large = jax.scipy.linalg.block_diag(
      *[R_batch[i] for i in range(num_pixels)]
  )
  T_large = jax.scipy.linalg.block_diag(
      *[T_batch[i] for i in range(num_pixels)]
  )

  # Flatten j_up and j_down
  j_up_large = j_up_batch.reshape(-1)
  j_down_large = j_down_batch.reshape(-1)

  # Call doubling_step with large matrices
  (
      r_thick_large,
      t_thick_large,
      j_up_thick_large,
      j_down_thick_large,
      e_thick,
  ) = doubling_step(
      R_large, T_large, j_up_large, j_down_large, e_batch, num_doubling_steps
  )

  # Extract diagonal blocks back using reshaping and advanced indexing
  r_thick_reshaped = r_thick_large.reshape(num_pixels, n, num_pixels, n)
  t_thick_reshaped = t_thick_large.reshape(num_pixels, n, num_pixels, n)

  idx = jnp.arange(num_pixels)
  r_thick_batch = r_thick_reshaped[idx, :, idx, :]
  t_thick_batch = t_thick_reshaped[idx, :, idx, :]

  # Reshape j_up and j_down back to (N, 2n)
  j_up_thick_batch = j_up_thick_large.reshape(num_pixels, n)
  j_down_thick_batch = j_down_thick_large.reshape(num_pixels, n)

  return (
      r_thick_batch,
      t_thick_batch,
      j_up_thick_batch,
      j_down_thick_batch,
      e_thick,
      mu,
      w,
  )


def solve_layer_for_scan(
    wavelength,
    r_eff_con,
    r_eff_nat,
    r_eff_water,
    tau_ray,
    tau_ice_con,
    tau_ice_nat,
    tau_wv,
    tau_water,
    omega_ray,
    n_streams,
    num_doubling_steps,
    mu_0,
    omega0_ice_con_grid,
    legendre_ice_con_grid,
    omega0_ice_nat_grid,
    legendre_ice_nat_grid,
    temperature=None,
    is_thermal=False,
):
  """Solve radiative transfer for a single layer, handling Bi-Modal Ice, Rayleigh, Water Vapor, and Liquid Cloud mixing.

  Returns full matrices (R, T, j_up, j_down, e, mu, w).
  """
  # Safe r_eff for interpolator
  r_eff_con_safe = jnp.maximum(r_eff_con, 2.0)
  r_eff_nat_safe = jnp.maximum(r_eff_nat, 2.0)
  r_eff_water_safe = jnp.maximum(r_eff_water, 2.0)

  omega_ice_con = jnp.interp(
      r_eff_con_safe, ice_properties._SIZES, omega0_ice_con_grid
  )
  legendre_ice_con = batch_interp_1d(
      r_eff_con_safe, ice_properties._SIZES, legendre_ice_con_grid.T
  )

  omega_ice_nat = jnp.interp(
      r_eff_nat_safe, ice_properties._SIZES, omega0_ice_nat_grid
  )
  legendre_ice_nat = batch_interp_1d(
      r_eff_nat_safe, ice_properties._SIZES, legendre_ice_nat_grid.T
  )

  # Force beta_0 to 1.0 for energy conservation
  legendre_ice_con = legendre_ice_con.at[0].set(1.0)
  legendre_ice_nat = legendre_ice_nat.at[0].set(1.0)

  # Rayleigh properties
  legendre_ray = jnp.zeros_like(legendre_ice_con)
  legendre_ray = legendre_ray.at[0].set(1.0)
  legendre_ray = legendre_ray.at[2].set(0.5)

  # Liquid Water Cloud properties (Mie scattering approximation in thermal IR)
  goes_waves = satellite_config.GOES_ABI_CONFIG.wavelengths
  is_84 = jnp.abs(wavelength - goes_waves[0]) < 0.1
  is_103 = jnp.abs(wavelength - goes_waves[1]) < 0.1
  omega_base = jnp.where(is_84, 0.5, jnp.where(is_103, 0.4, 0.3))

  # Continuous dynamic albedo and asymmetry parameter driven by droplet sizing
  omega_water = jnp.clip(
      omega_base * (10.0 / r_eff_water_safe) ** 0.15, 0.05, 0.9
  )
  g_water = jnp.clip(0.73 + 0.008 * r_eff_water_safe, 0.65, 0.92)
  legendre_water = legendre_hg(g_water)
  legendre_water = legendre_water.at[0].set(1.0)

  # Total mixed optical depth includes liquid water and water vapor (gas-only)
  tau_mix = tau_ray + tau_ice_con + tau_ice_nat + tau_water + tau_wv

  # Handle zero tau_mix safely
  is_non_empty = tau_mix > 0.0
  safe_tau_mix = jnp.where(is_non_empty, tau_mix, 1.0)

  # Water vapor does not scatter (omega_wv = 0.0)
  # So it only contributes to the denominator of omega_mix
  # Scattering optical depth now includes liquid water cloud scattering!
  scattering_od = (
      omega_ray * tau_ray
      + omega_ice_con * tau_ice_con
      + omega_ice_nat * tau_ice_nat
      + omega_water * tau_water
  )
  omega_mix = scattering_od / safe_tau_mix

  # Legendre mixing includes liquid water!
  is_scattering = scattering_od > 0.0
  safe_scattering_od = jnp.where(is_scattering, scattering_od, 1.0)

  legendre_mix = (
      legendre_ray * omega_ray * tau_ray
      + legendre_ice_con * omega_ice_con * tau_ice_con
      + legendre_ice_nat * omega_ice_nat * tau_ice_nat
      + legendre_water * omega_water * tau_water
  ) / safe_scattering_od

  # Now call deltam_truncate ONCE
  tau_star, omega_star, legendre_star = deltam_truncate(
      tau_mix, omega_mix, legendre_mix, n_streams
  )

  mu, w = get_quadrature(n_streams)

  # Now call legendre_phase_matrix ONCE
  P_df, P_db, P_sf, P_sb = legendre_phase_matrix(legendre_star, mu, w, mu_0)

  return solve_layer_generic(
      n_streams,
      tau_star,
      omega_star,
      P_df,
      P_db,
      P_sf,
      P_sb,
      num_doubling_steps,
      mu_0,
      temperature,
      wavelength,
      is_thermal,
  )


def stack_atmosphere(
    profile_tau_ray,
    profile_tau_ice_con,
    profile_tau_ice_nat,
    profile_tau_wv,
    profile_tau_water,
    profile_omega_ray,
    profile_r_eff_con,
    profile_r_eff_nat,
    profile_r_eff_water,
    profile_T,
    wavelength,
    n_streams,
    num_doubling_steps,
    mu_0,
    T_surf=None,
    is_thermal=False,
    omega0_ice_con_grid=None,
    legendre_ice_con_grid=None,
    omega0_ice_nat_grid=None,
    legendre_ice_nat_grid=None,
    emissivity=1.0,
    sharded_axes=None,
    use_checkpointing=True,
):
  """Stack multiple layers using jax.lax.scan."""
  init_R = jnp.zeros((2 * n_streams, 2 * n_streams))
  init_T = jnp.eye(2 * n_streams)
  init_j_up = jnp.zeros(2 * n_streams)
  init_j_down = jnp.zeros(2 * n_streams)
  init_e = 1.0

  if sharded_axes is not None:
    init_R = jax.lax.pcast(init_R, sharded_axes, to='varying')
    init_T = jax.lax.pcast(init_T, sharded_axes, to='varying')
    init_j_up = jax.lax.pcast(init_j_up, sharded_axes, to='varying')
    init_j_down = jax.lax.pcast(init_j_down, sharded_axes, to='varying')
    init_e = jax.lax.pcast(init_e, sharded_axes, to='varying')

  if is_thermal and T_surf is not None:
    B_surf = planck_function(wavelength, T_surf)
    init_j_up = init_j_up.at[::2].set(B_surf * emissivity)

  init_carry = (init_R, init_T, init_j_up, init_j_down, init_e)

  def scan_body(carry, layer_inputs):
    (
        tau_ray_i,
        tau_ice_con_i,
        tau_ice_nat_i,
        tau_wv_i,
        tau_water_i,
        omega_ray_i,
        r_eff_con_i,
        r_eff_nat_i,
        r_eff_water_i,
        temp_i,
    ) = layer_inputs

    r_layer, t_layer, j_up_layer, j_down_layer, e_layer, _, _ = (
        solve_layer_for_scan(
            wavelength,
            r_eff_con_i,
            r_eff_nat_i,
            r_eff_water_i,
            tau_ray_i,
            tau_ice_con_i,
            tau_ice_nat_i,
            tau_wv_i,
            tau_water_i,
            omega_ray_i,
            n_streams,
            num_doubling_steps,
            mu_0,
            temperature=temp_i,
            is_thermal=is_thermal,
            omega0_ice_con_grid=omega0_ice_con_grid,
            legendre_ice_con_grid=legendre_ice_con_grid,
            omega0_ice_nat_grid=omega0_ice_nat_grid,
            legendre_ice_nat_grid=legendre_ice_nat_grid,
        )
    )

    # r_layer is Layer 1 (top), carry is Layer 2 (bottom)
    new_carry = adding_step(
        r_layer,
        t_layer,
        j_up_layer,
        j_down_layer,
        e_layer,
        carry[0],
        carry[1],
        carry[2],
        carry[3],
        carry[4],
    )
    return new_carry, None

  # Reverse profiles to go from surface to TOA
  layer_inputs = (
      profile_tau_ray[::-1],
      profile_tau_ice_con[::-1],
      profile_tau_ice_nat[::-1],
      profile_tau_wv[::-1],
      profile_tau_water[::-1],
      profile_omega_ray[::-1],
      profile_r_eff_con[::-1],
      profile_r_eff_nat[::-1],
      profile_r_eff_water[::-1],
      profile_T[::-1],
  )
  if use_checkpointing:
    scan_body_fn = jax.checkpoint(scan_body)
  else:
    scan_body_fn = scan_body

  final_state, _ = jax.lax.scan(scan_body_fn, init_carry, layer_inputs)

  return final_state[2]  # Return final j_up (TOA upwelling radiance)
