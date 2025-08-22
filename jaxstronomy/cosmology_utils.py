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

"""Cosmology calculations in jax.

This module includes functions to calculate basic cosmology quantities all
implemented in jax.

Peak value calculations follow colossus implementation:
https://bitbucket.org/bdiemer/colossus/src/master/
"""

from typing import Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from jaxstronomy import power_spectrum

SPEED_OF_LIGHT = 2.99792458e8
# Linear density threshold for collapse.
DELTA_COLLAPSE_UNCORRECTED = 1.68647
# Critical density of the universe at redshift 0 in units of
# M_sun * h ** 2 / Mpc **3
RHO_CRIT_0 = 2.77536627245708E11
# Max value of integration for growth factor.
GROWTH_Z_MAX = 3e3


def _e_z(cosmology_params,
         z_values):
  """Calculate the hubble parameter.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion, potentially excluding lookup table.
    z_values: Redshifts at which to calculate the hubble parameter.

  Returns:
    The hubble parameter values in units of H_0 (km/s/Mpc).

  Notes:
    This assumes Lambda dark energy (i.e. no redshift dependence in the dark
    energy equation of state).
  """
  t = cosmology_params['omega_m_zero'] * (1.0 + z_values)**3.0
  t += cosmology_params['omega_de_zero']
  t += cosmology_params['omega_rad_zero'] * (1.0 + z_values)**4.0

  return jnp.sqrt(t)


def _e_z_rad_to_dark(cosmology_params,
                     z_values):
  """Calculate the hubble parameter with radiation treated as dark energy.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion, potentially excluding lookup table.
    z_values: Redshifts at which to calculate the hubble parameter.

  Returns:
    The hubble parameter values in units of H_0 (km/s/Mpc).

  Notes:
    Used for growth factor calculations.
  """
  t = cosmology_params['omega_m_zero'] * (1.0 + z_values)**3.0
  t += cosmology_params['omega_de_zero']
  t += cosmology_params['omega_rad_zero']
  return jnp.sqrt(t)


def _comoving_distance_numerical(init_cosmology_params,
                                 z_min, z_max):
  """Calculates the comoving distance at a given redshift numerically.

  Args:
    init_cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_min: Lower redshift of distance interval.
    z_max: Upper redhsift of distance interval.

  Returns:
    Comoving distance in units of Mpc.
  """
  # We probably want a better / less expensive integration scheme than this.
  z_samples = jnp.linspace(z_min, z_max, 1000)
  e_z_samples = _e_z(init_cosmology_params, z_samples)

  one_over_ez_int = jsp.integrate.trapezoid(1 / e_z_samples, z_samples)

  # Factor in the speed of light and the Hubble constant. The factor of 1e-3
  # comes from accounting for integral output being in Mpc * s / km and speed of
  # light being in m/s as well as h being H0/100.
  one_over_ez_int *= SPEED_OF_LIGHT * 1e-3
  one_over_ez_int /= init_cosmology_params['hubble_constant']

  return one_over_ez_int


def _sigma_k_integrand(init_cosmology_params,
                       log_k, radius):
  """Return the integrand for the variance of the linear density field.

  Args:
    init_cosmology_params: Cosmological parameters that define the universe's
      expansion, excluding the lookup table for the comoving_distance
      calculation.
    log_k: Log of the wavenumber values at which to evaluate integrand in units
      of h/Mpc.
    radius: Scale at which to calculate the rms variance in units of Mpc/h.

  Returns:
    Integrand at provided wavenumbers.
  """
  k = jnp.exp(log_k)
  # Fourier transform of the tophat filter function.
  k_times_r = k * radius
  # Use masking to default to 1 when below fidelity limit of float32.
  is_below_cut = k_times_r < 3e-2
  ft_filer = jnp.zeros_like(k)
  ft_filer += is_below_cut
  ft_filer += (
      jnp.logical_not(is_below_cut) * 3.0 / k_times_r**3 *
      (jnp.sin(k_times_r) - k_times_r * jnp.cos(k_times_r)))

  ps = power_spectrum.matter_power_spectrum(
      k, init_cosmology_params['omega_m_zero'],
      init_cosmology_params['omega_b_zero'],
      init_cosmology_params['temp_cmb_zero'],
      init_cosmology_params['hubble_constant'], init_cosmology_params['n_s'])

  return ps * ft_filer**2 * k**3


def _sigma_numerical(init_cosmology_params,
                     radius):
  """Calculate the variance of the linear density field numercially.

  Args:
    init_cosmology_params: Cosmological parameters that define the universe's
      expansion, excluding the lookup table for the comoving_distance
      calculation.
    radius: Scale at which to calculate the rms variance in units of Mpc/h.

  Returns:
    RMS variance of the linear density field.
  """
  # Integartion bins are in log momentum space (k).
  log_k_bins = jnp.linspace(
      jnp.log10(1e-6 / radius), jnp.log10(1e6 / radius), 1000)
  sigma_k_integrand = jnp.nan_to_num(
      _sigma_k_integrand(init_cosmology_params, log_k_bins, radius))
  sigma_squared = jsp.integrate.trapezoid(sigma_k_integrand, log_k_bins)

  return jnp.sqrt(sigma_squared / (2.0 * jnp.pi**2))


def _sigma_norm(init_cosmology_params):
  """Calculate the normalization of the variance of the linear density field.

  Args:
    init_cosmology_params: Cosmological parameters that define the universe's
      expansion, excluding the lookup table for the comoving_distance
      calculation.

  Returns:
    Normalized RMS variance of the linear density field.
  """
  # Normalize the raw output to sigma_eight.
  sigma_eight_raw = _sigma_numerical(init_cosmology_params, 8.0)

  return init_cosmology_params['sigma_eight'] / sigma_eight_raw


def _growth_factor_exact_unormalized(init_cosmology_params,
                                     z):
  """Calculate the linear growth factor, unormalized.

  Args:
    init_cosmology_params: Cosmological parameters that define the universe's
      expansion, excluding the lookup table for the comoving_distance
      calculation.
    z: Redshift at which to calculate the growth factor.

  Returns:
    Growth factor at input redshift.

  Notes:
    Approximations only hold to redshift 10.
  """
  z_samples = jnp.logspace(
      jnp.log10(jax.lax.max(z, 1e-6)), jnp.log10(GROWTH_Z_MAX), 10000)
  integral = jsp.integrate.trapezoid(
      (1.0 + z_samples) / _e_z_rad_to_dark(init_cosmology_params, z_samples)**3,
      z_samples)

  return (5.0 / 2.0 * init_cosmology_params['omega_m_zero'] *
          _e_z_rad_to_dark(init_cosmology_params, z) * integral)


def add_lookup_tables_to_cosmology_params(
    init_cosmology_params,
    z_lookup_max,
    dz,
    r_min,
    r_max,
    n_r_bins = 1000,
):
  """Add lookup tables to cosmology params.

  Args:
    init_cosmology_params: Cosmological parameters that define the universe's
      expansion, excluding the lookup tables.
    z_lookup_max: Maximum z to calculate in the lookup tables.
    dz: Step size in redshift space at which to calculate lookup tables.
    r_min: Logarithm of minimum lagrangian radius to include in lookup tables.
      In units of Mpc/h.
    r_max: Logarithm of maximum lagrangian radius to include in lookup tables.
      In units of Mpc/h.
    n_r_bins: Number of radial bins to include in sigma lookup tables.

  Returns:
    Cosmological parameters with lookup table.
  """
  # Jitted function for faster computation.
  comoving_distance_numerical = jax.jit(_comoving_distance_numerical)
  sigma_exact = jax.jit(_sigma_numerical)
  sigma_norm = jax.jit(_sigma_norm)
  growth_factor = jax.jit(_growth_factor_exact_unormalized)

  z_range = jnp.arange(0, z_lookup_max + dz, dz)
  radius_range = jnp.linspace(r_min, r_max, n_r_bins)

  # Normalize growth factor at redshift 0 to 1.
  growth_norm = 1.0 / growth_factor(init_cosmology_params, 0.0)
  sigma_norm = sigma_norm(init_cosmology_params)

  comoving_lookup_table = jax.vmap(
      jax.vmap(comoving_distance_numerical, in_axes=[None, None, 0]),
      in_axes=[None, 0, None])(init_cosmology_params, z_range, z_range)
  sigma_lookup_table = jax.vmap(
      sigma_exact, in_axes=[None, 0])(init_cosmology_params, radius_range)
  growth_lookup_table = (
      jax.vmap(growth_factor, in_axes=[None, 0])(init_cosmology_params,
                                                 z_range))

  cosmology_params = {
      'dz': dz,
      'z_lookup_max': z_lookup_max,
      'r_min': r_min,
      'r_max': r_max,
      'dr': (r_max - r_min) / (n_r_bins - 1),
      'comoving_lookup_table': comoving_lookup_table,
      'sigma_lookup_table': sigma_lookup_table * sigma_norm,
      'growth_lookup_table': growth_lookup_table * growth_norm
  }
  cosmology_params.update(init_cosmology_params)

  return cosmology_params


def comoving_distance(cosmology_params,
                      z_min, z_max):
  """Calculates the comoving distance at a given redshift using lookup.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_min: Lower redshift of distance interval.
    z_max: Upper redhsift of distance interval.

  Returns:
    Comoving distance in units of Mpc.

  Notes:
    If `z_min` or `z_max` are larger than `z_lookup_max` in `cosmology_params`
    the return will be equivalent to replacing the offending parameter(s) with
    `z_lookup_max`.
  """
  # Interpolate between the four nearest binds to the query.
  unrounded_i = z_min / cosmology_params['dz']
  unrounded_j = z_max / cosmology_params['dz']

  lookup_i_upper = jax.lax.min(
      jnp.ceil(unrounded_i).astype(int),
      len(cosmology_params['comoving_lookup_table'] - 1))
  lookup_i_lower = jax.lax.max(jnp.floor(unrounded_i).astype(int), 0)
  lookup_j_upper = jax.lax.min(
      jnp.ceil(unrounded_j).astype(int),
      len(cosmology_params['comoving_lookup_table'] - 1))
  lookup_j_lower = jax.lax.max(jnp.floor(unrounded_j).astype(int), 0)

  # Conduct a bilinear interpolation.
  frac_i = unrounded_i % 1
  frac_j = unrounded_j % 1

  # Replacing these lines with a matrix multiplication would be cleaner, but
  # leads to slightly slower code due to fraction matrix initialization.
  interpolated = (
      (1 - frac_i) * (1 - frac_j) *
      cosmology_params['comoving_lookup_table'][lookup_i_lower, lookup_j_lower])
  interpolated += (
      (frac_i) * (1 - frac_j) *
      cosmology_params['comoving_lookup_table'][lookup_i_upper, lookup_j_lower])
  interpolated += (
      (1 - frac_i) * (frac_j) *
      cosmology_params['comoving_lookup_table'][lookup_i_lower, lookup_j_upper])
  interpolated += (
      (frac_i) * (frac_j) *
      cosmology_params['comoving_lookup_table'][lookup_i_upper, lookup_j_upper])

  return interpolated


def angular_diameter_distance(cosmology_params,
                              z):
  """Calculate the angular diameter distance at a given redshift.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z: Redshift (i.e. distance) at which to conduct the calculation.

  Returns:
    Angular diameter distance in units of Mpc.
  """
  return comoving_distance(cosmology_params, 0.0, z) / (1.0 + z)


def angular_diameter_distance_between(
    cosmology_params,
    z_min,
    z_max,
):
  """Calculate the angular diameter distance between two redshifts.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_min: Redshift (i.e. distance) at which to start the calculation.
    z_max: Redhisft (i.e. distance) at which to stop the calculation.

  Returns:
    Angular diameter distance in units of Mpc.
  """
  return comoving_distance(cosmology_params, z_min, z_max) / (1.0 + z_max)


def kpc_per_arcsecond(cosmology_params,
                      z):
  """Calculate the physical kpc per arcsecond at a given redshift.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z: Redshift (i.e. distance) at which to conduct the calculation.

  Returns:
    Kpc per arcseconds.
  """
  radians_per_arcsecond = jnp.pi / 180 / 3600
  kpc_per_megapc = 1e3
  return (angular_diameter_distance(cosmology_params, z) *
          radians_per_arcsecond * kpc_per_megapc)


def comoving_to_angle(comv_x, comv_y,
                      cosmology_params,
                      z_lens):
  """Convert from comoving coordinates to angular units.

  Args:
    comv_x: Comoving x-coordinate.
    comv_y: Comoving y-coordinate.
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_lens: Redshift of the lens (i.e. current redshift).

  Returns:
    X- and y-coordinate in angular units (radians).
  """
  arcsecond_per_megaparsec = 1 / comoving_distance(cosmology_params, 0, z_lens)
  return comv_x * arcsecond_per_megaparsec, comv_y * arcsecond_per_megaparsec


def reduced_to_physical(
    reduced,
    cosmology_params,
    z_lens,
    z_source,
):
  """Transform from reduced deflection angle to physical coordinates.

  Args:
    reduced: Reduced deflection angle.
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z_lens: Redshift of the lens (i.e. current redshift).
    z_source: Redshift of the source.

  Returns:
    Physical coordinates.
  """
  return reduced * (
      angular_diameter_distance_between(cosmology_params, 0.0, z_source) /
      angular_diameter_distance_between(cosmology_params, z_lens, z_source))


def rho_matter(cosmology_params,
               z):
  """Return the matter density at redshift z.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z: Redshift at which to calculate the matter density.

  Returns:
    Matter density at redshift z in units M_sun * h ** 2 / Mpc ** 3.
  """
  return RHO_CRIT_0 * cosmology_params['omega_m_zero'] * (1.0 + z)**3


def collapse_overdensity(
    cosmology_params,
    z,
):
  """Return the collapse overdensity including first-order corrections.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    z: Redshift at which to calculate the collapse overdensity.

  Returns:
    Collapse overdensity.
  """
  return (DELTA_COLLAPSE_UNCORRECTED *
          (cosmology_params['omega_m_zero'] * (1.0 + z)**3 /
           (_e_z(cosmology_params, z))**2)**0.0055)


def lagrangian_radius(cosmology_params,
                      mass):
  """Return the lagrangian radius of halo of mass M.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    mass: Masses at which to calculate the lagrangian radius in units M_sun / h.

  Returns:
    Lagrangian radius in units of comoving Mpc / h.
  """
  # rho_matter returns in h units, so convert here for radius to be returned in
  # Mpc.
  return ((3.0 * mass / 4.0 / jnp.pi /
           rho_matter(cosmology_params, 0.0))**(1.0 / 3.0))


def sigma_tophat(cosmology_params,
                 lagrangian_r, z):
  """The RMS variance of the linear density field, including the growth factor.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    lagrangian_r: Lagrangian radius at which to evaluate the linear density
      field in units of comoving Mpc / h.
    z: Redshift at which to evaluate the linear density field.

  Returns:
    RMS variance of the linear density field.
  """
  # For both calculations, use linear interpolation between bins.
  lookup_z_unrounded = z / cosmology_params['dz']
  frac_z = lookup_z_unrounded % 1

  lookup_z_upper = jax.lax.min(
      jnp.ceil(lookup_z_unrounded).astype(int),
      len(cosmology_params['growth_lookup_table']) - 1)
  lookup_z_lower = jax.lax.max(jnp.floor(lookup_z_unrounded).astype(int), 0)

  lookup_sigma_unrounded = ((lagrangian_r - cosmology_params['r_min']) /
                            cosmology_params['dr'])
  frac_sigma = lookup_sigma_unrounded % 1

  lookup_sigma_upper = jax.lax.min(
      jnp.ceil(lookup_sigma_unrounded).astype(int),
      len(cosmology_params['sigma_lookup_table']) - 1)
  lookup_sigma_lower = jax.lax.max(
      jnp.floor(lookup_sigma_unrounded).astype(int), 0)

  sigma_no_growth = (
      frac_sigma * cosmology_params['sigma_lookup_table'][lookup_sigma_upper] +
      (1 - frac_sigma) *
      cosmology_params['sigma_lookup_table'][lookup_sigma_lower])
  growth = (
      frac_z * cosmology_params['growth_lookup_table'][lookup_z_upper] +
      (1 - frac_z) * cosmology_params['growth_lookup_table'][lookup_z_lower])

  return sigma_no_growth * growth


def derivative_log_sigma_log_r(
    cosmology_params,
    lagrangian_r,
    z,
):
  """Derivative of the log RMS variance with respect to log radius.

  Args:
    cosmology_params: Cosmological parameters that define the universe's
      expansion.
    lagrangian_r: Lagrangian radius at which to evaluate the derivative in units
      of comoving Mpc / h.
    z: Redshift at which to evaluate the derivative.

  Returns:
    Derivative of the log RMS variance.
  """
  # Calculate the derivative of sigma with respect to radius, then use the chain
  # rule to get the log of sigma with respect to the log of the radius.
  dr = cosmology_params['dr']

  # Calculate the derivative using the five-point stencil.
  derivative = -sigma_tophat(cosmology_params, lagrangian_r + 2 * dr, z)
  derivative += 8 * sigma_tophat(cosmology_params, lagrangian_r + dr, z)
  derivative += -8 * sigma_tophat(cosmology_params, lagrangian_r - dr, z)
  derivative += sigma_tophat(cosmology_params, lagrangian_r - 2 * dr, z)
  derivative /= 12 * dr

  sigma = sigma_tophat(cosmology_params, lagrangian_r, z)
  return derivative * lagrangian_r / sigma
