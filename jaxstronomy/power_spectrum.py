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

"""Eisenstein and Hu 1998 calculation of power spectrum in jax.

This module implements a jax calculation of the Eisenstein and Hu 1998 power
spectrum: https://arxiv.org/pdf/astro-ph/9709112.pdf
"""

from typing import Tuple

import jax.numpy as jnp


def _calculate_z_k_eq(omega_m_h_squared,
                      temp_cmb_scaled):
  """Calculate the redshift and wavenumber at matter-radiation equality.

  Args:
    omega_m_h_squared: Omega matter at redshift 0 scaled by h ** 2.
    temp_cmb_scaled: CMB temperature normalized by 2.7

  Returns:
    The redshift and wavenumber at matter-radiation equality.
  """
  # Matter-radiation transition redshift and wavenumber. Equations 2 and 3.
  z_eq = 2.5e4 * omega_m_h_squared / temp_cmb_scaled**4
  k_eq = 7.46e-2 * omega_m_h_squared / temp_cmb_scaled**2
  return z_eq, k_eq


def _baryon_to_photon_ratio(omega_b_h_squared, temp_cmb_scaled,
                            z):
  """Calculate the baryon to photon momentum density ratio.

  Args:
    omega_b_h_squared: Omega baryon at redshift 0 scaled by h ** 2.
    temp_cmb_scaled: CMB temperature normalized by 2.7
    z: Redshift at which to calculate the ratio.

  Returns:
    Baryon to photon momentum density ratio
  """
  # Equation 5.
  return 31.5 * omega_b_h_squared / temp_cmb_scaled**4 / (z / 1e3)


def _calculate_z_drag(omega_m_h_squared,
                      omega_b_h_squared):
  """Calculate the drag redshift.

  Args:
    omega_m_h_squared: Omega matter at redshift 0 scaled by h ** 2.
    omega_b_h_squared: Omega baryon at redshift 0 scaled by h ** 2.

  Returns:
    Drag redshift.
  """
  # Numerical recombination results. Equation 4.
  b_one = (0.313 * omega_m_h_squared**-0.419 *
           (1.0 + 0.607 * omega_m_h_squared**(0.674)))
  b_two = 0.238 * omega_m_h_squared**(0.223)
  return (1291.0 * omega_m_h_squared**0.251 /
          (1.0 + 0.659 * omega_m_h_squared**(0.828)) *
          (1.0 + b_one * omega_b_h_squared**(b_two)))


def _calculate_sound_horizon(omega_m_h_squared, omega_b_h_squared,
                             temp_cmb_scaled):
  """Calculate the sound horizon.

  Args:
    omega_m_h_squared: Omega matter at redshift 0 scaled by h ** 2.
    omega_b_h_squared: Omega baryon at redshift 0 scaled by h ** 2.
    temp_cmb_scaled: CMB temperature normalized by 2.7

  Returns:
    Sound horizon.
  """
  z_eq, k_eq = _calculate_z_k_eq(omega_m_h_squared, temp_cmb_scaled)
  z_drag = _calculate_z_drag(omega_m_h_squared, omega_b_h_squared)

  # Baryon to photon momentum ratio. Equation 5.
  ratio_drag = _baryon_to_photon_ratio(omega_b_h_squared, temp_cmb_scaled,
                                       z_drag)
  ratio_eq = _baryon_to_photon_ratio(omega_b_h_squared, temp_cmb_scaled, z_eq)
  # Sound horizon at drag epoch. Equation 6.
  return (2.0 / (3.0 * k_eq) * jnp.sqrt(6.0 / ratio_eq) * jnp.log(
      (jnp.sqrt(1.0 + ratio_drag) + jnp.sqrt(ratio_drag + ratio_eq)) /
      (1.0 + jnp.sqrt(ratio_eq))))


def _transfer_hypergeometric_terms(omega_m_h_squared,
                                   omega_m_zero, omega_cdm_zero,
                                   omega_ratio):
  """Calculate terms for hypergeometric expansion of transfer function.

  Args:
    omega_m_h_squared: Omega matter at redshift 0 scaled by h ** 2.
    omega_m_zero: Omega matter at redshift 0.
    omega_cdm_zero: Omega CDM at reshift 0.
    omega_ratio: Ratio of Omega baryon at redshift 0 to Omega matter at redshift
      0.

  Returns:
    Alpha and beta for hypergeometric expansion of transfer function.
  """
  # Equations 10, 11, and 12
  alpha_one = ((46.9 * omega_m_h_squared)**0.670 *
               (1.0 + (32.1 * omega_m_h_squared)**-0.532))
  alpha_two = ((12.0 * omega_m_h_squared)**0.424 *
               (1.0 + (45.0 * omega_m_h_squared)**-0.582))

  alpha_c = alpha_one**(-omega_ratio) * alpha_two**(-omega_ratio**3)
  beta_one = 0.944 / (1.0 + (458.0 * omega_m_h_squared)**-0.708)
  beta_two = (0.395 * omega_m_h_squared)**-0.0266
  beta_c = (1.0 / (1.0 + beta_one *
                   ((omega_cdm_zero / omega_m_zero)**beta_two - 1.0)))

  return alpha_c, beta_c


def _transfer_cdm(k, omega_m_h_squared,
                  temp_cmb_scaled, alpha_c,
                  beta_c):
  """Calculate the CDM trasnfer function.

  Args:
    k: Wavenumber values at which to evaluate the matter power spectrum in units
      of 1 / Mpc.
    omega_m_h_squared: Omega matter at redshift 0 scaled by h ** 2.
    temp_cmb_scaled: CMB temperature normalized by 2.7
    alpha_c: Hypergeometric function term alpha for CDM transfer function.
    beta_c: Hypergeometric function term beta for CDM trasnfer function.

  Returns:
    CDM transfer function.
  """
  _, k_eq = _calculate_z_k_eq(omega_m_h_squared, temp_cmb_scaled)

  # Equation 10
  q = k / (13.41 * k_eq)

  # Equations 19 and 20.
  const = 14.2 / alpha_c + 386.0 / (1 + 69.9 * q**1.08)
  return (jnp.log(jnp.e + 1.8 * beta_c * q) /
          (jnp.log(jnp.e + 1.8 * beta_c * q) + const * q**2))


def _transfer_baryon(k, omega_m_h_squared,
                     omega_b_h_squared,
                     temp_cmb_scaled):
  """Calculate the baryon trasnfer function.

  Args:
    k: Wavenumber values at which to evaluate the matter power spectrum in units
      of 1 / Mpc.
    omega_m_h_squared: Omega matter at redshift 0 scaled by h ** 2.
    omega_b_h_squared: Omega baryon at redshift 0 scaled by h ** 2.
    temp_cmb_scaled: CMB temperature normalized by 2.7

  Returns:
    Baryon transfer function.
  """
  z_eq, k_eq = _calculate_z_k_eq(omega_m_h_squared, temp_cmb_scaled)
  z_drag = _calculate_z_drag(omega_m_h_squared, omega_b_h_squared)
  sound_horizon = _calculate_sound_horizon(omega_m_h_squared, omega_b_h_squared,
                                           temp_cmb_scaled)
  ratio_drag = _baryon_to_photon_ratio(omega_b_h_squared, temp_cmb_scaled,
                                       z_drag)
  omega_ratio = omega_b_h_squared / omega_m_h_squared

  # Silk damping scale. Equation 7.
  k_silk = (1.6 * omega_b_h_squared**0.52 * omega_m_h_squared**0.73 *
            (1.0 + (10.4 * omega_m_h_squared)**-0.95))

  # Terms for silk dampening. Equations 14 and 15.
  scale_ratio = (1.0 + z_eq) / (1.0 + z_drag)
  ratio_term = (
      scale_ratio * (-6.0 * jnp.sqrt(1.0 + scale_ratio) +
                     (2.0 + 3.0 * scale_ratio) * jnp.log(
                         (jnp.sqrt(1.0 + scale_ratio) + 1.0) /
                         (jnp.sqrt(1.0 + scale_ratio) - 1.0))))
  alpha_b = (2.07 * k_eq * sound_horizon * (1 + ratio_drag)**(-3.0 / 4.0) *
             ratio_term)

  # Baryon contribution to transfer function. Equations 21, 22, 23, and 24.
  k_sound = k * sound_horizon
  beta_node = 8.41 * omega_m_h_squared**0.435
  sound_horizon_node = (
      sound_horizon / (1.0 + (beta_node / (k_sound))**3.0)**(1.0 / 3.0))
  beta_b = (
      0.5 + omega_ratio +
      (3.0 - 2.0 * omega_ratio) * jnp.sqrt((17.2 * omega_m_h_squared)**2 + 1.0))

  # Another repeat value in the final calculation.
  k_sound_node = k * sound_horizon_node

  transfer_baryon = (
      _transfer_cdm(k, omega_m_h_squared, temp_cmb_scaled, 1.0, 1.0) /
      (1.0 + (k_sound / 5.2)**2))
  transfer_baryon += (
      alpha_b / (1.0 + (beta_b / (k_sound))**3) * jnp.exp(-(k / k_silk)**1.4))
  transfer_baryon *= jnp.sin(k_sound_node) / (k_sound_node)

  return transfer_baryon


def transfer_function(k, omega_m_zero, omega_b_zero,
                      temp_cmb_zero,
                      hubble_constant):
  """Calculate the transfer function for the matter power spectrum.

  Args:
    k: Wavenumber values at which to evaluate the matter power spectrum in units
      of 1 / Mpc.
    omega_m_zero: Omega matter at redshift 0.
    omega_b_zero: Omega baryon at redshift 0.
    temp_cmb_zero: Temperature of the CMB at redshift 0 in units of Kelvin.
    hubble_constant: The Hubble constant in units of km / s / Mpc

  Returns:
    Transfer function for the matter power spectrum.
  """
  # A couple variable combinations that are used multiple times.
  omega_cdm_zero = (omega_m_zero - omega_b_zero)
  omega_m_h_squared = (omega_m_zero * (hubble_constant / 100.0)**2)
  omega_b_h_squared = (omega_b_zero * (hubble_constant / 100.0)**2)
  omega_ratio = (omega_b_zero / omega_m_zero)
  temp_cmb_scaled = temp_cmb_zero / 2.7

  # Sound horizon at drag epoch. Equation 6.
  sound_horizon = _calculate_sound_horizon(omega_m_h_squared, omega_b_h_squared,
                                           temp_cmb_scaled)

  # Hypergeometric term for CDM transfer function. Equations 10, 11, and 12
  alpha_c, beta_c = _transfer_hypergeometric_terms(omega_m_h_squared,
                                                   omega_m_zero, omega_cdm_zero,
                                                   omega_ratio)

  # CDM contribution to transfer function. Equations 17 and 18.
  interp_ratio = 1.0 / (1.0 + (k * sound_horizon / 5.4)**4)
  transfer_cdm = (
      interp_ratio *
      _transfer_cdm(k, omega_m_h_squared, temp_cmb_scaled, 1.0, beta_c) +
      (1.0 - interp_ratio) *
      _transfer_cdm(k, omega_m_h_squared, temp_cmb_scaled, alpha_c, beta_c))

  # Baryon contribution to transfer function. Equations 14, 15, 21, 22, 23, and
  # 24.
  transfer_baryon = _transfer_baryon(k, omega_m_h_squared, omega_b_h_squared,
                                     temp_cmb_scaled)

  return (omega_ratio * transfer_baryon +
          omega_cdm_zero / omega_m_zero * transfer_cdm)


def matter_power_spectrum(k, omega_m_zero,
                          omega_b_zero, temp_cmb_zero,
                          hubble_constant, n_s):
  """Calculate the matter power spectrum following Eisenstein and Hu 1998.

  Args:
    k: Wavenumber values at which to evaluate the matter power spectrum in units
      of h / Mpc.
    omega_m_zero: Omega matter at redshift 0.
    omega_b_zero: Omega baryon at redshift 0.
    temp_cmb_zero: Temperature of the CMB at redshift 0 in units of Kelvin.
    hubble_constant: The Hubble constant in units of km / s / Mpc
    n_s: Scalar tilt of the initial power spectrum.

  Returns:
    Matter power spectrum evaluated at provided wavenumbers.
  """
  # All the transfer function calculations are in units of 1/Mpc not h/Mpc in
  # accordance with the paper.
  k_no_h = k * hubble_constant / 100.0
  tf = transfer_function(k_no_h, omega_m_zero, omega_b_zero, temp_cmb_zero,
                         hubble_constant)
  return tf**2 * k**n_s
