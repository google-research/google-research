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

"""Unified JAX-Differentiable Cloud and Ice Microphysics Parameterizations."""

# pylint: disable=invalid-name

import dataclasses
import json
import os
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from . import ice_properties

# Eagerly load ice property grids to ensure _SIZES is initialized before JAX
# tracing.
ice_properties._ensure_loaded()  # pylint: disable=protected-access


# Load calibrated parameters from params_992.json at module load time
_package_dir = os.path.dirname(os.path.realpath(__file__))
_params_path = os.path.join(_package_dir, "data/params_992.json")
if os.path.exists(_params_path):
  with open(_params_path, "r") as f:
    _loaded_params = json.load(f)
else:
  _loaded_params = {}


# ==============================================================================
# Microphysics Parameters PyTree Dataclass
# ==============================================================================
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class MicrophysicsParams:
  """Configuration container for cloud and ice microphysics physical parameters.

  Registered as a JAX PyTree to allow seamless tracing of physical quantities
  (pquant.Unit) during dimensional tests, while falling back to raw floats
  in production with zero overhead.

  All calibrated empirical parameters default to the optimal values loaded
  from params_992.json, making it the single source of truth.
  """

  # --- Fundamental Physical Constants ---
  rho_w: Any = 1000.0  # Water density [kg/m^3]
  k_val: Any = 0.81  # Droplet concentration shape factor
  nc_ocean: Any = 80.0e6  # Cloud condensation nuclei (ocean) [1/m^3]
  nc_land: Any = 300.0e6  # Cloud condensation nuclei (land) [1/m^3]
  r_ref: Any = 10.0  # Reference radius [microns]
  rho_ice: Any = 917.0  # Ice density [kg/m^3]
  sigmoid_scale: Any = 22.0  # Convective transition scale [m^2/kg]

  # --- Calibrated Empirical Parameters (Loaded from params_992.json) ---
  snow_scatter_factor: Any = _loaded_params.get("snow_scatter_factor", 315.0)
  iwp_conv_thresh: Any = _loaded_params.get("iwp_conv_thresh", 0.168)
  wyser_offset: Any = _loaded_params.get("wyser_offset", -1.62)
  wyser_slope: Any = _loaded_params.get("wyser_slope", 1.18e-3)
  cext_water_scale: Any = _loaded_params.get("cext_water_scale", 1.45)
  ciwc_multiplier: Any = _loaded_params.get("ciwc_multiplier", 5.25)
  cswc_multiplier: Any = _loaded_params.get("cswc_multiplier", 3.45)
  r_eff_multiplier: Any = _loaded_params.get("r_eff_multiplier", 1.0)
  ica_ice_scale: Any = _loaded_params.get("ica_ice_scale", 4.82)
  ica_liq_scale: Any = _loaded_params.get("ica_liq_scale", 3.92)

  # --- Wyser Polynomial Coefficients (effective diameter in microns) ---
  wyser_coeff_0: Any = 377.4  # [microns] (Length)
  wyser_coeff_1: Any = 203.3  # [microns] (Length)
  wyser_coeff_2: Any = 37.91  # [microns] (Length)
  wyser_coeff_3: Any = 2.3696  # [microns] (Length)

  def tree_flatten(self):
    """Flattens the PyTree into its children nodes and auxiliary data."""
    children = (
        self.rho_w,
        self.k_val,
        self.nc_ocean,
        self.nc_land,
        self.r_ref,
        self.rho_ice,
        self.sigmoid_scale,
        self.snow_scatter_factor,
        self.iwp_conv_thresh,
        self.wyser_offset,
        self.wyser_slope,
        self.cext_water_scale,
        self.ciwc_multiplier,
        self.cswc_multiplier,
        self.r_eff_multiplier,
        self.ica_ice_scale,
        self.ica_liq_scale,
        self.wyser_coeff_0,
        self.wyser_coeff_1,
        self.wyser_coeff_2,
        self.wyser_coeff_3,
    )
    return children, None

  @classmethod
  def tree_unflatten(cls, _aux_data, children):
    return cls(*children)


# ==============================================================================
# Warm Liquid Microphysics
# ==============================================================================
def compute_liquid_properties(
    clwc,
    crwc,
    rho_air,
    dz,
    wavelength,
    is_land,
    params = MicrophysicsParams(),
):
  """Computes liquid water cloud optical depth and effective radius."""
  total_liquid = clwc + 0.40 * crwc
  lwc_gm3 = jnp.maximum(total_liquid * rho_air, 1e-8)

  # Warm stratocumulus cloud droplet sizing parameterization
  # Citation: Martin et al. (1994),
  # https://doi.org/10.1175/1520-0469(1994)051<1823:TMAPOE>2.0.CO;2
  n_c = jnp.where(is_land > 0.5, params.nc_land, params.nc_ocean)
  d_eff = 2.0e6 * (
      (3.0 * lwc_gm3) / (4.0 * jnp.pi * params.k_val * n_c * params.rho_w)
  ) ** (1.0 / 3.0)
  d_eff = jnp.clip(d_eff, 7.5, 42.0)

  # Satellite-agnostic channel wavelength matching
  is_8um = jnp.abs(wavelength - 8.5) < 0.4
  is_10um = jnp.abs(wavelength - 10.4) < 0.4
  q_ext = jnp.where(is_8um, 1.85, jnp.where(is_10um, 2.78, 3.88))

  q_ext = q_ext * (d_eff / params.r_ref) ** 0.14

  k_ext = (3.0 * q_ext) / (params.rho_w * d_eff * 1e-6)
  tau_water = params.cext_water_scale * k_ext * total_liquid * rho_air * dz
  return tau_water, d_eff / 2.0


# ==============================================================================
# Cold Ice Microphysics
# ==============================================================================
def compute_ice_properties(
    ciwc,
    cswc,
    T_prof,
    rho_air,
    dz,
    cext_grid,
    sizes_grid,
    ice_spectral_shift_factor=1.0,
    params = MicrophysicsParams(),
):
  """Computes ice crystal effective radius and optical depth.

  Args:
    ciwc: Cloud ice water content.
    cswc: Cloud snow water content.
    T_prof: Temperature profile.
    rho_air: Air density.
    dz: Layer thickness.
    cext_grid: Extinction cross-section grid.
    sizes_grid: Grid of effective sizes.
    ice_spectral_shift_factor: Spectral shift factor for this specific channel.
    params: MicrophysicsParams PyTree container.

  Returns:
    A tuple of (r_eff, tau_ice) representing:
      r_eff: Ice crystal effective radius [microns].
      tau_ice: Ice cloud optical depth.
  """
  # Ice-Snow coupling adapted from ECMWF IFS Documentation CY41R2,
  # Part IV, Chapter 2 (Radiation)
  # Multipliers calibrated by ERA search for Candidate 992.
  total_ice = ciwc * params.ciwc_multiplier + cswc * params.cswc_multiplier
  iwc_gm3 = jnp.maximum(total_ice * rho_air * 1000.0, 1e-6)
  T_c = T_prof - 273.15

  # Adapted from Wyser (1998),
  # https://doi.org/10.1175/1520-0442(1998)011<1793:TERIIC>2.0.CO;2
  # Parameters (offset, slope) calibrated by ERA search for Candidate 992.
  log_iwc = jnp.log10(jnp.maximum(iwc_gm3, 1e-5) / 50.0)
  b = jnp.clip(
      params.wyser_offset
      + params.wyser_slope * jnp.maximum(-T_c, 0.0) ** 1.5 * log_iwc,
      -1.8,
      1.8,
  )
  d_eff = (
      params.wyser_coeff_0
      + params.wyser_coeff_1 * b
      + params.wyser_coeff_2 * b**2
      + params.wyser_coeff_3 * b**3
  )
  r_eff = jnp.clip(d_eff / 2.0, 11.0, 480.0)

  cext_val = jnp.interp(r_eff, sizes_grid, cext_grid)
  r_eff_m = r_eff * 1e-6
  cext_m2 = cext_val * 1e-12
  tau_ice = (3.0 * cext_m2 * total_ice * rho_air * dz) / (
      4.0 * jnp.pi * params.rho_ice * r_eff_m**3
  )
  tau_ice = tau_ice + (params.snow_scatter_factor * cswc * rho_air * dz)

  # Apply convective spectral equalization via IWP mask
  iwp_path = jnp.sum(total_ice * rho_air * dz)
  conv_mask = jax.nn.sigmoid(
      (iwp_path - params.iwp_conv_thresh) * params.sigmoid_scale
  )
  tau_ice = tau_ice * (1.0 + (ice_spectral_shift_factor - 1.0) * conv_mask)

  return r_eff, tau_ice


# ==============================================================================
# Helper Methods (Ocean Emissivity & Phase Masking)
# ==============================================================================
# Gauss-Hermite quadrature points and weights (normalized by sqrt(pi)) for M=5
_GH_X = np.array([0.0, -0.95857248, 0.95857248, -2.02018287, 2.02018287])
_GH_W = np.array([0.53333333, 0.39361932, 0.39361932, 0.01995324, 0.01995324])


@jax.jit
def _fresnel_emissivity_complex(cos_t, n):
  """Computes Fresnel emissivity for complex refractive index."""
  cos_t = cos_t[Ellipsis, None, :]  # (..., 1, 25)
  n = n[Ellipsis, :, None]  # (..., 3, 1)

  sin_t = jnp.sqrt(jnp.maximum(1.0 - cos_t**2, 0.0))
  cos_t_t = jnp.sqrt(1.0 - (sin_t / n) ** 2 + 0j)
  rs = (cos_t - n * cos_t_t) / (cos_t + n * cos_t_t)
  rp = (n * cos_t - cos_t_t) / (n * cos_t + cos_t_t)
  r = 0.5 * (jnp.abs(rs) ** 2 + jnp.abs(rp) ** 2)
  return 1.0 - r


@jax.jit
def compute_ocean_directional_emissivity(
    em_nadir,
    mu_view,
    wind_speed=None,
    ocean_em_floor=_loaded_params.get("ocean_em_floor", 0.772),
):
  """Computes view-angle and wind-speed dependent ocean emissivity."""
  if wind_speed is None:
    raise ValueError(
        "Wind speed data (u10, v10) must be provided for Cox-Munk ocean"
        " emissivity calculation."
    )

  r0 = jnp.clip(1.0 - em_nadir, 0.001, 0.1)
  n = (1.0 + jnp.sqrt(r0)) / (1.0 - jnp.sqrt(r0))

  # Complex refractive index approximation (imaginary part 0.05j at 10.3 um)
  n_complex = n + 0.05j

  # Cox-Munk rough ocean wave slope distribution integration
  # Citation: Cox & Munk (1954), https://doi.org/10.1364/JOSA.44.000838
  sin_theta = jnp.sqrt(jnp.maximum(1.0 - mu_view**2, 0.0))

  sigma2 = 0.003 + 0.00512 * wind_speed
  sigma = jnp.sqrt(sigma2)

  # Flattened grid of GH points for vectorized computation (M=5, 25 terms)
  x_grid = jnp.repeat(_GH_X, 5)
  y_grid = jnp.tile(_GH_X, 5)
  w_grid = jnp.repeat(_GH_W, 5) * jnp.tile(_GH_W, 5)

  # Compute slopes
  zx = jnp.sqrt(2.0) * sigma * x_grid
  zy = jnp.sqrt(2.0) * sigma * y_grid

  cos_ti_num = mu_view - zx * sin_theta
  is_visible = cos_ti_num > 0

  cos_ti = jnp.where(
      is_visible, cos_ti_num / jnp.sqrt(1.0 + zx**2 + zy**2), 1.0
  )

  # em shape: (..., 3, 25)
  em = _fresnel_emissivity_complex(cos_ti, n_complex)

  # weight shape: (..., 25)
  weight = w_grid * cos_ti_num
  weight = jnp.where(is_visible, weight, 0.0)

  # num shape: (..., 3)
  num = jnp.sum(em * weight[Ellipsis, None, :], axis=-1)
  # den shape: (...,)
  den = jnp.sum(weight, axis=-1)

  safe_den = jnp.where(den > 0, den, 1.0)
  em_rough = jnp.where(
      den[Ellipsis, None] > 0, num / safe_den[Ellipsis, None], ocean_em_floor
  )
  return jnp.maximum(em_rough, ocean_em_floor)


@jax.jit
def apply_thermodynamic_phase_mask(ciwc_raw, clwc_raw, T_prof):
  """Differentiably partitions cloud water into liquid and ice phases.

  Uses a sigmoid-based transition centered at -15C (258.15K) with a 5K width
  to calculate ice and liquid fractions, ensuring physical consistency.

  Args:
    ciwc_raw: Raw cloud ice water content [kg/kg].
    clwc_raw: Raw cloud liquid water content [kg/kg].
    T_prof: Temperature profile [K].

  Returns:
    ciwc_phys: Partitioned physical cloud ice water content [kg/kg].
    clwc_phys: Partitioned physical cloud liquid water content [kg/kg].
  """
  # Ice fraction: 0 at warm temp, 1 at cold temp
  # Centered at 258.15 K (-15 C), scaling factor 5.0
  f_ice = jax.nn.sigmoid((258.15 - T_prof) / 5.0)

  # Liquid fraction is the complement
  f_liquid = 1.0 - f_ice

  ciwc_phys = ciwc_raw * f_ice
  clwc_phys = clwc_raw * f_liquid

  return ciwc_phys, clwc_phys
