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

# pylint: skip-file
import dataclasses
import functools
import json

import jax
import jax.numpy as jnp

from . import adding_doubling
from . import ice_properties
from . import microphysics
from . import satellite_config


def _load_configs(params_input):
  """Constructs SatelliteConfig and MicrophysicsParams from a raw dictionary or path."""
  if isinstance(params_input, str):
    with open(params_input, "r") as f:
      params_dict = json.load(f)
  elif isinstance(params_input, dict):
    params_dict = params_input
  else:
    raise TypeError(f"Invalid params input type: {type(params_input)}")

  # 1. MicrophysicsParams
  micro_fields = {
      f.name: params_dict[f.name]
      for f in dataclasses.fields(microphysics.MicrophysicsParams)
      if f.name in params_dict
  }
  microphysics_params = microphysics.MicrophysicsParams(**micro_fields)

  # 2. SatelliteConfig
  # Determine base config (default to GOES)
  satellite_name = params_dict.get("satellite", "goes")
  base_config = satellite_config.SATELLITE_CONFIGS.get(
      satellite_name, satellite_config.GOES_ABI_CONFIG
  )

  sat_fields = {}
  for f in dataclasses.fields(satellite_config.SatelliteConfig):
    if f.name in params_dict:
      val = params_dict[f.name]
      if isinstance(val, list):
        val = jnp.array(val)
      sat_fields[f.name] = val
  if sat_fields:
    satellite_config_obj = dataclasses.replace(base_config, **sat_fields)
  else:
    satellite_config_obj = base_config

  # Extract surface biases from config object
  t_land_bias = satellite_config_obj.t_land_bias
  t_water_bias = satellite_config_obj.t_water_bias

  return satellite_config_obj, microphysics_params, t_land_bias, t_water_bias


@jax.jit
def inverse_planck_function(wavelength_um, radiance):
  """Calculates Brightness Temperature in Kelvin from Radiance."""
  h = 6.62607015e-34  # J s
  c = 2.99792458e8  # m/s
  k_b = 1.380649e-23  # J/K

  c1 = 2.0 * h * c**2 * 1e24
  c2 = h * c / (k_b * 1e-6)

  # Inverted Planck equation
  return c2 / (
      wavelength_um * jnp.log((c1 / (wavelength_um**5 * radiance)) + 1.0)
  )


def simulate_pixel(
    atmosphere,
    surface,
    geometry,
    n_streams,
    num_doubling_steps,
    satellite_config,
    microphysics_params = None,
    stack_fn=adding_doubling.stack_atmosphere,
):
  """Simulates a single column and returns BTs for the ASH RGB (calibrated)."""
  if microphysics_params is None:
    microphysics_params = microphysics.MicrophysicsParams()

  # Unpack AtmosphereState
  p_prof = atmosphere.p_prof
  T_prof = atmosphere.T_prof
  q_prof = atmosphere.q_prof
  clwc = atmosphere.clwc
  ciwc_nat = atmosphere.ciwc_nat
  cswc = atmosphere.cswc
  crwc = atmosphere.crwc

  # Unpack SurfaceState
  is_land = surface.is_land
  sd = surface.sd
  T_surf = surface.skt
  lat = surface.lat
  u10 = surface.u10
  v10 = surface.v10

  # Unpack GeometryState
  mu_0 = geometry.mu_0
  mu_view = geometry.mu_view

  # Unpack and convert SatelliteConfig to JAX arrays at tracing time
  # We need both a static tuple for database lookups and a JAX array for vmap mapping!
  wavelengths_static = tuple(satellite_config.wavelengths)
  wavelengths_jax = jnp.asarray(satellite_config.wavelengths)

  em_snow = jnp.asarray(satellite_config.em_snow)
  em_land = jnp.asarray(satellite_config.em_land)
  em_ocean = jnp.asarray(satellite_config.em_ocean)
  wv_linear = jnp.asarray(satellite_config.wv_linear)
  wv_quadratic = jnp.asarray(satellite_config.wv_quadratic)
  ice_spectral_shift = jnp.asarray(satellite_config.ice_spectral_shift)
  ocean_em_floor = satellite_config.ocean_em_floor
  tau_co2 = satellite_config.tau_co2
  tau_o3 = satellite_config.tau_o3
  wv_t_dep = satellite_config.wv_t_dep

  # Setup grids
  rho_air = p_prof / (287.058 * T_prof)
  dp = jnp.abs(jnp.gradient(p_prof))
  dz = dp / (rho_air * 9.80665)

  n_levels = len(p_prof)
  sizes_grid = jnp.array(ice_properties._SIZES)  # pylint: disable=protected-access

  # Ice slices (load both con/droxtal and nat/column using static wavelengths)
  omega0_ice_con_stacked, cext_ice_con_stacked, legendre_ice_con_stacked = (
      ice_properties.get_stacked_slices_droxtal(wavelengths_static)
  )
  omega0_ice_con_stacked = jnp.array(omega0_ice_con_stacked)
  cext_ice_con_stacked = jnp.array(cext_ice_con_stacked)
  legendre_ice_con_stacked = jnp.array(legendre_ice_con_stacked)

  omega0_ice_nat_stacked, cext_ice_nat_stacked, legendre_ice_nat_stacked = (
      ice_properties.get_stacked_slices_single_column(wavelengths_static)
  )
  omega0_ice_nat_stacked = jnp.array(omega0_ice_nat_stacked)
  cext_ice_nat_stacked = jnp.array(cext_ice_nat_stacked)
  legendre_ice_nat_stacked = jnp.array(legendre_ice_nat_stacked)

  # Calculate directional ocean emissivity via Fresnel reflection
  if u10 is None or v10 is None:
    raise ValueError(
        "Wind speed components u10 and v10 are strictly required for Cox-Munk"
        " rough ocean emissivity calculation."
    )
  wind_speed = jnp.sqrt(u10**2 + v10**2)

  em_ocean_dir = microphysics.compute_ocean_directional_emissivity(
      em_ocean, mu_view, wind_speed=wind_speed, ocean_em_floor=ocean_em_floor
  )

  # Emissivity based on land/ocean/snow
  is_snow_surf = sd > 0.01
  em_base = jnp.where(
      is_snow_surf, em_snow, jnp.where(is_land, em_land, em_ocean_dir)
  )

  # Refined ICA logic for cloud-clear column separation using PyTree parameters
  total_iwp = jnp.sum(
      (
          ciwc_nat * microphysics_params.ciwc_multiplier
          + cswc * microphysics_params.cswc_multiplier
      )
      * rho_air
      * dz
  )
  total_lwp = jnp.sum((clwc + 0.40 * crwc) * rho_air * dz)
  cc_ice = 1.0 - jnp.exp(-total_iwp * microphysics_params.ica_ice_scale)
  cc_liq = 1.0 - jnp.exp(-total_lwp * microphysics_params.ica_liq_scale)
  cc = jnp.clip(jnp.maximum(cc_ice, cc_liq), 0.01, 0.99)

  def run_one_wavelength(w, em, linear_wv, quad_wv, co2_off, o3_off):
    # Index of wavelength (GOES channels)
    idx = jnp.where(
        jnp.abs(w - wavelengths_static[0]) < 0.1,
        0,
        jnp.where(jnp.abs(w - wavelengths_static[1]) < 0.1, 1, 2),
    )

    # 1. Water cloud extinction
    tau_water, r_eff_water = microphysics.compute_liquid_properties(
        clwc,
        crwc,
        rho_air,
        dz,
        wavelength=w,
        is_land=is_land,
        params=microphysics_params,
    )

    # 2. Ice Cloud Extinction (Clean, no duplicate parameters passed!)
    r_eff_nat, tau_ice_nat_only = microphysics.compute_ice_properties(
        ciwc_nat,
        cswc,
        T_prof,
        rho_air,
        dz,
        cext_grid=cext_ice_nat_stacked[idx],
        sizes_grid=sizes_grid,
        ice_spectral_shift_factor=ice_spectral_shift[idx],
        params=microphysics_params,
    )

    tau_total = tau_water + tau_ice_nat_only
    tau_eff = -jnp.log(
        jnp.maximum(
            1.0 - cc * (1.0 - jnp.exp(-jnp.maximum(tau_total, 1e-7))), 1e-8
        )
    )
    w_i = tau_ice_nat_only / jnp.maximum(tau_total, 1e-9)
    tau_i_eff, tau_l_eff = tau_eff * w_i, tau_eff * (1.0 - w_i)

    # Enhanced gas absorption with pressure scaling
    t_corr = jnp.exp(wv_t_dep * (T_prof - 288.15) / 10.0)
    tau_wv = (linear_wv * q_prof + quad_wv * q_prof**2 * t_corr) * dp / 9.80665
    p_scale = (p_prof / 101325.0) ** 1.25
    gas_offset = jnp.where(
        jnp.abs(w - wavelengths_static[2]) < 0.1, co2_off, 0.0
    ) + jnp.where(
        jnp.abs(w - wavelengths_static[1]) < 0.1, o3_off, 0.0
    )
    tau_gas = tau_wv + (
        gas_offset * p_scale * dp / jnp.maximum(jnp.sum(dp), 1e-12)
    )

    # Call stack_atmosphere
    rad = stack_fn(
        jnp.zeros_like(T_prof),  # p_tau_ray=0
        jnp.zeros_like(T_prof),  # p_tau_ice_con=0
        tau_i_eff,
        tau_gas,
        tau_l_eff,
        jnp.zeros_like(T_prof),  # p_omega_ray=0
        jnp.ones_like(T_prof) * 1.5,  # p_r_con placeholder
        r_eff_nat,
        r_eff_water,
        T_prof,
        wavelength=w,
        n_streams=n_streams,
        num_doubling_steps=num_doubling_steps,
        mu_0=mu_0,
        T_surf=T_surf,
        is_thermal=True,
        emissivity=em,
        omega0_ice_con_grid=omega0_ice_con_stacked[idx],
        legendre_ice_con_grid=legendre_ice_con_stacked[idx],
        omega0_ice_nat_grid=omega0_ice_nat_stacked[idx],
        legendre_ice_nat_grid=legendre_ice_nat_stacked[idx],
    )

    # Angular interpolation of quadrature streams to viewing zenith
    mu_q, _ = adding_doubling.get_quadrature(n_streams)
    low = jnp.clip(jnp.searchsorted(mu_q, mu_view) - 1, 0, n_streams - 2)
    weight = (mu_view - mu_q[low]) / (mu_q[low + 1] - mu_q[low])
    r_perspective = (1.0 - weight) * rad[2 * low] + weight * rad[2 * (low + 1)]

    bt = inverse_planck_function(w, r_perspective)
    total_ice_tau = jnp.sum(tau_i_eff, axis=0)

    return bt, total_ice_tau

  # Vectorize over wavelengths using the JAX array
  wavelengths = wavelengths_jax
  co2_offs = jnp.array([0.0, 0.0, tau_co2])
  o3_offs = jnp.array([0.0, tau_o3, 0.0])

  bts, taus = jax.vmap(run_one_wavelength)(
      wavelengths, em_base, wv_linear, wv_quadratic, co2_offs, o3_offs
  )

  bt_0 = bts[0]
  bt_1 = bts[1]
  bt_2 = bts[2]
  tau_1 = taus[1]

  return bt_0, bt_1, bt_2, tau_1


def get_batch_simulator(
    params,
    n_streams,
    num_doubling_steps,
    mu_0,  # Kept in signature for backwards compatibility
    sharded_axes=None,
    use_checkpointing=True,
):
  """Returns a JIT-compiled 1D flat batch simulator (enforced config)."""
  satellite_config_obj, microphysics_params, t_land_bias, t_water_bias = (
      _load_configs(params)
  )

  # Define the local sim_pix function that binds these parameters
  def sim_pix(
      atmosphere,
      surface,
      geometry,
  ):
    # Apply surface temperature bias
    bias = jnp.where(surface.is_land, t_land_bias, t_water_bias)
    surface_biased = surface.replace(skt=surface.skt + bias)

    stack_fn_wrapped = functools.partial(
        adding_doubling.stack_atmosphere,
        sharded_axes=sharded_axes,
        use_checkpointing=use_checkpointing,
    )

    return simulate_pixel(
        atmosphere,
        surface_biased,
        geometry,
        n_streams=n_streams,
        num_doubling_steps=num_doubling_steps,
        satellite_config=satellite_config_obj,
        microphysics_params=microphysics_params,
        stack_fn=stack_fn_wrapped,
    )

  # Vmap over 1D batch (axis 0 of all inputs)
  vmap_sim = jax.vmap(sim_pix, in_axes=(0, 0, 0))
  return jax.jit(vmap_sim)


def get_vmap_simulator(params, n_streams, num_doubling_steps, mu_0):
  """Returns a JIT-compiled 2D grid simulator (independent vmap)."""
  satellite_config_obj, microphysics_params, t_land_bias, t_water_bias = (
      _load_configs(params)
  )

  def sim_pix(
      atmosphere,
      surface,
      geometry,
  ):
    bias = jnp.where(surface.is_land, t_land_bias, t_water_bias)
    surface_biased = surface.replace(skt=surface.skt + bias)

    return simulate_pixel(
        atmosphere,
        surface_biased,
        geometry,
        n_streams=n_streams,
        num_doubling_steps=num_doubling_steps,
        satellite_config=satellite_config_obj,
        microphysics_params=microphysics_params,
    )

  # Vmap over longitude (original axis 1), then latitude (original axis 0)
  # Since the outer vmap slices along axis 0 first, the inner vmap receives
  # a sliced array where the original axis 1 has become the new axis 0.
  # Thus, both must map over axis 0.
  vmap_lon = jax.vmap(sim_pix, in_axes=(0, 0, 0))
  vmap_lat_lon = jax.vmap(vmap_lon, in_axes=(0, 0, 0))

  return jax.jit(vmap_lat_lon)


def ash_rgb_compositor(bt_84, bt_103, bt_123):
  """Composes ASH RGB image from Brightness Temperatures."""
  # EUMETSAT Ash RGB composite recipe (Kerkmann et al., 2003)
  # Red: BT(12.3) - BT(10.3) | Green: BT(10.3) - BT(8.4) | Blue: BT(10.3)
  r = (bt_123 - bt_103 - (-4.0)) / (2.0 - (-4.0))
  g = (bt_103 - bt_84 - (-4.0)) / (5.0 - (-4.0))
  b = (bt_103 - 243.0) / (303.0 - 243.0)

  r = jnp.clip(r, 0.0, 1.0)
  g = jnp.clip(g, 0.0, 1.0)
  b = jnp.clip(b, 0.0, 1.0)

  # Stack to form RGB image (height, width, 3)
  return jnp.stack([r, g, b], axis=-1)
