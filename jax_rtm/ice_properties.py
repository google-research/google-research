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
import os
import jax
import jax.numpy as jnp
from jax.scipy import interpolate
import numpy as np
from .satellite_config import SATELLITE_CONFIGS

_DATA = None
_INTERP_OMEGA0_DROXTAL = None
_INTERP_CEXT_DROXTAL = None
_INTERP_LEGENDRE_DROXTAL = None

_INTERP_OMEGA0_COLUMN = None
_INTERP_CEXT_COLUMN = None
_INTERP_LEGENDRE_COLUMN = None

_SLICES_DROXTAL = {}
_SLICES_COLUMN = {}
_SIZES = None

_PRECOMPUTED_WAVELENGTHS = frozenset(
    {0.47, 0.64, 0.86, 8.4, 8.7, 10.3, 10.5, 12.3}
)


def _ensure_loaded(create_2d_interps=False):
  global _DATA
  global _INTERP_OMEGA0_DROXTAL, _INTERP_CEXT_DROXTAL, _INTERP_LEGENDRE_DROXTAL
  global _INTERP_OMEGA0_COLUMN, _INTERP_CEXT_COLUMN, _INTERP_LEGENDRE_COLUMN
  global _SLICES_DROXTAL, _SLICES_COLUMN, _SIZES

  if _DATA is None:
    # Check multiple locations to support both internal tests and public execution
    internal_path = os.path.join(
        os.path.dirname(__file__),
        "google",
        "test_data",
        "ping_yang_multi_habit.npz",
    )
    public_path = os.path.join(
        os.path.dirname(__file__), "data", "ping_yang_multi_habit.npz"
    )
    parent_internal_path = os.path.join(
        os.path.dirname(__file__),
        "../../google/test_data",
        "ping_yang_multi_habit.npz",
    )
    parent_public_path = os.path.join(
        os.path.dirname(__file__), "../../data", "ping_yang_multi_habit.npz"
    )

    if os.path.exists(internal_path):
      data_path = internal_path
    elif os.path.exists(public_path):
      data_path = public_path
    elif os.path.exists(parent_internal_path):
      data_path = parent_internal_path
    else:
      data_path = parent_public_path
    # Ice crystal single-scattering properties from Ping Yang's database
    # Citation: Yang et al. (2013), https://doi.org/10.1175/JAS-D-12-039.1
    _DATA = np.load(data_path)

    wavelengths = _DATA["wavelengths"]
    sizes = _DATA["sizes"]
    omega0_grid = _DATA["omega0"]
    cext_grid = _DATA["cext"]
    legendre_coeffs_grid = _DATA["legendre_coeffs"]

    # Start with precomputed wavelengths needed for unit tests and fast 1D slicing
    target_wavelengths = set(_PRECOMPUTED_WAVELENGTHS)

    # Dynamically add any additional wavelengths from satellite configurations
    for config in SATELLITE_CONFIGS.values():
      target_wavelengths.update(config.wavelengths)

    # Sort wavelengths for deterministic index lookup
    target_wavelengths = sorted(list(target_wavelengths))
    indices = [np.argmin(np.abs(wavelengths - w)) for w in target_wavelengths]

    _SIZES = sizes
    _SLICES_DROXTAL = {
        w: (
            omega0_grid[0, idx],
            cext_grid[0, idx],
            legendre_coeffs_grid[0, idx],
        )
        for w, idx in zip(target_wavelengths, indices)
    }
    _SLICES_COLUMN = {
        w: (
            omega0_grid[1, idx],
            cext_grid[1, idx],
            legendre_coeffs_grid[1, idx],
        )
        for w, idx in zip(target_wavelengths, indices)
    }

  if create_2d_interps and _INTERP_OMEGA0_DROXTAL is None:
    grid = (_DATA["wavelengths"], _DATA["sizes"])
    omega0_grid = _DATA["omega0"]
    cext_grid = _DATA["cext"]
    legendre_coeffs_grid = _DATA["legendre_coeffs"]

    # Habit 0 is droxtal
    _INTERP_OMEGA0_DROXTAL = interpolate.RegularGridInterpolator(
        grid, omega0_grid[0]
    )
    _INTERP_CEXT_DROXTAL = interpolate.RegularGridInterpolator(
        grid, cext_grid[0]
    )
    _INTERP_LEGENDRE_DROXTAL = interpolate.RegularGridInterpolator(
        grid, legendre_coeffs_grid[0]
    )

    # Habit 1 is single_column
    _INTERP_OMEGA0_COLUMN = interpolate.RegularGridInterpolator(
        grid, omega0_grid[1]
    )
    _INTERP_CEXT_COLUMN = interpolate.RegularGridInterpolator(
        grid, cext_grid[1]
    )
    _INTERP_LEGENDRE_COLUMN = interpolate.RegularGridInterpolator(
        grid, legendre_coeffs_grid[1]
    )


def get_interpolators():
  _ensure_loaded(create_2d_interps=True)
  return (
      (_INTERP_OMEGA0_DROXTAL, _INTERP_CEXT_DROXTAL, _INTERP_LEGENDRE_DROXTAL),
      (_INTERP_OMEGA0_COLUMN, _INTERP_CEXT_COLUMN, _INTERP_LEGENDRE_COLUMN),
  )


def _get_ice_properties_droxtal_raw(
    wavelength, r_eff, interp_omega0, interp_cext, interp_legendre
):
  """Raw 2D interpolation of Droxtal properties."""
  r_eff = jnp.asarray(r_eff)
  if r_eff.ndim > 0:
    w_arr = jnp.ones_like(r_eff) * wavelength
    points = jnp.stack([w_arr, r_eff], axis=-1)
    omega0 = interp_omega0(points)
    cext = interp_cext(points)
    legendre_coeffs = interp_legendre(points)
  else:
    point = jnp.array([[wavelength, r_eff]])
    omega0 = interp_omega0(point)[0]
    cext = interp_cext(point)[0]
    legendre_coeffs = interp_legendre(point)[0]
  return omega0, cext, legendre_coeffs


def _get_ice_properties_single_column_raw(
    wavelength, r_eff, interp_omega0, interp_cext, interp_legendre
):
  """Raw 2D interpolation of Single Column properties."""
  r_eff = jnp.asarray(r_eff)
  if r_eff.ndim > 0:
    w_arr = jnp.ones_like(r_eff) * wavelength
    points = jnp.stack([w_arr, r_eff], axis=-1)
    omega0 = interp_omega0(points)
    cext = interp_cext(points)
    legendre_coeffs = interp_legendre(points)
  else:
    point = jnp.array([[wavelength, r_eff]])
    omega0 = interp_omega0(point)[0]
    cext = interp_cext(point)[0]
    legendre_coeffs = interp_legendre(point)[0]
  return omega0, cext, legendre_coeffs


def get_ice_properties_droxtal(wavelength, r_eff):
  """Get ice optical properties for Droxtals (auto-loading defaults)."""
  if wavelength in _PRECOMPUTED_WAVELENGTHS:
    return get_ice_properties_droxtal_1d(wavelength, r_eff)

  _ensure_loaded(create_2d_interps=True)
  return _get_ice_properties_droxtal_raw(
      wavelength,
      r_eff,
      _INTERP_OMEGA0_DROXTAL,
      _INTERP_CEXT_DROXTAL,
      _INTERP_LEGENDRE_DROXTAL,
  )


def get_ice_properties_single_column(wavelength, r_eff):
  """Get ice optical properties for Single Columns (auto-loading defaults)."""
  if wavelength in _PRECOMPUTED_WAVELENGTHS:
    return get_ice_properties_single_column_1d(wavelength, r_eff)

  _ensure_loaded(create_2d_interps=True)
  return _get_ice_properties_single_column_raw(
      wavelength,
      r_eff,
      _INTERP_OMEGA0_COLUMN,
      _INTERP_CEXT_COLUMN,
      _INTERP_LEGENDRE_COLUMN,
  )


def get_ice_properties_droxtal_1d(wavelength, r_eff):
  _ensure_loaded()
  omega0_grid, cext_grid, legendre_grid = _SLICES_DROXTAL[wavelength]
  legendre_grid = jnp.array(legendre_grid)

  omega0 = jnp.interp(r_eff, _SIZES, omega0_grid)
  cext = jnp.interp(r_eff, _SIZES, cext_grid)

  legendre_coeffs = []
  for i in range(128):
    c = jnp.interp(r_eff, _SIZES, legendre_grid[:, i])
    legendre_coeffs.append(c)
  legendre_coeffs = jnp.array(legendre_coeffs)
  return omega0, cext, legendre_coeffs


def get_ice_properties_single_column_1d(wavelength, r_eff):
  _ensure_loaded()
  omega0_grid, cext_grid, legendre_grid = _SLICES_COLUMN[wavelength]
  legendre_grid = jnp.array(legendre_grid)

  omega0 = jnp.interp(r_eff, _SIZES, omega0_grid)
  cext = jnp.interp(r_eff, _SIZES, cext_grid)

  legendre_coeffs = []
  for i in range(128):
    c = jnp.interp(r_eff, _SIZES, legendre_grid[:, i])
    legendre_coeffs.append(c)
  legendre_coeffs = jnp.array(legendre_coeffs)
  return omega0, cext, legendre_coeffs


def get_stacked_slices_droxtal(wavelengths=None):
  if wavelengths is None:
    wavelengths = tuple(SATELLITE_CONFIGS["goes"].wavelengths)
  _ensure_loaded()
  omega0_stacked = jnp.stack([_SLICES_DROXTAL[w][0] for w in wavelengths])
  cext_stacked = jnp.stack([_SLICES_DROXTAL[w][1] for w in wavelengths])
  leg_stacked = jnp.stack([_SLICES_DROXTAL[w][2] for w in wavelengths])
  return omega0_stacked, cext_stacked, leg_stacked


def get_stacked_slices_single_column(wavelengths=None):
  if wavelengths is None:
    wavelengths = tuple(SATELLITE_CONFIGS["goes"].wavelengths)
  _ensure_loaded()
  omega0_stacked = jnp.stack([_SLICES_COLUMN[w][0] for w in wavelengths])
  cext_stacked = jnp.stack([_SLICES_COLUMN[w][1] for w in wavelengths])
  leg_stacked = jnp.stack([_SLICES_COLUMN[w][2] for w in wavelengths])
  return omega0_stacked, cext_stacked, leg_stacked
