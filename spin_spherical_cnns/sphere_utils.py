# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Utilities for handling spherical functions."""

import functools
from typing import Tuple, Union
import jax.numpy as jnp
import numpy as np

import spherical


Array = Union[np.ndarray, jnp.ndarray]


def make_equiangular_grid(resolution):
  """Make equiangular grid on the sphere.

  This is the grid used by Huffenberger and Wandelt, which includes both
  poles. For resolution n:

  longitude = 2*pi/n * j for 0 <= j < n
  colatitude = pi/(n-1) * j for 0 <= j < n

  Args:
    resolution: grid resolution.

  Returns:
    Two float64 (resolution, resolution) matrices with longitude and
    colatitude values.
  """
  longitude = np.linspace(0, 2*np.pi, resolution, endpoint=False)
  colatitude = np.linspace(0, np.pi, resolution)
  return np.meshgrid(longitude, colatitude)


def sphere_quadrature_weights(resolution):
  """Returns spherical quadrature weights per latitude.

  These roughly correspond to spherical pixel areas. The sum of all weights is
  the area of the unit sphere, 4pi. Areas at some colatitude band are roughly
  proportional to the spherical portion of the H&W quadrature weights.

  Args:
    resolution: indicates a (resolution, resolution) grid.
  Returns:
    An (resolution,) array with the cell area per latitude.
  """
  if resolution == 1:
    return jnp.ones(1) * 4 * np.pi

  areas = torus_quadrature_weights(resolution)[:resolution]
  areas = areas / areas.sum() * 4 * np.pi

  # The area computed is for each latitude band; we divide by resolution to get
  # the area of each cell.
  return jnp.array(areas / resolution)


def torus_quadrature_weights(resolution):
  """Computes quadrature weights to integrate over torus extended from sphere.

  We follow spinsfast_quadrature_weights() in spinsfast_forward_Imm.c.

  Args:
    resolution: Spherical resolution. The spherical function being extended is
       an array of shape (resolution, resolution).

  Returns:
    A 2*(resolution-1) float64 vector with the normalized quadrature
    weights. The first (resolution,) weights correspond to the spherical
    function being extended, which includes both poles (colatitudes in [0, pi],
    a closed interval). The last (resolution-2,) weights correspond to the
    extended portion, with colatitudes in (pi, 2pi), an open interval.
  """
  w_size = 2 * (resolution - 1)
  # Compute pos = [0, 1, ..., ell-1, -(ell-2), -(ell-3), ..., -1].
  pos = np.arange(w_size)
  pos[pos > w_size / 2] = (pos - w_size)[pos > w_size / 2]
  weights = np.zeros_like(pos, dtype=np.complex128)
  weights[-1] = 1j * np.pi / 2
  weights[1] = -1j * np.pi / 2

  is_even = (pos % 2) == 0
  weights[is_even] = 2. / (1 - pos * pos)[is_even]

  return np.fft.ifft(weights).real


def spin_spherical_mean(sphere_set):
  """Evaluates average over spin-weighted spherical functions.

  This uses the same quadrature scheme to integrate over the sphere as H&W for
  computing the spin-weighted spherical harmonics transform. This is necessary
  to satisfy the Parseval identity and reduce equivariance errors caused by
  mismatching quadrature rules.

  Args:
    sphere_set: A (batch_size, resolution, resolution, n_spins, n_channels)
      array of spin-weighted spherical functions (SWSF) with equiangular
      sampling.
  Returns:
    An (batch_size, n_spins, n_channels) array with the
    spherical averages.
  """
  latitude_axis = 1
  resolution = sphere_set.shape[latitude_axis]

  # Quadrature weights as defined by H&W sum up to 2.0 over latitude only; we
  # correct it here.
  weights = torus_quadrature_weights(resolution) / resolution / 2
  # Split weights in sphere and extension parts.
  weights_base, weights_extension = weights[:resolution], weights[resolution:]
  # Instead of explicitly extending the sphere as in
  # `spin_spherical_harmonics._extend_sphere_fft`, here we can simply
  # add the quadrature weights since we are summing over the spatial
  # dimensions anyway. This is significantly faster.
  weights = weights_base + np.pad(weights_extension[::-1], (1, 1))

  return jnp.einsum("bhwsc,h->bsc", sphere_set, weights.astype(np.float32))


def ell_max_from_resolution(resolution):
  """Returns the maximum degree for a spherical input of given resolution."""
  return resolution // 2 - 1


def ell_max_from_n_coeffs(n_coeffs):
  """Returns the maximum degree for an SWSFT with n_coeffs coefficients."""
  ell_max = int(np.sqrt(n_coeffs)) - 1
  if (ell_max + 1) ** 2 != n_coeffs:
    raise ValueError("n_coeffs must be a perfect square!")
  return ell_max


# The following two functions are slow and called numerous times during testing;
# we cache the outputs to avoid timeouts.
@functools.lru_cache()
def compute_wigner_delta(ell):
  r"""Computes Wigner \Delta.

  Wigner \Delta (\Delta_{m,n}^\ell) are Wigner-d matrices evaluated at pi/2.
  They allow using FFTs to speedup intermediate steps of the SWSH transform.

  Args:
    ell: Degree (int).

  Returns:
    An (2*ell+1, 2*ell+1) read-only float64 array with the Wigner
      Delta. Output is read-only to allow caching.

  Raises:
    ValueError: if ell larger than maximum allowed.
  """
  wigner_delta_array = spherical.wigner_d(np.exp(1j * np.pi/2),
                                          ell, ell).reshape(2*ell+1, 2*ell+1)
  wigner_delta_array.flags.writeable = False
  return wigner_delta_array


@functools.lru_cache()
def compute_all_wigner_delta(ell_max):
  r"""Computes all Wigner \Delta for 0 <= ell <= ell_max.

  Args:
    ell_max: Maximum degree (int).

  Returns:
    Tuple with ell_max+1 elements. Element at position ell is an
    (2*ell+1, 2*ell+1) float64 array with the Wigner \Delta of degree
    ell.
  """
  return tuple(compute_wigner_delta(ell) for ell in range(ell_max + 1))


def swsft_forward_constant(spin, ell, m):
  """Returns constant factor in SWSFT computation."""
  return (-1.0)**spin * np.sqrt((2 * ell + 1) / 4 / np.pi) * (1j)**(m + spin)
