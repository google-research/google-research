# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Spin-weighted spherical harmonics (SWSH) transforms in numpy.

This implements Fourier transforms for spin-weighted spherical functions
(SWSFT).  We follow the method and C implementation of Huffenberger and Wandelt,
"Fast and Exact Spin-s Spherical Harmonic Transforms," ApJS 189 255, referred to
as H&W, available at https://arxiv.org/abs/1007.3514, with the modifications
made by Mike Boyle and available on https://github.com/moble/spinsfast. We try
to use similar notation, sometimes using n instead of m' to index matrix
elemetns.

This file contains a naive, but correct implementation (tested offline against
spinsfast).  It is a starting point and test baseline for the JAX version.
"""

from typing import Sequence
import numpy as np

from spin_spherical_cnns import sphere_utils


def _extend_sphere_fft(sphere, spin):
  """Applies 2D FFT to a spherical function by extending it to a torus.

  Args:
    sphere: See swsft_forward_naive().
    spin: See swsft_forward_naive().

  Returns:
    Matrix of complex128 Fourier coefficients. If the input shape is (n, n), the
    output will be (2*n-2, n).

  Raises:
    ValueError: If input dimensions are not even.
  """
  n = sphere.shape[1]
  if n % 2 != 0:
    raise ValueError("Input sphere must have even height!")
  torus = (-1)**spin * np.roll(sphere[1:-1][::-1], n // 2, axis=1)
  torus = np.concatenate([sphere, torus], axis=0)
  weights = sphere_utils.torus_quadrature_weights(n)
  torus = weights[:, None] * torus
  coeffs = np.fft.fft2(torus) * 2 * np.pi / n

  return coeffs


def n_coeffs_from_ell_max(ell_max):
  """Returns the number of coefficients for an SWSFT with max degree ell_max."""
  return (ell_max + 1)**2


def _compute_Inm(sphere, spin):  # pylint: disable=invalid-name
  r"""Computes Inm.

  This evaluates
  Inm = \int_{S^2}e^{-in\theta}e^{-im\phi} f(\theta, \phi) sin\theta d\theta
  d\phi, as defined in H&W, Equation (8).
  It is used in intermediate steps of the SWSFT computation.

  Args:
    sphere: See swsft_forward_naive().
    spin: See swsft_forward_naive().

  Returns:
    The complex128 matrix Inm. If sphere is (n, n), output will be (n-1, n-1).

  """
  ell_max = sphere_utils.ell_max_from_resolution(sphere.shape[0])
  coeffs = _extend_sphere_fft(sphere, spin=spin)

  rows1 = np.concatenate([coeffs[:ell_max + 1, :ell_max + 1],
                          coeffs[:ell_max + 1, -ell_max:]],
                         axis=1)
  rows2 = np.concatenate([coeffs[-ell_max:, :ell_max + 1],
                          coeffs[-ell_max:, -ell_max:]],
                         axis=1)

  return np.concatenate([rows1, rows2], axis=0)


def _compute_Jnm(sphere, spin):  # pylint: disable=invalid-name
  """Computes Jnm (trimmed version of Inm).

  Jnm = I0m for n=0
      = Inm + (-1)^{m+s}I_(-n)m for n>0

  This matrix is defined in H&W, Equation (10).

  Args:
    sphere: See swsft_forward_naive().
    spin: See swsft_forward_naive().

  Returns:
    The complex128 matrix Jnm. If sphere is (n, n), output will be (n//2, n-1).

  Raises:
    ValueError: If sphere rank is not 2.
  """
  if sphere.ndim != 2:
    raise ValueError("Input sphere rank must be 2.")
  ell_max = sphere_utils.ell_max_from_resolution(sphere.shape[0])

  Inm = _compute_Inm(sphere, spin=spin)  # pylint: disable=invalid-name

  # now make a matrix with (-1)^{m+s} columns
  m = np.concatenate([np.arange(ell_max + 1),
                      -np.arange(ell_max + 1)[1:][::-1]])[None]
  signs = (-1.)**(m + spin)

  # only takes positive n
  Jnm = Inm[:ell_max + 1].copy()  # pylint: disable=invalid-name
  # make n = -n rowwise
  Jnm[1:] += signs * Inm[-ell_max:][::-1]

  return Jnm


def _swsft_forward_single(sphere, spin, ell, m):
  r"""Computes a single SWSFT coefficient.

  Compute _sa_m^\ell, where s is the spin weight, ell the degree and m the
  order.

  Args:
    sphere: See swsft_forward_naive().
    spin: See swsft_forward_naive().
    ell: Degree (int).
    m: Order (int).

  Returns:
    A complex128 coefficient.

  """
  delta = sphere_utils.compute_wigner_delta(ell)
  Jnm = _compute_Jnm(sphere, spin)  # pylint: disable=invalid-name
  coeff = 0
  for n in range(ell + 1):  # n here is sometimes called m'
    if abs(spin) >= delta.shape[1]:
      break
    delta_s = delta[n, abs(spin)]
    delta_m = delta[n, abs(m)]
    if spin > 0:  # index is (-s)
      delta_s *= (-1)**(ell + n)
    if m < 0:
      delta_m *= (-1)**(ell + n)
    coeff += delta_m * delta_s * Jnm[n, m]

  coeff *= sphere_utils.swsft_forward_constant(spin, ell, m)

  return coeff


def _get_swsft_coeff_index(ell, m):
  """Get index in coefficient array.

  Returns the index corresponding to (ell, m) in the coefficient array in the
  format returned by swsft_forward_naive()

  Args:
    ell: Degree (int).
    m: Order (int).

  Returns:
    An index (int).
  """
  return ell**2 + m + ell


def swsft_forward_naive(sphere, spin):
  """Spin-weighted spherical harmonics transform (forward).

  This is a naive and slow implementation but useful for testing; computing
  multiple coefficients in a vectorized fashion is much faster.

  Args:
    sphere: A (n, n) array representing a spherical function with
      equirectangular sampling, lat, long order.
    spin: Spin weight (int).

  Returns:
    A ((n/2)**2,) Array of complex128 coefficients sorted by increasing
    degrees. Coefficient of degree ell and order m is at position ell**2 + m +
    ell.
  """
  ell_max = sphere_utils.ell_max_from_resolution(sphere.shape[0])
  coeffs = []
  for ell in range(ell_max+1):
    for m in range(-ell, ell+1):
      coeffs.append(_swsft_forward_single(sphere, spin, ell, m))

  return np.array(coeffs)


def _compute_Gnm_naive(coeffs, spin):  # pylint: disable=invalid-name
  r"""Compute Gnm (not vectorized).

  The matrix Gnm, defined in H&W, Equation (13), is closely related to the 2D
  Fourier transform of a spin-weighted spherical function.

  Gnm = (-1)^s i^(m+s) \sum_\ell c \Delta_{-n,-s} \Delta_{-n,m} _sa_m^\ell,
  where c = \sqrt{(2\ell + 1) / (4\pi)}, and _sa_m^\ell is the coefficient at
  (ell, m).

  Args:
    coeffs: See swsft_backward_naive().
    spin: See swsft_backward_naive().

  Returns:
    The complex128 matrix Gnm. If coeffs has n**2 elements, the output is
    (2*n-1, 2*n-1).

  Raises:
    ValueError: If len(coeffs) is not a perfect square.
  """
  ell_max = sphere_utils.ell_max_from_n_coeffs(len(coeffs))
  deltas = sphere_utils.compute_all_wigner_delta(ell_max)
  Gnm = np.zeros((ell_max+1, 2*ell_max+1), dtype=np.complex128)  # pylint: disable=invalid-name
  for ell in range(abs(spin), ell_max+1):
    factor = np.sqrt((2*ell+1)/4/np.pi)
    for m in range(-ell, ell+1):
      # The following also fixes the signs because deltas should be evaluated at
      # negative n but we only store values for positive n.
      phase = (1j)**(m+spin) * (-1)**m
      index = _get_swsft_coeff_index(ell, m)
      Gnm[:ell+1, ell_max + m] += (phase * factor *
                                   deltas[ell][:, ell-spin] *
                                   deltas[ell][:, ell+m] *
                                   coeffs[index])
  # Use symmetry to obtain entries for negative n.
  signs = (-1.)**(spin + np.arange(-ell_max, ell_max+1))[None, :]
  return np.concatenate([signs * Gnm[1:][::-1], Gnm])


def swsft_backward_naive(coeffs, spin):
  """Spin-weighted spherical harmonics transform (backward).

  This is a naive and slow implementation, due to the non-vectorized Gnm
  computation.

  Args:
    coeffs: List of n**2 SWSFT coefficients.
    spin: Spin weight (int).

  Returns:
    A complex128 matrix corresponding to the (H&W) equiangular sampling of a
    spin-weighted spherical function.
  """
  ell_max = sphere_utils.ell_max_from_n_coeffs(len(coeffs))
  # Gnm is related to the 2D Fourier transform of the desired output.
  ft = _compute_Gnm_naive(coeffs, spin)
  # Rearrange order for ifft2.
  ft = np.fft.ifftshift(ft)
  # We insert zero rows and columns to ensure the IFFT output matches the
  # swsft_backward_naive() input resolution.
  ft = np.concatenate([ft[:, :ell_max+1],
                       np.zeros((2*ell_max+1, 1)),
                       ft[:, ell_max+1:]], axis=1)
  ft = np.concatenate([ft[:ell_max+1],
                       np.zeros_like(ft),
                       ft[ell_max+1:]])

  return np.fft.ifft2(ft)[:2*(ell_max+1)] * ft.size
