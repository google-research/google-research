# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for spin_spherical_cnns.np_spin_spherical_harmonics."""

from absl.testing import parameterized
import numpy as np
import scipy.special
import tensorflow as tf

from spin_spherical_cnns import np_spin_spherical_harmonics


class NpSpinSphericalHarmonicsTest(tf.test.TestCase, parameterized.TestCase):
  """Test cases for numpy spin-weighted spherical Fourier transforms."""

  def _check_nonzero_harmonic_coeffs(self, coeffs, ell, m):
    """Assert that only the coeff of degree ell and order m is nonzero."""
    indices = np.ones(len(coeffs)).astype('bool')
    # This should be the only nonzero coefficient.
    indices[np_spin_spherical_harmonics._get_swsft_coeff_index(ell, m)] = False
    self.assertAllClose(coeffs[indices], np.zeros_like(coeffs[indices]))
    self.assertNotAllClose(coeffs[~indices], np.zeros_like(coeffs[~indices]))

  def _make_equiangular_grid(self, n):
    """Make (n, n) equiangular grid on the sphere.

    This is the grid used by Huffenberger and Wandelt, which includes both
    poles.

    longitude = 2*pi/n * j for 0 <= j < n
    colatitude = pi/(n-1) * j for 0 <= j < n

    Args:
      n: grid resolution (int).

    Returns:
      Two float64 (n, n) matrices corresponding to longitude and colatitude.
    """
    longitude = np.linspace(0, 2*np.pi, n+1)[:n]
    colatitude = np.linspace(0, np.pi, n)
    return np.meshgrid(longitude, colatitude)

  @parameterized.parameters(4, 8)
  def test_constant_function_forward(self, width):
    """SWSFT of a constant has only one nonzero coefficient at ell=m=0."""
    sphere = np.ones((width, width))
    coeffs = np_spin_spherical_harmonics.swsft_forward_naive(sphere, 0)
    self._check_nonzero_harmonic_coeffs(coeffs, 0, 0)

  @parameterized.parameters((4, 1, 0),
                            (4, 1, -1),
                            (8, 1, -1),
                            (8, 2, 0),
                            (8, 3, 2))
  def test_spherical_harmonics_forward(self, width, ell, m):
    r"""SWSFT of Y_m^\ell has only one nonzero coefficient, at (ell, m)."""
    longitude_g, colatitude_g = self._make_equiangular_grid(width)
    sphere = scipy.special.sph_harm(m, ell, longitude_g, colatitude_g)
    coeffs = np_spin_spherical_harmonics.swsft_forward_naive(sphere, 0)
    self._check_nonzero_harmonic_coeffs(coeffs, ell, m)

  @parameterized.parameters(4, 8)
  def test_spin_weighted_spherical_harmonics_forward(self, width):
    r"""SWSFT of sY_m^\ell has only one nonzero coefficient, at (ell, m)."""
    longitude_g, colatitude_g = self._make_equiangular_grid(width)

    # We use some known expressions for spin-weighted spherical harmonics.

    # spin=1, ell=1, m=0: 1Y_0^1(a, b) \propto sin(b)
    Y110 = np.sin(colatitude_g)  # pylint: disable=invalid-name
    coeffs = np_spin_spherical_harmonics.swsft_forward_naive(Y110, 1)
    self._check_nonzero_harmonic_coeffs(coeffs, 1, 0)

    # spin=1, ell=1, m=1: 1Y_1^1(a, b) \propto (1-cos(b))exp(ia)
    Y111 = (1-np.cos(colatitude_g)) * np.exp(1j * longitude_g)  # pylint: disable=invalid-name
    coeffs = np_spin_spherical_harmonics.swsft_forward_naive(Y111, 1)
    self._check_nonzero_harmonic_coeffs(coeffs, 1, 1)

    # spin=1, ell=1, m=-1: 1Y_{-1}^1(a, b) \propto (1+cos(b))exp(-ia)
    Y11m1 = (1+np.cos(colatitude_g)) * np.exp(-1j * longitude_g)  # pylint: disable=invalid-name
    coeffs = np_spin_spherical_harmonics.swsft_forward_naive(Y11m1, 1)
    self._check_nonzero_harmonic_coeffs(coeffs, 1, -1)

  @parameterized.parameters((4, 1, 1),
                            (4, 1, -1),
                            (8, 1, 1),
                            (8, 2, 0),
                            (8, 3, -2))
  def test_spherical_harmonics_backward(self, width, ell, m):
    r"""ISWSFT is Y_m^\ell when the only nonzero coeffs is 1. at (ell, m)."""
    n_coeffs = (width//2)**2
    coeffs = np.zeros(n_coeffs, dtype='complex128')
    coeffs[np_spin_spherical_harmonics._get_swsft_coeff_index(ell, m)] = 1
    sphere = np_spin_spherical_harmonics.swsft_backward_naive(coeffs, 0)

    phi_g, theta_g = self._make_equiangular_grid(width)
    sphere_gt = scipy.special.sph_harm(m, ell, phi_g, theta_g)

    self.assertAllClose(sphere, sphere_gt)

  @parameterized.parameters((4, 0),
                            (4, 1),
                            (4, -1),
                            (16, 0),
                            (16, 2))
  def test_swsft_backward_forward(self, n_coeffs, spin):
    r"""Given coefficients, ISWSFT -> SWFT must return them."""
    coeffs_gt = np.linspace(-1, 1, n_coeffs) + 1j*np.linspace(0, 1, n_coeffs)
    # Coefficients for ell < abs(spin) are always zero.
    coeffs_gt[:spin**2] = 0
    sphere = np_spin_spherical_harmonics.swsft_backward_naive(coeffs_gt, spin)
    coeffs = np_spin_spherical_harmonics.swsft_forward_naive(sphere, spin)
    self.assertAllClose(coeffs, coeffs_gt)

  @parameterized.parameters((4, 0),
                            (4, 1),
                            (4, -1),
                            (8, 0),
                            (8, -2))
  def test_swsft_forward_backward(self, width, spin):
    r"""Given bandlimited spherical function, SWSFT -> ISWFT must return it."""
    sphere_gt = np.linspace(-1, 1, width**2).reshape((width, width))
    # This effectively bandlimits the spherical function.
    sphere_gt = np_spin_spherical_harmonics.swsft_backward_naive(
        np_spin_spherical_harmonics.swsft_forward_naive(sphere_gt, spin), spin)

    coeffs = np_spin_spherical_harmonics.swsft_forward_naive(sphere_gt, spin)
    sphere = np_spin_spherical_harmonics.swsft_backward_naive(coeffs, spin)
    self.assertAllClose(sphere, sphere_gt)


if __name__ == '__main__':
  tf.test.main()
