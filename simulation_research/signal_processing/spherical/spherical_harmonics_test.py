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

"""Tests for the library of spherical harmonics."""

import math

from absl.testing import absltest

import jax.numpy as jnp
import numpy as np
import scipy.special as sp_special

from simulation_research.signal_processing.spherical import spherical_harmonics


def _compute_spherical_harmonics(l_max, theta, phi, nonnegative_order=True):
  """Computes the spherical harmonics."""
  num_theta = theta.shape[0]
  num_phi = phi.shape[0]
  phi, theta = np.meshgrid(phi, theta)

  sph_harm = np.zeros((l_max + 1, l_max + 1, num_theta, num_phi), dtype=complex)
  if nonnegative_order:
    for l in np.arange(l_max + 1):
      for m in np.arange(l + 1):
        sph_harm[l, m, :, :] = sp_special.sph_harm(m, l, phi, theta)
  else:
    for l in np.arange(l_max + 1):
      for m in np.arange(l + 1):
        sph_harm[l, m, :, :] = sp_special.sph_harm(-m, l, phi, theta)

  return jnp.asarray(sph_harm)


class SphericalHarmonicsTest(absltest.TestCase):

  def testOrderZeroDegreeZero(self):
    """Tests the spherical harmonics of order zero and degree zero."""
    num_theta = 6
    num_phi = 4
    expected = (1.0 / jnp.sqrt(4.0 * math.pi) *
                jnp.ones((1, 1, num_theta, num_phi)))
    theta = jnp.linspace(0, math.pi, num_theta)
    phi = jnp.linspace(0, 2.0 * math.pi, num_phi)
    sph_harm = spherical_harmonics.SphericalHarmonics(
        l_max=0, theta=theta, phi=phi)
    actual = jnp.real(sph_harm.harmonics_nonnegative_order())
    np.testing.assert_allclose(actual, expected, rtol=1.1e-7, atol=3e-8)

  def testOrderOneDegreeZero(self):
    """Tests the spherical harmonics of order one and degree zero."""
    num_theta = 4
    num_phi = 6
    theta = jnp.linspace(0, math.pi, num_theta)
    phi = jnp.linspace(0, 2.0 * math.pi, num_phi)

    expected = jnp.sqrt(3.0 / (4.0 * math.pi)) * jnp.outer(
        jnp.cos(theta), jnp.ones_like(phi))
    sph_harm = spherical_harmonics.SphericalHarmonics(
        l_max=1, theta=theta, phi=phi)
    actual = jnp.real(sph_harm.harmonics_nonnegative_order()[1, 0, :, :])
    np.testing.assert_allclose(actual, expected, rtol=7e-8, atol=1.5e-8)

  def testOrderOneDegreeOne(self):
    """Tests the spherical harmonics of order one and degree one."""
    num_theta = 7
    num_phi = 8
    theta = jnp.linspace(0, math.pi, num_theta)
    phi = jnp.linspace(0, 2.0 * math.pi, num_phi)

    expected = -1.0 / 2.0 * jnp.sqrt(3.0 / (2.0 * math.pi)) * jnp.outer(
        jnp.sin(theta), jnp.exp(1j * phi))
    sph_harm = spherical_harmonics.SphericalHarmonics(
        l_max=1, theta=theta, phi=phi)
    actual = sph_harm.harmonics_nonnegative_order()[1, 1, :, :]
    np.testing.assert_allclose(
        jnp.abs(actual), jnp.abs(expected), rtol=1e-8, atol=6e-8)

  def testAgainstScipySpecialSphHarmNonnegativeOrder(self):
    """Tests the accuracy against scipy.special.sph_harm."""
    l_max = 64
    num_theta = 128
    num_phi = 128
    theta = jnp.linspace(0, math.pi, num_theta)
    phi = jnp.linspace(0, 2.0 * math.pi, num_phi)

    expected = _compute_spherical_harmonics(l_max=l_max, theta=theta, phi=phi)
    sph_harm = spherical_harmonics.SphericalHarmonics(
        l_max=l_max, theta=theta, phi=phi)
    actual = sph_harm.harmonics_nonnegative_order()
    np.testing.assert_allclose(
        jnp.abs(actual), jnp.abs(expected), rtol=1e-8, atol=9e-5)

  def testAgainstScipySpecialSphHarmNonpositiveOrder(self):
    """Tests the accuracy against scipy.special.sph_harm."""
    l_max = 64
    num_theta = 128
    num_phi = 128
    theta = jnp.linspace(0, math.pi, num_theta)
    phi = jnp.linspace(0, 2.0 * math.pi, num_phi)

    expected = _compute_spherical_harmonics(
        l_max=l_max, theta=theta, phi=phi, nonnegative_order=False)
    sph_harm = spherical_harmonics.SphericalHarmonics(
        l_max=l_max, theta=theta, phi=phi)
    actual = sph_harm.harmonics_nonpositive_order()
    np.testing.assert_allclose(
        jnp.abs(actual), jnp.abs(expected), rtol=1e-8, atol=9e-5)


if __name__ == '__main__':
  absltest.main()
