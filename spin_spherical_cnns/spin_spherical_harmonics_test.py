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

"""Tests for spin_spherical_harmonics.py."""

import functools
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from spin_spherical_cnns import np_spin_spherical_harmonics
from spin_spherical_cnns import sphere_utils
from spin_spherical_cnns import spin_spherical_harmonics


TransformerModule = spin_spherical_harmonics.SpinSphericalFourierTransformer


@functools.lru_cache()
def _get_transformer():
  transformer = spin_spherical_harmonics.SpinSphericalFourierTransformer(
      resolutions=(8, 16), spins=(0, -1, 1, -2))
  variables = transformer.init(jax.random.PRNGKey(0))
  return transformer, variables


class SpinSphericalHarmonicsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict(resolution=8, spin=-1),
                            dict(resolution=16, spin=0),
                            dict(resolution=16, spin=-1),
                            dict(resolution=16, spin=-2))
  def test_swsft_forward_matches_np(self, resolution, spin):
    transformer, variables = _get_transformer()
    sphere = (jnp.linspace(-1, 1, resolution**2)
              .reshape((resolution, resolution)))
    coeffs_np = np_spin_spherical_harmonics.swsft_forward_naive(sphere, spin)
    coeffs_jax = transformer.apply(variables, sphere, spin,
                                   method=TransformerModule.swsft_forward)

    self.assertAllClose(
        coeffs_jax, spin_spherical_harmonics.coefficients_to_matrix(coeffs_np))

  @parameterized.parameters(dict(resolution=8, spin=-1),
                            dict(resolution=16, spin=0),
                            dict(resolution=16, spin=1))
  def test_swsft_forward_matches_with_symmetry(self, resolution, spin):
    """Check whether the versions with and without symmetry match."""
    transformer, variables = _get_transformer()
    sphere = (jnp.linspace(-1, 1, resolution**2)
              .reshape((resolution, resolution)))
    forward = transformer.apply(variables, sphere, spin,
                                method=TransformerModule.swsft_forward)
    forward_with_symmetry = transformer.apply(
        variables, sphere, spin,
        method=TransformerModule.swsft_forward_with_symmetry)

    self.assertAllClose(forward, forward_with_symmetry)

  def test_swsft_forward_validate_raises(self):
    """Check that swsft_forward() raises exception if constants are invalid."""
    transformer = spin_spherical_harmonics.SpinSphericalFourierTransformer(
        resolutions=(4, 8), spins=(0, 1))
    variables = transformer.init(jax.random.PRNGKey(0))
    # Wrong resolution, right spin:
    resolution = 16
    sphere = (jnp.linspace(-1, 1, resolution**2)
              .reshape((resolution, resolution)))

    def forward_fun(sphere, spin):
      return transformer.apply(variables, sphere, spin=spin,
                               method=transformer.swsft_forward)

    self.assertRaises(ValueError, forward_fun,
                      sphere, spin=0)

    # Right resolution, wrong spin:
    resolution = 8
    sphere = (jnp.linspace(-1, 1, resolution**2)
              .reshape((resolution, resolution)))
    self.assertRaises(ValueError, forward_fun,
                      sphere, spin=2)

  @parameterized.parameters(dict(num_coefficients=16, spin=-1),
                            dict(num_coefficients=16, spin=1),
                            dict(num_coefficients=64, spin=0),
                            dict(num_coefficients=64, spin=-2))
  def test_swsft_backward_matches_np(self, num_coefficients, spin):
    transformer, variables = _get_transformer()
    coeffs_np = (jnp.linspace(-1, 1, num_coefficients) +
                 1j*jnp.linspace(0, 1, num_coefficients))
    coeffs_jax = spin_spherical_harmonics.coefficients_to_matrix(coeffs_np)
    sphere_np = np_spin_spherical_harmonics.swsft_backward_naive(coeffs_np,
                                                                 spin)
    sphere_jax = transformer.apply(variables, coeffs_jax, spin,
                                   method=TransformerModule.swsft_backward)

    self.assertAllClose(sphere_np, sphere_jax)

  @parameterized.parameters(dict(num_coefficients=16, spin=1),
                            dict(num_coefficients=16, spin=0),
                            dict(num_coefficients=64, spin=-2))
  def test_swsft_backward_matches_with_symmetry(self, num_coefficients, spin):
    transformer, variables = _get_transformer()
    coeffs = (jnp.linspace(-1, 1, num_coefficients) +
              1j*jnp.linspace(0, 1, num_coefficients))
    coeffs = spin_spherical_harmonics.coefficients_to_matrix(coeffs)
    sphere = transformer.apply(variables, coeffs, spin,
                               method=TransformerModule.swsft_backward)

    with_symmetry = transformer.apply(
        variables, coeffs, spin,
        method=TransformerModule.swsft_backward_with_symmetry)

    self.assertAllClose(sphere, with_symmetry)

  def test_swsft_backward_validate_raises(self):
    """Check that swsft_backward() raises exception if constants are invalid."""
    transformer = spin_spherical_harmonics.SpinSphericalFourierTransformer(
        resolutions=(4, 8), spins=(0, 1))
    variables = transformer.init(jax.random.PRNGKey(0))
    n_coeffs = 64  # Corresponds to resolution == 16.
    # Wrong ell_max, right spin:
    coeffs_np = jnp.linspace(-1, 1, n_coeffs) + 1j*jnp.linspace(0, 1, n_coeffs)
    coeffs_jax = spin_spherical_harmonics.coefficients_to_matrix(coeffs_np)
    def backward_fun(coeffs, spin):
      return transformer.apply(variables, coeffs, spin,
                               method=TransformerModule.swsft_backward)
    self.assertRaises(ValueError, backward_fun,
                      coeffs_jax, spin=1)

    n_coeffs = 16  # Corresponds to resolution == 8.
    # Right ell_max, wrong spin:
    coeffs_np = jnp.linspace(-1, 1, n_coeffs) + 1j*jnp.linspace(0, 1, n_coeffs)
    coeffs_jax = spin_spherical_harmonics.coefficients_to_matrix(coeffs_np)
    self.assertRaises(ValueError, backward_fun,
                      coeffs_jax, spin=-1)

  def test_swsft_forward_spins_channels_matches_swsft_forward(self):
    transformer, variables = _get_transformer()
    resolution = 16
    n_channels = 2
    spins = (0, 1)
    shape = (resolution, resolution, len(spins), n_channels)
    sphere_set = jnp.linspace(-1, 1, np.prod(shape)).reshape(shape)
    coefficients = transformer.apply(
        variables, sphere_set, spins,
        method=TransformerModule.swsft_forward_spins_channels)
    for channel in range(n_channels):
      for spin in spins:
        # Slices must match swsft_forward().
        sliced = transformer.apply(
            variables, sphere_set[Ellipsis, spin, channel], spin,
            method=TransformerModule.swsft_forward)
        self.assertAllClose(coefficients[Ellipsis, spin, channel], sliced)

  def test_swsft_forward_spins_channels_ell_max(self):
    """When given `ell_max`, coefficients must match the proper slice."""
    transformer, variables = _get_transformer()
    resolution = 16
    ell_max = 3
    n_channels = 2
    spins = (0, 1)
    shape = (resolution, resolution, len(spins), n_channels)
    sphere_set = jnp.linspace(-1, 1, np.prod(shape)).reshape(shape)
    coefficients = transformer.apply(
        variables, sphere_set, spins, ell_max=ell_max,
        method=TransformerModule.swsft_forward_spins_channels)

    coefficients_full = transformer.apply(
        variables, sphere_set, spins,
        method=TransformerModule.swsft_forward_spins_channels)

    num_ell = ell_max + 1
    self.assertAllClose(coefficients,
                        coefficients_full[:num_ell, num_ell:-num_ell])

  def test_swsft_forward_spins_channels_matches_with_symmetry(self):
    transformer, variables = _get_transformer()
    resolution = 16
    n_channels = 2
    spins = (0, 1)
    shape = (resolution, resolution, len(spins), n_channels)
    sphere_set = jnp.linspace(-1, 1, np.prod(shape)).reshape(shape)
    coefficients = transformer.apply(
        variables, sphere_set, spins,
        method=TransformerModule.swsft_forward_spins_channels)

    with_symmetry = transformer.apply(
        variables, sphere_set, spins,
        method=TransformerModule.swsft_forward_spins_channels_with_symmetry)

    self.assertAllClose(coefficients, with_symmetry)

  def test_swsft_backward_spins_channels_matches_swsft_backward(self):
    transformer, variables = _get_transformer()
    ell_max = 7
    n_channels = 2
    spins = (0, 1)
    shape = (ell_max+1, 2*ell_max+1, n_channels, len(spins))
    coefficients = jnp.linspace(-1, 1, np.prod(shape)).reshape(shape)
    sphere = transformer.apply(
        variables, coefficients, spins,
        method=TransformerModule.swsft_backward_spins_channels)
    for channel in range(n_channels):
      for spin in spins:
        # Slices must match swsft_backward().
        sliced = transformer.apply(
            variables, coefficients[Ellipsis, spin, channel], spin,
            method=TransformerModule.swsft_backward)

        self.assertAllClose(sphere[Ellipsis, spin, channel], sliced)

  @parameterized.parameters(2, 3)
  def test_coefficients_vector_to_matrix_has_right_shape(self, num_ell):
    num_coefficients = num_ell**2
    coefficients = jnp.ones(num_coefficients)
    matrix = spin_spherical_harmonics.coefficients_to_matrix(coefficients)
    target_shape = (num_ell, 2*num_ell - 1)

    self.assertEqual(matrix.shape, target_shape)

  @parameterized.parameters(0, 1, 2, 4, 7)
  def test_SpinSphericalFourierTransformer_wigner_delta_matches_np(self, ell):
    transformer, variables = _get_transformer()
    wigner_delta = transformer.apply(
        variables, ell, include_negative_m=True,
        method=TransformerModule._slice_wigner_deltas)[ell]
    wigner_delta_np = sphere_utils.compute_wigner_delta(ell)
    self.assertAllClose(wigner_delta, wigner_delta_np)

  @parameterized.parameters(dict(spin=0, ell=0),
                            dict(spin=0, ell=1),
                            dict(spin=1, ell=2),
                            dict(spin=-2, ell=3),
                            dict(spin=-1, ell=7))
  def test_SpinSphericalFourierTransformer_forward_constants_matches_np(self,
                                                                        spin,
                                                                        ell):
    _, variables = _get_transformer()
    swsft_forward_constants = variables['constants']['swsft_forward_constants']
    ell_max = swsft_forward_constants[str(spin)].shape[0] - 1
    slice_ell = slice(ell_max - ell, ell_max + ell + 1)
    constants = swsft_forward_constants[str(spin)][ell, slice_ell]
    constants_np = sphere_utils.swsft_forward_constant(spin, ell,
                                                       jnp.arange(-ell, ell+1))
    self.assertAllClose(constants, constants_np)

  @parameterized.parameters(4, 8, 9)
  def test_fourier_transform_2d(self, dimension):
    """Ensures that our DFT implementation matches the FFT."""
    x = jnp.linspace(0, 1, dimension**2).reshape(dimension, dimension)
    fft = jnp.fft.fft2(x)
    ours = spin_spherical_harmonics._fourier_transform_2d(x)
    self.assertAllClose(fft, ours, atol=5e-6)


if __name__ == '__main__':
  tf.test.main()
