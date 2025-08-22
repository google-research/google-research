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

"""Tests for utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxstronomy import lens_models
from jaxstronomy import utils

COSMOLOGY_DICT = {
    'omega_m_zero': 0.30966,
    'omega_de_zero': 0.688846306,
    'omega_rad_zero': 0.0014937,
    'hubble_constant': 67.66,
}


def _prepare_x_y_angular():
  rng = jax.random.PRNGKey(3)
  rng_x, rng_y = jax.random.split(rng)
  x = jax.random.normal(rng_x, shape=(3,))
  y = jax.random.normal(rng_y, shape=(3,))
  return x, y


def _prepare_kwargs_detector():
  return {'n_x': 2, 'n_y': 2, 'pixel_width': 0.04, 'supersampling_factor': 2}


class UtilsTest(chex.TestCase, parameterized.TestCase):
  """Runs tests of utility functions."""

  def test_coordinates_evaluate(self):
    parameters = _prepare_kwargs_detector()
    expected_x = jnp.array([
        -0.03, -0.01, 0.01, 0.03, -0.03, -0.01, 0.01, 0.03, -0.03, -0.01, 0.01,
        0.03, -0.03, -0.01, 0.01, 0.03
    ])
    expected_y = jnp.array([
        -0.03, -0.03, -0.03, -0.03, -0.01, -0.01, -0.01, -0.01, 0.01, 0.01,
        0.01, 0.01, 0.03, 0.03, 0.03, 0.03
    ])

    np.testing.assert_allclose(
        jnp.array(utils.coordinates_evaluate(**parameters)),
        jnp.array([expected_x, expected_y]))

  @chex.all_variants
  def test_unpack_parameters_xy(self):
    x, y = _prepare_x_y_angular()
    kwargs_lens = {
        'alpha_rs': 1.0,
        'scale_radius': 1.0,
        'center_x': 0.0,
        'center_y': 0.0,
        'fake_param': 19.2
    }
    expected = jnp.array([[-0.90657, -0.29612964, 0.22304466],
                          [0.44380534, -0.9504099, 0.97678715]])

    unpack_parameters_derivatives = self.variant(
        utils.unpack_parameters_xy(lens_models.NFW.derivatives,
                                   lens_models.NFW.parameters))

    np.testing.assert_allclose(
        unpack_parameters_derivatives(x, y, kwargs_lens), expected, rtol=1e-5)

  def test_downsampling(self):
    downsample = utils.downsample

    image = jnp.ones((12, 12))
    np.testing.assert_allclose(downsample(image, 3), jnp.ones((4, 4)))
    np.testing.assert_allclose(downsample(image, 4), jnp.ones((3, 3)))

    image = jax.random.normal(jax.random.PRNGKey(0), shape=(4, 4))
    expected = jnp.array([[0.37571156, 0.7770451], [-0.67193794, 0.014301866]])
    np.testing.assert_allclose(downsample(image, 2), expected)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_mag_{mag}_mzp_{mzp}', mag, mzp, expected) for mag, mzp, expected in
      zip([20.35, -2, 15], [18.0, 5.6, 8.8],
          [0.11481536214968811, 1096.4781961431852, 0.0033113112148259144])
  ])
  def test_magnitude_to_cps(self, mag, mzp, expected):
    magnitude_to_cps = self.variant(utils.magnitude_to_cps)

    self.assertAlmostEqual(magnitude_to_cps(mag, mzp), expected, places=5)


if __name__ == '__main__':
  absltest.main()
