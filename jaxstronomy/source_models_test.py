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

"""Tests for source_models.py.

Expected values are drawn from lenstronomy:
https://github.com/lenstronomy/lenstronomy.
"""

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxstronomy import source_models


def _prepare_x_y():
  rng = jax.random.PRNGKey(0)
  rng_x, rng_y = jax.random.split(rng)
  x = jax.random.normal(rng_x, shape=(3,))
  y = jax.random.normal(rng_y, shape=(3,))
  return x, y


def _prepare_image():
  rng = jax.random.PRNGKey(0)
  return jax.random.normal(rng, shape=(62, 64))


def _prepare__brightness_expected(sersic_radius, n_sersic):
  if sersic_radius == 2.2 and n_sersic == 1.0:
    return jnp.array([2.227407, 1.9476248, 2.9091272])
  elif sersic_radius == 2.3 and n_sersic == 2.0:
    return jnp.array([2.9430487, 2.4276965, 4.5396194])
  elif sersic_radius == 3.5 and n_sersic == 3.0:
    return jnp.array([5.8227673, 4.809319, 9.121668])
  else:
    raise ValueError(
        f'sersic_radius={sersic_radius} n_sersic={n_sersic} are not a supported'
        ' parameter combination.')


def _prepare_interpol_parameters():
  return {
      'image': _prepare_image(),
      'amp': 1.4,
      'center_x': 0.2,
      'center_y': -0.2,
      'angle': -np.pi / 6,
      'scale': 1.5
  }


def _prepare_sersic_elliptic_parameters():
  return {
      'amp': 1.4,
      'sersic_radius': 1.0,
      'n_sersic': 2.0,
      'axis_ratio': 0.2,
      'angle': np.pi / 6,
      'center_x': 0.2,
      'center_y': -0.2,
  }


class AllTest(absltest.TestCase):
  """Runs tests of __all__ property of source_models module."""

  def test_all(self):
    all_present = sorted(source_models.__all__)
    all_required = []
    for name, value in inspect.getmembers(source_models):
      if inspect.isclass(value):
        all_required.append(name)

    self.assertListEqual(all_present, sorted(all_required))


class InterpolTest(chex.TestCase, parameterized.TestCase):
  """Runs tests of Interpol brightness functions."""

  def test_parameters(self):
    annotated_parameters = sorted(source_models.Interpol.parameters)
    correct_parameters = sorted(_prepare_interpol_parameters().keys())
    self.assertListEqual(annotated_parameters, correct_parameters)

  @chex.all_variants
  def test_function(self):
    x, y = _prepare_x_y()
    parameters = _prepare_interpol_parameters()
    expected = jnp.array([1.71894064, 0.4886814, 2.13953358])

    function = self.variant(source_models.Interpol.function)

    np.testing.assert_allclose(
        function(x, y, **parameters), expected, rtol=1e-5)

  @chex.all_variants
  def test__image_interpolation(self):
    x, y = _prepare_x_y()
    image = _prepare_image()
    expected = jnp.array([0.31192497, 0.64870896, 1.48134785])

    image_interpolation = self.variant(
        source_models.Interpol._image_interpolation)

    np.testing.assert_allclose(
        image_interpolation(x, y, image), expected, rtol=1e-5)

  @chex.all_variants
  def test__coord_to_image_pixels(self):
    x, y = _prepare_x_y()
    parameters = _prepare_interpol_parameters()
    expected = jnp.array([[-0.48121729, 0.51891287, -0.29162392],
                          [0.75206837, -0.48633681, -0.46977397]])

    coord_to_image_pixels = self.variant(
        source_models.Interpol._coord_to_image_pixels)

    np.testing.assert_allclose(
        jnp.asarray(
            coord_to_image_pixels(x, y, parameters['center_x'],
                                  parameters['center_y'], parameters['angle'],
                                  parameters['scale'])),
        expected,
        rtol=1e-6)


class SersicEllipticTest(chex.TestCase, parameterized.TestCase):
  """Runs tests of elliptical Sersic brightness functions."""

  def test_parameters(self):
    annotated_parameters = sorted(source_models.SersicElliptic.parameters)
    correct_parameters = sorted(_prepare_sersic_elliptic_parameters().keys())
    self.assertListEqual(annotated_parameters, correct_parameters)

  @chex.all_variants
  def test_function(self):
    x, y = _prepare_x_y()
    parameters = _prepare_sersic_elliptic_parameters()
    expected = jnp.array([0.13602875, 0.20377299, 5.802394])

    function = self.variant(source_models.SersicElliptic.function)

    np.testing.assert_allclose(
        function(x, y, **parameters), expected, rtol=1e-5)

  @chex.all_variants
  def test__get_distance_from_center(self):
    x, y = _prepare_x_y()
    parameters = _prepare_sersic_elliptic_parameters()
    expected = jnp.array([2.6733015, 2.3254495, 0.37543342])

    get_distance_from_center = self.variant(
        source_models.SersicElliptic._get_distance_from_center)

    np.testing.assert_allclose(
        get_distance_from_center(x, y, parameters['axis_ratio'],
                                 parameters['angle'], parameters['center_x'],
                                 parameters['center_y']),
        expected,
        rtol=1e-5)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_sr_{sr}_ns_{ns}', sr, ns)
      for sr, ns in zip([2.2, 2.3, 3.5], [1.0, 2.0, 3.0])
  ])
  def test__brightness(self, sersic_radius, n_sersic):
    x, y = _prepare_x_y()
    radius = jnp.sqrt(x**2 + y**2)
    expected = _prepare__brightness_expected(sersic_radius, n_sersic)

    brightness = self.variant(source_models.SersicElliptic._brightness)

    np.testing.assert_allclose(
        brightness(radius, sersic_radius, n_sersic), expected, rtol=1e-5)

  @chex.all_variants
  @parameterized.named_parameters([
      (f'_n_s_{n_sersic}', n_sersic, expected)
      for n_sersic, expected in zip([1., 2., 3.], [1.6721, 3.6713, 5.6705])
  ])
  def test__b_n(self, n_sersic, expected):
    b_n = self.variant(source_models.SersicElliptic._b_n)

    np.testing.assert_allclose(b_n(n_sersic), expected, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
