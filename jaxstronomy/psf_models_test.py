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

"""Tests for psf_models.py."""

import inspect

from absl.testing import absltest
from absl.testing import parameterized
import chex
import dm_pix
import jax
import jax.numpy as jnp
import numpy as np
from scipy import signal
from jaxstronomy import psf_models


def _prepare_image():
  rng = jax.random.PRNGKey(0)
  return jax.random.normal(rng, shape=(32, 32))


def _prepare_gaussian_parameters():
  return {'fwhm': 0.03, 'pixel_width': 0.04}


def _prepare_pixel_parameters():
  x = jnp.arange(-2, 2, 0.05)
  kernel = jnp.outer(jnp.exp(-x**2), jnp.exp(-x**2))
  return {'kernel_point_source': kernel}


class AllTest(absltest.TestCase):
  """Runs tests of __all__ property of psf_models module."""

  def test_all(self):
    all_present = sorted(psf_models.__all__)
    all_required = []
    for name, value in inspect.getmembers(psf_models):
      if inspect.isclass(value):
        all_required.append(name)

    self.assertListEqual(all_present, sorted(all_required))


class GaussianTest(chex.TestCase, parameterized.TestCase):
  """Runs tests of Gaussian derivative functions."""

  def test_parameters(self):
    annotated_parameters = sorted(psf_models.Gaussian.parameters)
    correct_parameters = sorted(_prepare_gaussian_parameters().keys())
    self.assertListEqual(annotated_parameters, correct_parameters)

  @chex.all_variants
  def test_convolve(self):
    image = _prepare_image()
    parameters = _prepare_gaussian_parameters()

    convolve = self.variant(psf_models.Gaussian.convolve)

    # Pulled from lenstronomy
    sigma_expected = 0.3184956751

    # As an additional consistency check, use a different channel axis for test
    # call to dm_pix.gaussian_blur.
    np.testing.assert_allclose(
        convolve(image, parameters),
        dm_pix.gaussian_blur(
            jnp.expand_dims(image, axis=0),
            sigma_expected,
            kernel_size=30,
            channel_axis=0)[0],
        rtol=1e-5)

  class PixelTest(chex.TestCase, parameterized.TestCase):
    """Runs tests of Pixel derivative functions."""

    def test_parameters(self):
      annotated_parameters = sorted(psf_models.Pixel.parameters)
      correct_parameters = sorted(_prepare_pixel_parameters().keys())
      self.assertListEqual(annotated_parameters, correct_parameters)

    @chex.all_variants
    def test_convolve(self):
      image = _prepare_image()
      parameters = _prepare_pixel_parameters()

      convolve = self.variant(psf_models.Pixel.convolve)

      # Not much to test here other than parameters being passed through
      # correctly / matching non-jax scipy.
      np.testing.assert_allclose(
          convolve(image, parameters),
          signal.convolve(
              image,
              parameters['kernel_point_source'] /
              jnp.sum(parameters['kernel_point_source']),
              mode='same'),
          rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
