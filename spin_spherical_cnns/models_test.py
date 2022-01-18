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

"""Tests for spin_spherical_cnns.models."""

import functools
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from spin_spherical_cnns import models
from spin_spherical_cnns import spin_spherical_harmonics
from spin_spherical_cnns import test_utils


# Pseudo-random number generator keys to deterministically initialize
# parameters. The initialization could cause flakiness in the unlikely event
# that JAX changes its pseudo-random algorithm.
_JAX_RANDOM_KEY = np.array([0, 0], dtype=np.uint32)


@functools.lru_cache()
def _get_transformer():
  return spin_spherical_harmonics.SpinSphericalFourierTransformer(
      resolutions=[4, 8, 16], spins=[0, -1, 1, 2])


def _mean_absolute_error(x, y):
  return jnp.abs(x - y).mean()


def _normalized_mean_absolute_error(x, y):
  return _mean_absolute_error(x, y) / jnp.abs(x).mean()


class SpinSphericalBlockTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(1, 2)
  def test_downsampling_factor_output_shape(self, downsampling_factor):
    transformer = _get_transformer()
    num_channels = 2
    spins_in = [0]
    spins_out = [0, 1]
    batch_size = 2
    resolution = 8
    model = models.SpinSphericalBlock(num_channels=num_channels,
                                      spins_in=spins_in,
                                      spins_out=spins_out,
                                      downsampling_factor=downsampling_factor,
                                      axis_name=None,
                                      transformer=transformer)

    shape = [batch_size, resolution, resolution, len(spins_in), num_channels]
    inputs = jnp.ones(shape)
    params = model.init(_JAX_RANDOM_KEY, inputs, train=False)
    outputs = model.apply(params, inputs, train=False)
    shape_out = (batch_size,
                 resolution // downsampling_factor,
                 resolution // downsampling_factor,
                 len(spins_out), num_channels)

    self.assertEqual(outputs.shape, shape_out)

  @parameterized.parameters(dict(shift=1, train=False),
                            dict(shift=5, train=True, num_filter_params=2),
                            dict(shift=2, train=True, downsampling_factor=2))
  def test_azimuthal_equivariance(self, shift, train,
                                  downsampling_factor=1,
                                  num_filter_params=None):
    resolution = 8
    transformer = _get_transformer()
    spins = (0, 1, 2)
    shape = (2, resolution, resolution, len(spins), 2)

    sphere, _ = test_utils.get_spin_spherical(transformer, shape, spins)
    rotated_sphere = jnp.roll(sphere, shift, axis=2)

    model = models.SpinSphericalBlock(num_channels=2,
                                      spins_in=spins,
                                      spins_out=spins,
                                      downsampling_factor=downsampling_factor,
                                      num_filter_params=num_filter_params,
                                      axis_name=None,
                                      transformer=transformer)
    params = model.init(_JAX_RANDOM_KEY, sphere, train=False)

    # Add negative bias so that the magnitude nonlinearity is active.
    params = params.unfreeze()
    for key, value in params['params']['batch_norm_nonlin'].items():
      if 'magnitude_nonlin' in key:
        value['bias'] -= 0.1

    output, _ = model.apply(params, sphere, train=train,
                            mutable=['batch_stats'])
    rotated_output, _ = model.apply(params, rotated_sphere, train=train,
                                    mutable=['batch_stats'])
    shifted_output = jnp.roll(output, shift // downsampling_factor, axis=2)

    self.assertAllClose(rotated_output, shifted_output, atol=1e-6)

  @parameterized.parameters(False, True)
  def test_equivariance(self, train):
    resolution = 8
    spins = (0, -1, 2)
    transformer = _get_transformer()
    model = models.SpinSphericalBlock(num_channels=2,
                                      spins_in=spins,
                                      spins_out=spins,
                                      downsampling_factor=1,
                                      axis_name=None,
                                      transformer=transformer)

    init_args = dict(train=False)
    apply_args = dict(train=train, mutable=['batch_stats'])

    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer, model, resolution, spins,
        init_args=init_args, apply_args=apply_args)

    # Tolerance needs to be high here due to approximate equivariance. We check
    # the mean absolute error.
    self.assertLess(
        _normalized_mean_absolute_error(coefficients_1, coefficients_2),
        0.1)


class SpinSphericalClassifierTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict(num_classes=3, num_filter_params=None),
                            dict(num_classes=5, num_filter_params=[3, 2]))
  def test_shape(self, num_classes, num_filter_params):
    transformer = _get_transformer()
    resolutions = [8, 4]
    spins = [[0], [0, 1]]
    channels = [1, 2]
    batch_size = 2
    model = models.SpinSphericalClassifier(num_classes,
                                           resolutions,
                                           spins,
                                           channels,
                                           num_filter_params=num_filter_params,
                                           axis_name=None,
                                           input_transformer=transformer)
    resolution = resolutions[0]
    shape = [batch_size, resolution, resolution, len(spins[0]), channels[0]]
    inputs = jnp.ones(shape)
    params = model.init(_JAX_RANDOM_KEY, inputs, train=False)
    outputs = model.apply(params, inputs, train=False)

    self.assertEqual(outputs.shape, (batch_size, num_classes))

  @parameterized.parameters(2, 4)
  def test_azimuthal_invariance(self, shift):
    # Make a simple two-layer classifier with pooling for testing.
    resolutions = [8, 4]
    transformer = _get_transformer()
    spins = [[0, -1], [0, 1, 2]]
    channels = [2, 3]
    shape = [2, resolutions[0], resolutions[0], len(spins[0]), channels[0]]
    sphere, _ = test_utils.get_spin_spherical(transformer, shape, spins[0])
    rotated_sphere = jnp.roll(sphere, shift, axis=2)

    model = models.SpinSphericalClassifier(num_classes=5,
                                           resolutions=resolutions,
                                           spins=spins,
                                           widths=channels,
                                           axis_name=None,
                                           input_transformer=transformer)
    params = model.init(_JAX_RANDOM_KEY, sphere, train=False)

    output, _ = model.apply(params, sphere, train=True,
                            mutable=['batch_stats'])
    rotated_output, _ = model.apply(params, rotated_sphere, train=True,
                                    mutable=['batch_stats'])

    # The classifier should be rotation-invariant.
    self.assertAllClose(rotated_output, output, atol=1e-6)

  def test_invariance(self):
    # Make a simple two-layer classifier with pooling for testing.
    resolutions = [16, 8]
    transformer = _get_transformer()
    spins = [[0, -1], [0, 1, 2]]
    channels = [2, 3]
    shape = [2, resolutions[0], resolutions[0], len(spins[0]), channels[0]]
    pair = test_utils.get_rotated_pair(transformer, shape, spins[0],
                                       alpha=1.0, beta=2.0, gamma=3.0)

    model = models.SpinSphericalClassifier(num_classes=5,
                                           resolutions=resolutions,
                                           spins=spins,
                                           widths=channels,
                                           axis_name=None,
                                           input_transformer=transformer)

    params = model.init(_JAX_RANDOM_KEY, pair.sphere, train=False)

    output, _ = model.apply(params, pair.sphere, train=True,
                            mutable=['batch_stats'])
    rotated_output, _ = model.apply(params, pair.rotated_sphere, train=True,
                                    mutable=['batch_stats'])

    # The classifier should be rotation-invariant. Here the tolerance is high
    # because the local pooling introduces equivariance errors.
    self.assertAllClose(rotated_output, output, atol=1e-1)
    self.assertLess(_normalized_mean_absolute_error(output, rotated_output),
                    0.1)


class CNNClassifierTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(3, 5)
  def test_shape(self, num_classes):
    resolutions = [8, 4]
    channels = [1, 2]
    batch_size = 2
    model = models.CNNClassifier(num_classes,
                                 resolutions,
                                 channels,
                                 axis_name=None)
    resolution = resolutions[0]
    shape = [batch_size, resolution, resolution, 1, channels[0]]
    inputs = jnp.linspace(-1, 1, np.prod(shape)).reshape(shape)
    params = model.init(_JAX_RANDOM_KEY, inputs, train=False)
    outputs = model.apply(params, inputs, train=False)

    self.assertEqual(outputs.shape, (batch_size, num_classes))

if __name__ == '__main__':
  tf.test.main()
