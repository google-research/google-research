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

"""Tests for spin_spherical_cnns.layers."""

import functools
from absl.testing import parameterized
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from spin_spherical_cnns import layers
from spin_spherical_cnns import sphere_utils
from spin_spherical_cnns import spin_spherical_harmonics
from spin_spherical_cnns import test_utils


# Pseudo-random number generator keys to deterministically initialize
# parameters. The initialization could cause flakiness in the unlikely event
# that JAX changes its pseudo-random algorithm.
_JAX_RANDOM_KEY = np.array([0, 0], dtype=np.uint32)


@functools.lru_cache()
def _get_transformer():
  return spin_spherical_harmonics.SpinSphericalFourierTransformer(
      resolutions=[4, 8, 16], spins=(0, -1, 1, 2))


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict(resolution=8,
                                 spins_in=[0], spins_out=[0, 1, 2]),
                            dict(resolution=8, spins_in=[0], spins_out=[0]),
                            dict(resolution=8, spins_in=[1], spins_out=[1]),
                            dict(resolution=8, spins_in=[0, 1], spins_out=[0]),
                            dict(resolution=8, spins_in=[0, 1], spins_out=[1]),
                            dict(resolution=16,
                                 spins_in=[0, 1], spins_out=[0, 1]),
                            dict(resolution=16, spins_in=[0], spins_out=[0, 1]))
  def test_swsconv_spatial_spectral_is_equivariant(self,
                                                   resolution,
                                                   spins_in,
                                                   spins_out):
    """Tests the SO(3)-equivariance of _swsconv_spatial_spectral()."""
    transformer = _get_transformer()
    num_channels_in, num_channels_out = 2, 3
    # Euler angles.
    alpha, beta, gamma = 1.0, 2.0, 3.0
    shape = (1, resolution, resolution, len(spins_in), num_channels_in)
    pair = test_utils.get_rotated_pair(transformer,
                                       shape=shape,
                                       spins=spins_in,
                                       alpha=alpha,
                                       beta=beta,
                                       gamma=gamma)
    # Get rid of the batch dimension.
    sphere = pair.sphere[0]
    rotated_sphere = pair.rotated_sphere[0]

    # Filter is defined by its spectral coefficients.
    ell_max = resolution // 2 - 1
    shape = [ell_max+1,
             len(spins_in), len(spins_out),
             num_channels_in, num_channels_out]
    # Make more arbitrary reproducible complex inputs.
    filter_coefficients = jnp.linspace(-0.5 + 0.2j, 0.2,
                                       np.prod(shape)).reshape(shape)

    sphere_out = layers._swsconv_spatial_spectral(transformer,
                                                  sphere,
                                                  filter_coefficients,
                                                  spins_in, spins_out)

    rotated_sphere_out = layers._swsconv_spatial_spectral(transformer,
                                                          rotated_sphere,
                                                          filter_coefficients,
                                                          spins_in, spins_out)

    # Now since the convolution is SO(3)-equivariant, the same rotation that
    # relates the inputs must relate the outputs. We apply it spectrally.
    coefficients_out = transformer.swsft_forward_spins_channels(sphere_out,
                                                                spins_out)

    # This is R(f) * g (in the spectral domain).
    rotated_coefficients_out_1 = transformer.swsft_forward_spins_channels(
        rotated_sphere_out, spins_out)

    # And this is R(f * g) (in the spectral domain).
    rotated_coefficients_out_2 = test_utils.rotate_coefficients(
        coefficients_out, alpha, beta, gamma)

    # There is some loss of precision on the Wigner-D computation for rotating
    # the coefficients, hence we use a slighly higher tolerance.
    self.assertAllClose(rotated_coefficients_out_1, rotated_coefficients_out_2,
                        atol=1e-5)


class SpinSphericalConvolutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dict(batch_size=2, resolution=8,
           spins_in=[0], spins_out=[0, 1, 2],
           n_channels_in=1, n_channels_out=3,
           num_filter_params=None),
      dict(batch_size=3, resolution=8,
           spins_in=[0, 1, 2], spins_out=[0],
           n_channels_in=3, n_channels_out=1,
           num_filter_params=2),
      dict(batch_size=2, resolution=16,
           spins_in=[0], spins_out=[0, 1],
           n_channels_in=2, n_channels_out=3,
           num_filter_params=4))
  def test_shape(self,
                 batch_size,
                 resolution,
                 spins_in, spins_out,
                 n_channels_in, n_channels_out,
                 num_filter_params):
    """Checks that SpinSphericalConvolution outputs the right shape."""
    transformer = _get_transformer()
    shape = (batch_size, resolution, resolution, len(spins_in), n_channels_in)
    inputs = (jnp.linspace(-0.5, 0.7 + 0.5j, np.prod(shape))
              .reshape(shape))

    model = layers.SpinSphericalConvolution(transformer=transformer,
                                            spins_in=spins_in,
                                            spins_out=spins_out,
                                            features=n_channels_out,
                                            num_filter_params=num_filter_params)
    params = model.init(_JAX_RANDOM_KEY, inputs)
    out = model.apply(params, inputs)

    self.assertEqual(out.shape, (batch_size, resolution, resolution,
                                 len(spins_out), n_channels_out))

  def test_equivariance(self):
    resolution = 8
    spins = (0, -1, 2)
    transformer = _get_transformer()
    model = layers.SpinSphericalConvolution(transformer=transformer,
                                            spins_in=spins,
                                            spins_out=spins,
                                            features=2)

    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer, model, resolution, spins)

    self.assertAllClose(coefficients_1, coefficients_2, atol=1e-6)

  @parameterized.parameters(1, 2, 4)
  def test_localized_kernel_shape(self, num_filter_params):
    resolution = 16
    spins_in = (0, 1)
    spins_out = (0, 1, 2)
    num_channels_in = 3
    num_channels_out = 4
    transformer = _get_transformer()
    inputs = jnp.ones((2, resolution, resolution,
                       len(spins_in), num_channels_in))

    model = layers.SpinSphericalConvolution(
        transformer=transformer,
        spins_in=spins_in,
        spins_out=spins_out,
        features=num_channels_out,
        num_filter_params=num_filter_params)
    params = model.init(_JAX_RANDOM_KEY, inputs)

    expected_shape = (len(spins_in), len(spins_out),
                      num_channels_in, num_channels_out,
                      num_filter_params)
    self.assertEqual(params["params"]["kernel"].shape, expected_shape)

  def test_localized_with_all_params_match_nonlocalized(self):
    """Check that filters with ell_max+1 params equal nonlocalized."""
    # We use fewer params to enforce a smooth spectrum for localized
    # filters. When the number of params is maximum (ell_max+1), no smoothness
    # is enforced and the filters match their nonlocalized counterparts.
    resolution = 16
    spins_in = (0, 1)
    spins_out = (0, 1, 2)
    n_channels_out = 4
    transformer = _get_transformer()
    inputs = jnp.ones((2, resolution, resolution, len(spins_in), 3))

    model = layers.SpinSphericalConvolution(transformer=transformer,
                                            spins_in=spins_in,
                                            spins_out=spins_out,
                                            features=n_channels_out)
    params = model.init(_JAX_RANDOM_KEY, inputs)

    # The parameters for localized filters are transposed for performance
    # reasons. This custom initializer undoes the transposing so the init is the
    # same between localized and nonlocalized.
    def _transposed_initializer(key, shape, dtype=jnp.complex64):
      del dtype
      shape = [shape[-1], *shape[:-1]]
      weights = layers.default_initializer(key, shape)
      return weights.transpose(1, 2, 3, 4, 0)

    ell_max = resolution // 2 - 1
    model_localized = layers.SpinSphericalConvolution(
        transformer=transformer,
        spins_in=spins_in,
        spins_out=spins_out,
        features=n_channels_out,
        num_filter_params=ell_max + 1,
        initializer=_transposed_initializer)
    params_localized = model_localized.init(_JAX_RANDOM_KEY, inputs)

    self.assertAllClose(params["params"]["kernel"].transpose(1, 2, 3, 4, 0),
                        params_localized["params"]["kernel"])


# Default initialization is zero. We include bias so that some entries are
# actually rectified.
def _magnitude_nonlinearity_nonzero_initializer(*args, **kwargs):
  return -0.1 * nn.initializers.ones(*args, **kwargs)


class MagnitudeNonlinearityTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([(1, 8, 8, 3, 4)],
                            [(2, 4, 4, 5, 6)])
  def test_magnitude_thresholding(self, input_shape):
    small_row = 3
    inputs = jnp.ones(input_shape)
    inputs = inputs.at[:, small_row].set(0.1)

    model = layers.MagnitudeNonlinearity()
    params = model.init(_JAX_RANDOM_KEY, inputs)

    # With zero bias output must match input.
    bias = params["params"]["bias"].at[:].set(0.0)
    inputs_unchanged = model.apply(params, inputs)
    self.assertAllClose(inputs, inputs_unchanged)

    # We run again with bias = -0.2; now out[small_row] must be zero.
    bias_value = -0.2
    bias = params["params"]["bias"].at[:].set(bias_value)
    params_changed = flax.core.FrozenDict({"params": {"bias": bias}})
    inputs_changed = model.apply(params_changed, inputs)
    self.assertAllEqual(inputs_changed[:, small_row],
                        np.zeros_like(inputs[:, small_row]))
    # All other rows have the bias added.
    self.assertAllClose(inputs_changed[:, :small_row],
                        inputs[:, :small_row] + bias_value)
    self.assertAllClose(inputs_changed[:, small_row+1:],
                        inputs[:, small_row+1:] + bias_value)

  @parameterized.parameters(1, 5)
  def test_azimuthal_equivariance(self, shift):
    resolution = 8
    spins = (0, -1, 2)
    transformer = _get_transformer()

    model = layers.MagnitudeNonlinearity(
        bias_initializer=_magnitude_nonlinearity_nonzero_initializer)

    output_1, output_2 = test_utils.apply_model_to_azimuthally_rotated_pairs(
        transformer, model, resolution, spins, shift)

    self.assertAllClose(output_1, output_2)

  def test_equivariance(self):
    resolution = 16
    spins = (0, -1, 2)
    transformer = _get_transformer()

    model = layers.MagnitudeNonlinearity(
        bias_initializer=_magnitude_nonlinearity_nonzero_initializer)
    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer, model, resolution, spins)
    # Tolerance needs to be high here due to approximate equivariance. We also
    # check the mean absolute error.
    self.assertAllClose(coefficients_1, coefficients_2, atol=1e-1)
    self.assertLess(abs(coefficients_1 - coefficients_2).mean(), 5e-3)


def _evaluate_magnitudenonlinearity_versions(spins):
  """Evaluates MagnitudeNonlinearity and MagnitudeNonlinearityLeakyRelu."""
  transformer = _get_transformer()
  inputs, _ = test_utils.get_spin_spherical(transformer,
                                            shape=(2, 8, 8, len(spins), 2),
                                            spins=spins)
  model = layers.MagnitudeNonlinearity(
      bias_initializer=_magnitude_nonlinearity_nonzero_initializer)
  params = model.init(_JAX_RANDOM_KEY, inputs)
  outputs = model.apply(params, inputs)

  model_relu = layers.MagnitudeNonlinearityLeakyRelu(
      spins=spins,
      bias_initializer=_magnitude_nonlinearity_nonzero_initializer)
  params_relu = model_relu.init(_JAX_RANDOM_KEY, inputs)
  outputs_relu = model_relu.apply(params_relu, inputs)

  return inputs, outputs, outputs_relu


class MagnitudeNonlinearityLeakyReluTest(tf.test.TestCase):

  def test_spin0_matches_relu(self):
    """Zero spin matches real leaky_relu, others match MagnitudeNonlinearity."""
    spins = [0, -1, 2]
    inputs, outputs, outputs_relu = _evaluate_magnitudenonlinearity_versions(
        spins)

    self.assertAllEqual(outputs[Ellipsis, 1:, :], outputs_relu[Ellipsis, 1:, :])
    self.assertAllEqual(outputs_relu[Ellipsis, 0, :],
                        nn.leaky_relu(inputs[Ellipsis, 0, :].real))


class SphericalPoolingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(4, 8)
  def test_constant_latitude_values(self, resolution):
    """Average for constant-latitude values tilts towards largest area."""
    inputs = jnp.zeros([2, resolution, resolution, 1, 1])
    first_latitude = 1
    second_latitude = 2
    inputs = inputs.at[:, 0].set(first_latitude)
    inputs = inputs.at[:, 1].set(second_latitude)

    model = layers.SphericalPooling(stride=2)
    params = model.init(_JAX_RANDOM_KEY, inputs)

    pooled = model.apply(params, inputs)
    # Since both the area and the value in the second band are larger than the
    # first, the output values should be larger than the unweighted average.
    unweighted = (first_latitude + second_latitude) / 2
    self.assertAllGreater(pooled[:, 0], unweighted)

    # Now we make the second value smaller, so average must be smaller than the
    # unweighted.
    second_latitude = 0
    inputs = inputs.at[:, 1].set(second_latitude)
    unweighted = (first_latitude + second_latitude) / 2
    pooled = model.apply(params, inputs)
    self.assertAllLess(pooled[:, 0], unweighted)

  @parameterized.parameters(dict(shift=2, stride=2),
                            dict(shift=4, stride=2),
                            dict(shift=4, stride=4))
  def test_azimuthal_equivariance(self, shift, stride):
    resolution = 16
    spins = (0, -1, 2)
    transformer = _get_transformer()
    model = layers.SphericalPooling(stride=stride)

    output_1, output_2 = test_utils.apply_model_to_azimuthally_rotated_pairs(
        transformer, model, resolution, spins, shift)

    self.assertAllClose(output_1, output_2)

  @parameterized.parameters(8, 16)
  def test_SphericalPooling_matches_spin_spherical_mean(self, resolution):
    """SphericalPooling with max stride must match spin_spherical_mean."""
    shape = [2, resolution, resolution, 3, 4]
    spins = [0, -1, 2]
    inputs, _ = test_utils.get_spin_spherical(_get_transformer(), shape, spins)
    spherical_mean = sphere_utils.spin_spherical_mean(inputs)

    model = layers.SphericalPooling(stride=resolution)
    params = model.init(_JAX_RANDOM_KEY, inputs)
    pooled = model.apply(params, inputs)

    # Tolerance here is higher because of slightly different quadratures.
    self.assertAllClose(spherical_mean, pooled[:, 0, 0], atol=1e-3)


def _batched_spherical_variance(inputs):
  """Computes variances over the sphere and batch dimensions."""
  # Assumes mean=0 as in SpinSphericalBatchNormalization.
  return sphere_utils.spin_spherical_mean(inputs * inputs.conj()).mean(axis=0)


class SphericalBatchNormalizationTest(tf.test.TestCase,
                                      parameterized.TestCase):

  def test_output_and_running_variance(self):
    momentum = 0.9
    input_shape = (2, 6, 5, 4, 3, 2)
    real, imaginary = jax.random.normal(_JAX_RANDOM_KEY, input_shape)
    inputs = real + 1j * imaginary

    model = layers.SphericalBatchNormalization(momentum=momentum,
                                               use_running_stats=False,
                                               use_bias=False,
                                               centered=False)

    # Output variance must be one.
    output, initial_params = model.init_with_output(_JAX_RANDOM_KEY, inputs)
    output_variance = _batched_spherical_variance(output)
    with self.subTest(name="OutputVarianceIsOne"):
      self.assertAllClose(output_variance, jnp.ones_like(output_variance),
                          atol=1e-5)

    output, variances = model.apply(initial_params, inputs,
                                    mutable=["batch_stats"])
    # Running variance is between variance=1 and variance of input.
    input_variance = _batched_spherical_variance(inputs)
    momentum_variance = momentum * 1.0 + (1.0 - momentum) * input_variance
    with self.subTest(name="RunningVariance"):
      self.assertAllClose(momentum_variance,
                          variances["batch_stats"]["variance"])

  @parameterized.parameters(False, True)
  def test_equivariance(self, train):
    resolution = 16
    spins = (0, -1, 2)
    transformer = _get_transformer()
    model = layers.SphericalBatchNormalization(use_bias=False,
                                               centered=False)
    init_args = dict(use_running_stats=True)
    apply_args = dict(use_running_stats=not train, mutable=["batch_stats"])

    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer, model, resolution, spins,
        init_args=init_args,
        apply_args=apply_args)

    self.assertAllClose(coefficients_1, coefficients_2, atol=1e-6)


class SpinSphericalBatchNormalizationTest(tf.test.TestCase,
                                          parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_equivariance(self, train):
    resolution = 16
    spins = (0, 1)
    transformer = _get_transformer()
    model = layers.SpinSphericalBatchNormalization(spins=spins)
    init_args = dict(use_running_stats=True)
    apply_args = dict(use_running_stats=not train, mutable=["batch_stats"])

    coefficients_1, coefficients_2, _ = test_utils.apply_model_to_rotated_pairs(
        transformer, model, resolution, spins,
        init_args=init_args,
        apply_args=apply_args)

    self.assertAllClose(coefficients_1, coefficients_2, atol=1e-5)


if __name__ == "__main__":
  tf.test.main()
