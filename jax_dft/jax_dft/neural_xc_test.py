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

# Lint as: python3
"""Tests for jax_dft.neural_xc."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
from jax.config import config
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np

from jax_dft import neural_xc
from jax_dft import scf
from jax_dft import utils


# Set the default dtype as float64
config.update('jax_enable_x64', False)


class NetworkTest(parameterized.TestCase):

  def test_negativity_transform(self):
    init_fn, apply_fn = neural_xc.negativity_transform()

    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=(-1, 3, 1))

    self.assertEqual(output_shape, (-1, 3, 1))
    self.assertEqual(init_params, ())
    self.assertTrue(np.all(
        apply_fn(init_params, jnp.array([[[-0.5], [0.], [0.5]]])) <= 0.278))

  @parameterized.parameters(0.5, 1.5, 2.)
  def test_exponential_function_normalization(self, width):
    # Check the normalization of the exponential function is correct.
    grids = jnp.arange(-256, 257) * 0.08
    self.assertAlmostEqual(
        jnp.sum(neural_xc._exponential_function(grids, width)) * 0.08,
        1., places=2)

  def test_exponential_function_channels(self):
    self.assertEqual(
        neural_xc._exponential_function_channels(
            displacements=jnp.array(np.random.rand(11, 11)),
            widths=jnp.array([1., 2., 3.])).shape,
        (11, 11, 3))

  def test_exponential_global_convolution(self):
    init_fn, apply_fn = neural_xc.exponential_global_convolution(
        num_channels=2, grids=jnp.linspace(-1, 1, 5), minval=0.1, maxval=2.)

    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=(-1, 5, 1))
    output = apply_fn(init_params, jnp.array(np.random.rand(1, 5, 1)))

    self.assertEqual(output_shape, (-1, 5, 2))
    self.assertLen(init_params, 1)
    self.assertEqual(init_params[0].shape, (2,))
    self.assertEqual(output.shape, (1, 5, 2))

  @parameterized.parameters((0.5, 0.77880025), (1., 1.), (100., 0.))
  def test_self_interaction_weight(self, density_integral, expected_weight):
    grids = jnp.linspace(-5, 5, 11)
    self.assertAlmostEqual(
        neural_xc.self_interaction_weight(
            reshaped_density=density_integral * utils.gaussian(
                grids=grids, center=1., sigma=1.)[jnp.newaxis, :, jnp.newaxis],
            dx=utils.get_dx(grids),
            width=1.),
        expected_weight)

  def test_self_interaction_layer_one_electron(self):
    grids = jnp.linspace(-5, 5, 11)
    density = utils.gaussian(grids=grids, center=1., sigma=1.)
    reshaped_density = density[jnp.newaxis, :, jnp.newaxis]

    init_fn, apply_fn = neural_xc.self_interaction_layer(
        grids=grids, interaction_fn=utils.exponential_coulomb)
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=((-1, 11, 1), (-1, 11, 1)))

    self.assertEqual(output_shape, (-1, 11, 1))
    self.assertAlmostEqual(init_params, (1.,))
    np.testing.assert_allclose(
        # The features (second input) is not used for one electron.
        apply_fn(
            init_params, (reshaped_density, jnp.ones_like(reshaped_density))),
        -0.5 * scf.get_hartree_potential(
            density=density,
            grids=grids,
            interaction_fn=utils.exponential_coulomb)[
                jnp.newaxis, :, jnp.newaxis])

  def test_self_interaction_layer_large_num_electrons(self):
    grids = jnp.linspace(-5, 5, 11)
    density = 100. * utils.gaussian(grids=grids, center=1., sigma=1.)
    reshaped_density = density[jnp.newaxis, :, jnp.newaxis]
    features = np.random.rand(*reshaped_density.shape)

    init_fn, apply_fn = neural_xc.self_interaction_layer(
        grids=grids, interaction_fn=utils.exponential_coulomb)
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=((-1, 11, 1), (-1, 11, 1)))

    self.assertEqual(output_shape, (-1, 11, 1))
    self.assertAlmostEqual(init_params, (1.,))
    np.testing.assert_allclose(
        # The output is completely the features (second input).
        apply_fn(init_params, (reshaped_density, features)), features)

  def test_wrap_network_with_self_interaction_layer_one_electron(self):
    grids = jnp.linspace(-5, 5, 9)
    density = utils.gaussian(
        grids=grids, center=1., sigma=1.)
    reshaped_density = density[jnp.newaxis, :, jnp.newaxis]

    init_fn, apply_fn = neural_xc.wrap_network_with_self_interaction_layer(
        network=neural_xc.build_unet(
            num_filters_list=[2, 4],
            core_num_filters=4,
            activation='swish'),
        grids=grids,
        interaction_fn=utils.exponential_coulomb)
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=((-1, 9, 1)))

    self.assertEqual(output_shape, (-1, 9, 1))
    np.testing.assert_allclose(
        apply_fn(init_params, reshaped_density),
        -0.5 * scf.get_hartree_potential(
            density=density,
            grids=grids,
            interaction_fn=utils.exponential_coulomb)[
                jnp.newaxis, :, jnp.newaxis])

  def test_wrap_network_with_self_interaction_layer_large_num_electrons(self):
    grids = jnp.linspace(-5, 5, 9)
    density = 100. * utils.gaussian(
        grids=grids, center=1., sigma=1.)
    reshaped_density = density[jnp.newaxis, :, jnp.newaxis]
    inner_network_init_fn, inner_network_apply_fn = neural_xc.build_unet(
        num_filters_list=[2, 4],
        core_num_filters=4,
        activation='swish')

    init_fn, apply_fn = neural_xc.wrap_network_with_self_interaction_layer(
        network=(inner_network_init_fn, inner_network_apply_fn),
        grids=grids,
        interaction_fn=utils.exponential_coulomb)
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=((-1, 9, 1)))

    self.assertEqual(output_shape, (-1, 9, 1))
    self.assertEqual(apply_fn(init_params, reshaped_density).shape, (1, 9, 1))
    np.testing.assert_allclose(
        apply_fn(init_params, reshaped_density),
        inner_network_apply_fn(
            # The initial parameters of the inner network.
            init_params[1][1], reshaped_density))

  @parameterized.parameters('relu', 'elu', 'softplus', 'swish')
  def test_downsampling_block(self, activation):
    init_fn, apply_fn = neural_xc.downsampling_block(
        num_filters=32, activation=activation)
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=(-1, 9, 1))
    self.assertEqual(output_shape, (-1, 5, 32))

    output = apply_fn(
        init_params, jnp.array(np.random.randn(6, 9, 1)))
    self.assertEqual(output.shape, (6, 5, 32))

  def test_linear_interpolation(self):
    init_fn, apply_fn = neural_xc.linear_interpolation()
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=(-1, 3, 2))
    self.assertEqual(output_shape, (-1, 5, 2))
    self.assertEmpty(init_params)

    np.testing.assert_allclose(
        apply_fn((), jnp.array([[[1, 2], [3, 4], [5, 6]]], dtype=float)),
        [[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]]
    )

  def test_linear_interpolation_transpose(self):
    init_fn, apply_fn = neural_xc.linear_interpolation_transpose()
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=(-1, 5, 2))
    self.assertEqual(output_shape, (-1, 3, 2))
    self.assertEmpty(init_params)

    np.testing.assert_allclose(
        apply_fn((), jnp.array([[[1.0, 2], [3, 4], [5, 6], [7, 8], [9, 10]]])),
        [[[1.5, 2.5], [5, 6], [8.5, 9.5]]]
    )

  def test_upsampling_block(self):
    init_fn, apply_fn = neural_xc.upsampling_block(
        num_filters=32, activation='softplus')
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=(-1, 9, 1))
    self.assertEqual(output_shape, (-1, 17, 32))

    output = apply_fn(
        init_params, jnp.array(np.random.randn(6, 9, 1)))
    self.assertEqual(output.shape, (6, 17, 32))

  def test_build_unet(self):
    init_fn, apply_fn = neural_xc.build_unet(
        num_filters_list=[2, 4, 8],
        core_num_filters=16,
        activation='softplus')
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=(-1, 9, 1))
    self.assertEqual(output_shape, (-1, 9, 1))

    output = apply_fn(
        init_params, jnp.array(np.random.randn(6, 9, 1)))
    self.assertEqual(output.shape, (6, 9, 1))

  def test_build_sliding_net(self):
    init_fn, apply_fn = neural_xc.build_sliding_net(
        window_size=3,
        num_filters_list=[2, 4, 8],
        activation='softplus')
    output_shape, init_params = init_fn(
        random.PRNGKey(0), input_shape=(-1, 9, 1))
    self.assertEqual(output_shape, (-1, 9, 1))

    output = apply_fn(
        init_params, jnp.array(np.random.randn(6, 9, 1)))
    self.assertEqual(output.shape, (6, 9, 1))

  def test_build_sliding_net_invalid_window_size(self):
    with self.assertRaisesRegex(
        ValueError, 'window_size cannot be less than 1, but got 0'):
      neural_xc.build_sliding_net(
          window_size=0,
          num_filters_list=[2, 4, 8],
          activation='softplus')


class LDAFunctionalTest(parameterized.TestCase):

  def setUp(self):
    super(LDAFunctionalTest, self).setUp()
    self.grids = jnp.linspace(-5, 5, 11)
    self.density = utils.gaussian(grids=self.grids, center=1., sigma=1.)

  def test_local_density_approximation(self):
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(16), stax.Elu, stax.Dense(1)))
    init_params = init_fn(rng=random.PRNGKey(0))
    xc_energy_density = xc_energy_density_fn(self.density, init_params)

    # The first layer of the network takes 1 feature (density).
    self.assertEqual(init_params[0][0].shape, (1, 16))
    self.assertEqual(xc_energy_density.shape, (11,))

  def test_local_density_approximation_wrong_output_shape(self):
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(16), stax.Elu, stax.Dense(3)))
    init_params = init_fn(rng=random.PRNGKey(0))

    with self.assertRaisesRegex(
        ValueError,
        r'The output shape of the network '
        r'should be \(-1, 1\) but got \(11, 3\)'):
      xc_energy_density_fn(self.density, init_params)


class GlobalFunctionalTest(parameterized.TestCase):

  def setUp(self):
    super(GlobalFunctionalTest, self).setUp()
    self.grids = jnp.linspace(-5, 5, 17)
    self.density = utils.gaussian(grids=self.grids, center=1., sigma=1.)

  def test_spatial_shift_input(self):
    np.testing.assert_allclose(
        neural_xc._spatial_shift_input(
            features=jnp.array([[
                [11., 21.], [12., 22.], [13., 23.], [14., 24.], [15., 25.]
            ]]),
            num_spatial_shift=4),
        [
            # No shift.
            [[11., 21.], [12., 22.], [13., 23.], [14., 24.], [15., 25.]],
            # Shift by 1.
            [[12., 22.], [13., 23.], [14., 24.], [15., 25.], [0., 0.]],
            # Shift by 2.
            [[13., 23.], [14., 24.], [15., 25.], [0., 0.], [0., 0.]],
            # Shift by 3.
            [[14., 24.], [15., 25.], [0., 0.], [0., 0.], [0., 0.]],
        ])

  def test_reverse_spatial_shift_output(self):
    np.testing.assert_allclose(
        neural_xc._reverse_spatial_shift_output(
            array=jnp.array([
                [1., 2., 3., 4., 5.],
                [12., 13., 14., 15., 0.],
                [23., 24., 25., 0., 0.],
            ])),
        [
            [1., 2., 3., 4., 5.],
            [0., 12., 13., 14., 15.],
            [0., 0., 23., 24., 25.],
        ])

  @parameterized.parameters(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
  def test_is_power_of_two_true(self, number):
    self.assertTrue(neural_xc._is_power_of_two(number))

  @parameterized.parameters(0, 3, 6, 9)
  def test_is_power_of_two_false(self, number):
    self.assertFalse(neural_xc._is_power_of_two(number))

  @parameterized.parameters('relu', 'elu', 'softplus', 'swish')
  def test_global_functional_with_unet(self, activation):
    init_fn, xc_energy_density_fn = (
        neural_xc.global_functional(
            neural_xc.build_unet(
                num_filters_list=[4, 2],
                core_num_filters=4,
                activation=activation),
            grids=self.grids))
    init_params = init_fn(rng=random.PRNGKey(0))
    xc_energy_density = xc_energy_density_fn(self.density, init_params)
    self.assertEqual(xc_energy_density.shape, (17,))

  @parameterized.parameters('relu', 'elu', 'softplus', 'swish')
  def test_global_functional_with_sliding_net(self, activation):
    init_fn, xc_energy_density_fn = (
        neural_xc.global_functional(
            neural_xc.build_sliding_net(
                window_size=3,
                num_filters_list=[4, 2, 2],
                activation=activation),
            grids=self.grids))
    init_params = init_fn(rng=random.PRNGKey(0))
    xc_energy_density = xc_energy_density_fn(self.density, init_params)
    self.assertEqual(xc_energy_density.shape, (17,))

  def test_global_functional_wrong_num_spatial_shift(self):
    with self.assertRaisesRegex(
        ValueError, 'num_spatial_shift can not be less than 1 but got 0'):
      neural_xc.global_functional(
          neural_xc.build_unet(
              num_filters_list=[4, 2],
              core_num_filters=4,
              activation='swish'),
          grids=self.grids,
          # Wrong num_spatial_shift
          num_spatial_shift=0)

  def test_global_functional_wrong_num_grids(self):
    with self.assertRaisesRegex(
        ValueError,
        'The num_grids must be power of two plus one for global functional '
        'but got 6'):
      neural_xc.global_functional(
          neural_xc.build_unet(
              num_filters_list=[4, 2],
              core_num_filters=4,
              activation='softplus'),
          # grids with wrong num_grids.
          grids=jnp.linspace(-1, 1, 6))

  def test_global_functional_with_unet_wrong_output_shape(self):
    init_fn, xc_energy_density_fn = (
        neural_xc.global_functional(
            stax.serial(
                neural_xc.build_unet(
                    num_filters_list=[4, 2],
                    core_num_filters=4,
                    activation='softplus'),
                # Additional conv layer to make the output shape wrong.
                neural_xc.Conv1D(
                    1, filter_shape=(3,), strides=(2,), padding='VALID')
                ),
            grids=self.grids))
    init_params = init_fn(rng=random.PRNGKey(0))

    with self.assertRaisesRegex(
        ValueError,
        r'The output shape of the network '
        r'should be \(-1, 17\) but got \(1, 8\)'):
      xc_energy_density_fn(self.density, init_params)

  def test_global_functional_with_sliding_net_wrong_output_shape(self):
    init_fn, xc_energy_density_fn = (
        neural_xc.global_functional(
            stax.serial(
                neural_xc.build_sliding_net(
                    window_size=3,
                    num_filters_list=[4, 2, 2],
                    activation='softplus'),
                # Additional conv layer to make the output shape wrong.
                neural_xc.Conv1D(
                    1, filter_shape=(1,), strides=(2,), padding='VALID')
                ),
            grids=self.grids))
    init_params = init_fn(rng=random.PRNGKey(0))

    with self.assertRaisesRegex(
        ValueError,
        r'The output shape of the network '
        r'should be \(-1, 17\) but got \(1, 9\)'):
      xc_energy_density_fn(self.density, init_params)


if __name__ == '__main__':
  absltest.main()
