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

"""Tests for jax_dft.utils."""

from absl.testing import absltest
from absl.testing import parameterized
from jax.config import config
import jax.numpy as jnp
import numpy as np

from jax_dft import constants
from jax_dft import utils


# Set the default dtype as float64
config.update('jax_enable_x64', True)


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      (2, [4., 1., 0., 0.]),
      (-2, [0., 0., 1., 2.]),
      )
  def test_shift(self, offset, expected_output):
    np.testing.assert_allclose(
        utils.shift(jnp.array([1., 2., 4., 1.]), offset=offset),
        expected_output)

  def test_get_dx(self):
    self.assertAlmostEqual(utils.get_dx(jnp.linspace(0, 1, 11)), 0.1)

  def test_get_dx_incorrect_ndim(self):
    with self.assertRaisesRegex(
        ValueError, 'grids.ndim is expected to be 1 but got 2'):
      utils.get_dx(jnp.array([[-0.1], [0.], [0.1]]))

  @parameterized.parameters(
      (0., 2., 1 / (np.sqrt(2 * np.pi) * 2)),
      (3., 0.5, 1 / (np.sqrt(2 * np.pi) * 0.5)),
      )
  def test_gaussian(self, center, sigma, expected_max_value):
    gaussian = utils.gaussian(
        grids=jnp.linspace(-10, 10, 201), center=center, sigma=sigma)
    self.assertAlmostEqual(float(jnp.sum(gaussian) * 0.1), 1, places=5)
    self.assertAlmostEqual(float(jnp.amax(gaussian)), expected_max_value)

  @parameterized.parameters(-1., 0., 1.)
  def test_soft_coulomb(self, center):
    grids = jnp.linspace(-10, 10, 201)
    soft_coulomb_interaction = utils.soft_coulomb(grids - center)
    self.assertAlmostEqual(float(jnp.amax(soft_coulomb_interaction)), 1)
    self.assertAlmostEqual(
        float(grids[jnp.argmax(soft_coulomb_interaction)]), center)

  @parameterized.parameters(-1., 0., 1.)
  def test_exponential_coulomb(self, center):
    grids = jnp.linspace(-10, 10, 201)
    soft_coulomb_interaction = utils.exponential_coulomb(grids - center)
    self.assertAlmostEqual(
        float(jnp.amax(soft_coulomb_interaction)),
        constants.EXPONENTIAL_COULOMB_AMPLITUDE)
    self.assertAlmostEqual(
        float(grids[jnp.argmax(soft_coulomb_interaction)]), center)

  def test_get_atomic_chain_potential_soft_coulomb(self):
    potential = utils.get_atomic_chain_potential(
        grids=jnp.linspace(-10, 10, 201),
        locations=jnp.array([0., 1.]),
        nuclear_charges=jnp.array([2, 1]),
        interaction_fn=utils.soft_coulomb)
    # -2 / jnp.sqrt(10 ** 2 + 1) - 1 / jnp.sqrt(11 ** 2 + 1) = -0.28954318
    self.assertAlmostEqual(float(potential[0]), -0.28954318)
    # -2 / jnp.sqrt(0 ** 2 + 1) - 1 / jnp.sqrt(1 ** 2 + 1) = -2.70710678
    self.assertAlmostEqual(float(potential[100]), -2.70710678)
    # -2 / jnp.sqrt(10 ** 2 + 1) - 1 / jnp.sqrt(9 ** 2 + 1) = -0.30943896
    self.assertAlmostEqual(float(potential[200]), -0.30943896)

  def test_get_atomic_chain_potential_exponential_coulomb(self):
    potential = utils.get_atomic_chain_potential(
        grids=jnp.linspace(-10, 10, 201),
        locations=jnp.array([0., 1.]),
        nuclear_charges=jnp.array([2, 1]),
        interaction_fn=utils.exponential_coulomb)
    # -2 * 1.071295 * jnp.exp(-np.abs(10) / 2.385345) - 1.071295 * jnp.exp(
    #     -np.abs(11) / 2.385345) = -0.04302427
    self.assertAlmostEqual(float(potential[0]), -0.04302427)
    # -2 * 1.071295 * jnp.exp(-np.abs(0) / 2.385345) - 1.071295 * jnp.exp(
    #     -np.abs(1) / 2.385345) = -2.84702559
    self.assertAlmostEqual(float(potential[100]), -2.84702559)
    # -2 * 1.071295 * jnp.exp(-np.abs(10) / 2.385345) - 1.071295 * jnp.exp(
    #     -np.abs(9) / 2.385345) = -0.05699946
    self.assertAlmostEqual(float(potential[200]), -0.05699946)

  @parameterized.parameters(
      ([[-0.1], [0.], [0.1]], [1, 3], [1, 2],
       'grids.ndim is expected to be 1 but got 2'),
      ([-0.1, 0., 0.1], [[1], [3]], [1, 2],
       'locations.ndim is expected to be 1 but got 2'),
      ([-0.1, 0., 0.1], [1, 3], [[1], [2]],
       'nuclear_charges.ndim is expected to be 1 but got 2'),
      )
  def test_get_atomic_chain_potential_incorrect_ndim(
      self, grids, locations, nuclear_charges, expected_message):
    with self.assertRaisesRegex(ValueError, expected_message):
      utils.get_atomic_chain_potential(
          grids=jnp.array(grids),
          locations=jnp.array(locations),
          nuclear_charges=jnp.array(nuclear_charges),
          interaction_fn=utils.exponential_coulomb)

  @parameterized.parameters(
      # One pair of soft coulomb interaction.
      # 1 * 2 / jnp.sqrt((1 - 3) ** 2 + 1)
      ([1, 3], [1, 2], utils.soft_coulomb, 0.89442719),
      # Three pairs of soft coulomb interaction.
      # 1 * 1 / jnp.sqrt((1 + 2) ** 2 + 1)
      # + 1 * 2 / jnp.sqrt((3 + 2) ** 2 + 1)
      # + 1 * 2 / jnp.sqrt((3 - 1) ** 2 + 1)
      ([-2, 1, 3], [1, 1, 2], utils.soft_coulomb, 1.602887227),
      # One pair of exponential interaction.
      # 1 * 2 * 1.071295 * jnp.exp(-np.abs(1 - 3) / 2.385345)
      ([1, 3], [1, 2], utils.exponential_coulomb, 0.92641057),
      # Three pairs of exponential interaction.
      # 1 * 1 * 1.071295 * jnp.exp(-np.abs(1 + 2) / 2.385345)
      # + 1 * 2 * 1.071295 * jnp.exp(-np.abs(3 + 2) / 2.385345)
      # + 1 * 2 * 1.071295 * jnp.exp(-np.abs(3 - 1) / 2.385345)
      ([-2, 1, 3], [1, 1, 2], utils.exponential_coulomb, 1.49438414),
      )
  def test_get_nuclear_interaction_energy(
      self, locations, nuclear_charges, interaction_fn, ecpected_energy):
    self.assertAlmostEqual(
        float(utils.get_nuclear_interaction_energy(
            locations=jnp.array(locations),
            nuclear_charges=jnp.array(nuclear_charges),
            interaction_fn=interaction_fn)),
        ecpected_energy)

  @parameterized.parameters(
      ([[1, 3], [0, 0]], [[1, 2], [1, 1]],
       utils.soft_coulomb, [0.89442719, 1.]),
      ([[1, 3], [0, 0]], [[1, 2], [1, 1]],
       utils.exponential_coulomb, [0.92641057, 1.071295]),
      )
  def test_get_nuclear_interaction_energy_batch(
      self, locations, nuclear_charges, interaction_fn, ecpected_energies):
    np.testing.assert_allclose(
        utils.get_nuclear_interaction_energy_batch(
            locations=jnp.array(locations),
            nuclear_charges=jnp.array(nuclear_charges),
            interaction_fn=interaction_fn),
        ecpected_energies)

  @parameterized.parameters(
      ([[1], [3]], [1, 2], 'locations.ndim is expected to be 1 but got 2'),
      ([1, 3], [[1], [2]],
       'nuclear_charges.ndim is expected to be 1 but got 2'),
      )
  def test_get_nuclear_interaction_energy_incorrect_ndim(
      self, locations, nuclear_charges, expected_message):
    with self.assertRaisesRegex(ValueError, expected_message):
      utils.get_nuclear_interaction_energy(
          locations=jnp.array(locations),
          nuclear_charges=jnp.array(nuclear_charges),
          interaction_fn=utils.exponential_coulomb)

  @parameterized.parameters(-0.1, 0.0, 0.1, 0.2, 0.3)
  def test_float_value_in_array_true(self, value):
    self.assertTrue(utils._float_value_in_array(
        value, array=jnp.array([-0.1, 0.0, 0.1, 0.2, 0.3])))

  @parameterized.parameters(-0.15, 0.05, 0.12)
  def test_float_value_in_array_false(self, value):
    self.assertFalse(utils._float_value_in_array(
        value, array=jnp.array([-0.1, 0.0, 0.1, 0.2, 0.3])))

  def test_flip_and_average_the_front_of_array_center_on_grids(self):
    np.testing.assert_allclose(
        utils.flip_and_average(
            locations=jnp.array([-0.1, 0.3]),
            grids=jnp.array([-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            array=jnp.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8])),
        # The center is 0.1, which is the grid point with index 3.
        # The array on the grids [-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        # are flipped:
        # [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5]
        # -> [0.5, 0.3, 0.2, 0.7, 0.6, 0.2, 0.1]
        # The averaged array is
        # [0.3, 0.25, 0.4, 0.7, 0.4, 0.25, 0.3]
        # Replace the corresponding range (slice(0, 7)) in the original array:
        # [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8]
        # -> [0.3, 0.25, 0.4, 0.7, 0.4, 0.25, 0.3, 0.1, 0.8]
        [0.3, 0.25, 0.4, 0.7, 0.4, 0.25, 0.3, 0.1, 0.8])

  def test_flip_and_average_the_back_of_array_center_on_grids(self):
    np.testing.assert_allclose(
        utils.flip_and_average(
            locations=jnp.array([0.4, 0.6]),
            grids=jnp.array([-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            array=jnp.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8])),
        # The center is 0.5, which is the grid point with index 7.
        # The array on the grids [0.4, 0.5, 0.6] are flipped:
        # [0.5, 0.1, 0.8]
        # -> [0.8, 0.1, 0.5]
        # The averaged array is
        # [0.65, 0.1, 0.65]
        # Replace the corresponding range (slice(6, 9)) in the original array:
        # [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8]
        # -> [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.65, 0.1, 0.65]
        [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.65, 0.1, 0.65])

  def test_flip_and_average_the_front_of_array_center_not_on_grids(self):
    np.testing.assert_allclose(
        utils.flip_and_average(
            locations=jnp.array([-0.1, 0.2]),
            grids=jnp.array([-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            array=jnp.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8])),
        # The center is 0.05, which is the grid point between index 2 and 3.
        # The array on the grids [-0.2, -0.1, 0., 0.1, 0.2, 0.3]
        # are flipped:
        # [0.1, 0.2, 0.6, 0.7, 0.2, 0.3]
        # -> [0.3, 0.2, 0.7, 0.6, 0.2, 0.1]
        # The averaged array is
        # [0.2, 0.2, 0.65, 0.65, 0.2, 0.2]
        # Replace the corresponding range (slice(0, 6)) in the original array:
        # [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8]
        # -> [0.2, 0.2, 0.65, 0.65, 0.2, 0.2, 0.5, 0.1, 0.8]
        [0.2, 0.2, 0.65, 0.65, 0.2, 0.2, 0.5, 0.1, 0.8])

  def test_flip_and_average_the_back_of_array_center_not_on_grids(self):
    np.testing.assert_allclose(
        utils.flip_and_average(
            locations=jnp.array([0.4, 0.5]),
            grids=jnp.array([-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            array=jnp.array([0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8])),
        # The center is 0.45, which is the grid point between index 6 and 7.
        # The array on the grids [0.3, 0.4, 0.5, 0.6] are flipped:
        # [0.3, 0.5, 0.1, 0.8]
        # -> [0.8, 0.1, 0.5, 0.3]
        # The averaged array is
        # [0.55, 0.3, 0.3, 0.55]
        # Replace the corresponding range (slice(5, 9)) in the original array:
        # [0.1, 0.2, 0.6, 0.7, 0.2, 0.3, 0.5, 0.1, 0.8]
        # -> [0.1, 0.2, 0.6, 0.7, 0.2, 0.55, 0.3, 0.3, 0.55]
        [0.1, 0.2, 0.6, 0.7, 0.2, 0.55, 0.3, 0.3, 0.55])

  def test_flip_and_average_location_not_on_grids(self):
    with self.assertRaisesRegex(
        ValueError, r'Location 0\.25 is not on the grids'):
      utils.flip_and_average(
          # 0.25 is not on the grids.
          locations=jnp.array([0.0, 0.25]),
          grids=jnp.array([-0.1, 0.0, 0.1, 0.2, 0.3]),
          # Values of array do not matter in this test.
          array=jnp.array([0.1, 0.2, 0.6, 0.7, 0.2]))

  def test_location_center_at_grids_center_point_true(self):
    self.assertTrue(
        utils.location_center_at_grids_center_point(
            locations=jnp.array([-0.5, 0.5]),
            grids=jnp.array([-0.4, -0.2, 0., 0.2, 0.4])))

  def test_location_center_at_grids_center_point_false(self):
    # The center of the location is not at the center point of the grids.
    self.assertFalse(
        utils.location_center_at_grids_center_point(
            locations=jnp.array([-0.5, 0.6]),
            grids=jnp.array([-0.4, -0.2, 0., 0.2, 0.4])))
    # The number of grids is not odd number, so there is no single center point
    # on the grids.
    self.assertFalse(
        utils.location_center_at_grids_center_point(
            locations=jnp.array([-0.5, 0.5]),
            grids=jnp.array([-0.4, -0.2, 0., 0.2])))

  def test_compute_distances_between_nuclei(self):
    np.testing.assert_allclose(
        utils.compute_distances_between_nuclei(
            locations=np.array([
                [-1., 1., 3.5, 5.],
                [-4., 0., 3.5, 10.],
                [-2., -1., 3.5, 55.],
            ]),
            nuclei_indices=(1, 2)),
        [2.5, 3.5, 4.5])

  def test_compute_distances_between_nuclei_wrong_locations_ndim(self):
    with self.assertRaisesRegex(
        ValueError, 'The ndim of locations is expected to be 2 but got 3'):
      utils.compute_distances_between_nuclei(
          # Values of locations are not used in this test.
          locations=np.array([
              [[-1.], [1.], [3.5], [5.]],
              [[-4.], [0.], [3.5], [10.]],
              [[-2.], [-1.], [3.5], [55.]],
          ]),
          # Unused in this test.
          nuclei_indices=(1, 2))

  def test_compute_distances_between_nuclei_wrong_nuclei_indices_size(self):
    with self.assertRaisesRegex(
        ValueError, 'The size of nuclei_indices is expected to be 2 but got 4'):
      utils.compute_distances_between_nuclei(
          # Values of locations are not used in this test.
          locations=np.array([
              [-1., 1., 3.5, 5.],
              [-4., 0., 3.5, 10.],
              [-2., -1., 3.5, 55.],
          ]),
          # Wrong length of nuclei_indices.
          nuclei_indices=(1, 2, 3, 4))


if __name__ == '__main__':
  absltest.main()
