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

"""Tests for jax_dft.jit_scf."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
from jax import tree_util
from jax.config import config
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np

from jax_dft import jit_scf
from jax_dft import neural_xc
from jax_dft import scf
from jax_dft import utils


# Set the default dtype as float64
config.update('jax_enable_x64', True)


class FlipAndAverageOnCenterTest(absltest.TestCase):

  def test_flip_and_average_on_center(self):
    np.testing.assert_allclose(
        jit_scf._flip_and_average_on_center(jnp.array([1., 2., 3.])),
        [2., 2., 2.])

  def test_flip_and_average_on_center_fn(self):
    averaged_fn = jit_scf._flip_and_average_on_center_fn(
        lambda x: jnp.array([4., 5., 6.]))
    np.testing.assert_allclose(
        averaged_fn(jnp.array([1., 2., 3.])), [5., 5., 5.])


class ConnectionWeightsTest(parameterized.TestCase):

  @parameterized.parameters(
      (5, 2,
       [[1., 0., 0., 0., 0.],
        [0.5, 0.5, 0., 0., 0.],
        [0., 0.5, 0.5, 0., 0.],
        [0., 0., 0.5, 0.5, 0.],
        [0., 0., 0., 0.5, 0.5]]),
      (5, 4,
       [[1., 0., 0., 0., 0.],
        [0.5, 0.5, 0., 0., 0.],
        [0.33333334, 0.33333334, 0.33333334, 0., 0.],
        [0.25, 0.25, 0.25, 0.25, 0.],
        [0., 0.25, 0.25, 0.25, 0.25]]),
      )
  def test_connection_weights(
      self, num_iterations, num_mixing_iterations, expected_mask):
    np.testing.assert_allclose(
        jit_scf._connection_weights(num_iterations, num_mixing_iterations),
        expected_mask)


class KohnShamIterationTest(parameterized.TestCase):

  def setUp(self):
    super(KohnShamIterationTest, self).setUp()
    self.grids = jnp.linspace(-5, 5, 101)
    self.num_electrons = 2

  def _create_testing_initial_state(self, interaction_fn):
    locations = jnp.array([-0.5, 0.5])
    nuclear_charges = jnp.array([1, 1])
    return scf.KohnShamState(
        density=self.num_electrons * utils.gaussian(
            grids=self.grids, center=0., sigma=1.),
        # Set initial energy as inf, the actual value is not used in Kohn-Sham
        # calculation.
        total_energy=jnp.inf,
        locations=locations,
        nuclear_charges=nuclear_charges,
        external_potential=utils.get_atomic_chain_potential(
            grids=self.grids,
            locations=locations,
            nuclear_charges=nuclear_charges,
            interaction_fn=interaction_fn),
        grids=self.grids,
        num_electrons=self.num_electrons)

  def _test_state(self, state, initial_state):
    # The density in the next state should normalize to number of electrons.
    self.assertAlmostEqual(
        float(jnp.sum(state.density) * utils.get_dx(self.grids)),
        self.num_electrons)
    # The total energy should be finite after one iteration.
    self.assertTrue(jnp.isfinite(state.total_energy))
    self.assertLen(state.hartree_potential, len(state.grids))
    self.assertLen(state.xc_potential, len(state.grids))
    # locations, nuclear_charges, external_potential, grids and num_electrons
    # remain unchanged.
    np.testing.assert_allclose(initial_state.locations, state.locations)
    np.testing.assert_allclose(
        initial_state.nuclear_charges, state.nuclear_charges)
    np.testing.assert_allclose(
        initial_state.external_potential, state.external_potential)
    np.testing.assert_allclose(initial_state.grids, state.grids)
    self.assertEqual(initial_state.num_electrons, state.num_electrons)
    self.assertGreater(state.gap, 0)

  @parameterized.parameters(True, False)
  def test_kohn_sham_iteration_neural_xc(self, enforce_reflection_symmetry):
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(8), stax.Elu, stax.Dense(1)))
    params_init = init_fn(rng=random.PRNGKey(0))
    initial_state = self._create_testing_initial_state(
        utils.exponential_coulomb)
    next_state = jit_scf.kohn_sham_iteration(
        state=initial_state,
        num_electrons=self.num_electrons,
        xc_energy_density_fn=tree_util.Partial(
            xc_energy_density_fn, params=params_init),
        interaction_fn=utils.exponential_coulomb,
        enforce_reflection_symmetry=enforce_reflection_symmetry)
    self._test_state(next_state, initial_state)


class KohnShamTest(parameterized.TestCase):

  def setUp(self):
    super(KohnShamTest, self).setUp()
    self.grids = jnp.linspace(-5, 5, 101)
    self.num_electrons = 2
    self.locations = jnp.array([-0.5, 0.5])
    self.nuclear_charges = jnp.array([1, 1])

  def _create_testing_external_potential(self, interaction_fn):
    return utils.get_atomic_chain_potential(
        grids=self.grids,
        locations=self.locations,
        nuclear_charges=self.nuclear_charges,
        interaction_fn=interaction_fn)

  def _test_state(self, state, external_potential):
    # The density in the final state should normalize to number of electrons.
    self.assertAlmostEqual(
        float(jnp.sum(state.density) * utils.get_dx(self.grids)),
        self.num_electrons)
    # The total energy should be finite after iterations.
    self.assertTrue(jnp.isfinite(state.total_energy))
    self.assertLen(state.hartree_potential, len(state.grids))
    self.assertLen(state.xc_potential, len(state.grids))
    # locations, nuclear_charges, external_potential, grids and num_electrons
    # remain unchanged.
    np.testing.assert_allclose(state.locations, self.locations)
    np.testing.assert_allclose(state.nuclear_charges, self.nuclear_charges)
    np.testing.assert_allclose(
        external_potential, state.external_potential)
    np.testing.assert_allclose(state.grids, self.grids)
    self.assertEqual(state.num_electrons, self.num_electrons)
    self.assertGreater(state.gap, 0)

  @parameterized.parameters(
      (jnp.inf, [False, True, True]), (-1, [False, False, False]))
  def test_kohn_sham_neural_xc_density_mse_converge_tolerance(
      self, density_mse_converge_tolerance, expected_converged):
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(8), stax.Elu, stax.Dense(1)))
    params_init = init_fn(rng=random.PRNGKey(0))

    states = jit_scf.kohn_sham(
        locations=self.locations,
        nuclear_charges=self.nuclear_charges,
        num_electrons=self.num_electrons,
        num_iterations=3,
        grids=self.grids,
        xc_energy_density_fn=tree_util.Partial(
            xc_energy_density_fn, params=params_init),
        interaction_fn=utils.exponential_coulomb,
        initial_density=self.num_electrons * utils.gaussian(
            grids=self.grids, center=0., sigma=0.5),
        density_mse_converge_tolerance=density_mse_converge_tolerance)

    np.testing.assert_array_equal(states.converged, expected_converged)

    for single_state in scf.state_iterator(states):
      self._test_state(
          single_state,
          self._create_testing_external_potential(utils.exponential_coulomb))

  @parameterized.parameters(2, 3, 4, 5)
  def test_kohn_sham_neural_xc_num_mixing_iterations(
      self, num_mixing_iterations):
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(8), stax.Elu, stax.Dense(1)))
    params_init = init_fn(rng=random.PRNGKey(0))

    states = jit_scf.kohn_sham(
        locations=self.locations,
        nuclear_charges=self.nuclear_charges,
        num_electrons=self.num_electrons,
        num_iterations=3,
        grids=self.grids,
        xc_energy_density_fn=tree_util.Partial(
            xc_energy_density_fn, params=params_init),
        interaction_fn=utils.exponential_coulomb,
        initial_density=self.num_electrons * utils.gaussian(
            grids=self.grids, center=0., sigma=0.5),
        num_mixing_iterations=num_mixing_iterations)

    for single_state in scf.state_iterator(states):
      self._test_state(
          single_state,
          self._create_testing_external_potential(utils.exponential_coulomb))


if __name__ == '__main__':
  absltest.main()
