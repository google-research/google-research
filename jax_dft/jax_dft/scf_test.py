# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint: python3
"""Tests for jax_dft.scf."""

import functools
import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
from jax import tree_util
from jax.config import config
from jax.experimental import stax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

from jax_dft import neural_xc
from jax_dft import np_utils
from jax_dft import scf
from jax_dft import utils


# Set the default dtype as float64
config.update('jax_enable_x64', True)


class ScfTest(parameterized.TestCase):

  def test_discrete_laplacian(self):
    np.testing.assert_allclose(
        scf.discrete_laplacian(6),
        [
            [-5. / 2, 4. / 3, -1. / 12, 0., 0., 0.],
            [4. / 3, -5. / 2, 4. / 3, -1. / 12, 0., 0.],
            [-1. / 12, 4. / 3, -5. / 2, 4. / 3, -1. / 12, 0.],
            [0., -1. / 12, 4. / 3, -5. / 2, 4. / 3, -1. / 12],
            [0., 0., -1. / 12, 4. / 3, -5. / 2, 4. / 3],
            [0., 0., 0., -1. / 12, 4. / 3, -5. / 2],
        ],
        atol=1e-6)

  def test_get_kinetic_matrix(self):
    np.testing.assert_allclose(
        scf.get_kinetic_matrix(grids=jnp.linspace(-10, 10, 6)),
        [
            [0.078125, -0.04166667, 0.00260417, 0., 0., 0.],
            [-0.04166667, 0.078125, -0.04166667, 0.00260417, 0., 0.],
            [0.00260417, -0.04166667, 0.078125, -0.04166667, 0.00260417, 0.],
            [0., 0.00260417, -0.04166667, 0.078125, -0.04166667, 0.00260417],
            [0., 0., 0.00260417, -0.04166667, 0.078125, -0.04166667],
            [0., 0., 0., 0.00260417, -0.04166667, 0.078125],
        ],
        atol=1e-6)

  @parameterized.parameters(
      # The normalized wavefunctions are
      # [[0., 0., 1. / sqrt(0.1), 0., 0.],
      #  [0., -1. / sqrt(0.2), 0., 1. / sqrt(0.2), 0.]]
      # Intensities
      # [[0., 0., 10., 0., 0.],
      #  [0., 5., 0., 5., 0.]]
      (1, np.array([0., 0., 10., 0., 0.])),
      (2, np.array([0., 0., 20., 0., 0.])),
      (3, np.array([0., 5., 20., 5., 0.])),
      (4, np.array([0., 10., 20., 10., 0.])),
      )
  def test_wavefunctions_to_density(self, num_electrons, expected_density):
    np.testing.assert_allclose(
        scf.wavefunctions_to_density(
            num_electrons=num_electrons,
            wavefunctions=jnp.array([
                [0., 0., 1., 0., 0.],
                [0., -1., 0., 1., 0.],
            ]),
            grids=jnp.arange(5) * 0.1),
        expected_density)

  @parameterized.parameters(
      (1, -1.),  # total_eigen_energies = -1.
      (2, -2.),  # total_eigen_energies = -1. - 1.
      (3, 0.),   # total_eigen_energies = -1. - 1. + 2.
      (4, 2.),   # total_eigen_energies = -1. - 1. + 2. + 2.
      (5, 7.),   # total_eigen_energies = -1. - 1. + 2. + 2. + 5.
      (6, 12.),  # total_eigen_energies = -1. - 1. + 2. + 2. + 5. + 5.
      )
  def test_get_total_eigen_energies(
      self, num_electrons, expected_total_eigen_energies):
    self.assertAlmostEqual(
        scf.get_total_eigen_energies(
            num_electrons=num_electrons,
            eigen_energies=jnp.array([-1., 2., 5.])),
        expected_total_eigen_energies)

  @parameterized.parameters(
      (1, 0.),  # gap = -1. - (-1.)
      (2, 3.),  # gap = 2. - (-1.)
      (3, 0.),   # gap = 2. - 2.
      (4, 7.),   # gap = 9. - 2.
      (5, 0.),   # gap = 9. - 9.
      (6, 78.),  # gap = 87. - 9.
      )
  def test_get_gap(self, num_electrons, expected_gap):
    self.assertAlmostEqual(
        scf.get_gap(
            num_electrons=num_electrons,
            eigen_energies=jnp.array([-1., 2., 9., 87.])),
        expected_gap)

  @parameterized.parameters(
      (1, 0.5, 0.),  # total_eigen_energies = 0.5
      (2, 1., 1.),   # total_eigen_energies = 0.5 + 0.5
      (3, 2.5, 0.),  # total_eigen_energies = 0.5 + 0.5 + 1.5
      (4, 4., 1.),   # total_eigen_energies = 0.5 + 0.5 + 1.5 + 1.5
      )
  def test_solve_noninteracting_system(
      self, num_electrons, expected_total_eigen_energies, expected_gap):
    # Quantum harmonic oscillator.
    grids = jnp.linspace(-10, 10, 1001)
    density, total_eigen_energies, gap = scf.solve_noninteracting_system(
        external_potential=0.5 * grids ** 2,
        num_electrons=num_electrons,
        grids=grids)
    self.assertTupleEqual(density.shape, (1001,))
    self.assertAlmostEqual(
        float(total_eigen_energies), expected_total_eigen_energies, places=7)
    self.assertAlmostEqual(float(gap), expected_gap, places=7)

  @parameterized.parameters(utils.soft_coulomb, utils.exponential_coulomb)
  def test_get_hartree_energy(self, interaction_fn):
    grids = jnp.linspace(-5, 5, 11)
    dx = utils.get_dx(grids)
    density = utils.gaussian(grids=grids, center=1., sigma=1.)

    # Compute the expected Hartree energy by nested for loops.
    expected_hartree_energy = 0.
    for x_0, n_0 in zip(grids, density):
      for x_1, n_1 in zip(grids, density):
        expected_hartree_energy += 0.5 * n_0 * n_1 * interaction_fn(
            x_0 - x_1) * dx ** 2

    self.assertAlmostEqual(
        float(scf.get_hartree_energy(
            density=density, grids=grids, interaction_fn=interaction_fn)),
        float(expected_hartree_energy))

  @parameterized.parameters(utils.soft_coulomb, utils.exponential_coulomb)
  def test_get_hartree_potential(self, interaction_fn):
    grids = jnp.linspace(-5, 5, 11)
    dx = utils.get_dx(grids)
    density = utils.gaussian(grids=grids, center=1., sigma=1.)

    # Compute the expected Hartree energy by nested for loops.
    expected_hartree_potential = np.zeros_like(grids)
    for i, x_0 in enumerate(grids):
      for x_1, n_1 in zip(grids, density):
        expected_hartree_potential[i] += np.sum(
            n_1 * interaction_fn(x_0 - x_1)) * dx

    np.testing.assert_allclose(
        scf.get_hartree_potential(
            density=density, grids=grids, interaction_fn=interaction_fn),
        expected_hartree_potential)

  def test_get_external_potential_energy(self):
    grids = jnp.linspace(-5, 5, 10001)
    self.assertAlmostEqual(
        float(scf.get_external_potential_energy(
            external_potential=-jnp.exp(-grids ** 2),
            density=jnp.exp(-(grids - 1) ** 2),
            grids=grids)),
        # Analytical solution:
        # integrate(-exp(-x^2) * exp(-(x - 1) ^ 2), {x, -inf, inf})
        #   = -sqrt(pi / (2 * e))
        -np.sqrt(np.pi / (2 * np.e)))

  def test_get_xc_energy(self):
    grids = jnp.linspace(-5, 5, 10001)
    # We use the form of 3d LDA exchange functional as an example. So the
    # correlation contribution is 0.
    # exchange energy = -0.73855 \int n^(4 / 3) dx
    # exchange energy density = -0.73855 n^(1 / 3)
    # Compute the exchange energy on density exp(-(x - 1) ^ 2:
    # -0.73855 * integrate(exp(-(x - 1) ^ 2) ^ (4 / 3), {x, -inf, inf})
    #   = -1.13367
    xc_energy_density_fn = lambda density: -0.73855 * density ** (1 / 3)
    density = jnp.exp(-(grids - 1) ** 2)
    self.assertAlmostEqual(
        float(scf.get_xc_energy(
            density=density,
            xc_energy_density_fn=xc_energy_density_fn,
            grids=grids)),
        -1.13367,
        places=5)

  def test_get_xc_potential(self):
    grids = jnp.linspace(-5, 5, 10001)
    # We use the form of 3d LDA exchange functional as an example. So the
    # correlation contribution is 0.
    # exchange energy = -0.73855 \int n^(4 / 3) dx
    # exchange potential should be -0.73855 * (4 / 3) n^(1 / 3)
    # by taking functional derivative on exchange energy.
    xc_energy_density_fn = lambda density: -0.73855 * density ** (1 / 3)
    density = jnp.exp(-(grids - 1) ** 2)
    np.testing.assert_allclose(
        scf.get_xc_potential(
            density,
            xc_energy_density_fn=xc_energy_density_fn,
            grids=grids),
        -0.73855 * (4 / 3) * density ** (1 / 3))

  def test_get_xc_potential_hartree(self):
    grids = jnp.linspace(-5, 5, 10001)
    density = utils.gaussian(grids=grids, center=1., sigma=1.)
    def half_hartree_potential(density):
      return 0.5 * scf.get_hartree_potential(
          density=density,
          grids=grids,
          interaction_fn=utils.exponential_coulomb)

    np.testing.assert_allclose(
        scf.get_xc_potential(
            density=density,
            xc_energy_density_fn=half_hartree_potential,
            grids=grids),
        scf.get_hartree_potential(
            density, grids=grids, interaction_fn=utils.exponential_coulomb))


class KohnShamStateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_save_and_load_state(self):
    # Make up a random KohnShamState.
    state = scf.KohnShamState(
        density=np.random.random((5, 100)),
        total_energy=np.random.random(5,),
        locations=np.random.random((5, 2)),
        nuclear_charges=np.random.random((5, 2)),
        external_potential=np.random.random((5, 100)),
        grids=np.random.random((5, 100)),
        num_electrons=np.random.randint(10, size=5),
        hartree_potential=np.random.random((5, 100)))
    save_dir = os.path.join(self.test_dir, 'test_state')

    scf.save_state(save_dir, state)
    loaded_state = scf.load_state(save_dir)

    # Check fields.
    self.assertEqual(state._fields, loaded_state._fields)
    # Check values.
    for field in state._fields:
      value = getattr(state, field)
      if value is None:
        self.assertIsNone(getattr(loaded_state, field))
      else:
        np.testing.assert_allclose(value, getattr(loaded_state, field))


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

  @parameterized.parameters(
      (utils.soft_coulomb, True),
      (utils.soft_coulomb, False),
      (utils.exponential_coulomb, True),
      (utils.exponential_coulomb, False),
      )
  def test_kohn_sham_iteration(
      self, interaction_fn, enforce_reflection_symmetry):
    initial_state = self._create_testing_initial_state(interaction_fn)
    next_state = scf.kohn_sham_iteration(
        state=initial_state,
        num_electrons=self.num_electrons,
        # Use 3d LDA exchange functional and zero correlation functional.
        xc_energy_density_fn=tree_util.Partial(
            lambda density: -0.73855 * density ** (1 / 3)),
        interaction_fn=interaction_fn,
        enforce_reflection_symmetry=enforce_reflection_symmetry)
    self._test_state(next_state, initial_state)

  @parameterized.parameters(
      (utils.soft_coulomb, True),
      (utils.soft_coulomb, False),
      (utils.exponential_coulomb, True),
      (utils.exponential_coulomb, False),
      )
  def test_kohn_sham_iteration_neural_xc(
      self, interaction_fn, enforce_reflection_symmetry):
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(8), stax.Elu, stax.Dense(1)))
    params_init = init_fn(rng=random.PRNGKey(0))
    initial_state = self._create_testing_initial_state(interaction_fn)
    next_state = scf.kohn_sham_iteration(
        state=initial_state,
        num_electrons=self.num_electrons,
        xc_energy_density_fn=tree_util.Partial(
            xc_energy_density_fn, params=params_init),
        interaction_fn=interaction_fn,
        enforce_reflection_symmetry=enforce_reflection_symmetry)
    self._test_state(next_state, initial_state)

  def test_kohn_sham_iteration_neural_xc_energy_loss_gradient(self):
    # The network only has one layer.
    # The initial params contains weights with shape (1, 1) and bias (1,).
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(1)))
    init_params = init_fn(rng=random.PRNGKey(0))
    initial_state = self._create_testing_initial_state(
        utils.exponential_coulomb)
    target_energy = 2.
    spec, flatten_init_params = np_utils.flatten(init_params)

    def loss(flatten_params, initial_state, target_energy):
      state = scf.kohn_sham_iteration(
          state=initial_state,
          num_electrons=self.num_electrons,
          xc_energy_density_fn=tree_util.Partial(
              xc_energy_density_fn,
              params=np_utils.unflatten(spec, flatten_params)),
          interaction_fn=utils.exponential_coulomb,
          enforce_reflection_symmetry=True)
      return (state.total_energy - target_energy) ** 2

    grad_fn = jax.grad(loss)

    params_grad = grad_fn(
        flatten_init_params,
        initial_state=initial_state,
        target_energy=target_energy)

    # Check gradient values.
    np.testing.assert_allclose(params_grad, [-1.40994668, -2.58881225])

    # Check whether the gradient values match the numerical gradient.
    np.testing.assert_allclose(
        optimize.approx_fprime(
            xk=flatten_init_params,
            f=functools.partial(
                loss, initial_state=initial_state, target_energy=target_energy),
            epsilon=1e-9),
        params_grad, atol=3e-4)

  def test_kohn_sham_iteration_neural_xc_density_loss_gradient(self):
    # The network only has one layer.
    # The initial params contains weights with shape (1, 1) and bias (1,).
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(1)))
    init_params = init_fn(rng=random.PRNGKey(0))
    initial_state = self._create_testing_initial_state(
        utils.exponential_coulomb)
    target_density = (
        utils.gaussian(grids=self.grids, center=-0.5, sigma=1.)
        + utils.gaussian(grids=self.grids, center=0.5, sigma=1.))
    spec, flatten_init_params = np_utils.flatten(init_params)

    def loss(flatten_params, initial_state, target_density):
      state = scf.kohn_sham_iteration(
          state=initial_state,
          num_electrons=self.num_electrons,
          xc_energy_density_fn=tree_util.Partial(
              xc_energy_density_fn,
              params=np_utils.unflatten(spec, flatten_params)),
          interaction_fn=utils.exponential_coulomb,
          enforce_reflection_symmetry=False)
      return jnp.sum(jnp.abs(state.density - target_density)) * utils.get_dx(
          self.grids)

    grad_fn = jax.grad(loss)

    params_grad = grad_fn(
        flatten_init_params,
        initial_state=initial_state,
        target_density=target_density)

    # Check gradient values.
    np.testing.assert_allclose(params_grad, [0.2013181, 0.], atol=1e-7)

    # Check whether the gradient values match the numerical gradient.
    np.testing.assert_allclose(
        optimize.approx_fprime(
            xk=flatten_init_params,
            f=functools.partial(
                loss,
                initial_state=initial_state,
                target_density=target_density),
            epsilon=1e-9),
        params_grad, atol=1e-4)

  def test_kohn_sham_iteration_neural_xc_density_loss_gradient_symmetry(self):
    # The network only has one layer.
    # The initial params contains weights with shape (1, 1) and bias (1,).
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(1)))
    init_params = init_fn(rng=random.PRNGKey(0))
    initial_state = self._create_testing_initial_state(
        utils.exponential_coulomb)
    target_density = (
        utils.gaussian(grids=self.grids, center=-0.5, sigma=1.)
        + utils.gaussian(grids=self.grids, center=0.5, sigma=1.))
    spec, flatten_init_params = np_utils.flatten(init_params)

    def loss(flatten_params, initial_state, target_density):
      state = scf.kohn_sham_iteration(
          state=initial_state,
          num_electrons=self.num_electrons,
          xc_energy_density_fn=tree_util.Partial(
              xc_energy_density_fn,
              params=np_utils.unflatten(spec, flatten_params)),
          interaction_fn=utils.exponential_coulomb,
          enforce_reflection_symmetry=True)
      return jnp.sum(jnp.abs(state.density - target_density)) * utils.get_dx(
          self.grids)

    grad_fn = jax.grad(loss)

    params_grad = grad_fn(
        flatten_init_params,
        initial_state=initial_state,
        target_density=target_density)

    # Check gradient values.
    np.testing.assert_allclose(params_grad, [0.2013181, 0.], atol=1e-7)

    # Check whether the gradient values match the numerical gradient.
    np.testing.assert_allclose(
        optimize.approx_fprime(
            xk=flatten_init_params,
            f=functools.partial(
                loss,
                initial_state=initial_state,
                target_density=target_density),
            epsilon=1e-9),
        params_grad, atol=1e-4)


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

  @parameterized.parameters(utils.soft_coulomb, utils.exponential_coulomb)
  def test_kohn_sham(self, interaction_fn):
    state = scf.kohn_sham(
        locations=self.locations,
        nuclear_charges=self.nuclear_charges,
        num_electrons=self.num_electrons,
        num_iterations=3,
        grids=self.grids,
        # Use 3d LDA exchange functional and zero correlation functional.
        xc_energy_density_fn=tree_util.Partial(
            lambda density: -0.73855 * density ** (1 / 3)),
        interaction_fn=interaction_fn)
    for single_state in scf.state_iterator(state):
      self._test_state(
          single_state,
          self._create_testing_external_potential(interaction_fn))

  @parameterized.parameters(
      (-1., [False, False, False]),
      (jnp.inf, [True, True, True]),
      )
  def test_kohn_sham_convergence(
      self, density_mse_converge_tolerance, expected_converged):
    state = scf.kohn_sham(
        locations=self.locations,
        nuclear_charges=self.nuclear_charges,
        num_electrons=self.num_electrons,
        num_iterations=3,
        grids=self.grids,
        # Use 3d LDA exchange functional and zero correlation functional.
        xc_energy_density_fn=tree_util.Partial(
            lambda density: -0.73855 * density ** (1 / 3)),
        interaction_fn=utils.exponential_coulomb,
        density_mse_converge_tolerance=density_mse_converge_tolerance)
    np.testing.assert_allclose(state.converged, expected_converged)

  @parameterized.parameters(utils.soft_coulomb, utils.exponential_coulomb)
  def test_kohn_sham_neural_xc(self, interaction_fn):
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(8), stax.Elu, stax.Dense(1)))
    params_init = init_fn(rng=random.PRNGKey(0))
    state = scf.kohn_sham(
        locations=self.locations,
        nuclear_charges=self.nuclear_charges,
        num_electrons=self.num_electrons,
        num_iterations=3,
        grids=self.grids,
        xc_energy_density_fn=tree_util.Partial(
            xc_energy_density_fn, params=params_init),
        interaction_fn=interaction_fn)
    for single_state in scf.state_iterator(state):
      self._test_state(
          single_state,
          self._create_testing_external_potential(interaction_fn))

  def test_kohn_sham_neural_xc_energy_loss_gradient(self):
    # The network only has one layer.
    # The initial params contains weights with shape (1, 1) and bias (1,).
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(1)))
    init_params = init_fn(rng=random.PRNGKey(0))
    target_energy = 2.
    spec, flatten_init_params = np_utils.flatten(init_params)

    def loss(flatten_params, target_energy):
      state = scf.kohn_sham(
          locations=self.locations,
          nuclear_charges=self.nuclear_charges,
          num_electrons=self.num_electrons,
          num_iterations=3,
          grids=self.grids,
          xc_energy_density_fn=tree_util.Partial(
              xc_energy_density_fn,
              params=np_utils.unflatten(spec, flatten_params)),
          interaction_fn=utils.exponential_coulomb)
      final_state = scf.get_final_state(state)
      return (final_state.total_energy - target_energy) ** 2

    grad_fn = jax.grad(loss)

    params_grad = grad_fn(flatten_init_params, target_energy=target_energy)

    # Check gradient values.
    np.testing.assert_allclose(params_grad, [-3.908153, -5.448675], atol=1e-6)

    # Check whether the gradient values match the numerical gradient.
    np.testing.assert_allclose(
        optimize.approx_fprime(
            xk=flatten_init_params,
            f=functools.partial(loss, target_energy=target_energy),
            epsilon=1e-8),
        params_grad, atol=5e-3)

  def test_kohn_sham_neural_xc_density_loss_gradient(self):
    # The network only has one layer.
    # The initial params contains weights with shape (1, 1) and bias (1,).
    init_fn, xc_energy_density_fn = neural_xc.local_density_approximation(
        stax.serial(stax.Dense(1)))
    init_params = init_fn(rng=random.PRNGKey(0))
    target_density = (
        utils.gaussian(grids=self.grids, center=-0.5, sigma=1.)
        + utils.gaussian(grids=self.grids, center=0.5, sigma=1.))
    spec, flatten_init_params = np_utils.flatten(init_params)

    def loss(flatten_params, target_density):
      state = scf.kohn_sham(
          locations=self.locations,
          nuclear_charges=self.nuclear_charges,
          num_electrons=self.num_electrons,
          num_iterations=3,
          grids=self.grids,
          xc_energy_density_fn=tree_util.Partial(
              xc_energy_density_fn,
              params=np_utils.unflatten(spec, flatten_params)),
          interaction_fn=utils.exponential_coulomb,
          density_mse_converge_tolerance=-1)
      final_state = scf.get_final_state(state)
      return jnp.sum(
          jnp.abs(final_state.density - target_density)) * utils.get_dx(
              self.grids)

    grad_fn = jax.grad(loss)

    params_grad = grad_fn(flatten_init_params, target_density=target_density)

    # Check gradient values.
    np.testing.assert_allclose(params_grad, [0.2643006, 0.], atol=2e-6)
    # Check whether the gradient values match the numerical gradient.
    np.testing.assert_allclose(
        optimize.approx_fprime(
            xk=flatten_init_params,
            f=functools.partial(loss, target_density=target_density),
            epsilon=1e-9),
        params_grad, atol=3e-4)


class GetInitialDensityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.states = scf.KohnShamState(
        density=np.random.random((5, 100)),
        total_energy=np.random.random(5,),
        locations=np.random.random((5, 2)),
        nuclear_charges=np.random.random((5, 2)),
        external_potential=np.random.random((5, 100)),
        grids=np.random.random((5, 100)),
        num_electrons=np.random.randint(10, size=5))

  def test_get_initial_density_exact(self):
    np.testing.assert_allclose(
        scf.get_initial_density(self.states, 'exact'),
        self.states.density)

  def test_get_initial_density_noninteracting(self):
    initial_density = scf.get_initial_density(self.states, 'noninteracting')
    self.assertEqual(initial_density.shape, (5, 100))

  def test_get_initial_density_unknown(self):
    with self.assertRaisesRegex(
        ValueError, 'Unknown initialization method foo'):
      scf.get_initial_density(self.states, 'foo')


if __name__ == '__main__':
  absltest.main()
