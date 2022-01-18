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

"""Tests for research.biology.collaborations.xc.solver1d.single_electron."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from solver1d.solver1d import single_electron


class GetDxTest(absltest.TestCase):

  def test_get_dx(self):
    np.testing.assert_allclose(
        single_electron.get_dx(np.linspace(0, 1, 11)), 0.1)


class VwGridTest(absltest.TestCase):

  def test_quantum_harmonic_oscillator(self):
    grids = np.linspace(-5, 5, 501)
    dx = single_electron.get_dx(grids)
    # Exact ground state density of the quantum harmonic oscillator (k=1):
    # \pi^(-1/2) \exp(-x^2)
    density = np.sqrt(1 / np.pi) * np.exp(-grids ** 2)
    # Its corresponding kinetic energy is 0.25.
    np.testing.assert_allclose(single_electron.vw_grid(density, dx),
                               0.25,
                               atol=1e-7)


class PoschlTellerTest(parameterized.TestCase):

  @parameterized.parameters([(0.5, 1., 0.2),
                             (1., 1.2, 0.),
                             (1.5, 1.1, -0.1),
                             (2, 1., 0.1)])
  def test_poschl_teller(self, lam, a, center):
    grids = np.linspace(-10, 10, 1001)
    vp = single_electron.poschl_teller(grids, lam, a, center)
    # Potential should have same shape with grids.
    self.assertTrue(vp.shape, grids.shape)
    # Potential should be zero at two ends of the grids.
    self.assertAlmostEqual(vp[0], 0)
    self.assertAlmostEqual(vp[-1], 0)
    # Minimum of the potential at x=center.
    self.assertAlmostEqual(grids[np.argmin(vp)], center)

  def test_poschl_teller_invalid_lam(self):
    with self.assertRaisesRegexp(
        ValueError, 'lam is expected to be positive but got -1.50'):
      single_electron.poschl_teller(np.linspace(-10, 10, 1001), lam=-1.5)

  def test_valid_poschl_teller_level_lambda(self):
    with self.assertRaisesRegexp(
        ValueError, 'level is expected to be greater or equal to 1, but got 0'):
      single_electron._valid_poschl_teller_level_lambda(level=0, lam=2.)

    # level=1 is valid.
    single_electron._valid_poschl_teller_level_lambda(level=1, lam=2.)

    with self.assertRaisesRegexp(
        ValueError, 'lam 1.50 can hold 2 levels, but got level 3'):
      single_electron._valid_poschl_teller_level_lambda(level=3, lam=1.5)

    single_electron._valid_poschl_teller_level_lambda(level=2, lam=1.5)

    with self.assertRaisesRegexp(
        ValueError, 'lam is expected to be positive but got -1.00'):
      single_electron._valid_poschl_teller_level_lambda(level=1, lam=-1.)

  @parameterized.parameters([(1, 1., -0.5),
                             (1, 2., -2.),
                             (2, 2., -0.5),
                             (1, 3., -4.5),
                             (2, 3., -2.),
                             (3, 3., -0.5)])
  def test_poschl_teller_eigen_energy_int_lambda(
      self, level, lam, expected_energy):
    # For integer value of lambda, the eigen energy is
    # E = -\mu^2 / 2
    # \mu = \lambda, \lambda - 1, ..., 1
    # https://en.wikipedia.org/wiki/P%C3%B6schl%E2%80%93Teller_potential
    self.assertAlmostEqual(
        single_electron.poschl_teller_eigen_energy(level, lam), expected_energy)

  @parameterized.parameters([(1, 0.5, -0.125),
                             (1, 1.5, -1.125),
                             (2, 1.5, -0.125),
                             (1, 2.5, -3.125),
                             (2, 2.5, -1.125),
                             (3, 2.5, -0.125)])
  def test_poschl_teller_eigen_energy_float_lambda(
      self, level, lam, expected_energy):
    # NOTE(leeley): I calculated some eigen energy for float lambda by the
    # expression in Phys. Rev. B 95, 115115
    self.assertAlmostEqual(
        single_electron.poschl_teller_eigen_energy(level, lam), expected_energy)

  @parameterized.parameters([(1, 3., -4.5),
                             (2, 3., -6.5),
                             (3, 3., -7.),
                             (1, 2.5, -3.125),
                             (2, 2.5, -4.25),
                             (3, 2.5, -4.375)])
  def test_poschl_teller_energy(
      self, level, lam, expected_energy):
    self.assertAlmostEqual(
        single_electron.poschl_teller_energy(level, lam), expected_energy)


class SolverBaseTest(absltest.TestCase):

  def test_invalid_num_electrons(self):
    grids = np.linspace(-5, 5, 501)
    potential_fn = functools.partial(single_electron.harmonic_oscillator, k=1)

    with self.assertRaisesRegexp(ValueError, 'num_electrons is not an integer'):
      single_electron.SolverBase(grids, potential_fn, num_electrons=1.5)

    with self.assertRaisesRegexp(ValueError,
                                 'num_electrons must be greater or equal '
                                 'to 1, but got 0'):
      single_electron.SolverBase(grids, potential_fn, num_electrons=0)


class EigenSolverTest(parameterized.TestCase):

  @parameterized.parameters([(1, 0.5, 0.25),
                             (2, 2, 1),
                             (3, 4.5, 2.25),
                             (4, 8, 4)])
  def test_harmonic_oscillator(self,
                               num_electrons,
                               expected_total_energy,
                               expected_kinetic_energy):
    grids = np.linspace(-5, 5, 501)
    potential_fn = functools.partial(single_electron.harmonic_oscillator, k=1)
    solver = single_electron.EigenSolver(
        grids=grids, potential_fn=potential_fn, num_electrons=num_electrons)

    solver.solve_ground_state()
    # Integrate the density over the grid points.
    norm = np.sum(solver.density) * solver.dx

    np.testing.assert_allclose(norm, num_electrons)
    # i-th eigen-energy of the quantum harmonic oscillator is
    # 0.5 + (i - 1).
    # The total energy for num_electrons states is
    # (0.5 + 0.5 + num_electrons - 1) / 2 * num_electrons
    # = num_electrons ** 2 / 2
    np.testing.assert_allclose(solver.total_energy,
                               expected_total_energy, atol=1e-3)
    # Kinetic energy should equal to half of the total energy.
    # num_electrons ** 2 / 4
    np.testing.assert_allclose(solver.kinetic_energy,
                               expected_kinetic_energy, atol=1e-3)

  def test_gaussian_dips(self):
    grids = np.linspace(-5, 5, 501)
    coeff = np.array([1., 1.])
    sigma = np.array([1., 1.])
    mu = np.array([-2., 2.])
    potential_fn = functools.partial(single_electron.gaussian_dips,
                                     coeff=coeff,
                                     sigma=sigma,
                                     mu=mu)
    solver = single_electron.EigenSolver(grids, potential_fn)

    solver.solve_ground_state()
    # Integrate the density over the grid points.
    norm = np.sum(solver.density) * solver.dx
    # Exact kinetic energy from von Weizsacker functional.
    t_vw = single_electron.vw_grid(solver.density, solver.dx)

    np.testing.assert_allclose(norm, 1.)
    # Kinetic energy should equal to the exact solution from
    # von Weizsacker kinetic energy functional.
    np.testing.assert_allclose(solver.kinetic_energy, t_vw, atol=1e-4)

  @parameterized.parameters([(1, 3., -4.5, 1e-4),
                             (2, 3., -6.5, 5e-4),
                             (3, 3., -7., 1e-3),
                             (1, 2.5, -3.125, 1e-4),
                             (2, 2.5, -4.25, 5e-4),
                             (3, 2.5, -4.375, 1e-3)])
  def test_poschl_teller(
      self, num_electrons, lam, expected_energy, atol):
    solver = single_electron.EigenSolver(
        grids=np.linspace(-10, 10, 1001),
        potential_fn=functools.partial(single_electron.poschl_teller, lam=lam),
        num_electrons=num_electrons)

    solver.solve_ground_state()

    self.assertAlmostEqual(np.sum(solver.density) * solver.dx, num_electrons)
    np.testing.assert_allclose(solver.total_energy, expected_energy, atol=atol)


class SparseEigenSolverTest(parameterized.TestCase):

  def test_additional_levels_negative(self):
    with self.assertRaisesRegexp(
        ValueError,
        'additional_levels is expected to be non-negative, but got -1'):
      single_electron.SparseEigenSolver(
          grids=np.linspace(-5, 5, 501),
          potential_fn=functools.partial(
              single_electron.harmonic_oscillator, k=1),
          num_electrons=3,
          additional_levels=-1)

  def test_additional_levels_too_large(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'additional_levels is expected to be smaller than '
        r'num_grids - num_electrons \(498\), but got 499'):
      single_electron.SparseEigenSolver(
          grids=np.linspace(-5, 5, 501),
          potential_fn=functools.partial(
              single_electron.harmonic_oscillator, k=1),
          num_electrons=3,
          additional_levels=499)

  @parameterized.parameters([(1, 0.5, 0.25),
                             (2, 2, 1),
                             (3, 4.5, 2.25),
                             (4, 8, 4)])
  def test_harmonic_oscillator(self,
                               num_electrons,
                               expected_total_energy,
                               expected_kinetic_energy):
    grids = np.linspace(-5, 5, 501)
    potential_fn = functools.partial(single_electron.harmonic_oscillator, k=1)
    solver = single_electron.SparseEigenSolver(
        grids=grids, potential_fn=potential_fn, num_electrons=num_electrons)

    solver.solve_ground_state()
    # Integrate the density over the grid points.
    norm = np.sum(solver.density) * solver.dx

    np.testing.assert_allclose(norm, num_electrons)
    # i-th eigen-energy of the quantum harmonic oscillator is
    # 0.5 + (i - 1).
    # The total energy for num_electrons states is
    # (0.5 + 0.5 + num_electrons - 1) / 2 * num_electrons
    # = num_electrons ** 2 / 2
    np.testing.assert_allclose(solver.total_energy,
                               expected_total_energy, atol=1e-3)
    # Kinetic energy should equal to half of the total energy.
    # num_electrons ** 2 / 4
    np.testing.assert_allclose(solver.kinetic_energy,
                               expected_kinetic_energy, atol=1e-3)

  def test_gaussian_dips(self):
    grids = np.linspace(-5, 5, 501)
    coeff = np.array([1., 1.])
    sigma = np.array([1., 1.])
    mu = np.array([-2., 2.])
    potential_fn = functools.partial(single_electron.gaussian_dips,
                                     coeff=coeff,
                                     sigma=sigma,
                                     mu=mu)
    solver = single_electron.SparseEigenSolver(grids, potential_fn)

    solver.solve_ground_state()
    # Integrate the density over the grid points.
    norm = np.sum(solver.density) * solver.dx
    # Exact kinetic energy from von Weizsacker functional.
    t_vw = single_electron.vw_grid(solver.density, solver.dx)

    np.testing.assert_allclose(norm, 1.)
    # Kinetic energy should equal to the exact solution from
    # von Weizsacker kinetic energy functional.
    np.testing.assert_allclose(solver.kinetic_energy, t_vw, atol=1e-4)

  @parameterized.parameters([(1, 3., -4.5, 1e-4),
                             (2, 3., -6.5, 5e-4),
                             (3, 3., -7., 1e-3),
                             (1, 2.5, -3.125, 1e-4),
                             (2, 2.5, -4.25, 5e-4),
                             (3, 2.5, -4.375, 1e-3)])
  def test_poschl_teller(
      self, num_electrons, lam, expected_energy, atol):
    # NOTE(leeley): Default additional_levels cannot converge to the correct
    # eigenstate, even for the first eigenstate. I set the additional_levels
    # to 20 so it can converge to the correct eigenstate and reach the same
    # accuracy of the EigenSolver.
    solver = single_electron.SparseEigenSolver(
        grids=np.linspace(-10, 10, 1001),
        potential_fn=functools.partial(single_electron.poschl_teller, lam=lam),
        num_electrons=num_electrons,
        additional_levels=20)

    solver.solve_ground_state()

    self.assertAlmostEqual(np.sum(solver.density) * solver.dx, num_electrons)
    np.testing.assert_allclose(solver.total_energy, expected_energy, atol=atol)


class Solved1dSolverToExampleTest(absltest.TestCase):

  def setUp(self):
    super(Solved1dSolverToExampleTest, self).setUp()
    self.grids = np.linspace(-1, 1, 10)
    self.potential_params = {
        'mu': np.array([0.]),
        'sigma': np.array([0.5]),
        'coeff': np.array([1.])}
    self.potential_fn = functools.partial(single_electron.gaussian_dips,
                                          **self.potential_params)

  def test_solver_should_be_solved(self):
    solver = single_electron.EigenSolver(self.grids, self.potential_fn)
    # The solver is not solved.
    with self.assertRaisesRegexp(ValueError, r'Input solver is not solved.'):
      single_electron.solved_1dsolver_to_example(solver, self.potential_params)

  def test_example_output(self):
    solver = single_electron.EigenSolver(self.grids, self.potential_fn)
    solver.solve_ground_state()

    example = single_electron.solved_1dsolver_to_example(solver,
                                                         self.potential_params)

    self.assertLen(example.features.feature['density'].float_list.value, 10)
    self.assertLen(example.features.feature['potential'].float_list.value, 10)
    np.testing.assert_allclose(
        example.features.feature['kinetic_energy'].float_list.value,
        [0.82937312])
    np.testing.assert_allclose(
        example.features.feature['total_energy'].float_list.value, [0.07467839])
    np.testing.assert_allclose(
        example.features.feature['dx'].float_list.value,
        # dx = (1 + 1) / (10 - 1)
        [0.2222222222])
    np.testing.assert_allclose(
        example.features.feature['mu'].float_list.value,
        self.potential_params['mu'])
    np.testing.assert_allclose(
        example.features.feature['sigma'].float_list.value,
        self.potential_params['sigma'])
    np.testing.assert_allclose(
        example.features.feature['coeff'].float_list.value,
        self.potential_params['coeff'])


if __name__ == '__main__':
  absltest.main()
