# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Solvers for non-interacting 1d system.

Solve non-interacting 1d system numerically on grids. Each eigenstate will be
occupied by one electron.

Note both solver (EigenSolver, SparseEigenSolver) here are based on directly
diagonalizing the Hamiltonian matrix, which are straightforward to understand,
but not as accurate as other delicate numerical methods, like density matrix
renormalization group (DMRG).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import six
from six.moves import range
import tensorflow as tf


def get_dx(grids):
  """Gets the grid spacing from grids array.

  Args:
    grids: Numpy array with shape (num_grids,).

  Returns:
    Grid spacing.
  """
  return (grids[-1] - grids[0]) / (len(grids) - 1)


def vw_grid(density, dx):
  """von Weizsacker kinetic energy functional on grid.

  Args:
    density: numpy array, density on grid.
      (num_grids,)
    dx: grid spacing.

  Returns:
    kinetic_energy: von Weizsacker kinetic energy.
  """
  gradient = np.gradient(density) / dx
  return np.sum(0.125 * gradient * gradient / density) * dx


def quadratic(mat, x):
  """Compute the quadratic form of matrix and vector.

  Args:
    mat: matrix.
      (n, n)
    x: vector.
      (n,)

  Returns:
    output: scalar value as result of x A x.T.
  """
  return np.dot(x, np.dot(mat, x))


def gaussian_dips(grids, coeff, sigma, mu):
  """Potential of sum of Gaussian dips.

  The i-th Gaussian dip is
    -coeff[i] * np.exp(-(grids - mu[i]) ** 2 / (2 * sigma[i] ** 2))

  Args:
    grids: numpy array of grid points for evaluating 1d potential.
      (num_grids,)
    coeff: numpy array of coefficient for each gaussian dip.
      (num_dips,)
    sigma: numpy array of standard deviation for each gaussian dip.
      (num_dips,)
    mu: numpy array of mean for each gaussian dip.
      (num_dips,)

  Returns:
    vp: Potential on grid.
      (num_grids,)
  """
  grids = np.expand_dims(grids, axis=0)
  coeff = np.expand_dims(coeff, axis=1)
  sigma = np.expand_dims(sigma, axis=1)
  mu = np.expand_dims(mu, axis=1)

  vps = -coeff * np.exp(-(grids - mu) ** 2 / (2 * sigma ** 2))
  vp = np.sum(vps, axis=0)
  return vp


def harmonic_oscillator(grids, k=1.):
  """Potential of quantum harmonic oscillator.

  Args:
    grids: numpy array of grid points for evaluating 1d potential.
      (num_grids,)
    k: strength constant for potential vp = 0.5 * k * grids ** 2

  Returns:
    vp: Potential on grid.
      (num_grid,)
  """
  vp = 0.5 * k * grids ** 2
  return vp


def poschl_teller(grids, lam, a=1., center=0.):
  r"""Poschl-Teller potential.

  Poschl-Teller potential is a special class of potentials for which the
  one-dimensional Schrodinger equation can be solved in terms of Special
  functions.

  https://en.wikipedia.org/wiki/P%C3%B6schl%E2%80%93Teller_potential

  The general form of the potential is

  v(x) = -\frac{\lambda(\lambda + 1)}{2} a^2 \frac{1}{\cosh^2(a x)}

  It holds M=ceil(\lambda) levels, where \lambda is a positive float.

  Args:
    grids: numpy array of grid points for evaluating 1d potential.
      (num_grids,)
    lam: float, lambda in the Poschl-Teller potential function.
    a: float, coefficient in the Poschl-Teller potential function.
    center: float, the center of the potential.

  Returns:
    Potential on grid with shape (num_grid,)

  Raises:
    ValueError: If lam is not positive.
  """
  if lam <= 0:
    raise ValueError('lam is expected to be positive but got %4.2f.' % lam)
  return -lam * (lam + 1) * a ** 2 / (2 * np.cosh(a * (grids - center)) ** 2)


def _valid_poschl_teller_level_lambda(level, lam):
  """Checks whether level and lambda is valid.

  Args:
    level: positive integer, the ground state is level=1.
    lam: positive float, lambda.

  Raises:
    ValueError: If lam is not positive; level is less than 1 or level is greater
      than the total number of levels the potential can hold.
  """
  if lam <= 0:
    raise ValueError('lam is expected to be positive but got %4.2f.' % lam)
  level = int(level)
  if level < 1:
    raise ValueError(
        'level is expected to be greater or equal to 1, but got %d.' % level)
  if level > np.ceil(lam):
    raise ValueError(
        'lam %4.2f can hold %d levels, but got level %d.'
        % (lam, np.ceil(lam), level))


def poschl_teller_energy(level, lam, a=1.):
  """Analytic solution of the total energy filled up to level-th eigenstate.

  The solution can be found in second row of Table 1 in

  Leading corrections to local approximations. II. The case with turning points
  Raphael F. Ribeiro and Kieron Burke, Phys. Rev. B 95, 115115
  https://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.115115

  Args:
    level: positive integer, the ground state is level=1.
    lam: positive float, lambda.
    a: float, coefficient in Poschl-Teller potential.

  Returns:
    Float, the total energy from first to the level-th eigenstate.
  """
  total_energy = 0.
  for i in range(1, int(level) + 1):
    total_energy += poschl_teller_eigen_energy(i, lam, a)
  return total_energy


def poschl_teller_eigen_energy(level, lam, a=1.):
  """Analytic solution of the level-th eigen energy for Poschl-Teller potential.

  This is the energy level for Poschl-Teller potential with float lambda. The
  solution can be found in second row of Table 1 in

  Leading corrections to local approximations. II. The case with turning points
  Raphael F. Ribeiro and Kieron Burke, Phys. Rev. B 95, 115115
  https://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.115115

  Args:
    level: positive integer, the ground state is level=1.
    lam: positive float, lambda.
    a: float, coefficient in Poschl-Teller potential.

  Returns:
    Float, the energy of the level-th eigenstate.
  """
  level = int(level)
  _valid_poschl_teller_level_lambda(level, lam)
  a2 = a ** 2  # a square
  return -a2 * (np.sqrt(lam * (lam + 1) / a2 + 0.25) - level + 0.5) ** 2 / 2


class SolverBase(object):
  """Base Solver for non-interacting 1d system.

  Subclasses should define solve_ground_state method.
  """

  def __init__(self, grids, potential_fn, num_electrons=1):
    """Initialize the solver with potential function and grid.

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
          (num_grids,)
      potential_fn: potential function taking grids as argument.
      num_electrons: Integer, the number of electrons in the system. Must be
          greater or equal to 1.

    Raises:
      ValueError: If num_electrons is less than 1; or num_electrons is not
          an integer.
    """
    # 1d grids.
    self.grids = grids
    self.dx = get_dx(grids)
    self.num_grids = len(grids)
    # Potential on grid.
    self.vp = potential_fn(grids)
    if not isinstance(num_electrons, int):
      raise ValueError('num_electrons is not an integer.')
    elif num_electrons < 1:
      raise ValueError('num_electrons must be greater or equal to 1, but got %d'
                       % num_electrons)
    else:
      self.num_electrons = num_electrons
    # Solver is unsolved by default.
    self._solved = False

  def is_solved(self):
    """Returns whether this solver has been solved."""
    return self._solved

  def solve_ground_state(self):
    """Solve ground state. Need to be implemented in subclasses.

    Compute attributes:
    total_energy, kinetic_energy, potential_energy, density, wave_function.

    Returns:
      self
    """
    raise NotImplementedError('Must be implemented by subclass.')


class EigenSolver(SolverBase):
  """Represents the Hamiltonian as a matrix and diagonalizes it directly.
  """

  def __init__(self, grids, potential_fn, num_electrons=1):
    """Initialize the solver with potential function and grid.

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      potential_fn: potential function taking grids as argument.
      num_electrons: Integer, the number of electrons in the system.
    """
    super(EigenSolver, self).__init__(grids, potential_fn, num_electrons)
    self._set_matrices()

  def _set_matrices(self):
    """Sets matrix attributes.

    Attributes:
      _t_mat: Scipy sparse matrix, kinetic matrix in Hamiltonian.
      _v_mat: Scipy sparse matrix, potential matrix in Hamiltonian.
      _h: Scipy sparse matrix, Hamiltonian matrix.
    """
    # Kinetic matrix
    self._t_mat = self.get_kinetic_matrix()
    # Potential matrix
    self._v_mat = self.get_potential_matrix()
    # Hamiltonian matrix
    self._h = self._t_mat + self._v_mat

  def get_kinetic_matrix(self):
    """Kinetic matrix.

    Returns:
      mat: Kinetic matrix.
        (num_grids, num_grids)
    """
    mat = np.eye(self.num_grids)
    idx = np.arange(self.num_grids)
    mat[idx[:-1], idx[:-1] + 1] = -0.5
    mat[idx[1:], idx[1:] - 1] = -0.5
    return mat / (self.dx * self.dx)

  def get_potential_matrix(self):
    """Potential matrix.

    Returns:
      mat: Potential matrix.
        (num_grids, num_grids)
    """
    return np.diag(self.vp)

  def _update_ground_state(self, eigenvalues, eigenvectors, quadratic_function):
    """Helper function to solve_ground_state() method.

    Updates the attributes total_energy, wave_function, density, kinetic_energy,
    potential_enenrgy and _solved from the eigensolver's output (w, v).

    Args:
      eigenvalues: Numpy array with shape [num_eigenstates,], the eigenvalues in
          ascending order.
      eigenvectors: Numpy array with shape [num_grids, num_eigenstates], each
          column eigenvectors[:, i] is the normalized eigenvector corresponding
          to the eigenvalue eigenvalues[i].
      quadratic_function: Callable, compute the quadratic form of matrix and
          vector.

    Returns:
      self
    """
    self.total_energy = 0.
    self.wave_function = np.zeros((self.num_electrons, self.num_grids))
    self.density = np.zeros(self.num_grids)
    self.kinetic_energy = 0.
    self.potential_energy = 0.

    for i in range(self.num_electrons):
      self.total_energy += eigenvalues[i]
      self.wave_function[i] = eigenvectors.T[i] / np.sqrt(self.dx)
      self.density += self.wave_function[i] ** 2
      self.kinetic_energy += quadratic_function(
          self._t_mat, self.wave_function[i]) * self.dx
      self.potential_energy += quadratic_function(
          self._v_mat, self.wave_function[i]) * self.dx

    self._solved = True
    return self

  def solve_ground_state(self):
    """Solve ground state by diagonalize the Hamiltonian matrix directly.

    Compute attributes:
    total_energy, kinetic_energy, potential_energy, density, wave_function.

    Returns:
      self
    """
    eigenvalues, eigenvectors = np.linalg.eigh(self._h)
    return self._update_ground_state(eigenvalues, eigenvectors, quadratic)


class SparseEigenSolver(EigenSolver):
  """Represents the Hamiltonian as a matrix and solve with sparse eigensolver.
  """

  def __init__(self,
               grids,
               potential_fn,
               num_electrons=1,
               additional_levels=5):
    """Initialize the solver with potential function and grid.

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      potential_fn: potential function taking grids as argument.
      num_electrons: Integer, the number of electrons in the system.
      additional_levels: Integer, non-negative number. For numerical accuracy of
        eigen energies for the first num_electrons,
        num_electrons + additional_levels will be solved.

    Raises:
      ValueError: If additional_levels is negative.
    """
    super(SparseEigenSolver, self).__init__(grids, potential_fn, num_electrons)
    if additional_levels < 0:
      raise ValueError('additional_levels is expected to be non-negative, but '
                       'got %d.' % additional_levels)
    elif additional_levels > self.num_grids - self.num_electrons:
      raise ValueError('additional_levels is expected to be smaller than '
                       'num_grids - num_electrons (%d), but got %d.'
                       % (self.num_grids - self.num_electrons,
                          additional_levels))
    self._additional_levels = additional_levels
    self._set_matrices()

  def get_kinetic_matrix(self):
    """Kinetic matrix.

    Returns:
      mat: Kinetic matrix.
        (num_grids, num_grids)
    """
    grid_factor = 1 / (self.dx * self.dx)
    elements = -0.5 * grid_factor * np.ones(self.num_grids - 1)
    return (sparse.eye(self.num_grids) * grid_factor +
            sparse.diags(elements, offsets=1) +
            sparse.diags(elements, offsets=-1))

  def get_potential_matrix(self):
    """Potential matrix.

    Returns:
      mat: Potential matrix.
        (num_grids, num_grids)
    """
    return sparse.diags(self.vp, offsets=0)

  def _sparse_quadratic(self, sparse_matrix, vector):
    """Compute quadratic of a sparse matrix and a dense vector.

    As of Numpy 1.7, np.dot is not aware of sparse matrices, scipy suggests to
    use the matrix dot method: sparse_matrix.dot(vector).

    Args:
      sparse_matrix: Scipy sparse matrix with shape [dim, dim].
      vector: Numpy array with shape [dim].

    Returns:
      Float, quadratic form of the input matrix and vector.
    """
    return np.dot(vector, sparse_matrix.dot(vector))

  def solve_ground_state(self):
    """Solve ground state by sparse eigensolver.

    Compute attributes:
    total_energy, kinetic_energy, potential_energy, density, wave_function.

    Returns:
      self
    """
    # NOTE(leeley): linalg.eigsh is built on ARPACK. ArpackNoConvergence will be
    # raised if convergence is not obtained.
    # eigsh will solve 5 more eigenstates than self.num_electrons to reduce the
    # numerical error for the last few eigenstates.
    eigenvalues, eigenvectors = linalg.eigsh(
        self._h, k=self.num_electrons + self._additional_levels, which='SM')
    return self._update_ground_state(
        eigenvalues, eigenvectors, self._sparse_quadratic)


def solved_1dsolver_to_example(solver, params):
  """Converts an solved solver with a name to a tf.Example proto.

  Args:
    solver: A Solver instance with attribute solved=True.
    params: dict, other parameters to store in the tf.Example proto.

  Returns:
    example: A tf.Example proto with the following populated fields:
      density, kinetic_energy, total_energy, potential, and other keys in params
      dict.

  Raises:
    ValueError: If the solver is not solved.
  """
  if not solver.is_solved():
    raise ValueError('Input solver is not solved.')

  example = tf.train.Example()
  example.features.feature['density'].float_list.value.extend(
      list(solver.density))
  example.features.feature['kinetic_energy'].float_list.value.append(
      solver.kinetic_energy)
  example.features.feature['total_energy'].float_list.value.append(
      solver.total_energy)
  example.features.feature['dx'].float_list.value.append(
      solver.dx)
  example.features.feature['potential'].float_list.value.extend(list(solver.vp))
  for key, value in six.iteritems(params):
    if isinstance(value, (list, np.ndarray)):
      example.features.feature[key].float_list.value.extend(list(value))
    else:
      example.features.feature[key].float_list.value.append(value)

  return example
