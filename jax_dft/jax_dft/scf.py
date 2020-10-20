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

# python3
"""Functions for self-consistent field calculation."""

import functools
import typing
from typing import Optional, Union

import jax
from jax import tree_util
import jax.numpy as jnp
from jax_dft.jax_dft import utils


ArrayLike = Union[float, bool, jnp.ndarray]


def discrete_laplacian(num_grids):
  """Uses finite difference to approximate Laplacian operator.

  Use five-point estimation of the Laplacian operator.

  Generation of finite difference formulas on arbitrarily spaced grids
  Fornberg, Bengt. Mathematics of computation 51.184 (1988): 699-706.
  https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/

  The Laplacian operator is represented as a penta-diagonal matrix with elements
  (-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12).

  Args:
    num_grids: Integer, the number of grids.

  Returns:
    Float numpy array with shape (num_grids, num_grids).
  """
  # TODO(leeley): Add an option for selecting particular order of accuracy.
  # https://en.wikipedia.org/wiki/Finite_difference_coefficient
  return (
      jnp.diag(-2.5 * jnp.ones(num_grids))
      + jnp.diag(4. / 3 * jnp.ones(num_grids - 1), k=1)
      + jnp.diag(4. / 3 * jnp.ones(num_grids - 1), k=-1)
      + jnp.diag(-1. / 12 * jnp.ones(num_grids - 2), k=2)
      + jnp.diag(-1. / 12 * jnp.ones(num_grids - 2), k=-2))


@jax.jit
def get_kinetic_matrix(grids):
  """Gets kinetic matrix.

  Args:
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float numpy array with shape (num_grids, num_grids).
  """
  dx = utils.get_dx(grids)
  return -0.5 * discrete_laplacian(grids.size) / (dx * dx)


@functools.partial(jax.jit, static_argnums=(0,))
def _wavefunctions_to_density(num_electrons, wavefunctions, grids):
  """Converts wavefunctions to density."""
  # Reduce the amount of computation by removing most of the unoccupid states.
  wavefunctions = wavefunctions[:num_electrons]
  # Normalize the wavefunctions.
  wavefunctions = wavefunctions / jnp.sqrt(jnp.sum(
      wavefunctions ** 2, axis=1, keepdims=True) * utils.get_dx(grids))
  # Each eigenstate has spin up and spin down.
  intensities = jnp.repeat(wavefunctions ** 2, repeats=2, axis=0)
  return jnp.sum(intensities[:num_electrons], axis=0)


def wavefunctions_to_density(num_electrons, wavefunctions, grids):
  """Converts wavefunctions to density.

  Note each eigenstate contains two states: spin up and spin down.

  Args:
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    wavefunctions: Float numpy array with shape (num_eigen_states, num_grids).
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float numpy array with shape (num_grids,).
  """
  return _wavefunctions_to_density(num_electrons, wavefunctions, grids)


def get_total_eigen_energies(num_electrons, eigen_energies):
  """Gets the total eigen energies of the first num_electrons states.

  Note each eigenstate contains two states: spin up and spin down.

  Args:
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    eigen_energies: Float numpy array with shape (num_eigen_states,).

  Returns:
    Float.
  """
  return jnp.sum(jnp.repeat(eigen_energies, repeats=2)[:num_electrons])


def get_gap(num_electrons, eigen_energies):
  """Gets the HOMO–LUMO gap.

  The energy difference between the highest occupied molecule orbital (HOMO)
  and the lowest un-occupied molecular orbital (LUMO).

  Args:
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupied.
    eigen_energies: Float numpy array with shape (num_eigen_states,).

  Returns:
    Float.
  """
  double_occupied_eigen_energies = jnp.repeat(eigen_energies, repeats=2)
  lumo = double_occupied_eigen_energies[num_electrons]
  homo = double_occupied_eigen_energies[num_electrons - 1]
  return lumo - homo


@functools.partial(jax.jit, static_argnums=(1,))
def _solve_noninteracting_system(external_potential, num_electrons, grids):
  """Solves noninteracting system."""
  eigen_energies, wavefunctions_transpose = jnp.linalg.eigh(
      # Hamiltonian matrix.
      get_kinetic_matrix(grids) + jnp.diag(external_potential))
  density = wavefunctions_to_density(
      num_electrons, jnp.transpose(wavefunctions_transpose), grids)
  total_eigen_energies = get_total_eigen_energies(
      num_electrons=num_electrons, eigen_energies=eigen_energies)
  gap = get_gap(num_electrons, eigen_energies)
  return density, total_eigen_energies, gap


def solve_noninteracting_system(external_potential, num_electrons, grids):
  """Solves noninteracting system.

  Args:
    external_potential: Float numpy array with shape (num_grids,).
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    grids: Float numpy array with shape (num_grids,).

  Returns:
    density: Float numpy array with shape (num_grids,).
        The ground state density.
    total_eigen_energies: Float, the total energy of the eigen states.
    gap: Float, the HOMO–LUMO gap.
  """
  return _solve_noninteracting_system(external_potential, num_electrons, grids)


@functools.partial(jax.jit, static_argnums=(2,))
def _get_hartree_energy(density, grids, interaction_fn):
  """Gets the Hartree energy."""
  n1 = jnp.expand_dims(density, axis=0)
  n2 = jnp.expand_dims(density, axis=1)
  r1 = jnp.expand_dims(grids, axis=0)
  r2 = jnp.expand_dims(grids, axis=1)
  return 0.5 * jnp.sum(
      n1 * n2 * interaction_fn(r1 - r2)) * utils.get_dx(grids) ** 2


def get_hartree_energy(density, grids, interaction_fn):
  r"""Gets the Hartree energy.

  U[n] = 0.5 \int dx \int dx' n(x) n(x') / \sqrt{(x - x')^2 + 1}

  Args:
    density: Float numpy array with shape (num_grids,).
    grids: Float numpy array with shape (num_grids,).
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.

  Returns:
    Float.
  """
  return _get_hartree_energy(density, grids, interaction_fn)


@functools.partial(jax.jit, static_argnums=(2,))
def _get_hartree_potential(density, grids, interaction_fn):
  """Gets the Hartree potential."""
  n1 = jnp.expand_dims(density, axis=0)
  r1 = jnp.expand_dims(grids, axis=0)
  r2 = jnp.expand_dims(grids, axis=1)
  return jnp.sum(n1 * interaction_fn(r1 - r2), axis=1) * utils.get_dx(grids)


def get_hartree_potential(density, grids, interaction_fn):
  r"""Gets the Hartree potential.

  v_H(x) = \int dx' n(x') / \sqrt{(x - x')^2 + 1}

  Args:
    density: Float numpy array with shape (num_grids,).
    grids: Float numpy array with shape (num_grids,).
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.

  Returns:
    Float numpy array with shape (num_grids,).
  """
  return _get_hartree_potential(density, grids, interaction_fn)


def get_external_potential_energy(external_potential, density, grids):
  """Gets external potential energy.

  Args:
    external_potential: Float numpy array with shape (num_grids,).
    density: Float numpy array with shape (num_grids,).
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float.
  """
  return jnp.dot(density, external_potential) * utils.get_dx(grids)


def get_xc_energy(density, xc_energy_density_fn, grids):
  r"""Gets xc energy.

  E_xc = \int density * xc_energy_density_fn(density) dx.

  Args:
    density: Float numpy array with shape (num_grids,).
    xc_energy_density_fn: function takes density and returns float numpy array
        with shape (num_grids,).
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float.
  """
  return jnp.dot(xc_energy_density_fn(density), density) * utils.get_dx(grids)


def get_xc_potential(density, xc_energy_density_fn, grids):
  """Gets xc potential.

  The xc potential is derived from xc_energy_density through automatic
  differentiation.

  Args:
    density: Float numpy array with shape (num_grids,).
    xc_energy_density_fn: function takes density and returns float numpy array
        with shape (num_grids,).
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float numpy array with shape (num_grids,).
  """
  return jax.grad(get_xc_energy)(
      density, xc_energy_density_fn, grids) / utils.get_dx(grids)


class KohnShamState(typing.NamedTuple):
  """A namedtuple containing the state of an Kohn-Sham iteration.

  Attributes:
    density: A float numpy array with shape (num_grids,).
    total_energy: Float, the total energy of Kohn-Sham calculation.
    locations: A float numpy array with shape (num_nuclei,).
    nuclear_charges: A float numpy array with shape (num_nuclei,).
    external_potential: A float numpy array with shape (num_grids,).
    grids: A float numpy array with shape (num_grids,).
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    hartree_potential: A float numpy array with shape (num_grids,).
    xc_potential: A float numpy array with shape (num_grids,).
    xc_energy_density: A float numpy array with shape (num_grids,).
    gap: Float, the Kohn-Sham gap.
    converged: Boolean, whether the state is converged.
  """

  density: jnp.ndarray
  total_energy: ArrayLike
  locations: jnp.ndarray
  nuclear_charges: jnp.ndarray
  external_potential: jnp.ndarray
  grids: jnp.ndarray
  num_electrons: ArrayLike
  hartree_potential: Optional[jnp.ndarray] = None
  xc_potential: Optional[jnp.ndarray] = None
  xc_energy_density: Optional[jnp.ndarray] = None
  gap: Optional[ArrayLike] = None
  converged: Optional[ArrayLike] = False




def _flip_and_average_fn(fn, locations, grids):
  """Flips and averages a function at the center of the locations."""
  def output_fn(array):
    output_array = utils.flip_and_average(
        locations=locations, grids=grids, array=array)
    return utils.flip_and_average(
        locations=locations, grids=grids, array=fn(output_array))
  return output_fn


def kohn_sham_iteration(
    state,
    num_electrons,
    xc_energy_density_fn,
    interaction_fn,
    enforce_reflection_symmetry):
  """One iteration of Kohn-Sham calculation.

  Note xc_energy_density_fn must be wrapped by jax.tree_util.Partial so this
  function can take a callable. When the arguments of this callable changes,
  e.g. the parameters of the neural network, kohn_sham_iteration() will not be
  recompiled.

  Args:
    state: KohnShamState.
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    xc_energy_density_fn: function takes density (num_grids,) and returns
        the energy density (num_grids,).
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.
    enforce_reflection_symmetry: Boolean, whether to enforce reflection
        symmetry. If True, the system are symmetric respecting to the center.

  Returns:
    KohnShamState, the next state of Kohn-Sham iteration.
  """
  if enforce_reflection_symmetry:
    xc_energy_density_fn = _flip_and_average_fn(
        xc_energy_density_fn, locations=state.locations, grids=state.grids)

  hartree_potential = get_hartree_potential(
      density=state.density,
      grids=state.grids,
      interaction_fn=interaction_fn)
  xc_potential = get_xc_potential(
      density=state.density,
      xc_energy_density_fn=xc_energy_density_fn,
      grids=state.grids)
  ks_potential = hartree_potential + xc_potential + state.external_potential
  xc_energy_density = xc_energy_density_fn(state.density)

  # Solve Kohn-Sham equation.
  density, total_eigen_energies, gap = solve_noninteracting_system(
      external_potential=ks_potential,
      num_electrons=num_electrons,
      grids=state.grids)

  total_energy = (
      # kinetic energy = total_eigen_energies - external_potential_energy
      total_eigen_energies
      - get_external_potential_energy(
          external_potential=ks_potential,
          density=density,
          grids=state.grids)
      # Hartree energy
      + get_hartree_energy(
          density=density,
          grids=state.grids,
          interaction_fn=interaction_fn)
      # xc energy
      + get_xc_energy(
          density=density,
          xc_energy_density_fn=xc_energy_density_fn,
          grids=state.grids)
      # external energy
      + get_external_potential_energy(
          external_potential=state.external_potential,
          density=density,
          grids=state.grids)
      )

  if enforce_reflection_symmetry:
    density = utils.flip_and_average(
        locations=state.locations, grids=state.grids, array=density)

  return state._replace(
      density=density,
      total_energy=total_energy,
      hartree_potential=hartree_potential,
      xc_potential=xc_potential,
      xc_energy_density=xc_energy_density,
      gap=gap)


def kohn_sham(
    locations,
    nuclear_charges,
    num_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density=None,
    alpha=0.5,
    alpha_decay=0.9,
    enforce_reflection_symmetry=False,
    num_mixing_iterations=2,
    density_mse_converge_tolerance=-1.):
  """Runs Kohn-Sham to solve ground state of external potential.

  Args:
    locations: Float numpy array with shape (num_nuclei,), the locations of
        atoms.
    nuclear_charges: Float numpy array with shape (num_nuclei,), the nuclear
        charges.
    num_electrons: Integer, the number of electrons in the system. The first
        num_electrons states are occupid.
    num_iterations: Integer, the number of Kohn-Sham iterations.
    grids: Float numpy array with shape (num_grids,).
    xc_energy_density_fn: function takes density (num_grids,) and returns
        the energy density (num_grids,).
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.
    initial_density: Float numpy array with shape (num_grids,), initial guess
        of the density for Kohn-Sham calculation. Default None, the initial
        density is non-interacting solution from the external_potential.
    alpha: Float between 0 and 1, density linear mixing factor, the fraction
        of the output of the k-th Kohn-Sham iteration.
        If 0, the input density to the k-th Kohn-Sham iteration is fed into
        the (k+1)-th iteration. The output of the k-th Kohn-Sham iteration is
        completely ignored.
        If 1, the output density from the k-th Kohn-Sham iteration is fed into
        the (k+1)-th iteration, equivalent to no density mixing.
    alpha_decay: Float between 0 and 1, the decay factor of alpha. The mixing
        factor after k-th iteration is alpha * alpha_decay ** k.
    enforce_reflection_symmetry: Boolean, whether to enforce reflection
        symmetry. If True, the density are symmetric respecting to the center.
    num_mixing_iterations: Integer, the number of density differences in the
        previous iterations to mix the density.
    density_mse_converge_tolerance: Float, the stopping criteria. When the MSE
        density difference between two iterations is smaller than this value,
        the Kohn Sham iterations finish. The outputs of the rest of the steps
        are padded by the output of the converged step. Set this value to
        negative to disable early stopping.

  Returns:
    KohnShamState, the states of all the Kohn-Sham iteration steps.
  """
  external_potential = utils.get_atomic_chain_potential(
      grids=grids,
      locations=locations,
      nuclear_charges=nuclear_charges,
      interaction_fn=interaction_fn)
  if initial_density is None:
    # Use the non-interacting solution from the external_potential as initial
    # guess.
    initial_density, _, _ = solve_noninteracting_system(
        external_potential=external_potential,
        num_electrons=num_electrons,
        grids=grids)
  # Create initial state.
  state = KohnShamState(
      density=initial_density,
      total_energy=jnp.inf,
      locations=locations,
      nuclear_charges=nuclear_charges,
      external_potential=external_potential,
      grids=grids,
      num_electrons=num_electrons)
  states = []
  differences = None
  converged = False
  for _ in range(num_iterations):
    if converged:
      states.append(state)
      continue

    old_state = state
    state = kohn_sham_iteration(
        state=old_state,
        num_electrons=num_electrons,
        xc_energy_density_fn=xc_energy_density_fn,
        interaction_fn=interaction_fn,
        enforce_reflection_symmetry=enforce_reflection_symmetry)
    density_difference = state.density - old_state.density
    if differences is None:
      differences = jnp.array([density_difference])
    else:
      differences = jnp.vstack([differences, density_difference])
    if jnp.mean(
        jnp.square(density_difference)) < density_mse_converge_tolerance:
      converged = True
    state = state._replace(converged=converged)
    # Density mixing.
    state = state._replace(
        density=old_state.density
        + alpha * jnp.mean(differences[-num_mixing_iterations:], axis=0))
    states.append(state)
    alpha *= alpha_decay

  return tree_util.tree_multimap(lambda *x: jnp.stack(x), *states)


def get_final_state(state):
  """Get the final state from states in KohnShamState.

  Args:
    state: KohnShamState contains a series of states in Kohn-Sham iterations.

  Returns:
    KohnShamState contains the final state.
  """
  return tree_util.tree_map(lambda x: x[-1], state)


def state_iterator(state):
  """Iterates over states in KohnShamState.

  Args:
    state: KohnShamState contains a series of states in Kohn-Sham iterations.

  Yields:
    KohnShamState.
  """
  leaves, treedef = tree_util.tree_flatten(state)
  for elements in zip(*leaves):
    yield treedef.unflatten(elements)


def get_initial_density(states, method):
  """Gets initial density for Kohn-Sham calculation.

  Args:
    states: KohnShamState contains a batch of states.
    method: String, the density initialization method.

  Returns:
    Float numpy array with shape (batch_size, num_grids).

  Raises:
    ValueError: If the initialization method is not exact or noninteracting.
  """
  if method == 'exact':
    return states.density
  elif method == 'noninteracting':
    solve = jax.vmap(solve_noninteracting_system, in_axes=(0, None, None))
    return solve(
        states.external_potential,
        states.num_electrons[0],
        states.grids[0])[0]
  else:
    raise ValueError(f'Unknown initialization method {method}')
