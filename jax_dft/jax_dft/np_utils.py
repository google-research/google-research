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

# Lint as: python3
"""Original numpy utility functions."""

from jax import tree_util
import numpy as np
from scipy import special

from jax_dft import constants
from jax_dft import utils


# TODO(shoyer): Remove flatten() and unflatten() after they are checked in to
# jax.
def flatten(params, dtype=np.float64):
  """Flattens the params to 1d original numpy array.

  Args:
    params: pytree.
    dtype: the data type of the output array.

  Returns:
    (tree, shapes), vec
      * tree: the structure of tree.
      * shapes: List of tuples, the shapes of leaves.
      * vec: 1d numpy array, the flatten vector of params.
  """
  leaves, tree = tree_util.tree_flatten(params)
  shapes = [leaf.shape for leaf in leaves]
  vec = np.concatenate([leaf.ravel() for leaf in leaves]).astype(dtype)
  return (tree, shapes), vec


def unflatten(spec, vec):
  """Unflattens the 1d original numpy array to pytree.

  Args:
    spec: (tree, shapes).
      * tree: the structure of tree.
      * shapes: List of tuples, the shapes of leaves.
    vec: 1d numpy array, the flatten vector of params.

  Returns:
    A pytree.
  """
  tree, shapes = spec
  sizes = [int(np.prod(shape)) for shape in shapes]
  leaves_flat = np.split(vec, np.cumsum(sizes)[:-1])
  leaves = [leaf.reshape(shape) for leaf, shape in zip(leaves_flat, shapes)]
  return tree_util.tree_unflatten(tree, leaves)


def _get_exact_h_atom_density(displacements, dx, energy=-0.670):
  """Gets exact Hydrogen atom density with exponential Coulomb interaction.

  One-dimensional mimicking of electronic structure: The case for exponentials.
  Physical Review B 91.23 (2015): 235141.
  https://arxiv.org/pdf/1504.05620.pdf

  Note this function returns an np array since special.jv is not implemented
  in jax.

  Args:
    displacements: Float numpy array with shape (num_nuclei, num_grids).
    dx: Float, the grid spacing.
    energy: Float, the ground state energy of Hydrogen atom.

  Returns:
    Float numpy array with shape (num_nuclei, num_grids).

  Raises:
    ValueError: If ndim of displacements is not 2.
  """
  if displacements.ndim != 2:
    raise ValueError(
        'displacements is expected to have ndim=2, but got %d'
        % displacements.ndim)
  # Defined after Equation 2.
  v = np.sqrt(-8 * energy / constants.EXPONENTIAL_COULOMB_KAPPA ** 2)
  # Equation 2.
  z = (
      2 * np.sqrt(2 * constants.EXPONENTIAL_COULOMB_AMPLITUDE)
      / constants.EXPONENTIAL_COULOMB_KAPPA
      * np.exp(
          -constants.EXPONENTIAL_COULOMB_KAPPA * np.abs(displacements) / 2))
  # Equation 3.
  raw_exact_density = special.jv(v, z) ** 2
  return raw_exact_density / (
      np.sum(raw_exact_density, axis=1, keepdims=True) * dx)


def spherical_superposition_density(grids, locations, nuclear_charges):
  """Builds initial guess of density by superposition of spherical densities.

  Args:
    grids: Float numpy array with shape (num_grids,).
    locations: Float numpy array with shape (num_nuclei,).
    nuclear_charges: Float numpy array with shape (num_nuclei,).

  Returns:
    Float numpy array with shape (num_grids,).
  """
  # (num_nuclei, num_grids)
  displacements = np.expand_dims(
      np.array(grids), axis=0) - np.expand_dims(np.array(locations), axis=1)
  density = _get_exact_h_atom_density(displacements, float(utils.get_dx(grids)))
  return np.dot(nuclear_charges, density)
