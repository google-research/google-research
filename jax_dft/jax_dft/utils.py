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

"""Utility functions."""

import jax
import jax.numpy as jnp

from jax_dft import constants


def shift(array, offset):
  """Shifts array by offset and pads zero on the edge.

  Args:
    array: Float numpy array with shape (num_grids,).
    offset: Integer, the offset moving to the left.

  Returns:
    Float numpy array with shape (num_grids,).
  """
  sliced = array[slice(offset, None) if offset >= 0 else slice(None, offset)]
  return jnp.pad(
      sliced,
      pad_width=(-min(offset, 0), max(offset, 0)),
      mode='constant',
      constant_values=0)


def get_dx(grids):
  """Gets the grid spacing from grids array.

  Args:
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Float, grid spacing.

  Raises:
    ValueError: If grids.ndim is not 1.
  """
  if grids.ndim != 1:
    raise ValueError(
        'grids.ndim is expected to be 1 but got %d' % grids.ndim)
  return (jnp.amax(grids) - jnp.amin(grids)) / (grids.size - 1)


def gaussian(grids, center, sigma=1.):
  """Gets Gaussian shape blob on grids.

  Args:
    grids: Float numpy array with shape (num_grids,).
    center: Float, the center of the Gaussian function.
    sigma: Float, the variance of Gaussian.

  Returns:
    Float numpy array with shape (num_grids,).
  """
  return 1 / jnp.sqrt(2 * jnp.pi) * jnp.exp(
      -0.5 * ((grids - center) / sigma) ** 2) / sigma


def soft_coulomb(
    displacements, soften_factor=constants.SOFT_COULOMB_SOFTEN_FACTOR):
  """Soft Coulomb interaction.

  Args:
    displacements: Float numpy array.
    soften_factor: Float, soften factor in soft coulomb.

  Returns:
    Float numpy array with the same shape of displacements.
  """
  return 1 / jnp.sqrt(displacements ** 2 + soften_factor)


def exponential_coulomb(
    displacements,
    amplitude=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
    kappa=constants.EXPONENTIAL_COULOMB_KAPPA):
  """Exponential Coulomb interaction.

  v(x) = amplitude * exp(-abs(x) * kappa)

  1d interaction described in
  One-dimensional mimicking of electronic structure: The case for exponentials.
  Physical Review B 91.23 (2015): 235141.
  https://arxiv.org/pdf/1504.05620.pdf

  This potential is used in
  Pure density functional for strong correlation and the thermodynamic limit
  from machine learning.
  Physical Review B 94.24 (2016): 245129.
  https://arxiv.org/pdf/1609.03705.pdf

  Args:
    displacements: Float numpy array.
    amplitude: Float, parameter of exponential Coulomb interaction.
    kappa: Float, parameter of exponential Coulomb interaction.

  Returns:
    Float numpy array with the same shape of displacements.
  """
  return amplitude * jnp.exp(-jnp.abs(displacements) * kappa)


def get_atomic_chain_potential(
    grids, locations, nuclear_charges, interaction_fn):
  """Gets atomic chain potential.

  Args:
    grids: Float numpy array with shape (num_grids,).
    locations: Float numpy array with shape (num_nuclei,),
        the locations of the nuclei.
    nuclear_charges: Float numpy array with shape (num_nuclei,),
        the charges of nuclei.
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.

  Returns:
    Float numpy array with shape (num_grids,).

  Raises:
    ValueError: If grids.ndim, locations.ndim or nuclear_charges.ndim is not 1.
  """
  if grids.ndim != 1:
    raise ValueError(
        'grids.ndim is expected to be 1 but got %d' % grids.ndim)
  if locations.ndim != 1:
    raise ValueError(
        'locations.ndim is expected to be 1 but got %d' % locations.ndim)
  if nuclear_charges.ndim != 1:
    raise ValueError(
        'nuclear_charges.ndim is expected to be 1 but got %d'
        % nuclear_charges.ndim)
  displacements = jnp.expand_dims(
      grids, axis=0) - jnp.expand_dims(locations, axis=1)
  return jnp.dot(-nuclear_charges, interaction_fn(displacements))


def get_nuclear_interaction_energy(locations, nuclear_charges, interaction_fn):
  """Gets nuclear interaction energy for atomic chain.

  Args:
    locations: Float numpy array with shape (num_nuclei,),
        the locations of the nuclei.
    nuclear_charges: Float numpy array with shape (num_nuclei,),
        the charges of nuclei.
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.

  Returns:
    Float.

  Raises:
    ValueError: If locations.ndim or nuclear_charges.ndim is not 1.
  """
  if locations.ndim != 1:
    raise ValueError(
        'locations.ndim is expected to be 1 but got %d' % locations.ndim)
  if nuclear_charges.ndim != 1:
    raise ValueError(
        'nuclear_charges.ndim is expected to be 1 but got %d'
        % nuclear_charges.ndim)
  # Convert locations and nuclear_charges to jax.numpy array.
  locations = jnp.array(locations)
  nuclear_charges = jnp.array(nuclear_charges)
  indices_0, indices_1 = jnp.triu_indices(locations.size, k=1)
  charges_products = nuclear_charges[indices_0] * nuclear_charges[indices_1]
  return jnp.sum(
      charges_products * interaction_fn(
          locations[indices_0] - locations[indices_1]))


def get_nuclear_interaction_energy_batch(
    locations, nuclear_charges, interaction_fn):
  """Gets nuclear interaction energy for atomic chain in batch.

  Args:
    locations: Float numpy array with shape (batch_size, num_nuclei),
        the locations of the nuclei.
    nuclear_charges: Float numpy array with shape (batch_size, num_nuclei),
        the charges of nuclei.
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.

  Returns:
    Float numpy array with shape (batch_size,).
  """
  return jax.vmap(get_nuclear_interaction_energy, in_axes=(0, 0, None))(
      locations, nuclear_charges, interaction_fn)


def _float_value_in_array(value, array, atol=1e-7):
  return any(jnp.abs(array - value) <= atol)


def flip_and_average(locations, grids, array):
  """Flips and average the array around center.

  This symmetry is applied to system with reflection symmetry, e.g. H2 or single
  H atom, to improve the convergence.

  Args:
    locations: Float numpy array with shape (num_nuclei,), the average of the
        locations are the center to filp the array.
    grids: Float numpy array with shape (num_grids,).
    array: Float numpy array with shape (num_grids,).

  Returns:
    Float numpy array with shape (num_grids,).

  Raises:
    ValueError: If any location is not on the grids.
  """
  for location in locations:
    if not _float_value_in_array(location, grids):
      raise ValueError('Location %4.2f is not on the grids.' % location)

  center = jnp.mean(locations)
  if _float_value_in_array(center, grids):
    # The center is on the grids.
    center_index = jnp.argmin(jnp.abs(grids - center))
    left_index = center_index
    right_index = center_index
  else:
    # The center is in the middle of two grid points.
    abs_distance_to_center = jnp.abs(grids - center)
    left_index = jnp.argmin(
        jnp.where(grids < center, abs_distance_to_center, jnp.inf))
    right_index = jnp.argmin(
        jnp.where(grids > center, abs_distance_to_center, jnp.inf))
  radius = min([left_index, len(grids) - right_index - 1])
  range_slice = slice(left_index - radius, right_index + radius + 1)
  array_to_flip = array[range_slice]
  return jax.ops.index_update(
      array,
      idx=range_slice,
      y=(array_to_flip + jnp.flip(array_to_flip)) / 2)


def location_center_at_grids_center_point(locations, grids):
  """Checks whether the center of the location is at the center of the grids.

  Args:
    locations: Float numpy array with shape (batch_size, num_nuclei),
        the average of the locations are the center to filp the density.
    grids: Float numpy array with shape (num_grids,).

  Returns:
    Boolean.
  """
  num_grids = grids.shape[0]
  return bool(
      # If num_grids is odd, there is no single center point on the grids.
      num_grids % 2
      and jnp.abs(jnp.mean(locations) - grids[num_grids // 2]) < 1e-8)


def compute_distances_between_nuclei(locations, nuclei_indices):
  """Computes the distances between nuclei.

  Args:
    locations: Float numpy array with shape (batch_size, num_nuclei),
        the locations of the nuclei.
    nuclei_indices: Tuple of two integers, the indices of nuclei to compute
        distance.

  Returns:
    Float numpy array with shape (batch_size,).

  Raises:
    ValueError: If the ndim of locations is not 2, or the size of
        nuclei_indices is not 2.
  """
  if locations.ndim != 2:
    raise ValueError(
        'The ndim of locations is expected to be 2 but got %d' % locations.ndim)
  size = len(nuclei_indices)
  if size != 2:
    raise ValueError(
        'The size of nuclei_indices is expected to be 2 but got %d' % size)
  return jnp.abs(
      locations[:, nuclei_indices[0]] - locations[:, nuclei_indices[1]])
