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

"""Test utilities."""

import dataclasses
import functools
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import sympy.physics.wigner

from spin_spherical_cnns import np_spin_spherical_harmonics
from spin_spherical_cnns import sphere_utils
from spin_spherical_cnns import spin_spherical_harmonics

Array = Union[np.ndarray, jnp.ndarray]


@functools.lru_cache()
def _stacked_wigner_ds(ell_max,
                       alpha, beta, gamma):
  """Stacks all Wigner-Ds for ell <= ell_max."""
  if beta == 0.0:
    raise ValueError('beta cannot be zero! sympy returns NaNs for beta==0.')

  sympy_wigner_ds = [sympy.physics.wigner.wigner_d(ell, alpha, beta, gamma)
                     for ell in range(ell_max + 1)]
  list_wigner_ds = []
  for ell, w in enumerate(sympy_wigner_ds):
    list_wigner_ds.append(jnp.pad(np.array(w).astype(np.complex64),
                                  ((ell_max - ell, ell_max - ell),
                                   (ell_max - ell, ell_max - ell))))
  return jnp.stack(list_wigner_ds, axis=0)


def rotate_coefficients(coefficients,
                        alpha,
                        beta,
                        gamma):
  """Rotates stacked coefficients by given Euler angles.

  Args:
    coefficients: An (..., ell_max, 2*ell_max+1, num_spins, num_channels) array
      of stacked coefficients.
    alpha: First Euler angle (ZYZ).
    beta: Second Euler angle (ZYZ).
    gamma: Third Euler angle (ZYZ).

  Returns:
    Rotated coefficient array with the same shape as `coefficients`.
  """
  *_, num_ell, _, _, _ = coefficients.shape
  ell_max = num_ell - 1
  wigner_ds = _stacked_wigner_ds(ell_max, alpha, beta, gamma)
  return jnp.einsum('lmn,...lnsc->...lmsc', wigner_ds, coefficients)


@dataclasses.dataclass
class RotatedPair:
  coefficients: jnp.ndarray
  sphere: jnp.ndarray
  rotated_coefficients: jnp.ndarray
  rotated_sphere: jnp.ndarray


def get_spin_spherical(
    transformer,
    shape,
    spins):
  """Returns set of spin-weighted spherical functions.

  Args:
    transformer: SpinSphericalFourierTransformer instance.
    shape: Desired shape (batch, latitude, longitude, spins, channels).
    spins: Desired spins.

  Returns:
    Array of spherical functions and array of their spectral coefficients.
  """
  # Make some arbitrary reproducible complex inputs.
  batch_size, resolution, _, num_spins, num_channels = shape
  if len(spins) != num_spins:
    raise ValueError('len(spins) must match desired shape.')
  ell_max = sphere_utils.ell_max_from_resolution(resolution)
  num_coefficients = np_spin_spherical_harmonics.n_coeffs_from_ell_max(ell_max)
  shape_coefficients = (batch_size, num_spins, num_channels, num_coefficients)
  # These numbers are chosen arbitrarily, but not randomly, since random
  # coefficients make for hard to visually interpret functions. Something
  # simpler like linspace(-1-1j, 1+1j) would have the same phase for all complex
  # numbers, which is also undesirable.
  coefficients = (jnp.linspace(-0.5, 0.7 + 0.5j,
                               np.prod(shape_coefficients))
                  .reshape(shape_coefficients))

  # Broadcast
  to_matrix = jnp.vectorize(spin_spherical_harmonics.coefficients_to_matrix,
                            signature='(i)->(j,k)')
  coefficients = to_matrix(coefficients)
  # Transpose back to (batch, ell, m, spin, channel) format.
  coefficients = jnp.transpose(coefficients, (0, 3, 4, 1, 2))

  # Coefficients for ell < |spin| are always zero.
  for i, spin in enumerate(spins):
    coefficients = coefficients.at[:, :abs(spin), :, i].set(0.0)

  # Convert to spatial domain.
  batched_backward_transform = jax.vmap(
      transformer.swsft_backward_spins_channels, in_axes=(0, None))
  sphere = batched_backward_transform(coefficients, spins)

  return sphere, coefficients


def get_rotated_pair(
    transformer,
    shape,
    spins,
    alpha,
    beta,
    gamma):
  """Returns pair of bandlimited rotated spin-weighted spherical functions.

  Useful for equivariance tests. The second output is rotated by (alpha, beta,
  gamma) with respect to the first.

  Args:
    transformer: SpinSphericalFourierTransformer instance.
    shape: Desired shape (batch, latitude, longitude, spins, channels).
    spins: Desired spins.
    alpha: First Euler angle (ZYZ).
    beta: Second Euler angle (ZYZ).
    gamma: Third Euler angle (ZYZ).

  Returns:
    RotatedPair of spherical functions and their spectral coefficients.
  """
  sphere, coefficients = get_spin_spherical(transformer, shape, spins)
  # Rotate in the spectral domain and invert again.
  rotated_coefficients = rotate_coefficients(coefficients, alpha, beta, gamma)

  # Convert to spatial domain.
  batched_backward_transform = jax.vmap(
      transformer.swsft_backward_spins_channels, in_axes=(0, None))
  rotated_sphere = batched_backward_transform(rotated_coefficients, spins)

  return RotatedPair(coefficients, sphere,
                     rotated_coefficients, rotated_sphere)


def apply_model_to_rotated_pairs(
    transformer,
    model,
    resolution,
    spins,
    init_args = None,
    apply_args = None,
):
  """Applies model to rotated pair of inputs and returns rotated coefficients.

  This is useful to evaluate equivariance errors. The model is initialized and
  applied to a pair of rotated inputs. The outputs are converted to spectral
  domain and one set of coefficients is rotated into the other. If the model is
  equivariant, both sets of output coefficients must match.

  Args:
    transformer: transformer: SpinSphericalFourierTransformer instance.
    model: linen module to evaluate.
    resolution: input spherical grid is (resolution, resolution).
    spins: A sequence of (n_spins,) input and output spin weights.
    init_args: extra arguments for `model.init`.
    apply_args: extra arguments for `model.apply`.

  Returns:
    rotated_coefficients_1: coefficient array of rotate(model(input)).
    rotated_coefficients_2: coefficient array of model(rotate(input)).
    pair: input pair and its coefficients.
  """
  if init_args is None:
    init_args = {}
  if apply_args is None:
    apply_args = {}

  key = np.array([0, 0], dtype=np.uint32)
  shape = (2, resolution, resolution, len(spins), 2)
  alpha, beta, gamma = 1.0, 2.0, 3.0
  pair = get_rotated_pair(transformer,
                          shape,
                          spins,
                          alpha, beta, gamma)

  params = model.init(key, pair.sphere, **init_args)
  # `apply` returns either `output` or `(output, vars)`.
  output = model.apply(params, pair.sphere, **apply_args)
  rotated_output = model.apply(params, pair.rotated_sphere, **apply_args)
  if isinstance(output, tuple):
    output, _ = output
    rotated_output, _ = rotated_output

  batched_transform = jax.vmap(transformer.swsft_forward_spins_channels,
                               in_axes=(0, None))

  coefficients = batched_transform(output, spins)
  rotated_coefficients_1 = rotate_coefficients(
      coefficients, alpha, beta, gamma)
  rotated_coefficients_2 = batched_transform(rotated_output, spins)

  return rotated_coefficients_1, rotated_coefficients_2, pair


def apply_model_to_azimuthally_rotated_pairs(
    transformer,
    model,
    resolution,
    spins,
    shift,
    init_args = None,
    apply_args = None,
):
  """Applies model to rotated pair of inputs and returns rotated outputs.

  Useful for equivariance tests where interpolations due to arbitrary rotations
  cause large errors. Azimuthal rotations by integer shifts can be performed
  exactly.

  The model is initialized and applied to a pair of azimuthally rotated
  inputs. One output is rotated into the other. If the model is azimuthally
  rotation equivariant, outputs must match.

  Args:
    transformer: transformer: SpinSphericalFourierTransformer instance.
    model: linen module to evaluate.
    resolution: input spherical grid is (resolution, resolution).
    spins: A sequence of (n_spins,) input and output spin weights.
    shift: Azimuthal rotation, in pixels.
    init_args: extra arguments for `model.init`.
    apply_args: extra arguments for `model.apply`.

  Returns:
    output: result of rotate(model(input)).
    rotated_output: result of model(rotate(input)).
  """
  if init_args is None:
    init_args = {}
  if apply_args is None:
    apply_args = {}

  key = np.array([0, 0], dtype=np.uint32)
  shape = (2, resolution, resolution, len(spins), 2)
  sphere, _ = get_spin_spherical(transformer, shape, spins)
  rotated_sphere = jnp.roll(sphere, shift, axis=2)

  params = model.init(key, sphere, **init_args)
  # `apply` returns either `output` or `(output, vars)`.
  output = model.apply(params, sphere, **apply_args)
  rotated_output = model.apply(params, rotated_sphere, **apply_args)
  if isinstance(output, tuple):
    output, _ = output
    rotated_output, _ = rotated_output

  # If there was subsampling, change shift accordingly.
  stride = resolution // output.shape[1]
  output = jnp.roll(output, shift // stride, axis=2)

  return output, rotated_output
