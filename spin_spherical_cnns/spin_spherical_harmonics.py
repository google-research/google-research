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

"""Spin-weighted spherical harmonics (SWSH) transforms in JAX.

Contains the fast, TPU-friendly, JAX implementation of the SWSFT. See
np_spin_spherical_harmonics.py for the slow numpy version.

The major difference is that here we avoid keeping lists of coefficients and
Wigner Deltas. In np_spin_spherical_harmonics.py, the arrangement concatenates
the 2*ell+1 coefficients per degree as a 1D array. For example, for ell_max=2 we
have coeffs=[c00, c1m1, c10, c11, c2m2, c2m1, c20, c21, c22]. Here in
spin_spherical_harmonics.py, each degree is zero-padded to make a (ell_max+1,
2*ell_max+1) matrix:
[0    0    c00 0   0  ]
[0    c1m1 c10 c11 0  ]
[c2m2 c2m1 c20 c21 c22]
The Wigner Deltas follow the same idea -- in np_spin_spherical_harmonics.py they
were lists of matrices, here they are zero-padded 3D arrays. This data structure
uses up to 3x as much memory but is 2x faster according to initial experiments
applying the forward SWSFT to (32, 32) inputs.

For efficiency, we use a SpinSphericalFourierTransformer instance to store all
constants needed in the forward and backward transforms, so the constants can be
computed once per model and reused. The instance can contain multiple
resolutions and spins, as required by a SWSCNN model, and also encapsulates the
forward and inverse transforms.
"""

import functools
from typing import Collection, Optional, Union
import jax
import jax.numpy as jnp
import numpy as np

from spin_spherical_cnns import sphere_utils

Array = Union[np.ndarray, jnp.ndarray]


class SpinSphericalFourierTransformer:
  r"""Handles spin-weighted spherical Fourier transforms (SWSFT).

  The forward and backward transforms use a number of constants that are better
  cached. This class stores constants and handles transforms for multiple
  different resolutions and spins, as required by a SWSCNN model. The idea is to
  instantiate a single object per model.

  For exampe, for a SWSCNN with three layers of resolutions 64x64, 32x32, 16x16,
  and spins per layer {0}, {0,1,2}, {1}, we would instantiate the constants as
  >>> transformer = SpinSphericalFourierTransformer([64, 32, 16], [0, 1, 2])
  and evaluate a transform for some input as
  >>> coefficients = transformer.swsft_forward(input, spin).

  Attributes:
    wigner_deltas: Zero-padded (ell_max+1, ell_max+1, 2*ell_max+1) array
      with stacked Wigner Deltas. Element at (ell, n, m) corresponds to
      \Delta_{n,m}^\ell. See also: sphere_utils.compute_all_wigner_delta().
    quadrature_weights: dict mapping resolutions (int) to quadrature weights
      as computed by torus_quadrature_weights().
    swsft_forward_constants: dict with spins as keys; entries are zero-padded
        (ell_max+1, 2*ell_max+1) arrays of per coefficient constants applied in
        the SWSFT transform. See also: sphere_utils.swsft_forward_constant().
  """

  def __init__(self,
               resolutions = None,
               spins = None):
    """Pre-compute constants.

    If resolutions and spins are not None, compute all constants. Otherwise do
    nothing. The latter behavior is used in set_attributes(), which is needed to
    make SpinSphericalFourierTransformer a jit-able JAX type.

    Args:
      resolutions: List of spherical resolutions.
      spins: List of spin-weights.

    Returns:
      None.
    """

    if resolutions is not None and spins is not None:
      self._compute_constants(resolutions, spins)

  def get_attributes(self):
    """Gets attributes. Needed to ensure ordering when using as JAX type."""
    return (self.wigner_deltas,
            self.quadrature_weights,
            self.swsft_forward_constants)

  @classmethod
  def set_attributes(cls,
                     wigner_deltas,
                     quadrature_weights,
                     swsft_forward_constants):
    """Sets attributes. Needed to ensure ordering when using as JAX type."""
    constants = cls()
    constants.wigner_deltas = wigner_deltas
    constants.quadrature_weights = quadrature_weights
    constants.swsft_forward_constants = swsft_forward_constants

    return constants

  def validate(self, resolution, spin):
    """Returns True iff constants are valid for given resolution and spin."""
    if int(spin) not in self.swsft_forward_constants.keys():
      return False
    if resolution not in self.quadrature_weights.keys():
      return False
    if resolution // 2 > self.wigner_deltas.shape[0]:
      return False

    return True

  def _slice_wigner_deltas(self, ell_max):
    """Returns sliced wigner_deltas as if max degree were ell_max."""
    middle = self.wigner_deltas.shape[0] - 1
    return self.wigner_deltas[:ell_max + 1,
                              :ell_max + 1,
                              (middle-ell_max):(middle+ell_max+1)]

  def _slice_forward_constants(self, ell_max, spin):
    """Returns sliced swsft_forward_constants as if max degree were ell_max."""
    forward_constants = self.swsft_forward_constants[int(spin)]
    middle = forward_constants.shape[0] - 1
    return forward_constants[:ell_max + 1, (middle-ell_max):(middle+ell_max+1)]

  def _compute_constants(self, resolutions, spins):
    """Computes constants (class attributes). See constructor docstring."""
    ells = [sphere_utils.ell_max_from_resolution(res) for res in resolutions]
    ell_max = max(ells)
    wigner_deltas = sphere_utils.compute_all_wigner_delta(ell_max)
    padded_deltas = []
    for ell, delta in enumerate(wigner_deltas):
      padded_deltas.append(jnp.pad(delta,
                                   ((0, ell_max - ell),
                                    (ell_max - ell, ell_max - ell))))
    self.wigner_deltas = jnp.stack(padded_deltas)

    self.quadrature_weights = {
        res: jnp.array(sphere_utils.torus_quadrature_weights(res))
        for res in resolutions}

    self.swsft_forward_constants = {}
    for spin in spins:
      constants_spin = []
      for ell in range(ell_max + 1):
        k_ell = sphere_utils.swsft_forward_constant(spin, ell,
                                                    jnp.arange(-ell, ell+1))
        k_ell = jnp.asarray(k_ell)
        constants_spin.append(k_ell)
      self.swsft_forward_constants[spin] = coefficients_to_matrix(
          jnp.concatenate(constants_spin))

  def _extend_sphere_fft(self, sphere, spin):
    """See np_spin_spherical_harmonics._extend_sphere_fft()."""
    n = sphere.shape[1]
    if n % 2 != 0:
      raise ValueError("Input sphere must have even height!")
    torus = (-1.0)**spin * jnp.roll(sphere[1:-1][::-1], n // 2, axis=1)
    torus = jnp.concatenate([sphere, torus], axis=0)
    weights = self.quadrature_weights[n]
    torus = weights[:, None] * torus
    coeffs = _fft2(torus) * 2 * jnp.pi / n

    return coeffs

  def _compute_Inm(self, sphere, spin):  # pylint: disable=invalid-name
    """See np_spin_spherical_harmonics._compute_Inm()."""
    ell_max = sphere_utils.ell_max_from_resolution(sphere.shape[0])
    coeffs = self._extend_sphere_fft(sphere, spin)

    rows1 = jnp.concatenate([coeffs[:ell_max + 1, :ell_max + 1],
                             coeffs[:ell_max + 1, -ell_max:]],
                            axis=1)
    rows2 = jnp.concatenate([coeffs[-ell_max:, :ell_max + 1],
                             coeffs[-ell_max:, -ell_max:]],
                            axis=1)

    return jnp.concatenate([rows1, rows2], axis=0)

  def _compute_Jnm(self, sphere, spin):  # pylint: disable=invalid-name
    """See np_spin_spherical_harmonics._compute_Jnm()."""
    ell_max = sphere_utils.ell_max_from_resolution(sphere.shape[0])
    Inm = self._compute_Inm(sphere, spin)  # pylint: disable=invalid-name

    # Make a matrix with (-1)^{m+s} columns.
    m = jnp.concatenate(
        [jnp.arange(ell_max + 1), -jnp.arange(ell_max + 1)[1:][::-1]])[None]
    signs = (-1.)**(m + spin)

    # Jnm only contains positive n.
    Jnm = Inm[:ell_max + 1]  # pylint: disable=invalid-name
    # Make n = -n rowwise.
    return Jnm.at[1:].add(signs * Inm[-ell_max:][::-1])

  def swsft_forward(self, sphere, spin):
    """Spin-weighted spherical harmonics transform (fast JAX version).

    Returns coefficients in zero-padded format:
    [0    0    c00 0   0  ]
    [0    c1m1 c10 c11 0  ]
    [c2m2 c2m1 c20 c21 c22]

    See also: np_spin_spherical_harmonics.swsft_forward_naive().

    Args:
      sphere: A (n, n) array representing a spherical function. Equirectangular
        sampling, lat, long order.
      spin: Spin weight.

    Returns:
      A (n//2, n-1) array of complex64 coefficients. The coefficient at degree
      ell and order m is at position [ell, ell_max+m].
    """
    if not self.validate(resolution=sphere.shape[0], spin=spin):
      raise ValueError("Constants are invalid for given input!")

    ell_max = sphere_utils.ell_max_from_resolution(sphere.shape[0])
    Jnm = self._compute_Jnm(sphere, spin)  # pylint: disable=invalid-name
    Jnm = jnp.concatenate([Jnm[:, -ell_max:], Jnm[:, :ell_max+1]], axis=1)  # pylint: disable=invalid-name
    deltas = self._slice_wigner_deltas(ell_max)
    deltas = deltas * deltas[Ellipsis, ell_max - spin][Ellipsis, None]
    forward_constants = self._slice_forward_constants(ell_max, spin)

    return jnp.einsum("ik,ijk,jk->ik", forward_constants, deltas, Jnm)

  def _compute_Gnm(self, coeffs, spin):  # pylint: disable=invalid-name
    """Computes Gnm (vectorized).

    See np_spin_spherical_harmonics._compute_Gnm_naive() for details.

    Args:
      coeffs: see swsft_backward().
      spin: see swsft_backward().

    Returns:
      A (2*ell_max+1, 2*ell_max+1) complex64 matrix, when coeffs has ell_max+1
      rows.
    """
    ell_max = coeffs.shape[0] - 1
    # Backward constants relates to forward via these signs.
    signs = (-1.)**(spin + jnp.arange(-ell_max, ell_max+1))[None, :]
    backward_constant = self._slice_forward_constants(ell_max, spin) * signs
    deltas = self._slice_wigner_deltas(ell_max)
    deltas_s = deltas[Ellipsis, ell_max - spin]

    # Bottom half of Gnm, since constants only store half of the Wigner Deltas.
    bottom_half = jnp.einsum("lm,lnm,ln->nm",
                             backward_constant * coeffs,
                             deltas,
                             deltas_s)
    # Use symmetry to complete the top half.
    return jnp.concatenate([bottom_half[1:][::-1] * signs,
                            bottom_half])

  def swsft_backward(self, coeffs, spin):
    """Inverse spin-weighted spherical harmonics transform (fast JAX version).

    See also np_spin_spherical_harmonics.swsft_backward_naive().

    Args:
      coeffs: An (ell_max+1, 2*ell_max+1) array of SWSFT coefficients, with the
        triangular zero-padded structure output by swsft_forward().
      spin: Spin weight.

    Returns:
      A (2*ell_max + 2, 2*ell_max + 2) complex64 array with equiangular sampling
      (H&W) of a spin-weighted spherical function.
    """
    ell_max = coeffs.shape[0] - 1
    if not self.validate(resolution=2*(ell_max + 1), spin=spin):
      raise ValueError("Constants are invalid for given input!")

    # Gnm is related to the 2D Fourier transform of the ISWSFT.
    ft = self._compute_Gnm(coeffs, spin)
    ft = jnp.fft.ifftshift(ft)

    # Zero-pad to final dimension before FFT. This is the torus extension, so we
    # only want the top half.
    ft = jnp.concatenate([ft[:, :ell_max+1],
                          jnp.zeros((2*ell_max+1, 1)),
                          ft[:, ell_max+1:]], axis=1)
    ft = jnp.concatenate([ft[:ell_max+1],
                          jnp.zeros_like(ft),
                          ft[ell_max+1:]])

    # Return the top half.
    return _ifft2(ft)[:2*(ell_max+1)] * ft.size

  @functools.partial(jax.vmap, in_axes=(None, -1, None), out_axes=-1)
  def swsft_forward_spins_channels(self,
                                   sphere_set,
                                   spins):
    """Applies swsft_forward() to multiple stacked spins and channels.

    Args:
      sphere_set: An (n, n, n_spins, n_channels) array representing a spherical
        functions. Equirectangular sampling, leading dimensions are lat, long.
      spins: An (n_spins,) list of int spin weights.

    Returns:
      An (n//2, n-1, n_spins, n_channels) complex64 array of coefficients.
    """
    return jnp.stack([self.swsft_forward(sphere_set[Ellipsis, i], spin)
                      for i, spin in enumerate(spins)], axis=-1)

  @functools.partial(jax.vmap, in_axes=(None, -1, None), out_axes=-1)
  def swsft_backward_spins_channels(self,
                                    coeffs_set,
                                    spins):
    """Applies swsft_backward() to multiple stacked spins and channels.

    Args:
      coeffs_set: An (ell_max+1, 2*ell_max+1, n_spins, n_channels) array of
        SWSFT coefficients.
      spins: An (n_spins,) list of int spin weights.

    Returns:
      A (2*ell_max + 2, 2*ell_max + 2, n_spins, n_channels) complex64 array of
      spin-weighted spherical functions.
    """
    return jnp.stack([self.swsft_backward(coeffs_set[Ellipsis, i], spin)
                      for i, spin in enumerate(spins)], axis=-1)

# This makes SpinSphericalFourierTransformer a jit-able JAX type. See
# https://github.com/google/jax/issues/806 for discussion.
jax.tree_util.register_pytree_node(
    SpinSphericalFourierTransformer,
    lambda cls: (cls.get_attributes(), None),
    lambda _, x: SpinSphericalFourierTransformer.set_attributes(*x))


def coefficients_to_matrix(coeffs):
  """Converts 1D array of coefficients to a 2D array, padding with zeros.

  For input [c00, c1m1, c10, c11, c2m2, c2m1, c20, c21, c22], this returns:
  [0    0    c00 0   0  ]
  [0    c1m1 c10 c11 0  ]
  [c2m2 c2m1 c20 c21 c22]

  Args:
    coeffs: List of n**2 SWSFT coefficients.

  Returns:
    A (n, 2n-1) array with one coefficient in the center of the first row, three
    in the second row, etcetera (see example above) and padded with zeros.
  """
  ell_max = sphere_utils.ell_max_from_n_coeffs(len(coeffs))
  matrix = []
  for ell in range(ell_max + 1):
    matrix.append(jnp.pad(coeffs[ell**2:(ell+1)**2],
                          ((ell_max - ell, ell_max - ell))))

  return jnp.stack(matrix)


def _fft2(x):
  """Computes the 2D FFT (because JAX does not have a 2D FFT TPU kernel)."""
  rowwise = jnp.fft.fft(x, axis=0)
  return jnp.fft.fft(rowwise, axis=1)


def _ifft2(x):
  """Computes the 2D IFFT (because JAX does not have a 2D FFT TPU kernel)."""
  rowwise = jnp.fft.ifft(x, axis=0)
  return jnp.fft.ifft(rowwise, axis=1)
