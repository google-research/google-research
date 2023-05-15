# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

from typing import Collection, Optional, Tuple, Union
from flax import linen as nn
import jax.numpy as jnp
import numpy as np

from spin_spherical_cnns import sphere_utils

Array = Union[np.ndarray, jnp.ndarray]


class SpinSphericalFourierTransformer(nn.Module):
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
    resolutions: List of spherical resolutions.
    spins: List of spin-weights.
  """
  resolutions: Tuple[int, Ellipsis]
  spins: Tuple[int, Ellipsis]

  def __call__(self):
    return

  def setup(self):
    self._compute_constants(self.resolutions, self.spins)

  def validate(self, resolution, spin):
    """Returns True iff constants are valid for given resolution and spin."""
    if str(spin) not in self.swsft_forward_constants.value.keys():
      return False
    if str(resolution) not in self.quadrature_weights.value.keys():
      return False
    if resolution // 2 > self.wigner_deltas.value.shape[0]:
      return False

    return True

  def _slice_wigner_deltas(self, ell_max, include_negative_m=False):
    """Returns sliced wigner_deltas as if max degree were ell_max."""
    middle = self.wigner_deltas.value.shape[0] - 1
    if include_negative_m:
      m_indices = slice(middle-ell_max, middle+ell_max+1)
    else:
      m_indices = slice(middle, middle+ell_max+1)
    return self.wigner_deltas.value[:ell_max + 1,
                                    m_indices,
                                    (middle-ell_max):(middle+ell_max+1)]

  def _slice_forward_constants(self, ell_max, spin):
    """Returns sliced swsft_forward_constants as if max degree were ell_max."""
    forward_constants = self.swsft_forward_constants.value[str(spin)]
    middle = forward_constants.shape[0] - 1
    return forward_constants[:ell_max + 1, (middle-ell_max):(middle+ell_max+1)]

  def _compute_constants(self, resolutions, spins):
    """Computes constants as `nn.Module.variable`."""
    ells = [sphere_utils.ell_max_from_resolution(res) for res in resolutions]
    ell_max = max(ells)
    wigner_deltas = sphere_utils.compute_all_wigner_delta(ell_max)
    padded_deltas = []
    for ell, delta in enumerate(wigner_deltas):
      padded_deltas.append(jnp.pad(delta,
                                   ((ell_max - ell, ell_max - ell),
                                    (ell_max - ell, ell_max - ell))))
    self.wigner_deltas = self.variable("constants",
                                       "wigner_deltas",
                                       lambda: jnp.stack(padded_deltas))

    def quad_init():
      return {str(r): jnp.array(sphere_utils.torus_quadrature_weights(r))
              for r in resolutions}
    self.quadrature_weights = self.variable(
        "constants", "quadrature_weights", quad_init)

    swsft_forward_constants = {}
    for spin in spins:
      constants_spin = []
      for ell in range(ell_max + 1):
        k_ell = sphere_utils.swsft_forward_constant(spin, ell,
                                                    jnp.arange(-ell, ell+1))
        k_ell = jnp.asarray(k_ell)
        constants_spin.append(k_ell)
      swsft_forward_constants[str(spin)] = coefficients_to_matrix(
          jnp.concatenate(constants_spin))
    self.swsft_forward_constants = self.variable(
        "constants",
        "swsft_forward_constants",
        lambda: swsft_forward_constants)

  def _extend_sphere_fft(self, sphere, spin):
    """See np_spin_spherical_harmonics._extend_sphere_fft()."""
    n = sphere.shape[1]
    if n % 2 != 0:
      raise ValueError("Input sphere must have even height!")
    torus = (-1.0)**spin * jnp.roll(sphere[1:-1][::-1], n // 2, axis=1)
    torus = jnp.concatenate([sphere, torus], axis=0)
    weights = self.quadrature_weights.value[str(n)]
    torus = jnp.einsum("i,i...->i...", weights, torus)
    coeffs = _fourier_transform_2d(torus) * 2 * jnp.pi / n

    return coeffs

  def _compute_Inm(self, sphere, spin, ell_max=None):  # pylint: disable=invalid-name
    """See np_spin_spherical_harmonics._compute_Inm()."""
    if ell_max is None:
      ell_max = sphere_utils.ell_max_from_resolution(sphere.shape[0])
    coeffs = self._extend_sphere_fft(sphere, spin)

    # Disable the type check here due to bug in pylint
    # (https://github.com/PyCQA/astroid/issues/791).
    # pylint: disable=invalid-unary-operand-type
    rows1 = jnp.concatenate([coeffs[:ell_max + 1, :ell_max + 1],
                             coeffs[:ell_max + 1, -ell_max:]],
                            axis=1)
    rows2 = jnp.concatenate([coeffs[-ell_max:, :ell_max + 1],
                             coeffs[-ell_max:, -ell_max:]],
                            axis=1)
    # pylint: enable=invalid-unary-operand-type

    return jnp.concatenate([rows1, rows2], axis=0)

  def _compute_Jnm(self, sphere, spin):  # pylint: disable=invalid-name
    """See np_spin_spherical_harmonics._compute_Jnm()."""
    return self._compute_Jnm_spins_channels(jnp.expand_dims(sphere, [2, 3]),
                                            [spin])[Ellipsis, 0, 0]

  def _compute_Jnm_spins_channels(self, sphere_set, spins):  # pylint: disable=invalid-name
    """Computes Jnm over different spins and channels."""
    ell_max = sphere_utils.ell_max_from_resolution(sphere_set.shape[0])
    expanded_spins = jnp.expand_dims(jnp.array(spins), [0, 1, 3])
    Inm = self._compute_Inm(sphere_set, expanded_spins)  # pylint: disable=invalid-name

    # Make a matrix with (-1)^{m+s} columns.
    m = jnp.concatenate(
        [jnp.arange(ell_max + 1), -jnp.arange(ell_max + 1)[1:][::-1]])
    signs = (-1.)**(jnp.expand_dims(m, (0, 2, 3)) +
                    expanded_spins)

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
    # This version uses more operations overall but is usually faster
    # on TPU due to less overhead.
    coefficients = self.swsft_forward_spins_channels(
        jnp.expand_dims(sphere, (2, 3)), [spin])
    return coefficients[Ellipsis, 0, 0]

  def swsft_forward_with_symmetry(
      self, sphere, spin):
    """Same as swsft, but with the intermediate Jnm computation."""
    # This version uses less operations overall but is usually slower
    # on TPU due to more overhead.
    if not self.validate(resolution=sphere.shape[0], spin=spin):
      raise ValueError("Constants are invalid for given input!")

    ell_max = sphere_utils.ell_max_from_resolution(sphere.shape[0])
    Jnm = self._compute_Jnm(sphere, spin)  # pylint: disable=invalid-name
    Jnm = jnp.concatenate([Jnm[:, -ell_max:], Jnm[:, :ell_max+1]], axis=1)  # pylint: disable=invalid-name
    deltas = self._slice_wigner_deltas(ell_max, include_negative_m=False)
    deltas = deltas * deltas[Ellipsis, ell_max - spin][Ellipsis, None]
    forward_constants = self._slice_forward_constants(ell_max, spin)

    return jnp.einsum("ik,ijk,jk->ik", forward_constants, deltas, Jnm)

  def _compute_Gnm_spins_channels(self,  # pylint: disable=invalid-name
                                  coeffs_set,
                                  spins):
    """Computes Gnm for multiple spins and channels.

    See np_spin_spherical_harmonics._compute_Gnm_naive() for details.

    Args:
      coeffs_set: An (ell_max+1, 2*ell_max+1, n_spins, n_channels) array of
        SWSFT coefficients.
      spins: An (n_spins,) list of int spin weights.

    Returns:
      A (2*ell_max+1, 2*ell_max+1, n_spins, n_channels) complex64
      matrix.
    """
    ell_max = coeffs_set.shape[0] - 1
    expanded_spins = jnp.expand_dims(jnp.array(spins), 0)
    # Backward constants relates to forward via these signs.
    signs = (-1.)**(expanded_spins +
                    jnp.expand_dims(jnp.arange(-ell_max, ell_max+1), 1))[None]
    constants = jnp.stack([self._slice_forward_constants(ell_max, spin)
                           for spin in spins], axis=-1) * signs
    deltas = self._slice_wigner_deltas(ell_max, include_negative_m=True)
    deltas_s = jnp.stack([deltas[Ellipsis, ell_max - spin]
                          for spin in spins], axis=-1)

    factors = jnp.einsum("lms,lnm,lns->lnms",
                         constants, deltas, deltas_s)
    return jnp.einsum("lnms,lmsc->nmsc",
                      factors, coeffs_set)

  def _compute_Gnm(self, coeffs, spin):  # pylint: disable=invalid-name
    """Computes Gnm for a single function. See `_compute_Gnm_spins_channels`."""
    return self._compute_Gnm_spins_channels(jnp.expand_dims(coeffs, [2, 3]),
                                            [spin])[Ellipsis, 0, 0]

  def _compute_Gnm_with_symmetry(self, coeffs, spin):  # pylint: disable=invalid-name
    """Same as `_compute_Gnm` but with fewer operations."""
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

  def _swsft_backward_base(self,
                           coeffs,
                           spin,
                           use_symmetry):
    """Implements `swsft_backward` and `swsft_backward_with_symmetry`."""
    ell_max = coeffs.shape[0] - 1
    if not self.validate(resolution=2*(ell_max + 1), spin=spin):
      raise ValueError("Constants are invalid for given input!")

    # Gnm is related to the 2D Fourier transform of the ISWSFT.
    if use_symmetry:
      ft = self._compute_Gnm_with_symmetry(coeffs, spin)
    else:
      ft = self._compute_Gnm(coeffs, spin)

    # Padding then shifting seem more efficient than the converse here.
    ft = jnp.pad(ft, [(ell_max+1, ell_max), (1, 0)])
    ft = jnp.fft.ifftshift(ft)

    # Since only half of the 2D IFFT is needed, it is faster to slice
    # after the first 1D IFFT.
    rowwise = jnp.fft.ifft(ft, axis=0)[:2*(ell_max + 1)]
    return jnp.fft.ifft(rowwise, axis=1) * ft.size

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
    return self._swsft_backward_base(coeffs, spin, use_symmetry=False)

  def swsft_backward_with_symmetry(
      self, coeffs, spin):
    """Same as `swsft_backward` but with fewer operations."""
    return self._swsft_backward_base(coeffs, spin, use_symmetry=True)

  def swsft_forward_spins_channels(
      self,
      sphere_set,
      spins,
      ell_max = None):
    """Applies swsft_forward() to multiple stacked spins and channels.

    Args:
      sphere_set: An (n, n, n_spins, n_channels) array representing a spherical
        functions. Equirectangular sampling, leading dimensions are lat, long.
      spins: An (n_spins,) list of int spin weights.
      ell_max: Maximum output frequency ell. If None, use `n//2 - 1`.

    Returns:
      An (ell_max+1, n-1, n_spins, n_channels) complex64 array of coefficients.
    """
    for spin in spins:
      if not self.validate(resolution=sphere_set.shape[0], spin=spin):
        raise ValueError("Constants are invalid for given input!")

    if ell_max is None:
      ell_max = sphere_utils.ell_max_from_resolution(sphere_set.shape[0])

    expanded_spins = jnp.expand_dims(jnp.array(spins), [0, 1, 3])
    Inm = jnp.fft.fftshift(  # pylint: disable=invalid-name
        self._compute_Inm(sphere_set, expanded_spins, ell_max=ell_max),
        axes=(0, 1))

    deltas = self._slice_wigner_deltas(ell_max, include_negative_m=True)
    deltas_s = jnp.stack([deltas[Ellipsis, ell_max - spin]
                          for spin in spins], axis=-1)
    forward_constants = jnp.stack([self._slice_forward_constants(ell_max, spin)
                                   for spin in spins], axis=-1)

    return jnp.einsum("lms,lnm,lns,nms...->lms...",
                      forward_constants, deltas, deltas_s, Inm)

  def swsft_forward_spins_channels_with_symmetry(
      self, sphere_set, spins):
    """Same as `swsft_forward_spins_channels`, but leveraging symmetry."""
    for spin in spins:
      if not self.validate(resolution=sphere_set.shape[0], spin=spin):
        raise ValueError("Constants are invalid for given input!")

    ell_max = sphere_utils.ell_max_from_resolution(sphere_set.shape[0])
    Jnm = self._compute_Jnm_spins_channels(sphere_set, spins)  # pylint: disable=invalid-name
    Jnm = jnp.concatenate([Jnm[:, -ell_max:], Jnm[:, :ell_max+1]], axis=1)  # pylint: disable=invalid-name

    deltas = self._slice_wigner_deltas(ell_max, include_negative_m=False)
    deltas_s = jnp.stack([deltas[Ellipsis, ell_max - spin]
                          for spin in spins], axis=-1)
    forward_constants = jnp.stack([self._slice_forward_constants(ell_max, spin)
                                   for spin in spins], axis=-1)

    return jnp.einsum("lms,lnm,lns,nms...->lms...",
                      forward_constants, deltas, deltas_s, Jnm)

  def swsft_backward_spins_channels(self,
                                    coeffs_set,
                                    spins):
    """Applies swsft_backward() to multiple stacked spins and channels."""
    ell_max = coeffs_set.shape[0] - 1
    for spin in spins:
      if not self.validate(resolution=2*(ell_max + 1), spin=spin):
        raise ValueError("Constants are invalid for given input!")

    ft = self._compute_Gnm_spins_channels(coeffs_set, spins)

    # Padding then shifting seem more efficient than the converse here.
    ft = jnp.pad(ft, [(ell_max+1, ell_max), (1, 0), (0, 0), (0, 0)])
    ft = jnp.fft.ifftshift(ft, axes=(0, 1))

    # For the dimensions we typically have here, computing the naive
    # FT by multiplying by the IDFT matrix is significantly faster
    # than the FFT (at least on TPU).
    idft_matrix = jnp.fft.ifft(jnp.eye(ft.shape[0]))
    # Since only half of the 2D IFFT is needed, it is faster to slice
    # after the first 1D IFFT.
    rowwise = jnp.einsum("ij,j...->i...", idft_matrix, ft)[:2*(ell_max + 1)]
    num_elements = ft.shape[0] * ft.shape[1]
    idft_matrix = jnp.fft.ifft(jnp.eye(ft.shape[1]))
    return jnp.einsum("ij,kj...->ki...", idft_matrix, rowwise) * num_elements


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


def _fourier_transform_2d(x):
  """Compute the 2D Fourier Transform for the first two dimensions."""
  # For the dimensions we typically have here, computing the naive
  # FT by multiplying by the DFT matrix is significantly faster than
  # the FFT (at least on TPU).
  dft_matrix = jnp.fft.fft(jnp.eye(x.shape[0]))
  rowwise = jnp.einsum("ij,j...->i...", dft_matrix, x)
  dft_matrix = jnp.fft.fft(jnp.eye(x.shape[1]))
  return jnp.einsum("ij,kj...->ki...", dft_matrix, rowwise)
