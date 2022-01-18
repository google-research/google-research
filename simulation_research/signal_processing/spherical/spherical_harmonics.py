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

r"""A library for computing spherical harmonics.

The spherical harmonics are special functions defined on the surface of a
sphere, which are often used to solve partial differential equations in many
scientific applications. A physical field defined on the surface of a sphere can
be written as a linear superposition of the spherical harmonics as the latter
form a complete set of orthogonal basis functions. The set of spherical
harmonics denoted `Y_l^m(θ, φ)` is often called Laplace's spherical
harmonics of degree `l` and order `m` and `θ` and `φ` are colatitude and
longitude, respectively. In addition, the spherical harmonics can be expressed
as `Y_l^m(θ, φ) = P_l^m(θ) \exp(i m φ)`, in which
`P_l^m(θ)` is the associated Legendre function with embedded normalization
constant \sqrt(1 / (4 𝛑)). We refer to the function f(θ, φ) with finite induced
norm as the signal on the sphere, where the colatitude θ ∈ [0, π] and longitude
φ ∈ [0, 2π). The signal on the sphere can be written as a linear superpostiion
of the spherical harmoincs, which form a complete set of orthonormal basis
functions for degree l ≥ 0 and order |m| ≤ l. In this library, θ and φ can be
non-uniformly sampled.
"""

import jax.numpy as jnp
import numpy as np

from simulation_research.signal_processing.spherical import associated_legendre_function


class SphericalHarmonics(object):
  """Computes the spherical harmonics on TPUs."""

  def __init__(self,
               l_max,
               theta,
               phi):
    """Constructor.

    Args:
      l_max: The maximum degree of the associated Legendre function. The degrees
        are `[0, 1, 2, ..., l_max]`. The orders `m` are `[-l_max, -l_max+1,
        0, 1, ..., l_max]`.
      theta: A vector containing the sampling points along the colatitude
        dimension. The associated Legendre functions are computed at
        `cos(θ)`.
      phi: A vector containing the sampling points along the longitude, at which
        the Vandermonde matrix is computed.
    """
    self.l_max = l_max
    self.theta = theta
    self._cos_theta = jnp.cos(theta)
    self.phi = phi
    self._legendre = associated_legendre_function.gen_normalized_legendre(
        self.l_max, self._cos_theta)
    self._vandermonde = self._gen_vandermonde_mat(self.l_max, self.phi)

  def _gen_vandermonde_mat(self, l_max, phi):
    """Generates the Vandermonde matrix exp(i m φ).

    The Vandermonde matrix has the first dimension along the degrees of the
    spherical harmonics and the second dimension along the longitude.

    Args:
      l_max: See `init`.
      phi: See `init`.

    Returns:
      A complex matrix.
    """
    nonnegative_degrees = jnp.arange(l_max+1)
    mat_dim0, mat_dim1 = jnp.meshgrid(nonnegative_degrees, phi, indexing='ij')
    num_phi = phi.shape[0]

    def vandermonde_fn(mat_dim0, mat_dim1, num_pts):
      coeff = 1j / num_pts
      return jnp.exp(coeff * jnp.multiply(mat_dim0, mat_dim1))

    return vandermonde_fn(mat_dim0, mat_dim1, num_phi)

  def harmonics_nonnegative_order(self):
    """Computes the spherical harmonics of nonnegative orders.

    Returns:
      A 4D complex tensor of shape `(l_max + 1, l_max + 1, num_theta, num_phi)`,
      where the dimensions are in the sequence of degree, order, colatitude, and
      longitude.
    """
    return jnp.einsum('ijk,jl->ijkl', self._legendre, self._vandermonde)

  def _gen_mask(self):
    """Generates the mask of (-1)^m, m = [0, 1, ..., l_max]."""
    mask = np.empty((self.l_max + 1,))
    mask[::2] = 1
    mask[1::2] = -1
    return jnp.asarray((mask))

  def harmonics_nonpositive_order(
      self, harmonics_nonnegative_order = None):
    """Computes the spherical harmonics of nonpositive orders.

    With normalization, the nonnegative order Associated Legendre functions are
    `P_l^{-m}(x) = (−1)^m P_l^m(x)`, which implies that
    `Y_l^{-m}(θ, φ) = (−1)^m conjugate(Y_l^m(θ, φ))`.

    Args:
      harmonics_nonnegative_order: A 4D complex tensor representing the
        harmonics of nonnegative orders, the shape of which is
        `(l_max + 1, l_max + 1, num_theta, num_phi)` andd the dimensions are in
        the sequence of degree, order, colatitude, and longitude.
    Returns:
      A 4D complex tensor of the same shape as `harmonics_nonnegative_order`
      representing the harmonics of nonpositive orders.
    """
    if harmonics_nonnegative_order is None:
      harmonics_nonnegative_order = self.harmonics_nonnegative_order()

    mask = self._gen_mask()
    return jnp.einsum(
        'j,ijkl->ijkl', mask, jnp.conjugate(harmonics_nonnegative_order))

  @property
  def associated_legendre_fn(self):
    """Associated Legendre function values.

    Returns:
      A 3D tensor of shape `(l_max + 1, l_max + 1, num_theta)` containing the
      values of the associated Legendre functions, the dimensions of which is in
      the sequence of degree, order, and colatitude.
    """
    return self._legendre
