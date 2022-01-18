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

"""Library of computing associated Legendre function of the first kind."""

import math

from jax import lax
import jax.numpy as jnp
import numpy as np


def gen_normalized_legendre(l_max,
                            x):
  r"""Computes the normalized associated Legendre functions (ALFs).

  The ALFs of the first kind are used in spherical harmonics. The spherical
  harmonic of degree `l` and order `m` can be written as
  `Y_l^m(θ, φ) = N_l^m * P_l^m(cos(θ)) * exp(i m φ)`, where `N_l^m` is the
  normalization factor and θ and φ are the colatitude and longitude,
  repectively. `N_l^m` is chosen in the way that the spherical harmonics form
  a set of orthonormal basis function of L^2(S^2). For the computational
  efficiency of spherical harmonics transform, the normalization factor is
  embedded into the computation of the ALFs. In addition, normalizing `P_l^m`
  avoids overflow/underflow and achieves better numerical stability. Three
  recurrence relations are used in the computation. Note that the factor of
  \sqrt(1 / (4 𝛑)) is used in the formulation.

  Args:
    l_max: The maximum degree of the associated Legendre function. Both the
      degrees and orders are `[0, 1, 2, ..., l_max]`.
    x: A vector of type `float32`, `float64` containing the sampled points in
      spherical coordinates, at which the ALFs are computed; `x` is essentially
      `cos(θ)`.

  Returns:
    The 3D array of shape `(l_max + 1, l_max + 1, len(x))` containing the
    normalized values of the ALFs at `x`.
  """
  dtype = lax.dtype(x)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
        'x.dtype={} is not supported, see docstring for supported types.'
        .format(dtype))

  if x.ndim != 1:
    raise ValueError('x must be a 1D array.')

  p = np.zeros((l_max + 1, l_max + 1, x.shape[0]))

  # The initial value p(0,0).
  initial_value = 0.5 / np.sqrt(math.pi)
  p[0, 0] = initial_value

  # Compute the diagonal entries p(l,l) with recurrence.
  y = np.sqrt(1.0 - x * x)
  for l in range(1, l_max + 1):
    a = -1.0 * np.sqrt(1.0 + 0.5 / l)
    p[l, l] = a * y * p[l - 1, l - 1]

  # Compute the off-diagonal entries with recurrence.
  for l in range(l_max):
    b = np.sqrt(2.0 * l + 3.0)
    p[l + 1, l] = b * x * p[l, l]

  # Compute the remaining entries with recurrence.
  for m in range(l_max + 1):
    for l in range(m + 2, l_max + 1):
      c0 = l * l
      c1 = m * m
      c2 = 2.0 * l
      c3 = (l - 1.0) * (l - 1.0)
      d0 = np.sqrt((4.0 * c0 - 1.0) / (c0 - c1))
      d1 = np.sqrt(((c2 + 1.0) * (c3 - c1)) / ((c2 - 3.0) * (c0 - c1)))
      p[l, m] = d0 * x * p[l - 1, m] - d1 * p[l - 2, m]

  return jnp.asarray(p)
