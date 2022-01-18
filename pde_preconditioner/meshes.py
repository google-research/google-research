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

# pylint: skip-file
import jax
import jax.numpy as np
from jax import random
import numpy as onp
import jax.ops
from . import multigrid
from . import equations

# defines the mesh for the experiments


class Mesh:

  def __init__(self, n):
    self.n = n
    grid = np.linspace(0, 1, num=n + 2)
    grid = grid[1:-1]
    self.x_mesh, self.y_mesh = onp.meshgrid(grid, grid, indexing='ij')
    self.bcmesh = np.linspace(0, 1, num=n)
    self.shape = (n, n)

  def bcR(self, rng=None, aspect_ratio=1.0):
    """bcR creates a random boundary condition for a rectangular domain defined

    by aspect_ratio. The boundary is a random 3rd order polynomial.
    rng variable allows to reproduce the results.
    The current boundary conditions are not periodic.
    """
    if rng is None:
      rng = random.PRNGKey(1)

    x = self.bcmesh
    n = self.n
    n_y = equations.num_row(n, aspect_ratio)
    y = np.linspace(0, 1, num=n_y)
    if rng is not None:
      coeffs = random.multivariate_normal(rng, np.zeros(16),
                                          np.diag(np.ones(16)))
    else:
      key = random.randint(random.PRNGKey(1), (1,), 1, 1000)
      coeffs = random.multivariate_normal(
          random.PRNGKey(key[0]), np.zeros(16), np.diag(np.ones(16)))
    left = coeffs[0] * y**3 + coeffs[1] * y**2 + coeffs[2] * y + coeffs[3]
    right = coeffs[4] * y**3 + coeffs[5] * y**2 + coeffs[6] * y + coeffs[7]
    lower = coeffs[8] * x**3 + coeffs[9] * x**2 + coeffs[10] * x + coeffs[11]
    upper = coeffs[12] * x**3 + coeffs[13] * x**2 + coeffs[14] * x + coeffs[15]
    shape = 2 * x.shape
    source = onp.zeros(shape)
    source[0, :] = upper
    source[n_y - 1, :] = lower
    source[0:n_y, -1] = right
    source[0:n_y, 0] = left
    # because this makes the correct order of boundary conditions
    return source * (n + 1)**2

  def bc(self, rng=None):
    """wrapper for bcR(aspect_ratio=1). Creates square boundaries."""
    return self.bcR(rng)

  def bcL(self, rng=None):
    """bcL creates a random boundary condition for a L-shaped domain.

    The boundary is a random 3rd order polynomial of sine functions.
    rng variable allows to reproduce the results. Sine functions are chosen so
    that the boundary is periodic and does not have discontinuities.
    """
    if rng is None:
      rng = random.PRNGKey(1)
    n = self.n
    x = onp.sin(self.bcmesh * np.pi)
    n_y = (np.floor((n + 1) / 2) - 1).astype(int)
    if rng is not None:
      coeffs = random.multivariate_normal(rng, np.zeros(16),
                                          np.diag(np.ones(16)))
    else:
      key = random.randint(random.PRNGKey(1), (1,), 1, 1000)
      coeffs = random.multivariate_normal(
          random.PRNGKey(key[0]), np.zeros(16), np.diag(np.ones(16)))
    left = coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x  #+ coeffs[3]
    right = coeffs[4] * x**3 + coeffs[5] * x**2 + coeffs[6] * x  #+ coeffs[7]
    lower = coeffs[8] * x**3 + coeffs[9] * x**2 + coeffs[10] * x  #+ coeffs[11]
    upper = coeffs[12] * x**3 + coeffs[13] * x**2 + coeffs[14] * x  #+ coeffs[15]
    shape = 2 * x.shape
    source = onp.zeros(shape)
    source[0, :] = upper
    source[n_y - 1, n_y - 1:] = lower[:n - n_y + 1]
    source[n_y - 1:, n_y - 1] = right[:n - n_y + 1]
    source[:, 0] = left
    source[-1, :n_y - 1] = right[n:n - n_y:-1]
    source[:n_y - 1, -1] = lower[n:n - n_y:-1]
    # because this makes the correct order of boundary conditions
    return source * (n + 1)**2

  def matvec_helmholtz(self, k, aspect_ratio, shapebc, shapebc_dual, x):
    x = equations.helmholtz(
        x.reshape(self.shape), k, 1. / (self.n + 1), aspect_ratio, shapebc,
        shapebc_dual)
    return x.ravel()

  def sine(self, j, k):
    return np.multiply(
        np.sin(k * self.y_mesh * np.pi), np.sin(j * self.x_mesh * np.pi))

  def polynomial(self, j, k, rng=None):
    if rng is None:
      rng = random.PRNGKey(1)
    coeffs = random.normal(rng, shape=(j + 1, k + 1))
    poly = random.normal(rng, shape=(self.shape))
    for xj in range(j + 1):
      for yk in range(k + 1):
        poly = poly.at[:, :].add(self.x_mesh**xj @ self.y_mesh**yk *
                                 coeffs[xj, yk])
    return poly

  def V_cycle_prec(self, x, k=0, aspect_ratio=1.0, shapebc='R'):
    return multigrid._V_Cycle(
        onp.zeros(self.shape).astype('float32'), x.reshape(self.shape), 1,
        shapebc, k, aspect_ratio).ravel()

  def V_cycle_prec_2(self, x, k=0):
    return multigrid._V_Cycle(
        onp.zeros(self.shape).astype('float32'), x.reshape(self.shape), 2,
        k).ravel()

  def V_cycle_prec_3(self, x, k=0):
    return multigrid._V_Cycle(
        onp.zeros(self.shape).astype('float32'), x.reshape(self.shape), 3,
        k).ravel()

  def V_cycle_GMRES(self, x, k=0, aspect_ratio=1.0, shapebc='R'):
    return multigrid._V_Cycle_GMRES(
        onp.zeros(self.shape).astype('float32'), x.reshape(self.shape), 1,
        shapebc, k, aspect_ratio).ravel()

  def masked_preconditioner(self, x, prec, shapebc='R', ratio=1.0):
    """Let G be the mask, M be the preconditioner. masked_preconditioner returns
    G(M(G(x)))+(I-G)(x)."""
    x = x.reshape(self.shape)
    if shapebc == 'R':
      mask_f = equations.make_mask
      mask_f_dual = equations.make_mask_dual
    elif shapebc == 'L':
      mask_f = equations.make_mask_L
      mask_f_dual = equations.make_mask_L_dual

    x_masked = np.multiply(x, mask_f(self.shape[0], ratio))
    x_dual = np.multiply(x, mask_f_dual(self.shape[0], ratio))
    x = np.multiply(
        prec(x_masked.ravel()).reshape(self.shape), mask_f(
            self.shape[0], ratio)) + x_dual
    return x.ravel()
