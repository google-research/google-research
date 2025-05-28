# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Unit tests for geopoly."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from internal import geopoly
import jax
from jax import random
import numpy as np


def is_same_basis(x, y, tol=1e-10):
  """Check if `x` and `y` describe the exact same basis modulo sign flips."""
  match = (
      np.minimum(geopoly.compute_sq_dist(x, y), geopoly.compute_sq_dist(x, -y))
      <= tol
  )
  return (
      np.all(np.array(x.shape) == np.array(y.shape))
      and np.all(np.sum(match, axis=0) == 1)
      and np.all(np.sum(match, axis=1) == 1)
  )


def spans_same_basis(x, y, tol=1e-10):
  """Check if `x` and `y` span the same linear basis."""
  x /= np.sqrt(np.sum(x**2, axis=0, keepdims=True))
  y /= np.sqrt(np.sum(y**2, axis=0, keepdims=True))
  match = (
      np.minimum(geopoly.compute_sq_dist(x, y), geopoly.compute_sq_dist(x, -y))
      <= tol
  )
  return np.all(np.any(match, axis=0)) and np.all(np.any(match, axis=1))


class GeopolyTest(parameterized.TestCase):

  def test_compute_sq_dist_reference(self):
    """Test against a simple reimplementation of compute_sq_dist."""
    num_points = 100
    num_dims = 10
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    mat0 = jax.random.normal(key, [num_dims, num_points])
    key, rng = random.split(rng)
    mat1 = jax.random.normal(key, [num_dims, num_points])

    sq_dist = geopoly.compute_sq_dist(mat0, mat1)

    sq_dist_ref = np.zeros([num_points, num_points])
    for i in range(num_points):
      for j in range(num_points):
        sq_dist_ref[i, j] = np.sum((mat0[:, i] - mat1[:, j]) ** 2)

    np.testing.assert_allclose(sq_dist, sq_dist_ref, atol=1e-5, rtol=1e-5)

  def test_compute_sq_dist_single_input(self):
    """Test that compute_sq_dist with a single input works correctly."""
    rng = random.PRNGKey(0)
    num_points = 100
    num_dims = 10
    key, rng = random.split(rng)
    mat0 = jax.random.normal(key, [num_dims, num_points])

    sq_dist = geopoly.compute_sq_dist(mat0)
    sq_dist_ref = geopoly.compute_sq_dist(mat0, mat0)
    np.testing.assert_allclose(sq_dist, sq_dist_ref)

  def test_compute_tesselation_weights_reference(self):
    """A reference implementation for triangle tesselation."""
    for v in range(1, 10):
      w = geopoly.compute_tesselation_weights(v)
      perm = np.array(list(itertools.product(range(v + 1), repeat=3)))
      w_ref = perm[np.sum(perm, axis=-1) == v, :] / v
      # Check that all rows of x are close to some row in x_ref.
      self.assertTrue(is_same_basis(w.T, w_ref.T))

  @parameterized.parameters(
      ('icosahedron', 1),
      ('icosahedron', 2),
      ('octahedron', 1),
      ('octahedron', 2),
      ('octahedron', 3),
  )
  def test_generate_basis_symmetries_removed(self, mode, tess):
    basis_full = geopoly.generate_basis(mode, tess, remove_symmetries=False)
    basis_symm = geopoly.generate_basis(mode, tess, remove_symmetries=True)
    self.assertTrue(spans_same_basis(basis_full.T, basis_symm.T))

  def test_generate_basis_golden(self):
    """A mediocre golden test against some arbitrary basis choices."""

    basis = geopoly.generate_basis('tetrahedron', 1)
    basis_golden = np.array([
        [-0.33333333, -0.81649658, -0.47140452],
        [-0.33333333, 0.81649658, -0.47140452],
        [-0.33333333, 0.00000000, 0.94280904],
        [1.00000000, 0.00000000, 0.00000000],
    ])
    self.assertTrue(is_same_basis(basis.T, basis_golden.T))

    basis = geopoly.generate_basis('tetrahedron', 2)
    basis_golden = np.array([
        [-0.33333333, -0.81649658, -0.47140452],
        [-0.57735027, 0.00000000, -0.81649658],
        [-0.33333333, 0.81649658, -0.47140452],
        [-0.57735027, -0.70710678, 0.40824829],
        [-0.57735027, 0.70710678, 0.40824829],
        [-0.33333333, 0.00000000, 0.94280904],
        [1.00000000, 0.00000000, 0.00000000],
    ])
    self.assertTrue(is_same_basis(basis.T, basis_golden.T))

    basis = geopoly.generate_basis('icosahedron', 2)
    basis_golden = np.array([
        [0.85065081, 0.00000000, 0.52573111],
        [0.80901699, 0.50000000, 0.30901699],
        [0.52573111, 0.85065081, 0.00000000],
        [1.00000000, 0.00000000, 0.00000000],
        [0.80901699, 0.50000000, -0.30901699],
        [0.85065081, 0.00000000, -0.52573111],
        [0.30901699, 0.80901699, -0.50000000],
        [0.00000000, 0.52573111, -0.85065081],
        [0.50000000, 0.30901699, -0.80901699],
        [0.00000000, 1.00000000, 0.00000000],
        [-0.52573111, 0.85065081, 0.00000000],
        [-0.30901699, 0.80901699, -0.50000000],
        [0.00000000, 0.52573111, 0.85065081],
        [-0.30901699, 0.80901699, 0.50000000],
        [0.30901699, 0.80901699, 0.50000000],
        [0.50000000, 0.30901699, 0.80901699],
        [0.50000000, -0.30901699, 0.80901699],
        [0.00000000, 0.00000000, 1.00000000],
        [-0.50000000, 0.30901699, 0.80901699],
        [-0.80901699, 0.50000000, 0.30901699],
        [-0.80901699, 0.50000000, -0.30901699],
    ])
    self.assertTrue(is_same_basis(basis.T, basis_golden.T))

    basis = geopoly.generate_basis('octahedron', 4)
    basis_golden = np.array([
        [0.00000000, 0.00000000, -1.00000000],
        [0.00000000, -0.31622777, -0.94868330],
        [0.00000000, -0.70710678, -0.70710678],
        [0.00000000, -0.94868330, -0.31622777],
        [0.00000000, -1.00000000, 0.00000000],
        [-0.31622777, 0.00000000, -0.94868330],
        [-0.40824829, -0.40824829, -0.81649658],
        [-0.40824829, -0.81649658, -0.40824829],
        [-0.31622777, -0.94868330, 0.00000000],
        [-0.70710678, 0.00000000, -0.70710678],
        [-0.81649658, -0.40824829, -0.40824829],
        [-0.70710678, -0.70710678, 0.00000000],
        [-0.94868330, 0.00000000, -0.31622777],
        [-0.94868330, -0.31622777, 0.00000000],
        [-1.00000000, 0.00000000, 0.00000000],
        [0.00000000, -0.31622777, 0.94868330],
        [0.00000000, -0.70710678, 0.70710678],
        [0.00000000, -0.94868330, 0.31622777],
        [0.40824829, -0.40824829, 0.81649658],
        [0.40824829, -0.81649658, 0.40824829],
        [0.31622777, -0.94868330, 0.00000000],
        [0.81649658, -0.40824829, 0.40824829],
        [0.70710678, -0.70710678, 0.00000000],
        [0.94868330, -0.31622777, 0.00000000],
        [0.31622777, 0.00000000, -0.94868330],
        [0.40824829, 0.40824829, -0.81649658],
        [0.40824829, 0.81649658, -0.40824829],
        [0.70710678, 0.00000000, -0.70710678],
        [0.81649658, 0.40824829, -0.40824829],
        [0.94868330, 0.00000000, -0.31622777],
        [0.40824829, -0.40824829, -0.81649658],
        [0.40824829, -0.81649658, -0.40824829],
        [0.81649658, -0.40824829, -0.40824829],
    ])
    self.assertTrue(is_same_basis(basis.T, basis_golden.T))


if __name__ == '__main__':
  absltest.main()
