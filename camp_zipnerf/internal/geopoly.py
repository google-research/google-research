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

"""Tools for constructing geodesic polyhedron, which are used as a basis."""

import itertools
import numpy as np


def compute_sq_dist(mat0, mat1=None):
  """Compute the squared Euclidean distance between all pairs of columns."""
  if mat1 is None:
    mat1 = mat0
  # Use the fact that ||x - y||^2 == ||x||^2 + ||y||^2 - 2 x^T y.
  sq_norm0 = np.sum(mat0**2, 0)
  sq_norm1 = np.sum(mat1**2, 0)
  sq_dist = sq_norm0[:, None] + sq_norm1[None, :] - 2 * mat0.T @ mat1
  sq_dist = np.maximum(0, sq_dist)  # Negative values must be numerical errors.
  return sq_dist


def compute_tesselation_weights(v):
  """Tesselate the vertices of a triangle by a factor of `v`."""
  if v < 1:
    raise ValueError(f'v {v} must be >= 1')
  int_weights = []
  for i in range(v + 1):
    for j in range(v + 1 - i):
      int_weights.append((i, j, v - (i + j)))
  int_weights = np.array(int_weights)
  weights = int_weights / v  # Barycentric weights.
  return weights


def tesselate_geodesic(base_verts, base_faces, v, eps=1e-4):
  """Tesselate the vertices of a geodesic polyhedron.

  Args:
    base_verts: tensor of floats, the vertex coordinates of the geodesic.
    base_faces: tensor of ints, the indices of the vertices of base_verts that
      constitute eachface of the polyhedra.
    v: int, the factor of the tesselation (v==1 is a no-op).
    eps: float, a small value used to determine if two vertices are the same.

  Returns:
    verts: a tensor of floats, the coordinates of the tesselated vertices.
  """
  if not isinstance(v, int):
    raise ValueError(f'v {v} must an integer')
  tri_weights = compute_tesselation_weights(v)

  verts = []
  for base_face in base_faces:
    new_verts = np.matmul(tri_weights, base_verts[base_face, :])
    new_verts /= np.sqrt(np.sum(new_verts**2, 1, keepdims=True))
    verts.append(new_verts)
  verts = np.concatenate(verts, 0)

  sq_dist = compute_sq_dist(verts.T)
  assignment = np.array([np.min(np.argwhere(d <= eps)) for d in sq_dist])
  unique = np.unique(assignment)
  verts = verts[unique, :]

  return verts


def generate_basis(
    base_shape, angular_tesselation, remove_symmetries=True, eps=1e-4
):
  """Generates a 3D basis by tesselating a geometric polyhedron.

  Args:
    base_shape: string, the name of the starting polyhedron, must be either
      'tetrahedron', 'icosahedron' or 'octahedron'.
    angular_tesselation: int, the number of times to tesselate the polyhedron,
      must be >= 1 (a value of 1 is a no-op to the polyhedron).
    remove_symmetries: bool, if True then remove the symmetric basis columns,
      which is usually a good idea because otherwise projections onto the basis
      will have redundant negative copies of each other.
    eps: float, a small number used to determine symmetries.

  Returns:
    basis: a matrix with shape [3, n].
  """

  if base_shape == 'tetrahedron':
    verts = np.array([
        (np.sqrt(8 / 9), 0, -1 / 3),
        (-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3),
        (-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3),
        (0, 0, 1),
    ])
    faces = np.array([(0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)])
  elif base_shape == 'icosahedron':
    a = (np.sqrt(5) + 1) / 2
    verts = np.array([
        (-1, 0, a),
        (1, 0, a),
        (-1, 0, -a),
        (1, 0, -a),
        (0, a, 1),
        (0, a, -1),
        (0, -a, 1),
        (0, -a, -1),
        (a, 1, 0),
        (-a, 1, 0),
        (a, -1, 0),
        (-a, -1, 0),
    ]) / np.sqrt(a + 2)
    faces = np.array([
        (0, 4, 1),
        (0, 9, 4),
        (9, 5, 4),
        (4, 5, 8),
        (4, 8, 1),
        (8, 10, 1),
        (8, 3, 10),
        (5, 3, 8),
        (5, 2, 3),
        (2, 7, 3),
        (7, 10, 3),
        (7, 6, 10),
        (7, 11, 6),
        (11, 0, 6),
        (0, 1, 6),
        (6, 1, 10),
        (9, 0, 11),
        (9, 11, 2),
        (9, 2, 5),
        (7, 2, 11),
    ])
  elif base_shape == 'octahedron':
    verts = np.array(
        [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
    )
    corners = np.array(list(itertools.product([-1, 1], repeat=3)))
    pairs = np.argwhere(compute_sq_dist(corners.T, verts.T) == 2)
    faces = np.sort(np.reshape(pairs[:, 1], [3, -1]).T, 1)
  else:
    raise ValueError(f'base_shape {base_shape} not supported')
  verts = tesselate_geodesic(verts, faces, angular_tesselation)

  if remove_symmetries:
    # Remove elements of `verts` that are reflections of each other.
    match = compute_sq_dist(verts.T, -verts.T) < eps
    verts = verts[~np.any(np.triu(match), axis=0), :]

  basis = verts[:, ::-1]
  return basis
