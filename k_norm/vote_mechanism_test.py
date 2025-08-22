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

"""Tests for vote_mechanism."""

from absl.testing import absltest
import numpy as np

from k_norm import vote_mechanism


class VoteMechanismTest(absltest.TestCase):

  def test_sample_from_simplex_norm(self):
    d = 5
    num_samples = 1000
    simplex_vertices = np.asarray([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    samples = [
        vote_mechanism.sample_from_simplex(simplex_vertices)
        for _ in range(num_samples)
    ]
    tol = 1e-10
    for sample in samples:
      np.testing.assert_array_less(np.zeros(d), sample)
      self.assertLess(np.linalg.norm(sample, ord=1), 1 + tol)

  def test_external_to_internal_direct_sum(self):
    subpermutohedron_1 = [1, 4]
    subpermutohedron_2 = [2, 3, 0]
    x_and_y = [[0.1, 0.2], [0.3, 0.4, 0.5]]
    embedded = vote_mechanism.external_to_internal_direct_sum(
        subpermutohedron_1, subpermutohedron_2, x_and_y
    )
    expected_embedded = np.asarray([0.5, 0.1, 0.3, 0.4, 0.2])
    np.testing.assert_array_equal(embedded, expected_embedded)

  def test_compute_face_class_weights(self):
    d = 4
    weights = vote_mechanism.compute_face_class_weights(d)
    # For each j in [1, 2, 3], we have
    # num_faces[j-1] = (4 choose j),
    # small_face_volumes[j-1] = j^(j - 3/2) * (4 - j)^(4 - j - 3/2),
    # pyramid_heights[j-1] = sqrt(j * (4 - j)^2 + j^2 * (4 - j)) / 2, and
    # face_class_weights[j-1] = product of all three.
    # Note that the 0 index of each is 0.
    num_faces = np.asarray([0, 4, 6, 4])
    small_face_volumes = np.asarray([0, 3**1.5, 2, 3**1.5])
    pyramid_heights = np.asarray([0, 12**0.5 / 2, 2, 12**0.5 / 2])
    face_class_weights = num_faces * small_face_volumes * pyramid_heights
    normalized_face_class_weights = face_class_weights / np.sum(
        face_class_weights
    )
    np.testing.assert_array_almost_equal(weights, normalized_face_class_weights)

  def test_sample_simplex_from_star_decomposition_dimensions(self):
    d = 5
    simplex = vote_mechanism.sample_simplex_from_star_decomposition(d)
    self.assertLen(simplex, 5)
    self.assertLen(simplex[0], 5)

  def test_sample_simplices_and_embedding_simplex_dimensions(self):
    face_class_weights = [0, 0.1, 0.2, 0.7]
    simplex_1, simplex_2, _, _ = vote_mechanism.sample_simplices_and_embedding(
        face_class_weights
    )
    self.assertEqual(len(simplex_1) + len(simplex_2), 4)

  def test_sample_simplices_and_embedding_subpermutohedra_dimensions(self):
    face_class_weights = [0, 0.1, 0.2, 0.7]
    _, _, subpermutohedron_1, subpermutohedron_2 = (
        vote_mechanism.sample_simplices_and_embedding(face_class_weights)
    )
    self.assertEqual(len(subpermutohedron_1) + len(subpermutohedron_2), 4)

  def test_sample_simplex_3_dimensions(self):
    # d = 5, j = 3
    simplex_1 = [[4, 2, 2], [4, 2.5, 2.5], [3, 3, 3]]
    simplex_2 = [[1, 0], [0.5, 0.5]]
    subpermutohedron_1 = [1, 3, 4]
    subpermutohedron_2 = [2, 0]
    simplex_3 = vote_mechanism.sample_simplex_3(
        simplex_1, simplex_2, subpermutohedron_1, subpermutohedron_2
    )
    self.assertLen(simplex_3, 4)
    self.assertLen(simplex_3[0], 5)

  def test_sample_simplex_3_volume(self):
    # d = 5, j = 3
    subpermutohedron_1 = [1, 3, 4]
    subpermutohedron_2 = [2, 0]
    simplex_1 = [[4, 2, 2], [4, 2.5, 2.5], [3, 3, 3]]
    simplex_2 = [[1, 0], [0.5, 0.5]]
    simplex_3 = np.asarray(vote_mechanism.sample_simplex_3(
        simplex_1, simplex_2, subpermutohedron_1, subpermutohedron_2
    ))
    # The volume of a rank-4 shape in R^5 is proportional to the determinant of
    # its Gram matrix. The Gram matrix M of a matrix with rows (v_1, ..., v_n)
    # is given by M_{ij} = <v_i, v_j>. We subtract v_1 because we want the
    # volume of the convex hull of (v_1, ..., v_n).
    simplex_3 = (simplex_3 - simplex_3[0])[1:]
    gram_matrix_simplex_3 = simplex_3.dot(simplex_3.T)
    volume = np.linalg.det(gram_matrix_simplex_3)
    num_trials = 1000
    for _ in range(num_trials):
      new_simplex_3 = np.asarray(vote_mechanism.sample_simplex_3(
          simplex_1, simplex_2, subpermutohedron_1, subpermutohedron_2
      ))
      new_simplex_3 = (new_simplex_3 - new_simplex_3[0])[1:]
      gram_matrix_new_simplex_3 = new_simplex_3.dot(new_simplex_3.T)
      new_volume = np.linalg.det(gram_matrix_new_simplex_3)
      self.assertAlmostEqual(volume, new_volume)

  def test_sample_vote_ball_dimension(self):
    d = 5
    cylinder_sample = vote_mechanism.sample_vote_ball(d)
    self.assertLen(cylinder_sample, 5)

  def test_sample_vote_ball_constraints(self):
    d = 3
    # We are sampling a cylinder defined by 8 hyperplane constraints consisting
    # of 4 symmetric pairs:
    # 1) { (2, 0, 1), (1, 0, 2), (0, 1, 2)}
    #  = {(0, 0, 0), (-1, 0, 1), (-2, 1, 1)} + (2, 0, 1). Then an orthogonal
    # vector z has z_1 = z_2 = z_3, and we use z = (1, 1, 1).
    # <(1, 1, 1), (2, 0, 1)> = 3, so the constraint is
    # {x | <(1, 1, 1), x> <= 3}. The symmetric constraint is
    # {x | <(1, 1, 1), x> >= -3}.
    # 2) {(1, 0, 2), (0, 1, 2), (-2, -1, 0)}
    #  = {(0, 0, 0), (-1, 1, 0), (-3, -1, -2)} + (1, 0, 2). Then an orthogonal
    # vector z has z_1 = z_2 = -z_3 / 2, and we use z = (1, 1, -2).
    # <(1, 1, -2), (1, 0, 2)> = -3, so the constraint is
    # {x | <(1, 1, -2), x> >= -3}. The symmetric constraint is
    # {x | <(1, 1, -2), x> <= 3}.
    # 3) {(0, 1, 2), (0, 2, 1), (-2, 0, -1)}
    # = {(0, 0, 0), (0, 1, -1), (-2, -1, -3)} + (0, 1, 2). Then an orthogonal
    # vector z has z_1 / 2 = -z_2 = -z_3, and we use z = (1, -1/2, -1/2).
    # <(1, -1/2, -1/2), (0, 1, 2)> = -3/2, so the constraint is
    # {x | <(1, -1/2, -1/2), x> >= -3/2}. The symmetric constraint is
    # {x | <(1, -1/2, -1/2), x> <= 3/2}.
    # 4) {(0, 2, 1), (1, 2, 0), (-1, 0, -2)}
    # = {(0, 0, 0), {(1, 0, -1), (-1, -2, -3)} + (0, 2, 1). Then an orthogonal
    # vector z has z_1 = z_3 = -z_2 / 2, and we use z = (1, -2, 1).
    # <(1, -2, 1), (0, 2, 1)> = -3, so the constraint is
    # {x | <(1, -2, 1), x> >= -3}. The symmetric constraint is
    # {x | <(1, -2, 1), x> <= 3}.
    num_samples = 10000
    samples = [vote_mechanism.sample_vote_ball(d) for _ in range(num_samples)]
    self.assertLess((np.max(np.dot(samples, [1, 1, 1]))), 3)
    self.assertLess(-3, np.min(np.dot(samples, [1, 1, 1])))
    self.assertLess((np.max(np.dot(samples, [1, 1, -2]))), 3)
    self.assertLess(-3, np.min(np.dot(samples, [1, 1, -2])))
    self.assertLess((np.max(np.dot(samples, [1, -1/2, -1/2]))), 3/2)
    self.assertLess(-3/2, np.min(np.dot(samples, [1, -1/2, -1/2])))
    self.assertLess((np.max(np.dot(samples, [1, -2, 1]))), 3)
    self.assertLess(-3, np.min(np.dot(samples, [1, -2, 1])))

if __name__ == '__main__':
  absltest.main()
