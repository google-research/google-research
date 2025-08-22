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

"""Tests for model_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from d3pm.text import model_utils


class ModelUtilsTest(parameterized.TestCase):

  def test_positional_embeddings(self):
    seq = jnp.ones((8, 32, 32, 128))
    embeddings = model_utils.get_timestep_embedding(np.array([0.5] * 8), 128)
    seq = seq + embeddings[:, None, None, :]

    self.assertEqual(seq.shape, ((8, 32, 32, 128)))

  @parameterized.parameters(True, False)
  def test_matrix_product(self, use_cache):
    """Tests the matrix_vector_product code."""

    key = jrandom.PRNGKey(0)
    dim = 50
    max_power = 25

    matrix = jrandom.normal(key, (dim, dim)) / 10
    vector = jnp.ones((dim,), dtype=jnp.float32)

    if use_cache:
      mpstate = model_utils.CachedMatrixPowerState.precompute(matrix, max_power)
    else:
      mpstate = model_utils.LazyMatrixPowerState(matrix)

    for t in range(max_power):
      result = mpstate.matrix_power_multiply(vector, t)
      expected = np.linalg.matrix_power(matrix, t) @ vector

      np.testing.assert_array_almost_equal(result, expected, decimal=1)

  @parameterized.parameters(True, False)
  def test_matrix_power(self, use_cache):
    """Tests the matrix_power_cached code."""

    key = jrandom.PRNGKey(0)
    dim = 50
    max_power = 25

    matrix = jrandom.normal(key, (dim, dim)) / 10

    if use_cache:
      mpstate = model_utils.CachedMatrixPowerState.precompute(matrix, max_power)
    else:
      mpstate = model_utils.LazyMatrixPowerState(matrix)

    for t in range(max_power):
      result = mpstate.matrix_power(t, precision=jax.lax.Precision.HIGHEST)
      expected = np.linalg.matrix_power(matrix, t)

      np.testing.assert_array_almost_equal(result, expected, decimal=1)


class NearestNeighborTest(absltest.TestCase):
  """Tests whether the nearest neighbor calculation work correctly."""

  def test_nearest_neighbor(self):
    embeddings = jnp.array([[1., 2.], [3., 4], [3., 5.], [1., -5.]])
    neighbors, distances = model_utils.get_nearest_neighbors(
        embeddings, num_chunks=2, k=2, return_distances=True)

    self.assertEqual(neighbors.shape, (4, 2))
    self.assertEqual(distances.shape, (4, 2))

    np.testing.assert_array_equal(neighbors,
                                  jnp.array([[1, 2], [2, 0], [1, 0], [0, 1]]))

    np.testing.assert_array_almost_equal(
        distances,
        jnp.array([[2.828427, 3.6055512], [1., 2.828427], [1., 3.6055512],
                   [6.9999995, 9.219544]]))

  def test_nearest_neighbor_chunk(self):
    expected = jnp.array([[1, 2], [2, 0], [1, 0], [0, 1]])
    embeddings = jnp.array([[1., 2.], [3., 4], [3., 5.], [1., -5.]])

    for num_chunks in range(1, 5):
      neighbors = model_utils.get_nearest_neighbors(
          embeddings, num_chunks=num_chunks, k=2, return_distances=False)

      self.assertEqual(neighbors.shape, (4, 2))

      np.testing.assert_array_equal(neighbors, expected)


if __name__ == '__main__':
  absltest.main()

if __name__ == '__main__':
  absltest.main()
