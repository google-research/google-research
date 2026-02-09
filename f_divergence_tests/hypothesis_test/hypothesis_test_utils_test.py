# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import tensorflow as tf
from f_divergence_tests.hypothesis_test import hypothesis_test_utils


class HypothesisTestUtilsTest(tf.test.TestCase):

  def test_get_permutations_returns_correct_values_and_shapes(self):
    key = jax.random.PRNGKey(0)
    num_permutations = 2
    m = 3
    n = 4
    true_assignments = jnp.concatenate((jnp.ones(m), -jnp.ones(n)))

    permuted_assignments, sorted_indices = (
        hypothesis_test_utils.get_permutations(
            key=key, num_permutations=num_permutations, m=m, n=n
        )
    )
    with self.subTest("returns_correct_shape"):
      self.assertAllEqual(
          permuted_assignments.shape, (num_permutations + 1, m + n)
      )
      self.assertAllEqual(sorted_indices.shape, (num_permutations + 1, m + n))

    with self.subTest("permutes_correctly"):
      # First two rows should be permuted
      self.assertNotAllEqual(true_assignments, permuted_assignments[0])
      self.assertNotAllEqual(true_assignments, permuted_assignments[1])
      self.assertNotAllEqual(list(range(m + n)), sorted_indices[0])
      self.assertNotAllEqual(list(range(m + n)), sorted_indices[1])

    # Last row should be the original order
    with self.subTest("returns_original_order_last"):
      self.assertAllEqual(true_assignments, permuted_assignments[-1])
      self.assertAllEqual(list(range(m + n)), sorted_indices[-1])


if __name__ == "__main__":
  absltest.main()
