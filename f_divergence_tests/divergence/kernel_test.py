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

"""Tests for functions to compute kernel distances."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import tensorflow as tf
from f_divergence_tests.divergence import kernel


class KernelTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.x = jnp.array([[1, 2], [3, 5]])
    self.y = jnp.array([[2, 1], [4, 7], [0, -2]])
    self.distances = {
        "l1": jnp.array([
            [2, 8, 5],
            [5, 3, 10],
        ]),
        "l2": jnp.array([
            [jnp.sqrt(2), jnp.sqrt(34), jnp.sqrt(17)],
            [jnp.sqrt(17), jnp.sqrt(5), jnp.sqrt(58)],
        ]),
    }

  @parameterized.named_parameters(
      dict(
          testcase_name="l1_sequential",
          norm="l1",
          min_memory=True,
      ),
      dict(
          testcase_name="l2_sequential",
          norm="l2",
          min_memory=True,
      ),
      dict(
          testcase_name="l1_vectorized",
          norm="l1",
          min_memory=False,
      ),
      dict(
          testcase_name="l2_vectorized",
          norm="l2",
          min_memory=False,
      ),
  )
  def test_get_distances_returns_correct_values(self, norm, min_memory):
    distances = kernel.get_distances(
        self.x, self.y, norm=norm, min_memory=min_memory
    )
    self.assertAllClose(distances, self.distances[norm], atol=1e-6)

  def test_get_distances_returns_correct_shape(self):
    distances = kernel.get_distances(self.x, self.y, norm="l1", min_memory=True)
    self.assertEqual(distances.shape, (self.x.shape[0], self.y.shape[0]))


if __name__ == "__main__":
  absltest.main()
