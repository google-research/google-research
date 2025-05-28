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

"""Tests for loss_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from internal import loss_utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


class LossUtilsTest(parameterized.TestCase):

  @parameterized.parameters((1e10,), (1e5,), (1,), (1e-5,), (1e-10,), (0.0))
  def test_eikonal_equation_gradients_are_finite(self, magnitude):
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    x = magnitude * random.normal(key, shape=(10000, 3))
    grad = jax.vmap(
        jax.grad(lambda x: jnp.sum(loss_utils.eikonal_equation(x)))
    )(x)
    np.testing.assert_equal(np.all(np.isfinite(grad)), True)

  def test_eikonal_equation(self):
    """Make sure eikonal equation is fully satisfired with normals."""
    rng = random.PRNGKey(0)
    for shape in [(45, 3), (4, 7, 3)]:
      key, rng = random.split(rng)
      vecs = random.normal(key, shape)

      loss = loss_utils.eikonal_equation(vecs)
      self.assertGreater(loss, 0.0)

      # Normalize vectors.
      normals = vecs / (jnp.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-10)
      loss_normalized = loss_utils.eikonal_equation(normals)
      np.testing.assert_allclose(loss_normalized, 0.0, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
