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

"""Tests for special2."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from scaling_transformer_inference_efficiency import special2


class Special2Test(absltest.TestCase):

  def test_softmax2(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (8,), jnp.float32)
    expected = jax.nn.softmax(x)
    actual = special2.softmax2(x * special2.LOG2_E)
    np.testing.assert_allclose(expected, actual, rtol=1e-6)

  def test_logsumexp2(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8), jnp.float32)
    expected = jax.scipy.special.logsumexp(x, axis=-1)
    actual = special2.logsumexp2(x * special2.LOG2_E) * special2.LN_2
    np.testing.assert_allclose(expected, actual, rtol=1e-6)

  def test_swish(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8), jnp.float32)
    expected = jax.nn.swish(x)
    actual = special2.swish2(x * 0.5)
    np.testing.assert_allclose(expected, actual, rtol=1e-6)

if __name__ == '__main__':
  absltest.main()
