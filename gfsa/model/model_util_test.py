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

"""Tests for gfsa.model.model_util."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from gfsa.model import model_util


class LossUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "min",
          "minval": 1,
          "maxval": None,
          "expected": [1., 1., 2., 3., 4.],
      }, {
          "testcase_name": "max",
          "minval": None,
          "maxval": 3,
          "expected": [0., 1., 2., 3., 3.],
      }, {
          "testcase_name": "both",
          "minval": 1,
          "maxval": 3,
          "expected": [1., 1., 2., 3., 3.],
      })
  def test_forward_clip(self, minval, maxval, expected):
    vals, tangents = jax.jvp(
        functools.partial(
            model_util.forward_clip, minval=minval, maxval=maxval),
        (jnp.arange(5).astype(jnp.float32),), (jnp.ones((5,)),))

    np.testing.assert_allclose(vals, expected)
    np.testing.assert_allclose(tangents, np.ones((5,)))

  def test_safe_logit(self):
    probs = jnp.array([0, 1e-20, 1e-3, 0.9, 1])
    logits = model_util.safe_logit(probs)
    self.assertTrue(np.all(np.isfinite(logits)))
    np.testing.assert_allclose(logits[1:3], jax.scipy.special.logit(probs[1:3]))

  def test_binary_logit_cross_entropy(self):
    logits = jnp.array([-10., -5., 0., 5., 10.])
    true_probs = jax.nn.sigmoid(logits)
    false_probs = jax.nn.sigmoid(-logits)
    true_nll = model_util.binary_logit_cross_entropy(logits,
                                                     jnp.ones([5], dtype=bool))
    false_nll = model_util.binary_logit_cross_entropy(
        logits, jnp.zeros([5], dtype=bool))

    np.testing.assert_allclose(true_nll, -jnp.log(true_probs), atol=1e-7)
    np.testing.assert_allclose(false_nll, -jnp.log(false_probs), atol=1e-7)

  def test_linear_cross_entropy(self):
    probs = jnp.array([0, 1e-20, 1e-3, 0.9, 1, 1, 1 - 1e-7, 1 - 1e-3, 0.1, 0])
    targets = jnp.array([True] * 5 + [False] * 5)
    losses = model_util.linear_cross_entropy(probs, targets)

    # Losses are clipped to be finite.
    self.assertTrue(np.all(np.isfinite(losses)))

    # Loss values make sense.
    np.testing.assert_allclose(
        losses[1:5], [-np.log(1e-20), -np.log(1e-3), -np.log(0.9), 0],
        atol=1e-5)
    self.assertGreater(losses[0], losses[1])
    # note: losses for false targets have especially low precision due to
    # rounding errors for small values close to 1.
    np.testing.assert_allclose(losses[6], -np.log(1e-7), atol=0.2)
    np.testing.assert_allclose(
        losses[7:10], [-np.log(1e-3), -np.log(0.9), 0], atol=1e-4)
    self.assertGreater(losses[5], losses[6])

    # Gradients are finite.
    gradients = jax.grad(
        lambda x: jnp.sum(model_util.linear_cross_entropy(x, targets)))(
            probs)
    self.assertTrue(np.all(np.isfinite(gradients)))


if __name__ == "__main__":
  absltest.main()
