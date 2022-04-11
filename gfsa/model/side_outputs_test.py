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

"""Tests for gfsa.model.side_outputs."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as np
from gfsa.model import side_outputs


@flax.deprecated.nn.module
def simple_add_model(a, b):
  side_outputs.SideOutput(a, name="a")
  side_outputs.SideOutput(b, name="b")
  return a + b


@flax.deprecated.nn.module
def encourage_discrete_model(logits, **kwargs):
  return side_outputs.encourage_discrete_logits(logits, name="foo", **kwargs)


class SideOutputsTest(parameterized.TestCase):

  def test_collect_side_outputs(self):
    with side_outputs.collect_side_outputs() as penalties:
      _ = simple_add_model.init(jax.random.PRNGKey(0), 3, 4)

    self.assertEqual(penalties, {
        "/a": 3,
        "/b": 4,
    })

  def test_side_outputs_no_op(self):
    out, _ = simple_add_model.init(jax.random.PRNGKey(0), 3, 4)
    self.assertEqual(out, 7)

  @parameterized.parameters("binary", "categorical")
  def test_encourage_discrete_logits(self, distribution_type):
    if distribution_type == "binary":
      logits = {
          "a": jnp.array([[0., 1.], [2., 3.]]),
          "b": jnp.array(4.),
      }
      p = jax.nn.sigmoid(jnp.arange(5, dtype=jnp.float32))
      expected_entropy = -jnp.mean(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))
    elif distribution_type == "categorical":
      logits = {
          "a": jnp.array([[0., 1.], [2., 3.]]),
          "b": jnp.array([4., 5.]),
      }
      p = jax.nn.softmax(jnp.arange(6, dtype=jnp.float32).reshape([3, 2]))
      expected_entropy = -jnp.mean(jnp.sum(p * jnp.log(p), axis=-1))

    # Penalty only.
    with side_outputs.collect_side_outputs() as penalties:
      out, _ = encourage_discrete_model.init(
          jax.random.PRNGKey(0),
          logits,
          distribution_type=distribution_type,
          regularize=True,
          perturb_scale=None)

    self.assertTrue(jnp.all(out["a"] == logits["a"]))
    self.assertTrue(jnp.all(out["b"] == logits["b"]))
    np.testing.assert_allclose(
        penalties["/foo_entropy"], expected_entropy, rtol=1e-6)

    # Perturbed only.
    with flax.deprecated.nn.stochastic(jax.random.PRNGKey(0)):
      out, _ = encourage_discrete_model.init(
          jax.random.PRNGKey(0),
          logits,
          distribution_type=distribution_type,
          regularize=False,
          perturb_scale=1)

    # Should be modified.
    self.assertTrue(jnp.all(out["a"] != logits["a"]))
    self.assertTrue(jnp.all(out["b"] != logits["b"]))


if __name__ == "__main__":
  absltest.main()
