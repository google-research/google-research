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

"""Tests for resnetv1."""

from jax import random
import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.model import Model
from abstract_nas.zoo import resnetv1


class Resnetv1Test(test.TestCase):

  def test_inference(self):
    graph, constants, _ = resnetv1.ResNet18(
        num_classes=10, input_resolution="small")
    model = Model(graph, constants)
    state = model.init(random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
    self.assertLen(state, 2)
    self.assertIn("params", state)
    self.assertIn("batch_stats", state)

    out = model.apply(state, {"input": jnp.ones((10, 32, 32, 3))})["fc/dense"]
    self.assertEqual(out.shape, (10, 10))

    output_dict, new_state = model.apply(
        state, {"input": jnp.ones((10, 32, 32, 3))}, mutable=["batch_stats"])
    self.assertEqual(output_dict["fc/dense"].shape, (10, 10))
    self.assertIn("batch_stats", new_state)


if __name__ == "__main__":
  test.main()
