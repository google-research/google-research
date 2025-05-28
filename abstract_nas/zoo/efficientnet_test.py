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

"""Tests for efficientnet."""

from typing import Any

import jax
from jax import random
import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.model import Model
from abstract_nas.zoo import efficientnet


def compute_num_params(params_cpu):
  return sum(p.size for p in jax.tree.flatten(params_cpu)[0])


# From //learning/faster_training/sidewinder/efficientnet/efficientnet_test.py
EFFICIENTNET_PARAMS = {
    "b0": 5288548,
    "b1": 7794184,
    "b2": 9109994,
    "b3": 12233232,
    "b4": 19341616,
    "b5": 30389784,
    "b6": 43040704,
    "b7": 66347960
}


class EfficientnetTest(test.TestCase):

  def test_inference(self):
    graph, constants, _ = efficientnet.EfficientNetB0(
        num_classes=10)
    for op in graph.ops:
      print(f"name={op.name}")
      print(f"input_names={op.input_names}")
    print()
    model = Model(graph, constants)
    state = model.init(random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
    self.assertLen(state, 2)
    self.assertIn("params", state)
    self.assertIn("batch_stats", state)

    inp = {"input": jnp.ones((10, 32, 32, 3))}

    out = model.apply(state, inp)["head/out"]
    self.assertEqual(out.shape, (10, 10))

    output_dict, new_state = model.apply(state, inp, mutable=["batch_stats"])
    self.assertEqual(output_dict["head/out"].shape, (10, 10))
    self.assertIn("batch_stats", new_state)

  def test_params_b0(self):
    graph, constants, _ = efficientnet.EfficientNetB0(
        num_classes=1000)
    model = Model(graph, constants)
    state = model.init(random.PRNGKey(0), jnp.ones((1, 224, 224, 3)))
    params = state["params"]
    num_params = compute_num_params(params)
    self.assertEqual(num_params, EFFICIENTNET_PARAMS["b0"])

  def test_params_b1(self):
    graph, constants, _ = efficientnet.EfficientNetB1(
        num_classes=1000)
    model = Model(graph, constants)
    state = model.init(random.PRNGKey(0), jnp.ones((1, 240, 240, 3)))
    params = state["params"]
    num_params = compute_num_params(params)
    self.assertEqual(num_params, EFFICIENTNET_PARAMS["b1"])

if __name__ == "__main__":
  test.main()
