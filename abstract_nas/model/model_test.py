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

"""Tests for model."""

from clu import parameter_overview
import flax
from jax import random
import jax.numpy as jnp

from absl.testing import absltest as test
from abstract_nas.model import Model
from abstract_nas.zoo import cnn


class ModelTest(test.TestCase):

  def setUp(self):
    super().setUp()

    graph, constants, _ = cnn.CifarNet()
    self.cnn = Model(graph, constants)
    self.cnn_state = self.cnn.init(random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))

  def test_cnn_inference(self):
    y = self.cnn.apply(self.cnn_state, jnp.ones((10, 32, 32, 3)))
    self.assertEqual(y.shape, (10, 10))

  def test_cnn_inference_dict(self):
    out = self.cnn.apply(self.cnn_state, {"input": jnp.ones((10, 32, 32, 3))})
    logits = out["fc/logits"]
    self.assertEqual(logits.shape, (10, 10))

  def test_cnn_params(self):
    params = flax.core.unfreeze(self.cnn_state)["params"]
    param_count = parameter_overview.count_parameters(params)
    self.assertEqual(param_count, 2192458)


if __name__ == "__main__":
  test.main()
