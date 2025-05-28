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

"""Tests for random_sequential."""

import random

from jax import numpy as jnp

from absl.testing import absltest as test
from abstract_nas.abstract.depth import DepthProperty
from abstract_nas.abstract.linear import LinopProperty
from abstract_nas.abstract.shape import ShapeProperty
from abstract_nas.evolution.mutator.random_sequential import RandomSequentialMutator
from abstract_nas.zoo import cnn


class RandomSequentialTest(test.TestCase):

  def test_construct(self):
    random.seed(0)
    for _ in range(20):
      graph, constants, _ = cnn.CifarNet()
      mutator = RandomSequentialMutator(
          [ShapeProperty(), DepthProperty(), LinopProperty()], p=0.5, max_len=3)
      mutated = mutator.mutate(
          graph, [(constants, None, {"input": jnp.ones((1, 32, 32, 3))})],
          abstract=True)
      self.assertLen(mutated, 1)


if __name__ == "__main__":
  test.main()
