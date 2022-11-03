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

"""Tests for Learned Interpreters workflows."""

from absl.testing import absltest
import jax.numpy as jnp

from ipagnn.adapters import common_adapters


class CommonAdaptersTest(absltest.TestCase):

  def test_compute_weighted_cross_entropy(self):
    logits = jnp.array([
        [[.8, .2, -.5],
         [.2, .5, -.1]],
        [[.1, -.2, .2],
         [.4, -.5, .1]],
    ])
    labels = jnp.array([
        [0, 1],
        [2, 2],
    ])
    common_adapters.compute_weighted_cross_entropy(logits, labels)


if __name__ == '__main__':
  absltest.main()
